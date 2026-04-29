"""StructuralMetrics — FFN delta, residual cosine, LN stats, embedding norms."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import MetricModule, _safe_skew
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class StructuralMetrics(MetricModule):
    """Structural probes for FFN, LayerNorm, residual connections, embeddings."""

    def __init__(self, inspector: ModelInspector, config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.ffn_probe_tokens = cfg.ffn_probe_tokens
        self.ffn_var_activity_threshold = cfg.ffn_var_activity_threshold

    @property
    def requires_hidden_states(self) -> bool:
        return True

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        hidden_states: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        outputs: Any = None,
        sublayer_capture: Any = None,
        **kwargs,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Prefer exact sublayer-boundary captures when available; fall
        # back to adjacent-hidden-state reconstruction otherwise. The
        # exact path reads FFN input / output and per-LayerNorm output
        # straight from forward hooks installed by ``SublayerCapture``.
        use_capture = (
            sublayer_capture is not None
            and getattr(sublayer_capture, "installed", False)
            and bool(getattr(sublayer_capture, "captures", {}))
        )

        try:
            hs_list = list(hidden_states) if hidden_states is not None else []
        except Exception:
            hs_list = []

        if not use_capture and len(hs_list) < 2:
            return metrics

        eps = 1e-6
        delta_means, cos_means, var_ratios = [], [], []
        ln_std_means, ln_mean_abs_means = [], []
        active_fracs, skew_vals = [], []

        if use_capture:
            n_layers = self.inspector.num_layers or 0
        else:
            n_layers = min(self.inspector.num_layers or len(hs_list) - 1,
                           len(hs_list) - 1)
        probe_tokens = self.ffn_probe_tokens

        for layer_idx in range(n_layers):
            h_in, h_out, ln_out = self._select_tap_tensors(
                layer_idx=layer_idx,
                use_capture=use_capture,
                sublayer_capture=sublayer_capture,
                hs_list=hs_list,
            )
            if h_in is None or h_out is None:
                continue

            flat_in = h_in.reshape(-1, h_in.size(-1))
            flat_out = h_out.reshape(-1, h_out.size(-1))

            if flat_in.size(0) > probe_tokens:
                flat_in = flat_in[:probe_tokens]
                flat_out = flat_out[:probe_tokens]

            if flat_in.numel() == 0 or flat_out.numel() == 0:
                continue

            # FFN delta
            delta = flat_out - flat_in
            delta_norm = torch.norm(delta, dim=-1)
            mean_delta = delta_norm.mean().item()
            metrics[f'ffn_delta_l{layer_idx}_mean'] = float(mean_delta)
            delta_means.append(mean_delta)

            # Residual cosine
            cos = torch.clamp(F.cosine_similarity(flat_in, flat_out, dim=-1), -1.0, 1.0)
            cos_mean = cos.mean().item()
            metrics[f'residual_cos_l{layer_idx}_mean'] = float(cos_mean)
            cos_means.append(cos_mean)

            # Variance ratio
            var_in = flat_in.var(dim=0, unbiased=False)
            var_out = flat_out.var(dim=0, unbiased=False)
            ratio = (var_out.mean() + eps) / (var_in.mean() + eps)
            metrics[f'ffn_var_ratio_l{layer_idx}'] = float(ratio.item())
            var_ratios.append(ratio.item())

            # LN-like stats: prefer the captured LayerNorm output when
            # available so we report the actual normalized distribution
            # rather than reconstructing it from FFN-input variance.
            if ln_out is not None:
                ln_flat = ln_out.reshape(-1, ln_out.size(-1))
                if ln_flat.size(0) > probe_tokens:
                    ln_flat = ln_flat[:probe_tokens]
                ln_std = float(ln_flat.std(dim=0, unbiased=False).mean().item())
                ln_mean_abs = float(ln_flat.mean(dim=0).abs().mean().item())
            else:
                std_out = torch.sqrt(var_out + eps)
                ln_std = float(std_out.mean().item())
                ln_mean_abs = float(flat_out.mean(dim=0).abs().mean().item())
            metrics[f'ln_std_l{layer_idx}_mean'] = ln_std
            ln_std_means.append(ln_std)
            metrics[f'ln_mean_abs_l{layer_idx}_mean'] = ln_mean_abs
            ln_mean_abs_means.append(ln_mean_abs)

            # Active dimension fraction
            active = (var_out > self.ffn_var_activity_threshold).float()
            active_frac = active.mean().item()
            metrics[f'ffn_active_dim_frac_l{layer_idx}'] = float(active_frac)
            active_fracs.append(active_frac)

            # Output skewness
            try:
                skew_val = _safe_skew(flat_out.detach().cpu().numpy().reshape(-1))
            except Exception:
                skew_val = 0.0
            metrics[f'ffn_out_skew_l{layer_idx}'] = float(skew_val)
            skew_vals.append(skew_val)

        # Aggregates
        if delta_means:
            metrics['ffn_delta_mean'] = float(np.mean(delta_means))
            metrics['residual_cos_mean'] = float(np.mean(cos_means))
            metrics['ffn_var_ratio_mean'] = float(np.mean(var_ratios))
            metrics['ln_std_mean'] = float(np.mean(ln_std_means))
            metrics['ln_mean_abs_mean'] = float(np.mean(ln_mean_abs_means))
            metrics['ffn_active_dim_frac_mean'] = float(np.mean(active_fracs))
            metrics['ffn_out_skew_mean'] = float(np.mean(skew_vals))

        # Embedding norms
        embedding = self.inspector.embedding
        if embedding is not None and input_ids is not None:
            embeds = embedding(input_ids)
            flat_embed = embeds.reshape(-1, embeds.size(-1))
            norms = torch.norm(flat_embed, dim=-1)
            metrics['embedding_norm_mean'] = float(norms.mean().item())
            metrics['embedding_norm_std'] = float(norms.std(unbiased=False).item())

        # First-layer drift
        if len(hs_list) > 1 and hs_list[0] is not None and hs_list[1] is not None:
            h0 = hs_list[0].reshape(-1, hs_list[0].size(-1))
            h1 = hs_list[1].reshape(-1, hs_list[1].size(-1))
            take = min(h0.size(0), h1.size(0))
            if take > 0:
                delta01 = torch.norm(h1[:take] - h0[:take], dim=-1)
                metrics['h1_delta_norm_mean'] = float(delta01.mean().item())

        return metrics

    def _select_tap_tensors(
        self,
        *,
        layer_idx: int,
        use_capture: bool,
        sublayer_capture: Any,
        hs_list: list,
    ):
        """Return ``(h_in, h_out, ln_out)`` for one layer.

        When sublayer captures are available we read FFN-sublayer
        boundaries directly (``ffn_in`` / ``ffn_out``) and the last
        captured LayerNorm output for the layer; otherwise we fall back
        to adjacent hidden states from ``output_hidden_states=True``.
        """
        if use_capture:
            ffn_in = sublayer_capture.get(layer_idx, "ffn_in")
            ffn_out = sublayer_capture.get(layer_idx, "ffn_out")
            ln_out = None
            for ord_idx in range(8):  # arbitrary upper bound; layers carry 1-3 LNs
                tap = sublayer_capture.get(layer_idx, f"ln{ord_idx}_out")
                if tap is not None:
                    ln_out = tap
            return ffn_in, ffn_out, ln_out

        if layer_idx + 1 >= len(hs_list):
            return None, None, None
        return hs_list[layer_idx], hs_list[layer_idx + 1], None
