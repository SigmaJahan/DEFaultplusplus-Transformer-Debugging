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
        **kwargs,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if hidden_states is None:
            return metrics

        try:
            hs_list = list(hidden_states)
        except Exception:
            return metrics

        if len(hs_list) < 2:
            return metrics

        eps = 1e-6
        delta_means, cos_means, var_ratios = [], [], []
        ln_std_means, ln_mean_abs_means = [], []
        active_fracs, skew_vals = [], []

        max_layers = min(self.inspector.num_layers or len(hs_list) - 1, len(hs_list) - 1)
        probe_tokens = self.ffn_probe_tokens

        for layer_idx in range(max_layers):
            h_in = hs_list[layer_idx]
            h_out = hs_list[layer_idx + 1]
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

            # LN-like stats
            std_out = torch.sqrt(var_out + eps)
            ln_std = std_out.mean().item()
            metrics[f'ln_std_l{layer_idx}_mean'] = float(ln_std)
            ln_std_means.append(ln_std)

            mean_out = flat_out.mean(dim=0)
            ln_mean_abs = mean_out.abs().mean().item()
            metrics[f'ln_mean_abs_l{layer_idx}_mean'] = float(ln_mean_abs)
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
