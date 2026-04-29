"""AttentionMetrics — entropy, sparsity, padding mass, head similarity, positional."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import MetricModule, _safe_skew, _safe_kurtosis
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class AttentionMetrics(MetricModule):
    """Attention-based feature extraction across sampled layers."""

    # Per-layer keys emitted by ``_compute_layer_metrics`` whenever
    # attention weights, attention_mask, and input_ids are present
    # (the standard runtime case). Used by ``static_feature_names``
    # to pin the schema at construction time.
    _PER_LAYER_KEYS = (
        "attention_entropy",
        "attention_entropy_mean",
        "attention_entropy_std",
        "attention_max_mean",
        "attention_max_std",
        "attention_sparsity",
        "attention_weight_magnitude",
        "attention_mass_leak",
        "attention_mass_leak_max",
        "attention_cross_example_leak",
        "attention_mass_future",
        "attention_mass_pad_mean",
        "attention_mass_pad_max",
        "head_similarity_mean",
        "head_similarity_std",
        "head_similarity_max",
        "positional_recv_mean",
        "positional_recv_var",
        "positional_recv_skew",
        "positional_recv_early",
        "positional_recv_mid",
        "positional_recv_late",
        "positional_recv_mid_over_early",
        "positional_recv_late_over_early",
        "attention_score_var",
        "attention_score_skew",
        "pre_softmax_score_mean",
        "pre_softmax_score_var",
        "pre_softmax_score_skew",
        "pre_softmax_score_kurt",
        "qkv_alignment_qk_cos_mean",
        "qkv_alignment_qv_cos_mean",
        "qkv_alignment_kv_cos_mean",
    )

    # Conditional per-layer keys emitted only when ``cls_token_id`` is
    # configured (the runtime can identify CLS / SEP tokens). Joined
    # into ``static_feature_names`` only when the flag is set so the
    # schema reflects what this collector will actually emit.
    _PER_LAYER_SPECIAL_KEYS = (
        "attention_mass_special_mean",
        "attention_mass_special_std",
    )

    _GLOBAL_ALIAS_KEYS = (
        "attention_entropy",
        "attention_entropy_mean",
        "mass_pad",
        "mass_leak",
        "cross_example_attention",
        "attention_mass_future",
        "pre_softmax_score_mean",
        "pre_softmax_score_var",
        "pre_softmax_score_skew",
        "pre_softmax_score_kurt",
        "head_similarity_mean",
        "head_similarity_max",
        "qkv_alignment_qk_cos_mean",
        "qkv_alignment_qv_cos_mean",
        "qkv_alignment_kv_cos_mean",
    )

    def static_feature_names(self) -> list[str]:
        sampled = self.inspector.get_sampled_layer_indices()
        per_layer = list(self._PER_LAYER_KEYS)
        if self.special_token_ids.get("cls") is not None:
            per_layer.extend(self._PER_LAYER_SPECIAL_KEYS)
        names: list[str] = []
        for layer_idx in sampled:
            for key in per_layer:
                names.append(f"L{layer_idx}_{key}")
        names.extend(self._GLOBAL_ALIAS_KEYS)
        return names

    def __init__(self, inspector: ModelInspector, config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.attention_leak_threshold = cfg.attention_leak_threshold
        self.position_cutoff = cfg.position_cutoff
        self.special_token_ids = {
            'pad': cfg.pad_token_id,
            'cls': cfg.cls_token_id,
            'sep': cfg.sep_token_id,
        }

    @property
    def requires_attention_weights(self) -> bool:
        return True

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        attention_weights: Any = None,
        hidden_states: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        sublayer_capture: Any = None,
        **kwargs,
    ) -> Dict[str, float]:
        if attention_weights is None:
            return {}

        attentions = attention_weights if isinstance(attention_weights, (list, tuple)) else [attention_weights]
        sample_layers = self.inspector.get_sampled_layer_indices()
        metrics: Dict[str, float] = {}
        self._sublayer_capture = sublayer_capture

        # Per-layer metrics with layer prefix
        alias_accum = {
            'entropy': [], 'mass_pad': [], 'mass_leak': [],
            'cross_example': [], 'mass_future': [],
            'pre_softmax_mean': [], 'pre_softmax_var': [],
            'pre_softmax_skew': [], 'pre_softmax_kurt': [],
            'head_sim_mean': [], 'head_sim_max': [],
            'qkv_qk': [], 'qkv_qv': [], 'qkv_kv': [],
        }

        for layer_idx in sample_layers:
            if layer_idx >= len(attentions):
                continue

            layer_input = None
            if hidden_states is not None and len(hidden_states) > layer_idx:
                layer_input = hidden_states[layer_idx]

            lm = self._compute_layer_metrics(
                attentions[layer_idx],
                model=model,
                layer_idx=layer_idx,
                layer_input=layer_input,
                attention_mask=attention_mask,
                input_ids=input_ids,
            )
            for k, v in lm.items():
                metrics[f'L{layer_idx}_{k}'] = v

            # Accumulate for global aliases
            if 'attention_entropy' in lm:
                alias_accum['entropy'].append(lm['attention_entropy'])
            if 'attention_mass_pad_mean' in lm:
                alias_accum['mass_pad'].append(lm['attention_mass_pad_mean'])
            if 'attention_mass_leak' in lm:
                alias_accum['mass_leak'].append(lm['attention_mass_leak'])
            if 'attention_cross_example_leak' in lm:
                alias_accum['cross_example'].append(lm['attention_cross_example_leak'])
            if 'attention_mass_future' in lm:
                alias_accum['mass_future'].append(lm['attention_mass_future'])
            if 'pre_softmax_score_mean' in lm:
                alias_accum['pre_softmax_mean'].append(lm['pre_softmax_score_mean'])
            if 'pre_softmax_score_var' in lm:
                alias_accum['pre_softmax_var'].append(lm['pre_softmax_score_var'])
            if 'pre_softmax_score_skew' in lm:
                alias_accum['pre_softmax_skew'].append(lm['pre_softmax_score_skew'])
            if 'pre_softmax_score_kurt' in lm:
                alias_accum['pre_softmax_kurt'].append(lm['pre_softmax_score_kurt'])
            if 'head_similarity_mean' in lm:
                alias_accum['head_sim_mean'].append(lm['head_similarity_mean'])
            if 'head_similarity_max' in lm:
                alias_accum['head_sim_max'].append(lm['head_similarity_max'])
            if 'qkv_alignment_qk_cos_mean' in lm:
                alias_accum['qkv_qk'].append(lm['qkv_alignment_qk_cos_mean'])
            if 'qkv_alignment_qv_cos_mean' in lm:
                alias_accum['qkv_qv'].append(lm['qkv_alignment_qv_cos_mean'])
            if 'qkv_alignment_kv_cos_mean' in lm:
                alias_accum['qkv_kv'].append(lm['qkv_alignment_kv_cos_mean'])

        # Global aliases (aggregated across sampled layers)
        def _mean(vals):
            return float(sum(vals) / len(vals)) if vals else 0.0

        metrics['attention_entropy'] = _mean(alias_accum['entropy'])
        metrics['attention_entropy_mean'] = metrics['attention_entropy']
        metrics['mass_pad'] = max(alias_accum['mass_pad']) if alias_accum['mass_pad'] else 0.0
        metrics['mass_leak'] = max(alias_accum['mass_leak']) if alias_accum['mass_leak'] else 0.0
        metrics['cross_example_attention'] = max(alias_accum['cross_example']) if alias_accum['cross_example'] else 0.0
        metrics['attention_mass_future'] = max(alias_accum['mass_future']) if alias_accum['mass_future'] else 0.0
        metrics['pre_softmax_score_mean'] = _mean(alias_accum['pre_softmax_mean'])
        metrics['pre_softmax_score_var'] = _mean(alias_accum['pre_softmax_var'])
        metrics['pre_softmax_score_skew'] = _mean(alias_accum['pre_softmax_skew'])
        metrics['pre_softmax_score_kurt'] = _mean(alias_accum['pre_softmax_kurt'])
        metrics['head_similarity_mean'] = max(alias_accum['head_sim_mean']) if alias_accum['head_sim_mean'] else 0.0
        metrics['head_similarity_max'] = max(alias_accum['head_sim_max']) if alias_accum['head_sim_max'] else 0.0
        metrics['qkv_alignment_qk_cos_mean'] = _mean(alias_accum['qkv_qk'])
        metrics['qkv_alignment_qv_cos_mean'] = _mean(alias_accum['qkv_qv'])
        metrics['qkv_alignment_kv_cos_mean'] = _mean(alias_accum['qkv_kv'])

        return metrics

    def _compute_layer_metrics(
        self,
        attn_weights: torch.Tensor,
        model: Optional[torch.nn.Module],
        layer_idx: int,
        layer_input: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        attn = attn_weights.detach().float()
        metrics: Dict[str, float] = {}

        # Entropy and sparsity
        probs = torch.clamp(attn, min=1e-12)
        log_probs = probs.log()
        head_entropy = -(probs * log_probs).sum(dim=-1).mean(dim=2)  # [batch, heads]
        head_entropy = head_entropy.mean(dim=0)  # [heads]

        max_weights = attn.max(dim=-1).values
        head_max = max_weights.mean(dim=2).mean(dim=0)

        metrics['attention_entropy_mean'] = float(head_entropy.mean().item())
        metrics['attention_entropy'] = metrics['attention_entropy_mean']
        metrics['attention_entropy_std'] = float(head_entropy.std(unbiased=False).item())
        metrics['attention_max_mean'] = float(head_max.mean().item())
        metrics['attention_max_std'] = float(head_max.std(unbiased=False).item())
        metrics['attention_sparsity'] = float((attn < 0.01).float().mean().item())
        metrics['attention_weight_magnitude'] = float(attn.abs().mean().item())

        metrics['attention_mass_leak'] = 0.0
        metrics['attention_mass_leak_max'] = 0.0
        metrics['attention_cross_example_leak'] = 0.0
        metrics['attention_mass_future'] = 0.0

        # Padding mass
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).float().unsqueeze(1).unsqueeze(2).to(attn.device)
            pad_mass = (attn * pad_mask).sum(dim=-1)
            metrics['attention_mass_pad_mean'] = float(pad_mass.mean().item())
            metrics['attention_mass_pad_max'] = float(pad_mass.max().item())

            cross_mask = self._compute_cross_example_mask(attention_mask)
            if cross_mask is not None:
                cross_mask = cross_mask.to(attn.device)
                cross_mass = (attn * cross_mask).sum(dim=-1)
                metrics['attention_mass_leak'] = float(cross_mass.mean().item())
                metrics['attention_mass_leak_max'] = float(cross_mass.max().item())
                leak_indicator = (cross_mass > self.attention_leak_threshold).float()
                metrics['attention_cross_example_leak'] = float(leak_indicator.mean().item())

        # Future attention mass
        future_mass = self._compute_future_attention_mass(attn, attention_mask)
        if future_mass is not None:
            metrics['attention_mass_future'] = float(future_mass)

        # Special token mass
        if input_ids is not None and self.special_token_ids.get('cls') is not None:
            special_mask = self._get_special_token_mask(input_ids)
            special_mask = special_mask.unsqueeze(1).unsqueeze(2).to(attn.device)
            special_mass = (attn * special_mask).sum(dim=-1)
            metrics['attention_mass_special_mean'] = float(special_mass.mean().item())
            metrics['attention_mass_special_std'] = float(special_mass.std(unbiased=False).item())

        # Head similarity
        metrics.update(self._compute_head_similarity(attn))

        # Positional profile
        if attention_mask is not None:
            metrics.update(self._compute_positional_profile(attn, attention_mask))

        # Score statistics proxy
        score_proxy = log_probs
        metrics['attention_score_var'] = float(score_proxy.var().item())
        metrics['attention_score_skew'] = float(_safe_skew(score_proxy.view(-1).cpu().numpy()))

        # Pre-softmax stats: prefer captured Q/K from sublayer hooks
        # (exact); fall back to recomputing the projections on the
        # layer input (approximate).
        cap = getattr(self, "_sublayer_capture", None)
        pre_stats = self._compute_pre_softmax_stats_from_capture(
            cap, layer_idx, attention_mask
        )
        if not pre_stats:
            pre_stats = self._compute_pre_softmax_stats(
                model, layer_idx, layer_input, attention_mask
            )
        metrics.update(pre_stats)

        # QKV alignment cosines: only available when Q, K, V are captured
        # post-projection by the sublayer hooks. Emits three direct
        # cosine-similarity statistics per sampled layer.
        metrics.update(self._compute_qkv_alignment(cap, layer_idx))

        return metrics

    @staticmethod
    def _compute_cross_example_mask(attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        mask = attention_mask.float()
        global_active = (mask.sum(dim=0, keepdim=True) > 0).float()
        other_example_mask = (1.0 - mask) * global_active
        if torch.all(other_example_mask <= 0):
            return None
        return other_example_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _compute_future_attention_mass(
        attn: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[float]:
        if attn.dim() != 4:
            return None
        batch_size, num_heads, query_len, key_len = attn.shape
        if query_len == 0 or key_len == 0:
            return None

        future_mask = torch.triu(
            torch.ones(query_len, key_len, device=attn.device, dtype=attn.dtype), diagonal=1
        )
        future_mass = attn * future_mask
        return float(future_mass.sum(dim=-1).mean().item())

    def _get_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        special_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        cls_id = self.special_token_ids.get('cls')
        sep_id = self.special_token_ids.get('sep')
        if cls_id is not None:
            special_mask = special_mask + (input_ids == cls_id).float()
        if sep_id is not None:
            special_mask = special_mask + (input_ids == sep_id).float()
        return (special_mask > 0).float()

    @staticmethod
    def _compute_head_similarity(attn: torch.Tensor) -> Dict[str, float]:
        batch_size, num_heads, seq_len, _ = attn.shape
        flattened = attn.reshape(batch_size, num_heads, -1)
        mean_patterns = flattened.mean(dim=0)
        sims = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                cos_sim = F.cosine_similarity(
                    mean_patterns[i].unsqueeze(0), mean_patterns[j].unsqueeze(0), dim=1
                )
                sims.append(float(cos_sim.item()))
        if not sims:
            return {'head_similarity_mean': 0.0, 'head_similarity_std': 0.0, 'head_similarity_max': 0.0}
        sims_arr = np.array(sims)
        return {
            'head_similarity_mean': float(np.mean(sims_arr)),
            'head_similarity_std': float(np.std(sims_arr)),
            'head_similarity_max': float(np.max(sims_arr)),
        }

    def _compute_positional_profile(
        self, attn: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        seq_len = min(attn.size(-1), self.position_cutoff)
        attn_cut = attn[..., :seq_len]
        key_mask = attention_mask[:, :seq_len].float()

        recv = attn_cut.mean(dim=2)
        recv = recv * key_mask.unsqueeze(1)
        total_mass = key_mask.sum()
        if total_mass <= 0:
            return {}

        recv_vec = recv.sum(dim=(0, 1)) / total_mass
        recv_np = recv_vec.detach().cpu().numpy()

        region_size = seq_len // 3 if seq_len >= 3 else seq_len
        early = recv_np[:region_size].mean() if region_size > 0 else 0.0
        mid = recv_np[region_size:2 * region_size].mean() if region_size > 0 else 0.0
        late = recv_np[2 * region_size:].mean() if region_size > 0 else 0.0

        return {
            'positional_recv_mean': float(np.mean(recv_np)),
            'positional_recv_var': float(np.var(recv_np)),
            'positional_recv_skew': float(_safe_skew(recv_np)),
            'positional_recv_early': float(early),
            'positional_recv_mid': float(mid),
            'positional_recv_late': float(late),
            'positional_recv_mid_over_early': float(mid / (early + 1e-8)),
            'positional_recv_late_over_early': float(late / (early + 1e-8)),
        }

    def _compute_pre_softmax_stats(
        self,
        model: Optional[torch.nn.Module],
        layer_idx: int,
        layer_input: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        if model is None or layer_input is None:
            return {}

        attn_module = self.inspector.get_attention_module(layer_idx)
        if attn_module is None:
            return {}

        qkv = self.inspector._attn_pattern
        if qkv is None or not qkv.qkv_names or qkv.qkv_style != 'separate':
            return {}

        # Try to find Q and K projection modules
        q_name, k_name = qkv.qkv_names[0], qkv.qkv_names[1]
        q_mod = self.inspector._find_submodule(attn_module, q_name)
        k_mod = self.inspector._find_submodule(attn_module, k_name)
        if q_mod is None or k_mod is None:
            return {}

        with torch.no_grad():
            hidden = layer_input.detach()
            batch_size, seq_len, hidden_size = hidden.shape
            n_heads = self.inspector.num_heads or 1
            dim_per_head = hidden_size // n_heads

            query = q_mod(hidden)
            key = k_mod(hidden)

            query = query.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)
            key = key.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_per_head)

            if attention_mask is not None:
                attn_mask = attention_mask.float()
                key_m = attn_mask.unsqueeze(1).unsqueeze(2)
                query_m = attn_mask.unsqueeze(1).unsqueeze(-1)
                valid_mask = (key_m * query_m).to(dtype=torch.bool)
                if valid_mask.sum() == 0:
                    return {}
                scores = scores.masked_select(valid_mask)
            else:
                scores = scores.reshape(-1)

            if scores.numel() == 0:
                return {}

            scores = scores.detach().cpu().float()
            return {
                'pre_softmax_score_mean': float(scores.mean().item()),
                'pre_softmax_score_var': float(scores.var(unbiased=False).item()),
                'pre_softmax_score_skew': float(_safe_skew(scores.numpy())),
                'pre_softmax_score_kurt': float(_safe_kurtosis(scores.numpy())),
            }

    def _compute_pre_softmax_stats_from_capture(
        self,
        sublayer_capture: Any,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute pre-softmax score stats from captured Q and K tensors."""
        if sublayer_capture is None or not getattr(sublayer_capture, "installed", False):
            return {}
        q = sublayer_capture.get(layer_idx, "q")
        k = sublayer_capture.get(layer_idx, "k")
        if q is None or k is None:
            return {}

        try:
            q_h, k_h = self._reshape_to_heads(q), self._reshape_to_heads(k)
        except Exception:
            return {}
        if q_h is None or k_h is None:
            return {}

        with torch.no_grad():
            dim_per_head = q_h.size(-1)
            scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(max(dim_per_head, 1))

            if attention_mask is not None and attention_mask.dim() == 2:
                attn_mask = attention_mask.float().to(scores.device)
                key_m = attn_mask.unsqueeze(1).unsqueeze(2)
                query_m = attn_mask.unsqueeze(1).unsqueeze(-1)
                valid_mask = (key_m * query_m).to(dtype=torch.bool)
                if valid_mask.sum() == 0:
                    return {}
                scores = scores.masked_select(valid_mask)
            else:
                scores = scores.reshape(-1)

            if scores.numel() == 0:
                return {}

            arr = scores.detach().cpu().float()
            return {
                'pre_softmax_score_mean': float(arr.mean().item()),
                'pre_softmax_score_var': float(arr.var(unbiased=False).item()),
                'pre_softmax_score_skew': float(_safe_skew(arr.numpy())),
                'pre_softmax_score_kurt': float(_safe_kurtosis(arr.numpy())),
            }

    def _compute_qkv_alignment(
        self,
        sublayer_capture: Any,
        layer_idx: int,
    ) -> Dict[str, float]:
        """Direct Q-K, Q-V, K-V cosine-similarity stats (T13).

        Reads post-projection Q, K, V tensors from the attention sublayer
        hooks and reports three head-averaged cosine values per layer.
        Returns an empty dict if any of Q/K/V was not captured.
        """
        if sublayer_capture is None or not getattr(sublayer_capture, "installed", False):
            return {}
        q = sublayer_capture.get(layer_idx, "q")
        k = sublayer_capture.get(layer_idx, "k")
        v = sublayer_capture.get(layer_idx, "v")
        if q is None or k is None or v is None:
            return {}

        try:
            q_h = self._reshape_to_heads(q)
            k_h = self._reshape_to_heads(k)
            v_h = self._reshape_to_heads(v)
        except Exception:
            return {}
        if q_h is None or k_h is None or v_h is None:
            return {}

        # Cosine similarity along the per-head feature dim, averaged over
        # batch / heads / sequence positions.
        def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
            sim = F.cosine_similarity(a, b, dim=-1)
            sim = torch.clamp(sim, -1.0, 1.0)
            return float(sim.mean().item())

        with torch.no_grad():
            qk = _mean_cos(q_h, k_h)
            qv = _mean_cos(q_h, v_h)
            kv = _mean_cos(k_h, v_h)

        return {
            'qkv_alignment_qk_cos_mean': qk,
            'qkv_alignment_qv_cos_mean': qv,
            'qkv_alignment_kv_cos_mean': kv,
        }

    def _reshape_to_heads(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Return ``tensor`` reshaped to ``(batch, heads, seq, head_dim)``.

        Accepts a captured projection output of shape ``(batch, seq, hidden)``
        or already-shaped ``(batch, heads, seq, head_dim)``.
        """
        if tensor is None:
            return None
        n_heads = self.inspector.num_heads or 1
        if tensor.dim() == 4:
            return tensor.detach().float()
        if tensor.dim() == 3:
            batch_size, seq_len, hidden = tensor.shape
            if hidden % n_heads != 0:
                return None
            head_dim = hidden // n_heads
            return tensor.detach().float().reshape(
                batch_size, seq_len, n_heads, head_dim
            ).transpose(1, 2)
        return None
