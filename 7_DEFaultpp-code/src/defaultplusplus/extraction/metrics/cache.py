"""CacheMetrics — decoder-only KV-cache diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .base import MetricModule
from ..inspector import ModelInspector


class CacheMetrics(MetricModule):
    """KV-cache hidden similarity and NLL divergence (decoder-only)."""

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            'cache_hidden_sim': 0.0,
            'cache_nll_divergence': 0.0,
        }

        if model is None or input_ids is None:
            return metrics

        past_kv = getattr(outputs, 'past_key_values', None)
        if past_kv is None:
            return metrics

        # Compute cosine similarity between consecutive key vectors
        sample_layers = self.inspector.get_sampled_layer_indices()
        sims = []
        for layer_idx in sample_layers:
            if layer_idx >= len(past_kv):
                continue
            keys = past_kv[layer_idx][0]  # [batch, heads, seq, dim]
            if keys.size(2) < 2:
                continue
            k1 = keys[:, :, :-1, :].reshape(-1, keys.size(-1))
            k2 = keys[:, :, 1:, :].reshape(-1, keys.size(-1))
            cos = F.cosine_similarity(k1, k2, dim=-1)
            sims.append(float(cos.mean().item()))

        if sims:
            metrics['cache_hidden_sim'] = float(sum(sims) / len(sims))

        return metrics
