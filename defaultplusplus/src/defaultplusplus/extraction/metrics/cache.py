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

        # HF transformers >=4.40 returns a ``DynamicCache`` object that
        # is iterable but not subscriptable. Older code paths still
        # return a tuple of (key, value) tensors per layer. Normalize
        # to a list of (key, value) pairs.
        try:
            kv_pairs = self._extract_kv_pairs(past_kv)
        except Exception:
            return metrics
        if not kv_pairs:
            return metrics

        sample_layers = self.inspector.get_sampled_layer_indices()
        sims = []
        for layer_idx in sample_layers:
            if layer_idx >= len(kv_pairs):
                continue
            keys = kv_pairs[layer_idx][0]
            if keys is None or keys.dim() < 4 or keys.size(2) < 2:
                continue
            k1 = keys[:, :, :-1, :].reshape(-1, keys.size(-1))
            k2 = keys[:, :, 1:, :].reshape(-1, keys.size(-1))
            cos = F.cosine_similarity(k1, k2, dim=-1)
            sims.append(float(cos.mean().item()))

        if sims:
            metrics['cache_hidden_sim'] = float(sum(sims) / len(sims))

        return metrics

    @staticmethod
    def _extract_kv_pairs(past_kv: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return per-layer (key, value) pairs from any HF cache shape.

        Handles three shapes:
          - tuple of tuples (legacy): ``((k0, v0), (k1, v1), ...)``;
          - ``DynamicCache`` (new): exposes ``.key_cache`` and
            ``.value_cache`` as parallel lists indexed by layer;
          - object exposing ``layers`` (mid-version): each layer has
            ``.keys`` and ``.values`` tensors.
        Anything else returns an empty list.
        """
        # Legacy tuple-of-tuples shape.
        if isinstance(past_kv, (list, tuple)) and past_kv:
            first = past_kv[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2 \
                    and isinstance(first[0], torch.Tensor):
                return [(layer[0], layer[1]) for layer in past_kv]

        # DynamicCache shape.
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            keys = list(getattr(past_kv, "key_cache"))
            values = list(getattr(past_kv, "value_cache"))
            return list(zip(keys, values))

        # Per-layer DynamicLayer objects with ``.keys`` / ``.values``.
        if hasattr(past_kv, "layers"):
            pairs = []
            for layer in past_kv.layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is not None and v is not None:
                    pairs.append((k, v))
            return pairs

        return []
