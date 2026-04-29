"""CacheMetrics — decoder-only KV-cache diagnostics."""

from __future__ import annotations

import inspect as _inspect
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .base import MetricModule
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class CacheMetrics(MetricModule):
    """KV-cache hidden similarity and NLL divergence (decoder-only)."""

    _STATIC_KEYS = (
        "cache_hidden_sim",
        "cache_nll_divergence",
    )

    def static_feature_names(self) -> list[str]:
        return list(self._STATIC_KEYS)

    def __init__(self, inspector: ModelInspector,
                 config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.cache_probe_interval = max(1, int(cfg.cache_probe_interval))
        self.cache_probe_positions = max(1, int(cfg.cache_probe_positions))
        self.cache_probe_max_seq_len = max(4, int(cfg.cache_probe_max_seq_len))

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        **kwargs,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            'cache_hidden_sim': 0.0,
            'cache_nll_divergence': 0.0,
        }

        if model is None or input_ids is None:
            return metrics

        # Cache NLL divergence: fresh vs cached forward at sampled
        # positions. Run on a separate cadence from the rest of cache
        # metrics because each probe costs two extra forward passes.
        if batch_idx % self.cache_probe_interval == 0:
            div = self._compute_cache_nll_divergence(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if div is not None:
                metrics['cache_nll_divergence'] = float(div)

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

    # ── cache_nll_divergence (T14) ───────────────────────────────────────
    def _compute_cache_nll_divergence(
        self,
        *,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[float]:
        """Symmetric-KL divergence between fresh and cached next-token
        distributions, averaged over a few sampled positions.

        For each sampled position ``t`` we run two forwards:

          * **fresh**:  full input ``[0..t]`` and read logits at
                        position ``t``.
          * **cached**: prefix forward on ``[0..t-1]`` to build a KV
                        cache, then a one-token forward on ``[t]`` with
                        ``past_key_values=`` and read logits at the
                        single output position.

        On a clean model the two distributions match within numerical
        noise; the cache-fault operators (``CST``, ``COB``, ``CTR``,
        ``CLK``) divert one or both, so the metric responds.
        """
        if input_ids.dim() != 2:
            return None
        seq_len = int(input_ids.size(1))
        if seq_len < 4:
            return None
        seq_len = min(seq_len, self.cache_probe_max_seq_len)

        sig_params = _safe_sig_params(model)
        # Skip the sig gate when the wrapped forward exposes only
        # ``**kwargs`` — common when DEForm injectors wrap the model.
        if sig_params is not None and not _accepts_var_keyword(sig_params):
            if "past_key_values" not in sig_params or "use_cache" not in sig_params:
                return None

        positions = self._sample_probe_positions(seq_len)
        if not positions:
            return None

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                divergences: List[float] = []
                for t in positions:
                    div = self._probe_position(
                        model=model,
                        input_ids=input_ids[:, :seq_len],
                        attention_mask=(
                            attention_mask[:, :seq_len]
                            if attention_mask is not None else None
                        ),
                        position=t,
                        sig_params=sig_params,
                    )
                    if div is not None:
                        divergences.append(div)
        except Exception:
            return None
        finally:
            model.train(was_training)

        if not divergences:
            return None
        return float(sum(divergences) / len(divergences))

    def _sample_probe_positions(self, seq_len: int) -> List[int]:
        """Sample ``cache_probe_positions`` indices in the second half
        of the sequence so the cache has meaningful context.
        """
        start = max(2, seq_len // 2)
        end = seq_len  # exclusive
        if end - start < 1:
            return []
        n = min(self.cache_probe_positions, end - start)
        if n <= 0:
            return []
        if n == 1:
            return [end - 1]
        # Evenly spaced positions in [start, end-1].
        step = max(1, (end - 1 - start) // (n - 1))
        return [min(end - 1, start + i * step) for i in range(n)]

    def _probe_position(
        self,
        *,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position: int,
        sig_params: Optional[Any],
    ) -> Optional[float]:
        if position <= 0 or position >= input_ids.size(1):
            return None

        prefix_ids = input_ids[:, :position]
        token_ids = input_ids[:, position:position + 1]

        prefix_mask = attention_mask[:, :position] if attention_mask is not None else None
        token_mask = attention_mask[:, position:position + 1] if attention_mask is not None else None
        full_mask = attention_mask[:, :position + 1] if attention_mask is not None else None

        accepts_var = _accepts_var_keyword(sig_params) if sig_params is not None else False
        accepts_mask = sig_params is None or accepts_var or "attention_mask" in sig_params
        accepts_position_ids = (
            sig_params is None or accepts_var or "position_ids" in sig_params
        )

        # Fresh forward over [0..t] — read logits at position t.
        fresh_kwargs = {"input_ids": input_ids[:, :position + 1]}
        if full_mask is not None and accepts_mask:
            fresh_kwargs["attention_mask"] = full_mask
        fresh_out = model(**fresh_kwargs)
        fresh_logits = _extract_logits(fresh_out)
        if fresh_logits is None or fresh_logits.dim() < 2:
            return None
        fresh_t = fresh_logits[:, position, :]

        # Cached forward — build the cache on [0..t-1], then run [t].
        warm_kwargs = {"input_ids": prefix_ids, "use_cache": True}
        if prefix_mask is not None and accepts_mask:
            warm_kwargs["attention_mask"] = prefix_mask
        warm_out = model(**warm_kwargs)
        past_kv = getattr(warm_out, "past_key_values", None)
        if past_kv is None:
            return None

        cached_kwargs = {
            "input_ids": token_ids,
            "past_key_values": past_kv,
            "use_cache": True,
        }
        if full_mask is not None and accepts_mask:
            # HF expects an attention_mask covering the full cached length
            # plus the new token when past_key_values is provided.
            cached_kwargs["attention_mask"] = full_mask
        if accepts_position_ids:
            cached_kwargs["position_ids"] = torch.full_like(token_ids, position)
        try:
            cached_out = model(**cached_kwargs)
        except TypeError:
            cached_kwargs.pop("position_ids", None)
            cached_out = model(**cached_kwargs)
        cached_logits = _extract_logits(cached_out)
        if cached_logits is None or cached_logits.dim() < 2:
            return None
        cached_t = cached_logits[:, -1, :]

        if fresh_t.shape != cached_t.shape:
            return None

        return _symmetric_kl(fresh_t, cached_t)

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


def _safe_sig_params(model: torch.nn.Module) -> Optional[Any]:
    """Return ``model.forward``'s parameter mapping, or ``None`` on failure."""
    try:
        return _inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None


def _accepts_var_keyword(params: Any) -> bool:
    """Return True iff ``params`` contains a ``**kwargs`` parameter."""
    try:
        return any(p.kind == _inspect.Parameter.VAR_KEYWORD
                   for p in params.values())
    except Exception:
        return False


def _extract_logits(output: Any) -> Optional[torch.Tensor]:
    """Pull the logits tensor out of an HF model output."""
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor,
                  eps: float = 1e-12) -> float:
    """Return the mean symmetric KL between softmax(p) and softmax(q).

    Computed in double precision for stability; the inputs are small
    (vocab-sized vectors at a handful of positions), so the cost is
    negligible.
    """
    p = F.log_softmax(p_logits.detach().to(torch.float64), dim=-1)
    q = F.log_softmax(q_logits.detach().to(torch.float64), dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    # KL(P||Q) + KL(Q||P) — clamp avoids -inf from log_softmax of all-zero
    # rows, which can happen with masked-out positions.
    pq = (p_prob * (p - q)).sum(dim=-1).clamp_min(0.0)
    qp = (q_prob * (q - p)).sum(dim=-1).clamp_min(0.0)
    div = (pq + qp).mean().item()
    if div != div or div in (float("inf"), float("-inf")):  # NaN/Inf guard
        return 0.0
    return float(max(div, 0.0))
