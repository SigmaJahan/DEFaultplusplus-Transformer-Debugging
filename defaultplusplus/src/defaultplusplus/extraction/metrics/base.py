"""MetricModule abstract base class and shared utilities."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..inspector import ModelInspector


class MetricModule(ABC):
    """Abstract base for all metric collection modules."""

    def __init__(self, inspector: ModelInspector):
        self.inspector = inspector

    @property
    def requires_attention_weights(self) -> bool:
        return False

    @property
    def requires_hidden_states(self) -> bool:
        return False

    @abstractmethod
    def collect(
        self,
        *,
        loss: Any = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        attention_weights: Any = None,
        hidden_states: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        epoch: int = 0,
        step_time: Optional[float] = None,
    ) -> Dict[str, float]:
        ...

    def static_feature_names(self) -> list[str]:
        """Return the set of raw metric keys this module always emits.

        The list is computed from the inspector (number of sampled
        layers, parameter groups) — it does not require any
        ``collect()`` call to have happened. Used by the fixed
        ``feature_names`` schema so a downstream classifier can pin
        its input dimensionality before any training run.

        Subclasses override to declare exactly which keys they emit.
        Modules that emit nothing (or whose emission set depends on
        unobservable runtime state) return ``[]``; those keys are
        treated as best-effort and not part of the fixed schema.
        """
        return []


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _safe_mean(tensor: torch.Tensor) -> float:
    """Mean with empty/NaN guard."""
    if tensor.numel() == 0:
        return 0.0
    val = tensor.float().mean().item()
    return 0.0 if math.isnan(val) or math.isinf(val) else val


def _safe_std(tensor: torch.Tensor) -> float:
    """Std with empty/NaN guard."""
    if tensor.numel() < 2:
        return 0.0
    val = tensor.float().std(unbiased=False).item()
    return 0.0 if math.isnan(val) or math.isinf(val) else val


def _safe_skew(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    try:
        from scipy.stats import skew
        return float(skew(values))
    except Exception:
        return 0.0


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(values, fisher=True, bias=False))
    except Exception:
        return 0.0


def _to_float(value: Any) -> float:
    """Convert tensor/number to Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Return logits tensor from HF output variants."""
    if hasattr(outputs, 'logits'):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs
