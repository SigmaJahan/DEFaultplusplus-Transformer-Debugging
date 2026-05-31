"""GradientMetrics — per-layer gradient norms, update ratios, vanish/explode."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch

from .base import MetricModule
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class GradientMetrics(MetricModule):
    """Gradient norms, update ratios, vanishing/exploding detection."""

    # Six global gradient statistics. Update ratios are reported per
    # component (the update_ratio_{group} keys) rather than as a global
    # total, which keeps the global set at six.
    _FIXED_KEYS = (
        "grad_norm_total",
        "grad_abs_min",
        "grad_abs_max",
        "grad_zero_ratio",
        "gradient_vanish",
        "gradient_explode",
    )

    def __init__(self, inspector: ModelInspector, config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.grad_vanish_threshold = cfg.grad_vanish_threshold
        self.grad_explode_threshold = cfg.grad_explode_threshold
        self.grad_activity_threshold = cfg.grad_activity_threshold
        self._previous_params: Dict[str, torch.Tensor] = {}

    def static_feature_names(self) -> list[str]:
        groups = self.inspector.get_parameter_groups()
        names = list(self._FIXED_KEYS)
        for group in groups:
            names.append(f"grad_norm_{group}")
            names.append(f"update_active_{group}")
            names.append(f"update_ratio_{group}")
        return names

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if model is None:
            return {}

        metrics: Dict[str, float] = {}
        metrics.update(self._compute_gradient_norms(model))
        metrics.update(self._compute_update_ratios(model))
        return metrics

    def _compute_gradient_norms(self, model: torch.nn.Module) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        layer_groups = self.inspector.get_parameter_groups()

        total_norm_sq = 0.0
        group_norm_sq = {g: 0.0 for g in layer_groups}
        total_elems = 0
        zero_elems = 0
        grad_abs_min = None
        grad_abs_max = None

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            gnorm_sq = grad.norm(2).item() ** 2
            total_norm_sq += gnorm_sq

            grad_abs = grad.abs().detach()
            g_min = grad_abs.min().item() if grad_abs.numel() > 0 else None
            g_max = grad_abs.max().item() if grad_abs.numel() > 0 else None
            if g_min is not None:
                grad_abs_min = g_min if grad_abs_min is None else min(grad_abs_min, g_min)
            if g_max is not None:
                grad_abs_max = g_max if grad_abs_max is None else max(grad_abs_max, g_max)
            zero_elems += int((grad_abs < self.grad_activity_threshold).sum().item())
            total_elems += grad_abs.numel()

            for group, patterns in layer_groups.items():
                if any(pattern in name for pattern in patterns):
                    group_norm_sq[group] += gnorm_sq

        for group, norm_sq in group_norm_sq.items():
            metrics[f'grad_norm_{group}'] = math.sqrt(norm_sq)
            metrics[f'update_active_{group}'] = 1.0 if norm_sq > self.grad_activity_threshold else 0.0

        metrics['grad_norm_total'] = math.sqrt(total_norm_sq)
        metrics['grad_abs_min'] = float(grad_abs_min) if grad_abs_min is not None else 0.0
        metrics['grad_abs_max'] = float(grad_abs_max) if grad_abs_max is not None else 0.0
        metrics['grad_zero_ratio'] = float(zero_elems / total_elems) if total_elems > 0 else 0.0
        metrics['gradient_vanish'] = 1.0 if metrics['grad_norm_total'] < self.grad_vanish_threshold else 0.0
        metrics['gradient_explode'] = 1.0 if metrics['grad_norm_total'] > self.grad_explode_threshold else 0.0
        return metrics

    def _compute_update_ratios(self, model: torch.nn.Module) -> Dict[str, float]:
        layer_groups = self.inspector.get_parameter_groups()
        default = {f'update_ratio_{g}': 0.0 for g in layer_groups}

        if not self._previous_params:
            self._initialize_previous_params(model)
            return default

        group_delta_sq = {g: 0.0 for g in layer_groups}
        group_weight_sq = {g: 0.0 for g in layer_groups}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            current = param.detach().to(device='cpu', dtype=torch.float32)
            previous = self._previous_params.get(name)
            if previous is None or previous.shape != current.shape:
                self._previous_params[name] = current
                continue

            delta = current - previous
            delta_norm_sq = float(torch.sum(delta * delta).item())
            weight_norm_sq = float(torch.sum(previous * previous).item())

            for group, patterns in layer_groups.items():
                if any(pattern in name for pattern in patterns):
                    group_delta_sq[group] += delta_norm_sq
                    group_weight_sq[group] += weight_norm_sq

            self._previous_params[name] = current

        eps = 1e-12
        metrics: Dict[str, float] = {}
        for group in layer_groups:
            denom = math.sqrt(group_weight_sq[group]) + eps
            metrics[f'update_ratio_{group}'] = math.sqrt(group_delta_sq[group]) / denom if denom > eps else 0.0

        return metrics

    def _initialize_previous_params(self, model: torch.nn.Module):
        self._previous_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._previous_params[name] = param.detach().to(device='cpu', dtype=torch.float32)
