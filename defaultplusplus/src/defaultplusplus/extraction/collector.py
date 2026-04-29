"""
MetricCollector — orchestrates all metric modules and epoch aggregation.

Ported from metric_collector.py with architecture-agnostic design.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .inspector import ModelInspector
from .aggregator import EpochAggregator, RunningMetrics, compute_window_features
from .sublayer_capture import SublayerCapture
from .metrics.training import TrainingMetrics
from .metrics.gradient import GradientMetrics
from .metrics.attention import AttentionMetrics
from .metrics.structural import StructuralMetrics
from .metrics.logit import LogitMetrics
from .metrics.positional import PositionalMetrics
from .metrics.cache import CacheMetrics
from ..config import ExtractionConfig


class MetricCollector:
    """Orchestrates all metric modules for a training run."""

    def __init__(
        self,
        inspector: ModelInspector,
        config: Optional[ExtractionConfig] = None,
    ):
        self.inspector = inspector
        self.config = config or ExtractionConfig()

        # Sublayer-boundary capture: forward hooks on FFN/LN/attn modules
        # plus the Q/K/V projections feed exact-tap tensors into
        # StructuralMetrics and AttentionMetrics. Installed lazily on the
        # first ``collect_step`` call so the constructor stays cheap and
        # standalone metric-module tests keep working.
        self.sublayer_capture = SublayerCapture(inspector)

        # Instantiate metric modules
        self._modules = [
            TrainingMetrics(inspector),
            GradientMetrics(inspector, self.config),
            AttentionMetrics(inspector, self.config),
            StructuralMetrics(inspector, self.config),
            LogitMetrics(inspector, self.config),
            PositionalMetrics(inspector, self.config),
        ]

        # Decoder-only module
        if inspector.arch_family == 'decoder':
            self._modules.append(CacheMetrics(inspector))

        self.epoch_aggregator = EpochAggregator()
        self.running_metrics = RunningMetrics(window_size=self.config.gradient_window)

        self.batch_counter = 0
        self.epoch_metrics_history: List[Dict[str, float]] = []
        self.validation_history: List[Dict[str, float]] = []
        self.validation_metric_history: Dict[str, List] = defaultdict(list)

    def collect_step(
        self,
        *,
        loss: Any = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
        epoch: int = 0,
        step_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Collect metrics for a single training step."""
        self.batch_counter += 1

        # Extract attention/hidden from outputs
        attention_weights = getattr(outputs, 'attentions', None)
        hidden_states = getattr(outputs, 'hidden_states', None)

        kwargs = dict(
            loss=loss, model=model, optimizer=optimizer, outputs=outputs,
            labels=labels, attention_weights=attention_weights,
            hidden_states=hidden_states, attention_mask=attention_mask,
            input_ids=input_ids, batch_idx=batch_idx, epoch=epoch,
            step_time=step_time, sublayer_capture=self.sublayer_capture,
        )

        metrics: Dict[str, float] = {}
        for module in self._modules:
            # Skip expensive modules on non-sampled steps
            if self.batch_counter % max(1, self.config.activation_interval) != 0:
                if module.requires_hidden_states and not isinstance(module, StructuralMetrics):
                    continue

            module_metrics = module.collect(**kwargs)
            metrics.update(module_metrics)

        # Running gradient stats
        grad_extras: Dict[str, float] = {}
        for key, value in metrics.items():
            if key.startswith('grad_norm_'):
                self.running_metrics.update(key, value)
                grad_extras[f'{key}_window_var'] = self.running_metrics.get_variance(key)
                grad_extras[f'{key}_gns'] = self.running_metrics.get_noise_scale(key)
        metrics.update(grad_extras)

        metrics['gradient_variance'] = self.running_metrics.get_variance('grad_norm_total')
        metrics['gradient_noise_scale'] = self.running_metrics.get_noise_scale('grad_norm_total')

        metrics['batch_idx'] = batch_idx
        metrics['epoch'] = epoch

        self.epoch_aggregator.update(metrics)
        # Captures from this forward have been consumed; drop refs so the
        # next step starts from a clean slate.
        self.sublayer_capture.clear()
        return metrics

    def finalize_epoch(self, epoch_idx: int) -> Dict[str, float]:
        """Finalize running statistics for the epoch."""
        epoch_metrics = self.epoch_aggregator.finalize_epoch(epoch_idx)
        epoch_metrics['epoch'] = epoch_idx
        self.epoch_metrics_history.append(epoch_metrics.copy())
        return epoch_metrics

    def record_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Store validation metrics for windowed features."""
        prefixed = {f'val_{k}': v for k, v in metrics.items()}
        prefixed['epoch'] = epoch
        self.validation_history.append(prefixed)
        for key, value in prefixed.items():
            if key == 'epoch':
                continue
            self.validation_metric_history[key].append((epoch + 1, value))

    def get_final_features(self) -> Dict[str, float]:
        """Aggregate final metrics, windows, and derived features."""
        if not self.epoch_metrics_history:
            return {}

        final: Dict[str, float] = {}
        last = self.epoch_metrics_history[-1]

        final['final_train_loss'] = last.get('train_loss_mean', 0.0)
        final['final_train_accuracy'] = last.get('accuracy_mean', 0.0)
        final['final_grad_norm_total'] = last.get('grad_norm_total_mean', 0.0)

        acc_series = [e.get('accuracy_mean', 0.0) for e in self.epoch_metrics_history]
        loss_series = [e.get('train_loss_mean', math.inf) for e in self.epoch_metrics_history]
        final['best_train_accuracy'] = max(acc_series) if acc_series else 0.0
        final['best_train_loss'] = min(loss_series) if loss_series else math.inf

        total_epochs = len(self.epoch_metrics_history)
        final.update(compute_window_features(self.epoch_aggregator.metric_history, total_epochs))
        final.update(compute_window_features(self.validation_metric_history, total_epochs))

        if self.validation_history:
            last_val = self.validation_history[-1]
            for key in ('val_accuracy', 'val_loss', 'val_perplexity', 'val_f1_score'):
                if key in last_val:
                    final[f'final_{key}'] = last_val[key]

        final['final_loss'] = final.get('final_val_loss', final.get('final_train_loss', 0.0))
        final['final_accuracy'] = final.get('final_val_accuracy', final.get('final_train_accuracy', 0.0))

        return final

    @property
    def feature_names(self) -> List[str]:
        """Return deterministic list of feature names from the last epoch."""
        if not self.epoch_metrics_history:
            return []
        last = self.epoch_metrics_history[-1]
        return sorted(k for k in last.keys() if k != 'epoch')

    def reset(self):
        """Reset all state."""
        self.epoch_aggregator.reset()
        self.epoch_metrics_history = []
        self.validation_history = []
        self.validation_metric_history = defaultdict(list)
        self.batch_counter = 0
        self.running_metrics.reset()
        # Tear down hooks; ``FeatureExtractor`` reinstalls them on reuse.
        self.sublayer_capture.remove()
