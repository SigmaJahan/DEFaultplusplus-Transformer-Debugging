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
            self._modules.append(CacheMetrics(inspector, self.config))

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

        # ``batch_idx`` / ``epoch`` are step bookkeeping the caller may
        # log; they are *not* real metrics, so they enter the per-step
        # output dict but never the aggregator's history. Skipping them
        # here keeps them out of the fixed feature_names schema.
        self.epoch_aggregator.update(metrics)
        metrics['batch_idx'] = batch_idx
        metrics['epoch'] = epoch
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
            # Emit ``final_<val_*>`` for every val_* key the registry knows
            # about plus the four legacy aliases. Keys not present in the
            # actual run are skipped here; the fixed schema does still
            # declare them so downstream classifiers see a stable column
            # set across every task.
            for key in _registry_validation_keys():
                if key in last_val:
                    final[f'final_{key}'] = last_val[key]

        final['final_loss'] = final.get('final_val_loss', final.get('final_train_loss', 0.0))
        final['final_accuracy'] = final.get('final_val_accuracy', final.get('final_train_accuracy', 0.0))

        return final

    @property
    def feature_names(self) -> List[str]:
        """Fully-determined feature names for this collector.

        The list is computable *before* any ``collect_step`` call and is
        stable across runs of the same (architecture, num_layers,
        sampled-layer strategy, parameter groups). Built by walking
        every metric module's :meth:`static_feature_names` declaration
        and crossing it with the epoch aggregator's
        ``_mean`` / ``_var`` / ``_count`` suffixes plus the
        finalize-time keys produced by :meth:`get_final_features`.

        Use :meth:`validate_feature_names` to fail closed when a
        downstream classifier's expected schema doesn't match.
        """
        return build_feature_names(
            modules=self._modules,
            inspector=self.inspector,
        )

    def validate_feature_names(self, expected: List[str]) -> None:
        """Raise ``ValueError`` if the live schema diverges from ``expected``.

        Used by downstream loaders (e.g. ``diagnosis.load_pretrained``)
        to verify that the runtime extractor produces the same feature
        columns the pretrained classifier was trained on.
        """
        live = self.feature_names
        live_set = set(live)
        expected_set = set(expected)
        missing = sorted(expected_set - live_set)
        unexpected = sorted(live_set - expected_set)
        if not missing and not unexpected:
            return
        parts = []
        if missing:
            parts.append(f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            parts.append(f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        raise ValueError(
            "feature_names schema mismatch (live vs expected): "
            + "; ".join(parts)
            + f" — total live={len(live)}, expected={len(expected)}."
        )

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


# ─────────────────────────────────────────────────────────────────────────
# Fixed feature_names schema
# ─────────────────────────────────────────────────────────────────────────
# Aggregator behavior we mirror here lives in
# ``extraction.aggregator.EpochAggregator.finalize_epoch`` (per-epoch
# mean/var/count) and ``compute_window_features`` (early/mid/late +
# slope + final). When you change either of those, this list must move
# in lockstep.

_WINDOW_NAMES = ("early", "mid", "late")

# Validation keys recognized by the ``final_<key>`` aliasing path in
# ``get_final_features``. The full set of windowed val_* keys is built
# dynamically from the task registry by :func:`_registry_validation_keys`
# so the schema covers every task DEFault++ supports without depending on
# the user passing the right metric names to ``record_validation``.
# Keep ``val_perplexity`` as a generic decoder shorthand even though no
# registry task emits it directly — HF callers sometimes derive it
# alongside ``val_loss`` and forward both.
_LEGACY_VALIDATION_FINAL_KEYS = ("val_accuracy", "val_loss", "val_perplexity")

_FINAL_FIXED_KEYS = (
    "final_train_loss",
    "final_train_accuracy",
    "final_grad_norm_total",
    "best_train_accuracy",
    "best_train_loss",
    "final_loss",
    "final_accuracy",
)


def _registry_validation_keys() -> set[str]:
    """Return the union of ``val_<raw_metric>`` keys across the task registry.

    Reading this from :mod:`benchmark.task_metrics` rather than from a
    static list means the schema automatically picks up new tasks as
    they are registered — the user never has to remember to forward an
    extra metric to :meth:`FeatureExtractor.record_validation`.
    """
    keys = set(_LEGACY_VALIDATION_FINAL_KEYS)
    try:
        # Local import: collector.py is in the runtime extraction path
        # and must not pull in the benchmark CLI module at import time.
        from ..benchmark.task_metrics import TASK_METRICS

        for spec in TASK_METRICS.values():
            for raw in spec.raw_metrics:
                keys.add(f"val_{raw}")
    except ImportError:  # pragma: no cover - benchmark module always present
        pass
    return keys


def build_feature_names(
    *,
    modules: List[Any],
    inspector: ModelInspector,
) -> List[str]:
    """Return the schema-pinned feature_names list for a collector.

    The output is the union of:
      * per-step keys each metric module declares via
        ``static_feature_names`` (varies with arch / num_layers);
      * gradient running-window extras
        (``<grad_norm_*>_window_var``, ``<grad_norm_*>_gns``,
        ``gradient_variance``, ``gradient_noise_scale``);
      * epoch aggregator suffixes (``<key>_mean / _var / _count``);
      * windowed features (``<key>_early_mean / _early_slope`` etc.,
        plus ``<key>_final``);
      * finalize-time fixed keys (``final_train_loss`` etc.) and the
        validation-prefixed keys when the run records them.

    The result is sorted for stable serialization. Ordering is fixed
    across runs, so the diagnostic model's input layer can be pinned
    against this list at training time.
    """
    # 1. Per-step raw keys from each metric module.
    raw_step: set[str] = set()
    for module in modules:
        try:
            raw_step.update(module.static_feature_names())
        except Exception:  # pragma: no cover - defensive
            continue

    # Running-window extras emitted in ``MetricCollector.collect_step``
    # for every grad_norm_* key the gradient module produced.
    grad_norm_keys = {k for k in raw_step if k.startswith("grad_norm_")}
    for key in grad_norm_keys:
        raw_step.add(f"{key}_window_var")
        raw_step.add(f"{key}_gns")
    raw_step.add("gradient_variance")
    raw_step.add("gradient_noise_scale")

    # 2. Windowed features emitted by ``compute_window_features``:
    # ``<key>_<window>_mean / _<window>_slope`` plus ``<key>_final``,
    # keyed by the raw step name (the aggregator's metric_history is
    # keyed that way).
    window_keys: set[str] = set()
    for key in raw_step:
        for win in _WINDOW_NAMES:
            window_keys.add(f"{key}_{win}_mean")
            window_keys.add(f"{key}_{win}_slope")
        window_keys.add(f"{key}_final")

    # Validation series: collector adds a ``val_`` prefix and feeds the
    # same window features path. Keys come from the task registry so
    # the schema covers every supported task without user wiring.
    val_keys = _registry_validation_keys()
    val_window_keys: set[str] = set()
    for key in val_keys:
        for win in _WINDOW_NAMES:
            val_window_keys.add(f"{key}_{win}_mean")
            val_window_keys.add(f"{key}_{win}_slope")
        val_window_keys.add(f"{key}_final")

    # 3. Finalize-time fixed keys plus per-validation final aliases.
    # ``final_<val_*>`` covers the four legacy aliases AND every task
    # registry key (so a CoLA run's ``final_val_matthews_correlation``
    # is in the schema even though it isn't a legacy alias).
    final_keys = set(_FINAL_FIXED_KEYS)
    for key in val_keys:
        final_keys.add(f"final_{key}")

    return sorted(window_keys | val_window_keys | final_keys)
