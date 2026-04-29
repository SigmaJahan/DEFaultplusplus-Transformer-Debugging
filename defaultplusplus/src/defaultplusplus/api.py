"""Public feature-extraction API for fine-tuning runs.

The :class:`FeatureExtractor` is the user-facing entry point of the
package. It wraps :class:`ModelInspector` + :class:`MetricCollector`
behind a small, training-loop-friendly surface so a caller can attach
DEFault++ to an existing fine-tuning script without knowing the
internals.

Three usage shapes are supported:

  Manual training loop
  ────────────────────
      from defaultplusplus import FeatureExtractor

      fx = FeatureExtractor(model, arch="encoder")
      for epoch in range(num_epochs):
          for step, batch in enumerate(loader):
              t0 = time.time()
              outputs = model(**batch, output_attentions=True,
                              output_hidden_states=True)
              outputs.loss.backward()
              optimizer.step(); optimizer.zero_grad()
              fx.step(loss=outputs.loss, outputs=outputs,
                      input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"],
                      optimizer=optimizer,
                      step_time=time.time() - t0)
          fx.epoch_end(epoch)
          fx.record_validation(epoch, val_metrics)
      feature_vector = fx.finalize()

  HuggingFace Trainer
  ───────────────────
      from defaultplusplus.hf_callback import DEFaultPlusCallback

      trainer = Trainer(
          model=model, args=args, ...,
          callbacks=[DEFaultPlusCallback(arch="encoder",
                                         out_path="features.json")],
      )
      trainer.train()

  As a context manager
  ────────────────────
      with FeatureExtractor(model, arch="encoder") as fx:
          for epoch in ...:
              for step, batch in ...:
                  ...
                  fx.step(...)
              fx.epoch_end(epoch)
      # fx.feature_vector is populated on exit.

The single source of truth for output keys is ``docs/SPEC.md``.
The collector emits exactly those keys plus the
layer-/step-/epoch-/training-phase aggregates produced by
``feature_construction``.
"""
from __future__ import annotations

import json
import logging
import math
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import ExtractionConfig
from .extraction.collector import MetricCollector
from .extraction.feature_construction import (
    EpochTrace,
    LayerInternalTrace,
    StepTrace,
    TrainingTrace,
    build_feature_vector,
)
from .extraction.inspector import ModelInspector


logger = logging.getLogger(__name__)


# Architectural family aliases the extractor accepts on construction.
_FAMILY_ALIASES: dict[str, str] = {
    "encoder": "encoder",
    "bert": "encoder",
    "roberta": "encoder",
    "distilbert": "encoder",
    "decoder": "decoder",
    "gpt": "decoder",
    "gpt2": "decoder",
    "gpt-neo": "decoder",
    "causal-lm": "decoder",
}


class FeatureExtractor(AbstractContextManager):
    """Collect DEFault++ training-time features from one fine-tuning run.

    Construct one extractor per fine-tuning run. The extractor inspects
    the model's structure on construction, then captures metrics on
    each call to :meth:`step` and rolls them up at each
    :meth:`epoch_end`. After the run, :meth:`finalize` returns the
    fixed-length feature vector that the diagnostic model consumes.

    Args:
        model:        any HuggingFace transformer (encoder or decoder
                      family).
        arch:         optional architectural family hint
                      (``"encoder"``, ``"decoder"``, or a model-family
                      alias). When omitted, the family is auto-detected
                      by :class:`ModelInspector`. Use this to fail
                      closed when the model belongs to an unsupported
                      family.
        config:       optional :class:`ExtractionConfig` overriding
                      collection thresholds, sampling cadence, and
                      special-token IDs.
        record_per_step: when True, the extractor keeps every per-step
                      metric dictionary in memory under
                      :attr:`step_history`. Useful for debugging and
                      offline analysis; off by default to keep memory
                      bounded on long runs.
    """

    SUPPORTED_FAMILIES: frozenset[str] = frozenset({"encoder", "decoder"})

    def __init__(self,
                 model: nn.Module,
                 arch: Optional[str] = None,
                 *,
                 config: Optional[ExtractionConfig] = None,
                 record_per_step: bool = False) -> None:
        self.model = model
        self.config = config or ExtractionConfig()
        self.record_per_step = record_per_step

        self.inspector = ModelInspector(model)
        detected_family = self.inspector.arch_family

        if arch is None:
            self.arch = detected_family
        else:
            requested = _FAMILY_ALIASES.get(arch.lower())
            if requested is None:
                raise ValueError(
                    f"Unknown arch hint {arch!r}. Pass one of "
                    f"{sorted(_FAMILY_ALIASES)} or omit to auto-detect.")
            if requested != detected_family:
                raise ValueError(
                    f"Requested arch={requested!r} but the inspector "
                    f"detected {detected_family!r}. Refusing to "
                    "continue: a wrong family hint silently corrupts "
                    "feature extraction.")
            self.arch = requested

        if self.arch not in self.SUPPORTED_FAMILIES:
            raise ValueError(
                f"Architecture family {self.arch!r} is not supported. "
                f"Supported families: {sorted(self.SUPPORTED_FAMILIES)}.")

        self.collector = MetricCollector(self.inspector, self.config)

        # Install sublayer-boundary hooks now so subsequent forward passes
        # populate the capture dict in time for ``step``.
        self.collector.sublayer_capture.install()

        self.step_history: list[dict[str, float]] = []
        self.feature_vector: Optional[dict[str, float]] = None
        self._last_step_time: Optional[float] = None
        self._epoch_counter: int = 0
        self._closed: bool = False

    # ── Public API ────────────────────────────────────────────────────────
    def step(self,
             *,
             loss: Any = None,
             outputs: Any = None,
             input_ids: Optional[torch.Tensor] = None,
             attention_mask: Optional[torch.Tensor] = None,
             labels: Optional[torch.Tensor] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             step_time: Optional[float] = None,
             batch_idx: Optional[int] = None,
             epoch: Optional[int] = None,
             ) -> dict[str, float]:
        """Capture metrics for one training step.

        Call this immediately after the forward and backward passes,
        before the optimizer step is consumed by another batch. The
        ``outputs`` object should carry attention weights and hidden
        states (HuggingFace returns these when ``output_attentions=True``
        and ``output_hidden_states=True`` are set on the call).

        Args:
            loss:           scalar loss tensor or ``float``. The collector
                            also accepts any object exposing ``.item()``.
            outputs:        HuggingFace model output. Reads ``attentions``
                            and ``hidden_states`` automatically.
            input_ids:      optional ``(batch, seq)`` token ids. Improves
                            attention mass and special-token masking
                            precision when the tokenizer's special token
                            IDs are also set in :attr:`config`.
            attention_mask: optional ``(batch, seq)`` mask used by the
                            attention metrics.
            labels:         optional supervision tensor. Required for
                            classification accuracy / margin metrics.
            optimizer:      the active optimizer. Required for update
                            ratio / activity metrics.
            step_time:      wall-clock seconds for this step. If omitted,
                            the extractor measures the time since the
                            previous :meth:`step` call.
            batch_idx:      optional batch index inside the current
                            epoch. Defaults to the running step counter.
            epoch:          optional epoch index. Defaults to the
                            extractor's internal epoch counter, which is
                            incremented by :meth:`epoch_end`.

        Returns:
            The per-step metrics dictionary emitted by the collector.
        """
        if self._closed:
            raise RuntimeError("FeatureExtractor.step called after finalize()")

        if step_time is None:
            now = time.perf_counter()
            if self._last_step_time is not None:
                step_time = now - self._last_step_time
            self._last_step_time = now
        else:
            self._last_step_time = time.perf_counter()

        metrics = self.collector.collect_step(
            loss=loss,
            model=self.model,
            optimizer=optimizer,
            outputs=outputs,
            labels=labels,
            attention_mask=attention_mask,
            input_ids=input_ids,
            batch_idx=batch_idx if batch_idx is not None else self.collector.batch_counter,
            epoch=epoch if epoch is not None else self._epoch_counter,
            step_time=step_time,
        )

        if self.record_per_step:
            self.step_history.append(metrics)

        return metrics

    def epoch_end(self, epoch: Optional[int] = None) -> dict[str, float]:
        """Roll up step-level metrics into an epoch summary.

        Args:
            epoch: optional epoch index. Defaults to the extractor's
                   internal counter.

        Returns:
            The epoch summary emitted by the collector.
        """
        if self._closed:
            raise RuntimeError("FeatureExtractor.epoch_end called after finalize()")
        idx = self._epoch_counter if epoch is None else int(epoch)
        summary = self.collector.finalize_epoch(idx)
        self._epoch_counter = idx + 1
        return summary

    def record_validation(self, epoch: int, metrics: dict[str, float]) -> None:
        """Attach validation-set metrics for the just-finished epoch.

        Validation metrics are folded into the windowed feature
        construction at :meth:`finalize`. Pass them directly from your
        evaluation loop; the keys ``accuracy`` / ``loss`` /
        ``perplexity`` / ``f1_score`` are recognized as level-1
        validation signals.
        """
        if self._closed:
            raise RuntimeError("FeatureExtractor.record_validation called after finalize()")
        self.collector.record_validation_metrics(int(epoch),
                                                 {k: float(v) for k, v in metrics.items()})

    def finalize(self) -> dict[str, float]:
        """Build the fixed-length feature vector for the run.

        Merges the collector's final-feature dictionary (which carries
        the contract keys directly) with the layer / step / epoch /
        training-phase aggregates produced by
        :func:`build_feature_vector`. Subsequent calls return the
        cached result.
        """
        if self.feature_vector is not None:
            return self.feature_vector

        flat_features = self.collector.get_final_features()
        trace_features = self._build_trace_features()

        merged: dict[str, float] = {}
        merged.update(flat_features)
        for k, v in trace_features.items():
            # Trace-aggregated keys carry a `__trace__` segment so they
            # cannot collide with the contract keys.
            merged[f"trace__{k}"] = v

        # Sanitize: every value finite, every key a str.
        cleaned: dict[str, float] = {}
        for key, value in merged.items():
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                f = 0.0
            cleaned[str(key)] = f

        self.feature_vector = cleaned
        return cleaned

    def to_json(self, path: str | Path) -> Path:
        """Serialize the finalized feature vector to JSON.

        Calls :meth:`finalize` if it has not been called already.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.finalize(), indent=2, sort_keys=True))
        return path

    def reset(self) -> None:
        """Discard all collected state so the extractor can be reused."""
        self.collector.reset()
        self.collector.sublayer_capture.install()
        self.step_history.clear()
        self.feature_vector = None
        self._last_step_time = None
        self._epoch_counter = 0
        self._closed = False

    # ── Context-manager glue ──────────────────────────────────────────────
    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Materialize the feature vector on clean exit so callers can
        # read ``self.feature_vector`` after the `with` block.
        if exc_type is None and not self._closed:
            try:
                self.finalize()
            except Exception:
                logger.exception("FeatureExtractor.finalize raised during __exit__")
        self._closed = True
        # Always tear down the sublayer hooks so the model is left clean.
        try:
            self.collector.sublayer_capture.remove()
        except Exception:  # pragma: no cover - defensive
            logger.exception("FeatureExtractor: sublayer_capture.remove() failed")
        return False  # do not suppress exceptions

    # ── Trace adapter ─────────────────────────────────────────────────────
    def _build_trace_features(self) -> dict[str, float]:
        """Convert collector state into a TrainingTrace and aggregate.

        The collector keeps step-level metric histories inside its
        ``epoch_aggregator``. We turn those plus the validation history
        into the typed traces consumed by ``build_feature_vector``,
        which produces the layer / step / epoch / training-phase
        aggregates that the diagnostic model expects.
        """
        epoch_history = self.collector.epoch_aggregator.metric_history
        if not epoch_history:
            return {}

        # Collapse per-epoch (epoch_idx, mean) pairs into 1-D arrays.
        # ``metric_history`` records the within-epoch mean of every key
        # the collector ever touched.
        step_level: dict[str, StepTrace] = {}
        for key, series in epoch_history.items():
            if not series:
                continue
            values = np.asarray([v for _, v in series], dtype=np.float64)
            if values.size == 0:
                continue
            step_level[key] = StepTrace(values)

        # Validation: per-epoch series keyed by ``val_*``.
        epoch_level: dict[str, EpochTrace] = {}
        for key, series in self.collector.validation_metric_history.items():
            if not series:
                continue
            values = np.asarray([v for _, v in series], dtype=np.float64)
            if values.size == 0:
                continue
            epoch_level[key] = EpochTrace(values)

        # No layer-internal traces here: the collector already encodes
        # per-layer metrics with an ``L{layer_idx}_`` prefix on the
        # step-level dict, so they flow through ``step_level`` above.
        layer_internal: dict[str, LayerInternalTrace] = {}

        boundaries = [i + 1 for i in range(len(self.collector.epoch_metrics_history))]
        if not boundaries:
            boundaries = [len(next(iter(step_level.values())).values)] if step_level else [0]

        trace = TrainingTrace(
            layer_internal=layer_internal,
            step_level=step_level,
            epoch_level=epoch_level,
            epoch_boundaries=boundaries,
        )
        return build_feature_vector(trace)

    # ── Convenience accessors ─────────────────────────────────────────────
    @property
    def epoch_metrics_history(self) -> list[dict[str, float]]:
        """Per-epoch summaries the collector has produced so far."""
        return list(self.collector.epoch_metrics_history)

    @property
    def feature_names(self) -> list[str]:
        """Stable list of contract feature names known to the collector."""
        return list(self.collector.feature_names)
