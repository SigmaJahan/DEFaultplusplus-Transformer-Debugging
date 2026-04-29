"""HuggingFace ``Trainer`` callback that drives a :class:`FeatureExtractor`.

Drop the callback into the ``callbacks=`` list of any HuggingFace
``Trainer`` instance and DEFault++ feature extraction will run for the
duration of the training job. The callback records per-step metrics
through the official trainer hooks, rolls them up at every epoch
boundary, and writes the final fixed-length feature vector to disk
when training completes.

Example:

    from transformers import Trainer
    from defaultplusplus.hf_callback import DEFaultPlusCallback

    trainer = Trainer(
        model=model, args=args, train_dataset=ds, eval_dataset=eval_ds,
        callbacks=[DEFaultPlusCallback(out_path="features.json")],
    )
    trainer.train()

The callback subclasses ``transformers.TrainerCallback`` and is only
imported when ``transformers`` is installed. Manual training loops
should use :class:`FeatureExtractor` directly.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from .api import FeatureExtractor
from .config import ExtractionConfig

logger = logging.getLogger(__name__)


# ── Lazy base resolution ─────────────────────────────────────────────────
def _resolve_base():
    """Return ``transformers.TrainerCallback`` or raise ``ImportError``."""
    try:
        from transformers import TrainerCallback
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "DEFaultPlusCallback requires the 'transformers' package. "
            "Install it with `pip install transformers` or use "
            "`FeatureExtractor` directly in a manual training loop."
        ) from e
    return TrainerCallback


_TrainerCallback = _resolve_base()


class DEFaultPlusCallback(_TrainerCallback):
    """Drive a :class:`FeatureExtractor` from inside a HF ``Trainer``.

    Attributes:
        out_path:    optional file path. When set, the finalized feature
                     vector is written here as JSON when training ends.
        arch:        optional architecture-family hint. See
                     :class:`FeatureExtractor`.
        config:      optional :class:`ExtractionConfig`.
        capture_attention: when True (default), the callback enables
                     ``output_attentions`` and ``output_hidden_states``
                     on the model's HF config so attention metrics fire.
                     Set False when memory is tight; layer-internal
                     attention metrics will then be dropped.
        feature_vector: populated after the trainer's ``on_train_end``
                     hook fires. ``None`` until then.
        extractor:   the underlying :class:`FeatureExtractor` instance,
                     created lazily in :meth:`on_train_begin`.
    """

    def __init__(self,
                 out_path: Optional[str | Path] = None,
                 *,
                 arch: Optional[str] = None,
                 config: Optional[ExtractionConfig] = None,
                 capture_attention: bool = True) -> None:
        super().__init__()
        self.out_path = Path(out_path) if out_path else None
        self.arch = arch
        self.config = config
        self.capture_attention = capture_attention

        self.extractor: Optional[FeatureExtractor] = None
        self.feature_vector: Optional[dict] = None
        self._step_clock: Optional[float] = None
        self._last_inputs: Optional[dict[str, Any]] = None
        self._last_outputs: Optional[Any] = None

    # ── Lifecycle hooks ───────────────────────────────────────────────────
    def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
        model = kwargs.get("model")
        if model is None:
            logger.warning("DEFaultPlusCallback: no model in on_train_begin; "
                           "skipping feature extraction for this run.")
            return control

        if self.capture_attention and hasattr(model, "config"):
            try:
                model.config.output_attentions = True
                model.config.output_hidden_states = True
            except Exception:  # pragma: no cover
                logger.exception("DEFaultPlusCallback: failed to enable "
                                 "output_attentions / output_hidden_states "
                                 "on the model config.")

        self.extractor = FeatureExtractor(
            model, arch=self.arch, config=self.config)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_clock = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.extractor is None:
            return control

        loss = kwargs.get("loss")
        outputs = self._last_outputs
        inputs = self._last_inputs or {}
        optimizer = kwargs.get("optimizer")

        step_time = None
        if self._step_clock is not None:
            step_time = time.perf_counter() - self._step_clock
            self._step_clock = None

        try:
            self.extractor.step(
                loss=loss,
                outputs=outputs,
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                optimizer=optimizer,
                step_time=step_time,
                batch_idx=int(state.global_step) if hasattr(state, "global_step") else None,
                epoch=int(state.epoch) if hasattr(state, "epoch") and state.epoch is not None else None,
            )
        except Exception:  # pragma: no cover
            logger.exception("DEFaultPlusCallback: step capture failed; "
                             "continuing the run with degraded features.")
        finally:
            self._last_outputs = None
            self._last_inputs = None
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.extractor is None:
            return control
        epoch = int(state.epoch) if hasattr(state, "epoch") and state.epoch is not None else None
        try:
            self.extractor.epoch_end(epoch)
        except Exception:  # pragma: no cover
            logger.exception("DEFaultPlusCallback: epoch_end failed.")
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if self.extractor is None:
            return control
        metrics = kwargs.get("metrics") or {}
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("eval_"):
                clean[k[len("eval_"):]] = float(v)
            elif k.startswith("test_"):
                clean[k[len("test_"):]] = float(v)
            else:
                clean[k] = float(v)
        if not clean:
            return control
        epoch = int(state.epoch) if hasattr(state, "epoch") and state.epoch is not None else 0
        try:
            self.extractor.record_validation(epoch, clean)
        except Exception:  # pragma: no cover
            logger.exception("DEFaultPlusCallback: record_validation failed.")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.extractor is None:
            return control
        try:
            self.feature_vector = self.extractor.finalize()
            if self.out_path is not None:
                self.extractor.to_json(self.out_path)
                logger.info("DEFaultPlusCallback: wrote %d features to %s",
                            len(self.feature_vector), self.out_path)
        except Exception:  # pragma: no cover
            logger.exception("DEFaultPlusCallback: finalize failed.")
        return control

    # ── Hand-off helpers for richer per-step metrics ──────────────────────
    # The HF ``Trainer`` does not pass batch inputs or model outputs into
    # ``on_step_end``. To capture attention weights and hidden states,
    # forward them in from a custom ``compute_loss`` override or a small
    # model wrapper. Without these, the callback still records gradient,
    # loss, runtime, and update-ratio metrics; only attention-internal
    # metrics degrade.
    def capture_inputs(self, inputs: dict[str, Any]) -> None:
        """Stash a reference to the current batch inputs."""
        self._last_inputs = inputs

    def capture_outputs(self, outputs: Any) -> None:
        """Stash a reference to the current model outputs."""
        self._last_outputs = outputs


__all__ = ["DEFaultPlusCallback"]
