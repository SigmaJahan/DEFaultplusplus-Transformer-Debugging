"""Single-configuration benchmark runner.

This is the executable entry point for one fault configuration on one
SLURM array task. It owns the paired clean / faulty fine-tuning loops,
the structural and statistical checks, and the call into the
feature-construction pipeline that produces the labeled instance.

The actual fine-tuning step is delegated to a pluggable
``FineTuneFn`` so that the runner can be unit-tested without HuggingFace
or GPU access. Production runs pass in a function that wraps the
project's existing fine-tuning loop and the metric collector.

Crash isolation
---------------
A benchmark run touches dozens of operators per model; some of them
genuinely break the model (NaN logits, OOM, shape mismatch from a
fault that the structural verifier would otherwise have caught). The
runner *never* lets a single configuration take down the whole batch:
any verifier failure or per-seed exception is captured into the
returned :class:`RunOutcome` with ``status != "ok"`` and the caller
(``DatasetWriter`` / CLI) drops the row from the dataset and records
the discard reason in the run log.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

import torch

from ..deform.fault_config import FaultConfiguration, Mutant
from ..deform.injection import FaultInjector
from ..deform.validation import StructuralVerifier, VerificationResult, is_killed


class FineTuneFn(Protocol):
    """Callable that fine-tunes a model under one seed and returns its
    test-set metric and the captured training trace.

    Implementations must:
      - load the requested HuggingFace model and downstream task,
      - apply the configured optimizer, schedule, and epoch budget,
      - if ``injector`` is given, enter it before any forward pass so
        the entire fine-tuning run is exposed to the fault,
      - emit a layer-/step-/epoch-level training trace through the
        metric collector,
      - return ``(test_metric, training_trace)``.

    Args:
        model_name: HuggingFace model name.
        task:       downstream task name.
        seed:       random seed.
        injector:   optional :class:`FaultInjector` to apply for the
                    duration of the run. ``None`` for clean runs.
    """
    def __call__(self,
                 model_name: str,
                 task: str,
                 seed: int,
                 injector: FaultInjector | None,
                 ) -> tuple[float, dict]: ...


class FeatureBuilderFn(Protocol):
    """Aggregator that turns paired training traces into one labeled
    instance for the diagnostic model.

    Production implementations live in
    :mod:`defaultplusplus.extraction.feature_construction` and follow
    the layer / step / epoch / training-phase pipeline.
    """
    def __call__(self,
                 clean_traces: list[dict],
                 faulty_traces: list[dict]
                 ) -> dict: ...


class RunStatus(str, Enum):
    """Why a configuration succeeded or was discarded.

    ``ok``                   completed all seed pairs and produced a
                              valid mutant for the dataset writer.
    ``verifier_failed``      pre-flight :class:`StructuralVerifier` said
                              the injector would not target the model
                              cleanly; no fine-tuning was attempted.
    ``runtime_error``        an exception propagated out of one of the
                              seed-paired fine-tunes.
    ``invalid_metric``       a fine-tune returned ``NaN`` / ``±Inf`` for
                              the test metric, so the kill-test would
                              be meaningless. Treated as a soft crash.
    """
    OK = "ok"
    VERIFIER_FAILED = "verifier_failed"
    RUNTIME_ERROR = "runtime_error"
    INVALID_METRIC = "invalid_metric"


@dataclass
class RunOutcome:
    """Per-configuration runner result.

    ``mutant`` is ``None`` when ``status != ok``; the row is meant to
    be dropped by the caller and recorded in the discard log built
    from :attr:`log`.
    """
    mutant: Optional[Mutant] = None
    duration_seconds: float = 0.0
    log: dict = field(default_factory=dict)
    status: RunStatus = RunStatus.OK
    discard_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.OK

    def discard_record(self) -> dict:
        """Compact JSON-serializable summary for the discard log."""
        return {
            "config_id": self.log.get("config_id"),
            "operator_id": self.log.get("operator_id"),
            "model": self.log.get("model"),
            "task": self.log.get("task"),
            "severity": self.log.get("severity"),
            "layers": self.log.get("layers"),
            "status": self.status.value,
            "reason": self.discard_reason,
            "duration_seconds": self.duration_seconds,
        }


VerifierFactory = Callable[[FaultConfiguration], Optional[VerificationResult]]


def run_one_configuration(
        config: FaultConfiguration,
        injector_factory: Callable[[FaultConfiguration], FaultInjector],
        fine_tune: FineTuneFn,
        feature_builder: FeatureBuilderFn,
        higher_is_better: bool,
        alpha: float = 0.05,
        verifier_factory: VerifierFactory | None = None,
        seeds: Iterable[int] | None = None,
        output_dir: Path | str | None = None,
        ) -> RunOutcome:
    """Run one fault configuration end to end and return a :class:`RunOutcome`.

    The runner discards the entire configuration on any of:

      * the optional pre-flight verifier reports ``ok=False``,
      * any seed (clean or faulty) raises an exception inside
        ``fine_tune``,
      * any seed returns a non-finite test metric (NaN / ±Inf).

    Discards are reported via :attr:`RunOutcome.status` and
    :attr:`RunOutcome.discard_reason`; the mutant field is ``None``
    so the caller skips the dataset row. Sign-flip kill testing only
    happens on a complete, finite-metric set of paired seeds — partial
    seed completion is never aggregated, since the paper's exact
    one-sided test at α=0.05 requires all five matched seeds.

    Args:
        config:           fault configuration to run.
        injector_factory: callable returning a fresh, unentered
                          ``FaultInjector`` for ``config``. A new
                          injector is constructed per seed pair so the
                          context manager invariants are preserved.
        fine_tune:        production fine-tuning callable that returns
                          ``(test_metric, training_trace)``.
        feature_builder:  aggregator that converts paired traces to the
                          labeled instance for the diagnostic model.
        higher_is_better: whether the test metric is accuracy-like
                          (encoder GLUE) or perplexity-like (decoder
                          language modeling).
        alpha:            kill threshold for the sign-flip permutation
                          test. Default 0.05.
        verifier_factory: optional factory that, given the config,
                          loads a *throwaway* model copy and returns a
                          :class:`VerificationResult` (or ``None`` to
                          skip verification). The runner never calls
                          fine-tune when verification fails.
        seeds:            override the seed list. By default uses the
                          single seed in ``config``; production callers
                          pass the full list of five matched seeds.
        output_dir:       if given, write a per-config status JSON
                          here. The status file lives next to the
                          dataset row produced by ``DatasetWriter``.
    """
    seed_list = list(seeds) if seeds is not None else [config.seed]
    t0 = time.time()
    log: dict = {
        "config_id": config.config_id(),
        "operator_id": config.operator_id,
        "model": config.model,
        "task": config.task,
        "severity": config.severity,
        "layers": list(config.layers) if config.layers else [],
        "n_seeds": len(seed_list),
    }

    # ── Stage 1: structural pre-flight verification ──────────────────
    if verifier_factory is not None:
        try:
            ver_result = verifier_factory(config)
        except Exception as exc:  # pragma: no cover - factory errors are rare
            return _build_discard_outcome(
                config, log, t0, RunStatus.VERIFIER_FAILED,
                f"verifier_factory raised {type(exc).__name__}: {exc}",
                output_dir,
            )
        if ver_result is not None and not ver_result.ok:
            log["structural_verification"] = "failed"
            log["verifier_message"] = ver_result.message
            return _build_discard_outcome(
                config, log, t0, RunStatus.VERIFIER_FAILED,
                ver_result.message or "structural verification failed",
                output_dir,
            )
        if ver_result is not None:
            log["structural_verification"] = "passed"

    clean_metrics: list[float] = []
    faulty_metrics: list[float] = []
    clean_traces: list[dict] = []
    faulty_traces: list[dict] = []

    # ── Stage 2: paired clean / faulty fine-tuning ───────────────────
    for seed in seed_list:
        # Clean runs are *not* protected here: clean failures usually
        # signal a model/dataset/environment problem unrelated to the
        # fault and should bubble up so the operator running the
        # benchmark fixes their environment.
        c_metric, c_trace = fine_tune(config.model, config.task, seed, None)
        if not _is_finite(c_metric):
            return _build_discard_outcome(
                config, log, t0, RunStatus.INVALID_METRIC,
                f"clean run produced non-finite metric (seed={seed}, value={c_metric!r})",
                output_dir,
            )
        clean_metrics.append(float(c_metric))
        clean_traces.append(c_trace)

        # Faulty runs are protected: any exception or non-finite metric
        # discards the whole configuration. We never aggregate a
        # partial set of seeds — the kill test would be invalid.
        try:
            faulty_injector = injector_factory(config)
            f_metric, f_trace = fine_tune(
                config.model, config.task, seed, faulty_injector
            )
        except Exception as exc:
            _release_cuda_cache()
            return _build_discard_outcome(
                config, log, t0, RunStatus.RUNTIME_ERROR,
                f"faulty run raised {type(exc).__name__} on seed={seed}: {exc}",
                output_dir,
            )

        if not _is_finite(f_metric):
            _release_cuda_cache()
            return _build_discard_outcome(
                config, log, t0, RunStatus.INVALID_METRIC,
                f"faulty run produced non-finite metric (seed={seed}, value={f_metric!r})",
                output_dir,
            )
        faulty_metrics.append(float(f_metric))
        faulty_traces.append(f_trace)

    # ── Stage 3: kill test + feature instance ────────────────────────
    killed, p = is_killed(clean_metrics, faulty_metrics,
                          higher_is_better=higher_is_better, alpha=alpha)

    feature_vector = feature_builder(clean_traces, faulty_traces)

    mutant = Mutant(
        config=config,
        clean_metrics=tuple(clean_metrics),
        faulty_metrics=tuple(faulty_metrics),
        p_value=p,
        killed=killed,
        feature_vector=feature_vector,
    )

    duration = time.time() - t0
    log.update({
        "status": RunStatus.OK.value,
        "p_value": p,
        "killed": killed,
        "duration_seconds": duration,
        "clean_metrics": clean_metrics,
        "faulty_metrics": faulty_metrics,
    })

    if output_dir is not None:
        _write_status_json(output_dir, config.config_id(), log)

    return RunOutcome(
        mutant=mutant,
        duration_seconds=duration,
        log=log,
        status=RunStatus.OK,
    )


def _is_finite(value: Any) -> bool:
    """Return True iff ``value`` is a real-valued finite number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _release_cuda_cache() -> None:
    """Free CUDA memory after a faulty run crashes (no-op on CPU/MPS)."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass


def _build_discard_outcome(
        config: FaultConfiguration,
        log: dict,
        t0: float,
        status: RunStatus,
        reason: str,
        output_dir: Path | str | None,
        ) -> RunOutcome:
    """Build a discard ``RunOutcome`` and persist its status JSON."""
    duration = time.time() - t0
    log = dict(log)
    log.update({
        "status": status.value,
        "discard_reason": reason,
        "duration_seconds": duration,
    })
    if output_dir is not None:
        _write_status_json(output_dir, config.config_id(), log)
    return RunOutcome(
        mutant=None,
        duration_seconds=duration,
        log=log,
        status=status,
        discard_reason=reason,
    )


def _write_status_json(output_dir: Path | str, config_id: str, log: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{config_id}.status.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
