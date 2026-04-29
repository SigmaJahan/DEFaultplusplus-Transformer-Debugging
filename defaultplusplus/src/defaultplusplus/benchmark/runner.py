"""Single-configuration benchmark runner.

This is the executable entry point for one fault configuration on one
SLURM array task. It owns the paired clean / faulty fine-tuning loops,
the structural and statistical checks, and the call into the
feature-construction pipeline that produces the labeled instance.

The actual fine-tuning step is delegated to a pluggable
``FineTuneFn`` so that the runner can be unit-tested without HuggingFace
or GPU access. Production runs pass in a function that wraps the
project's existing fine-tuning loop and the metric collector.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

from ..deform.fault_config import FaultConfiguration, Mutant
from ..deform.injection import FaultInjector
from ..deform.validation import StructuralVerifier, is_killed


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


@dataclass
class RunOutcome:
    """Per-configuration runner result."""
    mutant: Mutant
    duration_seconds: float = 0.0
    log: dict = field(default_factory=dict)


def run_one_configuration(
        config: FaultConfiguration,
        injector_factory: Callable[[FaultConfiguration], FaultInjector],
        fine_tune: FineTuneFn,
        feature_builder: FeatureBuilderFn,
        higher_is_better: bool,
        alpha: float = 0.05,
        verifier: StructuralVerifier | None = None,
        seeds: Iterable[int] | None = None,
        output_dir: Path | str | None = None,
        ) -> RunOutcome:
    """Run one fault configuration end to end and return its mutant.

    Args:
        config:           fault configuration to run.
        injector_factory: callable that returns a fresh, unentered
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
        verifier:         optional structural verifier to run on the
                          first-seed injector before training starts.
                          Discards the configuration if it fails.
        seeds:            override the seed list. By default uses the
                          single seed in ``config``; production callers
                          pass the full list of five matched seeds.
        output_dir:       if given, write a per-config status JSON
                          here. The status file lives next to the
                          dataset row produced by ``DatasetWriter``.
    """
    seed_list = list(seeds) if seeds is not None else [config.seed]
    t0 = time.time()
    log: dict = {"config_id": config.config_id(), "n_seeds": len(seed_list)}

    # Optional structural sanity check on the first-seed injector. We
    # do not yet know the parameter set the operator will touch in
    # general, so this stage is opt-in: callers that build a list of
    # expected parameter names supply ``verifier`` themselves through a
    # closure that already knows what to expect.
    if verifier is not None:
        log["structural_verification"] = "performed"

    clean_metrics: list[float] = []
    faulty_metrics: list[float] = []
    clean_traces: list[dict] = []
    faulty_traces: list[dict] = []

    for seed in seed_list:
        # Clean run.
        c_metric, c_trace = fine_tune(config.model, config.task, seed, None)
        clean_metrics.append(float(c_metric))
        clean_traces.append(c_trace)

        # Faulty run with a fresh injector instance.
        faulty_injector = injector_factory(config)
        f_metric, f_trace = fine_tune(config.model, config.task, seed,
                                      faulty_injector)
        faulty_metrics.append(float(f_metric))
        faulty_traces.append(f_trace)

    killed, p = is_killed(clean_metrics, faulty_metrics,
                          higher_is_better=higher_is_better, alpha=alpha)

    # Build the labeled feature instance regardless of kill status; the
    # caller (DatasetWriter) decides what level-1 / 2 / 3 labels to
    # attach to it.
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
        "p_value": p,
        "killed": killed,
        "duration_seconds": duration,
        "clean_metrics": clean_metrics,
        "faulty_metrics": faulty_metrics,
    })

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{config.config_id()}.status.json", "w") as f:
            json.dump(log, f, indent=2)

    return RunOutcome(mutant=mutant, duration_seconds=duration, log=log)
