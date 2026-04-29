"""Tests for the runner's discard-on-failure machinery.

Covers four discard paths the runner must enforce so a single broken
configuration cannot poison a benchmark batch:

  * verifier_failed     pre-flight verifier reports ok=False
  * runtime_error       fine_tune raises on a faulty seed
  * invalid_metric      fine_tune returns NaN/Inf
  * partial seeds       any one bad seed discards the whole config

Each test uses a fake ``fine_tune`` so the discard logic can be
exercised without HuggingFace or any real training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────
def _config(operator_id: str = "QZQ", seed: int = 42):
    from defaultplusplus.deform.fault_config import FaultConfiguration

    return FaultConfiguration(
        model="fake-model",
        task="fake-task",
        operator_id=operator_id,
        layers=(),
        severity="low",
        param_value=None,
        seed=seed,
    )


class _FakeInjector:
    """Minimal context manager that satisfies the runner's interface."""

    def __init__(self, *_args, **_kwargs):
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _injector_factory(_config):
    def _bind(_model=None):
        return _FakeInjector()
    # The runner calls injector_factory(config) and expects either a
    # FaultInjector or a callable that accepts a model and returns one.
    # Returning the _FakeInjector instance directly keeps the test focused.
    return _FakeInjector()


def _ok_fine_tune(model, task, seed, injector):
    # Faulty runs return a slightly different metric so the kill test
    # is non-degenerate. Clean=0.8 vs faulty=0.5 is a clean kill.
    return (0.5 if injector is not None else 0.8), {"seed": seed}


def _build_features(_clean_traces, _faulty_traces):
    return {"feat_0": 0.1, "feat_1": 0.2}


# ─────────────────────────────────────────────────────────────────────────
# OK path baseline (sanity check)
# ─────────────────────────────────────────────────────────────────────────
def test_ok_path_returns_status_ok_and_mutant() -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_ok_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        seeds=[1, 2, 3, 4, 5],
    )
    assert outcome.status == RunStatus.OK
    assert outcome.ok
    assert outcome.discard_reason is None
    assert outcome.mutant is not None
    assert outcome.mutant.feature_vector == {"feat_0": 0.1, "feat_1": 0.2}


# ─────────────────────────────────────────────────────────────────────────
# verifier_failed
# ─────────────────────────────────────────────────────────────────────────
def test_verifier_failure_discards_config_before_finetune() -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )
    from defaultplusplus.deform.validation import VerificationResult

    fine_tune_called = []

    def _fail_fine_tune(*args, **kwargs):
        fine_tune_called.append(True)
        return 0.5, {}

    def _verifier_factory(_config):
        return VerificationResult(ok=False, message="targeted no parameters")

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_fail_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        verifier_factory=_verifier_factory,
        seeds=[1, 2, 3, 4, 5],
    )

    assert outcome.status == RunStatus.VERIFIER_FAILED
    assert outcome.mutant is None
    assert "targeted no parameters" in (outcome.discard_reason or "")
    assert not fine_tune_called, "fine_tune must not be invoked after verifier fails"


def test_verifier_factory_exception_is_recorded_as_verifier_failed() -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )

    def _exploding_verifier(_config):
        raise RuntimeError("disk on fire")

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_ok_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        verifier_factory=_exploding_verifier,
        seeds=[1, 2, 3, 4, 5],
    )
    assert outcome.status == RunStatus.VERIFIER_FAILED
    assert "disk on fire" in (outcome.discard_reason or "")


def test_verifier_pass_runs_to_completion() -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )
    from defaultplusplus.deform.validation import VerificationResult

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_ok_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        verifier_factory=lambda _c: VerificationResult(ok=True),
        seeds=[1, 2, 3, 4, 5],
    )
    assert outcome.status == RunStatus.OK
    assert outcome.log.get("structural_verification") == "passed"


# ─────────────────────────────────────────────────────────────────────────
# runtime_error
# ─────────────────────────────────────────────────────────────────────────
def test_faulty_run_exception_discards_whole_config() -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )

    seeds_seen = []

    def _crashing_fine_tune(model, task, seed, injector):
        seeds_seen.append((seed, injector is not None))
        if injector is not None and seed == 3:
            raise ValueError("simulated NaN gradient")
        return (0.5 if injector is not None else 0.8), {"seed": seed}

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_crashing_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        seeds=[1, 2, 3, 4, 5],
    )

    assert outcome.status == RunStatus.RUNTIME_ERROR
    assert outcome.mutant is None
    assert "ValueError" in (outcome.discard_reason or "")
    assert "seed=3" in (outcome.discard_reason or "")

    # Crucially: the runner stops on the first crash and never aggregates
    # a partial set of seeds.
    faulty_calls = [s for s, is_faulty in seeds_seen if is_faulty]
    assert faulty_calls == [1, 2, 3], (
        "runner should bail after the first faulty crash, not run "
        "remaining seeds and aggregate a partial result"
    )


# ─────────────────────────────────────────────────────────────────────────
# invalid_metric
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_faulty_metric_is_discarded(bad_value: float) -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )

    def _nan_fine_tune(model, task, seed, injector):
        if injector is not None and seed == 2:
            return bad_value, {"seed": seed}
        return (0.5 if injector is not None else 0.8), {"seed": seed}

    outcome = run_one_configuration(
        config=_config(),
        injector_factory=_injector_factory,
        fine_tune=_nan_fine_tune,
        feature_builder=_build_features,
        higher_is_better=True,
        seeds=[1, 2, 3, 4, 5],
    )
    assert outcome.status == RunStatus.INVALID_METRIC
    assert outcome.mutant is None
    assert "non-finite" in (outcome.discard_reason or "")


def test_clean_run_failure_bubbles_up() -> None:
    """Clean failures are environment problems, not faults — the runner
    should not swallow them. The benchmark operator needs to see them."""
    from defaultplusplus.benchmark.runner import run_one_configuration

    def _bad_clean(model, task, seed, injector):
        if injector is None:
            raise RuntimeError("dataset not on disk")
        return 0.5, {}

    with pytest.raises(RuntimeError, match="dataset not on disk"):
        run_one_configuration(
            config=_config(),
            injector_factory=_injector_factory,
            fine_tune=_bad_clean,
            feature_builder=_build_features,
            higher_is_better=True,
            seeds=[1, 2],
        )


# ─────────────────────────────────────────────────────────────────────────
# Discard log file
# ─────────────────────────────────────────────────────────────────────────
def test_status_json_records_discard_reason(tmp_path: Path) -> None:
    from defaultplusplus.benchmark.runner import (
        RunStatus, run_one_configuration,
    )

    def _crash(model, task, seed, injector):
        if injector is not None:
            raise RuntimeError("kaboom")
        return 0.8, {"seed": seed}

    outcome = run_one_configuration(
        config=_config(operator_id="QZQ", seed=7),
        injector_factory=_injector_factory,
        fine_tune=_crash,
        feature_builder=_build_features,
        higher_is_better=True,
        seeds=[7, 8, 9, 10, 11],
        output_dir=tmp_path,
    )
    assert outcome.status == RunStatus.RUNTIME_ERROR

    files = list(tmp_path.glob("*.status.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["status"] == "runtime_error"
    assert record["operator_id"] == "QZQ"
    assert record["model"] == "fake-model"
    assert "kaboom" in record["discard_reason"]


def test_discard_record_payload_shape() -> None:
    from defaultplusplus.benchmark.runner import (
        RunOutcome, RunStatus,
    )

    outcome = RunOutcome(
        status=RunStatus.VERIFIER_FAILED,
        discard_reason="targeted no parameters",
        log={
            "config_id": "QZQ_lall_slow_seed42_fake_task",
            "operator_id": "QZQ",
            "model": "fake",
            "task": "task",
            "severity": "low",
            "layers": [],
        },
        duration_seconds=0.0,
    )
    record = outcome.discard_record()
    assert record["status"] == "verifier_failed"
    assert record["operator_id"] == "QZQ"
    assert record["reason"] == "targeted no parameters"


def test_cli_writes_discard_log_in_jsonl_format(tmp_path: Path) -> None:
    """The CLI helper persists one JSON object per discarded outcome."""
    from defaultplusplus.benchmark.cli import _write_discard_log
    from defaultplusplus.benchmark.runner import RunOutcome, RunStatus

    outcomes = [
        RunOutcome(
            status=RunStatus.VERIFIER_FAILED,
            discard_reason="no parameters matched",
            log={"config_id": "A", "operator_id": "QZQ",
                 "model": "m", "task": "t", "severity": "low", "layers": []},
        ),
        RunOutcome(
            status=RunStatus.RUNTIME_ERROR,
            discard_reason="ValueError: simulated NaN",
            log={"config_id": "B", "operator_id": "FCA",
                 "model": "m", "task": "t", "severity": "high", "layers": [3]},
        ),
    ]
    log_path = tmp_path / "out.discarded.jsonl"
    _write_discard_log(log_path, outcomes)

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["config_id"] == "A"
    assert first["status"] == "verifier_failed"
    assert second["config_id"] == "B"
    assert second["status"] == "runtime_error"
    assert second["layers"] == [3]


def test_cli_print_discard_summary_groups_by_status(capsys) -> None:
    from defaultplusplus.benchmark.cli import _print_discard_summary
    from defaultplusplus.benchmark.runner import RunOutcome, RunStatus

    outcomes = [
        RunOutcome(status=RunStatus.VERIFIER_FAILED, discard_reason="x", log={}),
        RunOutcome(status=RunStatus.VERIFIER_FAILED, discard_reason="y", log={}),
        RunOutcome(status=RunStatus.RUNTIME_ERROR,   discard_reason="z", log={}),
    ]
    _print_discard_summary(outcomes)
    out = capsys.readouterr().out
    assert "verifier_failed=2" in out
    assert "runtime_error=1" in out
