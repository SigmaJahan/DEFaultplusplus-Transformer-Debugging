"""Tests for the per-task metric registry (kill-test scoring)."""
from __future__ import annotations

import math

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────
# Registry coverage
# ─────────────────────────────────────────────────────────────────────────
def test_registry_covers_paper_tasks() -> None:
    from defaultplusplus.benchmark.task_metrics import TASK_METRICS

    expected = {
        # GLUE
        "sst2", "qnli", "rte", "mnli", "mrpc", "qqp", "stsb", "cola",
        # Decoder LM
        "lambada", "ptb", "wikitext2", "wikitext", "openwebtext",
    }
    missing = expected - set(TASK_METRICS)
    assert not missing, f"registry missing tasks: {sorted(missing)}"


def test_supported_tasks_filters_by_arch() -> None:
    from defaultplusplus.benchmark.task_metrics import supported_tasks

    encoders = set(supported_tasks("encoder"))
    decoders = set(supported_tasks("decoder"))
    assert "sst2" in encoders and "wikitext2" in decoders
    assert encoders.isdisjoint(decoders)


def test_get_task_spec_is_case_insensitive() -> None:
    from defaultplusplus.benchmark.task_metrics import get_task_spec

    a = get_task_spec("SST2")
    b = get_task_spec("sst2")
    assert a is b


def test_unknown_task_raises_with_helpful_message() -> None:
    from defaultplusplus.benchmark.task_metrics import get_task_spec

    with pytest.raises(KeyError, match="Unknown task"):
        get_task_spec("imagenet")


# ─────────────────────────────────────────────────────────────────────────
# higher_is_better aligns with the kill-test direction
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("task,direction", [
    ("sst2", True), ("qnli", True), ("rte", True), ("mnli", True),
    ("mrpc", True), ("qqp", True), ("stsb", True), ("cola", True),
    ("wikitext2", False), ("wikitext", False),
])
def test_higher_is_better_per_task(task: str, direction: bool) -> None:
    from defaultplusplus.benchmark.task_metrics import get_task_spec

    spec = get_task_spec(task)
    assert spec.higher_is_better is direction, (
        f"{task} should set higher_is_better={direction}; got {spec.higher_is_better}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Aggregator correctness
# ─────────────────────────────────────────────────────────────────────────
def test_sst2_aggregator_returns_accuracy_directly() -> None:
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation("sst2", {"eval_accuracy": 0.83, "eval_loss": 0.40})
    assert score == pytest.approx(0.83)


def test_mrpc_aggregator_averages_accuracy_and_f1() -> None:
    """The kill test sees (accuracy + F1) / 2 — the GLUE convention."""
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation("mrpc", {"eval_accuracy": 0.80, "eval_f1": 0.60})
    assert score == pytest.approx(0.70)


def test_qqp_aggregator_averages_accuracy_and_f1() -> None:
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation("qqp", {"eval_accuracy": 0.90, "eval_f1": 0.70})
    assert score == pytest.approx(0.80)


def test_stsb_aggregator_averages_pearson_and_spearman() -> None:
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation(
        "stsb", {"eval_pearson": 0.50, "eval_spearmanr": 0.30}
    )
    assert score == pytest.approx(0.40)


def test_cola_aggregator_uses_matthews_only() -> None:
    """CoLA reports MCC alone — accuracy is not part of the GLUE convention."""
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation(
        "cola", {"eval_matthews_correlation": 0.42, "eval_accuracy": 0.99}
    )
    assert score == pytest.approx(0.42)


def test_wikitext_aggregator_uses_eval_loss() -> None:
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation("wikitext2", {"eval_loss": 2.78})
    assert score == pytest.approx(2.78)


def test_aggregator_raises_when_required_metric_missing() -> None:
    """The kill-test fails closed if HF returned an unexpected eval shape."""
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    with pytest.raises(KeyError, match="missing required metric"):
        # MRPC needs both accuracy and F1; F1 absent here.
        score_evaluation("mrpc", {"eval_accuracy": 0.80})


def test_aggregator_accepts_unprefixed_keys() -> None:
    """``score_evaluation`` works whether HF returned ``accuracy`` or ``eval_accuracy``."""
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    score = score_evaluation("sst2", {"accuracy": 0.77})
    assert score == pytest.approx(0.77)


# ─────────────────────────────────────────────────────────────────────────
# build_compute_metrics
# ─────────────────────────────────────────────────────────────────────────
def test_compute_metrics_for_sst2_emits_accuracy() -> None:
    from defaultplusplus.benchmark.task_metrics import build_compute_metrics

    fn = build_compute_metrics("sst2")
    logits = np.array([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
    labels = np.array([1, 0, 1])
    out = fn((logits, labels))
    assert out == {"accuracy": 1.0}


def test_compute_metrics_for_mrpc_emits_accuracy_and_f1() -> None:
    """The compute_metrics callable must emit *both* metrics the
    aggregator needs — emitting only accuracy would silently break
    the GLUE composite."""
    from defaultplusplus.benchmark.task_metrics import build_compute_metrics

    fn = build_compute_metrics("mrpc")
    # 4 examples — both classes present, mixed correct/incorrect.
    logits = np.array([
        [-1.0, 1.0],   # pred=1
        [-1.0, 1.0],   # pred=1
        [1.0, -1.0],   # pred=0
        [1.0, -1.0],   # pred=0
    ])
    labels = np.array([1, 0, 1, 0])  # tp=1, fp=1, fn=1, tn=1 → acc=0.5, F1=0.5
    out = fn((logits, labels))
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_compute_metrics_for_cola_emits_matthews() -> None:
    from defaultplusplus.benchmark.task_metrics import build_compute_metrics

    fn = build_compute_metrics("cola")
    logits = np.array([
        [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0],
    ])
    labels = np.array([1, 1, 0, 0])  # perfect predictions → MCC=1.0
    out = fn((logits, labels))
    assert "matthews_correlation" in out
    assert out["matthews_correlation"] == pytest.approx(1.0)


def test_compute_metrics_for_stsb_emits_pearson_and_spearmanr() -> None:
    """Regression task: predictions are 1-d floats; correlations
    against the float labels feed the (pearson + spearman) / 2
    aggregator."""
    from defaultplusplus.benchmark.task_metrics import build_compute_metrics

    fn = build_compute_metrics("stsb")
    preds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    labels = np.array([0.5, 1.5, 1.8, 3.5, 4.0])
    out = fn((preds, labels))
    assert "pearson" in out and "spearmanr" in out
    assert out["pearson"] > 0.9
    assert out["spearmanr"] > 0.9


def test_compute_metrics_for_wikitext_returns_empty() -> None:
    """Decoder LM relies on HF's default eval_loss; compute_metrics is
    a no-op so HF doesn't try to argmax over a 1-d loss tensor."""
    from defaultplusplus.benchmark.task_metrics import build_compute_metrics

    fn = build_compute_metrics("wikitext2")
    out = fn((np.zeros((4, 8, 100)), np.zeros((4, 8))))
    assert out == {}


# ─────────────────────────────────────────────────────────────────────────
# CLI _validate_tasks
# ─────────────────────────────────────────────────────────────────────────
def test_validate_tasks_rejects_unknown_task() -> None:
    from defaultplusplus.benchmark.cli import _validate_tasks

    with pytest.raises(SystemExit, match="Unknown task"):
        _validate_tasks(("imagenet",), arch="encoder")


def test_validate_tasks_rejects_arch_mismatch() -> None:
    from defaultplusplus.benchmark.cli import _validate_tasks

    with pytest.raises(SystemExit, match="incompatible"):
        _validate_tasks(("wikitext2",), arch="encoder")


def test_validate_tasks_passes_for_supported_combo() -> None:
    from defaultplusplus.benchmark.cli import _validate_tasks

    # Should not raise.
    _validate_tasks(("sst2", "mrpc"), arch="encoder")
    _validate_tasks(("wikitext2",), arch="decoder")


def test_validate_tasks_rejects_empty_tasks() -> None:
    from defaultplusplus.benchmark.cli import _validate_tasks

    with pytest.raises(SystemExit, match="at least one"):
        _validate_tasks((), arch="encoder")


# ─────────────────────────────────────────────────────────────────────────
# Sanity: the composite is *strictly* more sensitive than accuracy alone
# ─────────────────────────────────────────────────────────────────────────
def test_mrpc_composite_responds_when_only_f1_drops() -> None:
    """Class-imbalance fault scenario: accuracy stays flat, F1 collapses.

    The single-metric kill test (accuracy only) would miss this; the
    composite catches it because (accuracy + F1) / 2 drops by F1's
    full magnitude / 2.
    """
    from defaultplusplus.benchmark.task_metrics import score_evaluation

    clean = score_evaluation("mrpc", {"eval_accuracy": 0.80, "eval_f1": 0.80})
    faulty = score_evaluation("mrpc", {"eval_accuracy": 0.80, "eval_f1": 0.30})
    assert clean - faulty == pytest.approx(0.25)
