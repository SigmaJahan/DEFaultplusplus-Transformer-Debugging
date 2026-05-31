"""Per-task metric registry for the kill test.

The sign-flip permutation test ([deform.validation][1]) expects a
single scalar per (clean, faulty) seed pair. For tasks where the
standard reporting convention is a single metric (SST-2 accuracy,
WikiText eval loss, CoLA Matthews correlation), that scalar is just
the raw metric. For tasks where the standard convention is a
composite (MRPC and QQP report (accuracy + F1) / 2; STS-B reports
(Pearson + Spearman) / 2), the registry returns the convention's
composite — never an arbitrary subset.

The registry is the single source of truth for three things:

  1. The HuggingFace Trainer's ``compute_metrics`` callable (we must
     emit every raw scalar the aggregator needs, not just one).
  2. The ``higher_is_better`` flag for the sign-flip test.
  3. The aggregator that collapses the raw scalars to one number.

Adding a new task means adding one entry to ``TASK_METRICS``; nothing
else in the runner / CLI needs to change.

[1]: defaultplusplus/deform/validation.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Spec
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TaskMetricSpec:
    """How to score one task for the kill test.

    Attributes:
        name:             registered lowercase task id (e.g. ``"sst2"``,
                          ``"wikitext2"``).
        arch:             ``"encoder"`` or ``"decoder"``.
        higher_is_better: passed through to the sign-flip permutation
                          test. Accuracy-style metrics use ``True``;
                          loss / perplexity-style use ``False``.
        raw_metrics:      tuple of HF Trainer eval-dict keys the
                          aggregator needs. Members are the un-prefixed
                          names (``"accuracy"``, ``"f1"``, ``"loss"``);
                          the registry resolves the ``"eval_"`` prefix
                          internally.
        aggregator:       maps a dict containing all ``raw_metrics`` to
                          one scalar. Must be a pure function — the
                          runner calls it inside the kill-test path.
    """
    name: str
    arch: str
    higher_is_better: bool
    raw_metrics: tuple[str, ...]
    aggregator: Callable[[Mapping[str, float]], float]


def _take(metrics: Mapping[str, float], key: str) -> float:
    """Return ``metrics[key]`` checking both ``key`` and ``eval_<key>``."""
    if key in metrics:
        return float(metrics[key])
    eval_key = f"eval_{key}"
    if eval_key in metrics:
        return float(metrics[eval_key])
    raise KeyError(
        f"task aggregator missing required metric {key!r}; "
        f"available keys: {sorted(metrics)}"
    )


def _accuracy(m: Mapping[str, float]) -> float:
    return _take(m, "accuracy")


def _log_perplexity(m: Mapping[str, float]) -> float:
    """Log-perplexity for a language-modeling eval set.

    On a fixed eval set the HF Trainer's mean cross-entropy ``eval_loss``
    equals the mean per-token negative log-likelihood, which is exactly
    log-perplexity (``log(perplexity) = mean NLL``). The decoder kill
    test therefore uses ``eval_loss`` directly as log-perplexity.
    """
    return _take(m, "loss")


def _matthews_correlation(m: Mapping[str, float]) -> float:
    return _take(m, "matthews_correlation")


def _accuracy_f1_avg(m: Mapping[str, float]) -> float:
    return 0.5 * (_take(m, "accuracy") + _take(m, "f1"))


def _pearson_spearman_avg(m: Mapping[str, float]) -> float:
    return 0.5 * (_take(m, "pearson") + _take(m, "spearmanr"))


# ─────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────
TASK_METRICS: dict[str, TaskMetricSpec] = {
    # ── Encoder / GLUE ──────────────────────────────────────────────────
    "sst2": TaskMetricSpec(
        name="sst2", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy",), aggregator=_accuracy,
    ),
    "qnli": TaskMetricSpec(
        name="qnli", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy",), aggregator=_accuracy,
    ),
    "rte": TaskMetricSpec(
        name="rte", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy",), aggregator=_accuracy,
    ),
    "mnli": TaskMetricSpec(
        name="mnli", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy",), aggregator=_accuracy,
    ),
    "mrpc": TaskMetricSpec(
        name="mrpc", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy", "f1"), aggregator=_accuracy_f1_avg,
    ),
    "qqp": TaskMetricSpec(
        name="qqp", arch="encoder", higher_is_better=True,
        raw_metrics=("accuracy", "f1"), aggregator=_accuracy_f1_avg,
    ),
    "stsb": TaskMetricSpec(
        name="stsb", arch="encoder", higher_is_better=True,
        raw_metrics=("pearson", "spearmanr"), aggregator=_pearson_spearman_avg,
    ),
    "cola": TaskMetricSpec(
        name="cola", arch="encoder", higher_is_better=True,
        raw_metrics=("matthews_correlation",), aggregator=_matthews_correlation,
    ),

    # ── Decoder / language modeling (log-perplexity = mean eval_loss) ────
    "lambada": TaskMetricSpec(
        name="lambada", arch="decoder", higher_is_better=False,
        raw_metrics=("loss",), aggregator=_log_perplexity,
    ),
    "ptb": TaskMetricSpec(
        name="ptb", arch="decoder", higher_is_better=False,
        raw_metrics=("loss",), aggregator=_log_perplexity,
    ),
    "wikitext2": TaskMetricSpec(
        name="wikitext2", arch="decoder", higher_is_better=False,
        raw_metrics=("loss",), aggregator=_log_perplexity,
    ),
    "wikitext": TaskMetricSpec(  # alias people commonly type
        name="wikitext", arch="decoder", higher_is_better=False,
        raw_metrics=("loss",), aggregator=_log_perplexity,
    ),
    "openwebtext": TaskMetricSpec(
        name="openwebtext", arch="decoder", higher_is_better=False,
        raw_metrics=("loss",), aggregator=_log_perplexity,
    ),
}


# ─────────────────────────────────────────────────────────────────────────
# Public accessors
# ─────────────────────────────────────────────────────────────────────────
def get_task_spec(task: str) -> TaskMetricSpec:
    """Return the registered spec for ``task`` (case-insensitive).

    Raises:
        KeyError: if the task is not registered. The error message
                  lists the supported task ids so a typo surfaces a
                  useful diagnostic instead of a silent fallback.
    """
    key = task.strip().lower()
    if key not in TASK_METRICS:
        raise KeyError(
            f"Unknown task {task!r}. Supported: {sorted(TASK_METRICS)}. "
            "To add a new task, register a TaskMetricSpec in task_metrics.py."
        )
    return TASK_METRICS[key]


def supported_tasks(arch: str | None = None) -> list[str]:
    """Return registered task ids, optionally filtered by ``arch``."""
    out = sorted(TASK_METRICS)
    if arch is None:
        return out
    arch = arch.lower()
    return [t for t in out if TASK_METRICS[t].arch == arch]


def score_evaluation(task: str, metrics: Mapping[str, float]) -> float:
    """Aggregate a Trainer.evaluate() dict into the one kill-test scalar."""
    spec = get_task_spec(task)
    return float(spec.aggregator(metrics))


# ─────────────────────────────────────────────────────────────────────────
# Trainer compute_metrics builder
# ─────────────────────────────────────────────────────────────────────────
def build_compute_metrics(task: str) -> Callable[[Any], dict[str, float]]:
    """Return a ``compute_metrics`` callable HF Trainer accepts.

    The callable must emit every raw key listed in the task's spec.
    For classification tasks we re-derive accuracy / F1 from
    (logits, labels); for regression tasks (STS-B) we report Pearson
    and Spearman correlations on the flat predictions.
    """
    spec = get_task_spec(task)
    needs_f1 = "f1" in spec.raw_metrics
    needs_acc = "accuracy" in spec.raw_metrics
    needs_mcc = "matthews_correlation" in spec.raw_metrics
    needs_pearson = "pearson" in spec.raw_metrics
    needs_spearman = "spearmanr" in spec.raw_metrics

    if not (needs_f1 or needs_acc or needs_mcc or needs_pearson or needs_spearman):
        # Tasks that only need eval_loss don't pass compute_metrics; HF
        # will return loss by default.
        return _identity_metrics

    is_regression = needs_pearson or needs_spearman

    def _compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        labels_arr = np.asarray(labels)

        out: dict[str, float] = {}
        if is_regression:
            preds = np.asarray(logits).reshape(-1).astype(np.float64)
            target = labels_arr.reshape(-1).astype(np.float64)
            if needs_pearson:
                out["pearson"] = _safe_pearson(preds, target)
            if needs_spearman:
                out["spearmanr"] = _safe_spearman(preds, target)
            return out

        preds = np.argmax(np.asarray(logits), axis=-1)
        if needs_acc:
            out["accuracy"] = float((preds == labels_arr).mean())
        if needs_f1:
            out["f1"] = _binary_f1(preds, labels_arr)
        if needs_mcc:
            out["matthews_correlation"] = _matthews(preds, labels_arr)
        return out

    return _compute


def _identity_metrics(_eval_pred: Any) -> dict[str, float]:
    return {}


# ─────────────────────────────────────────────────────────────────────────
# Statistics that don't require sklearn
# ─────────────────────────────────────────────────────────────────────────
def _binary_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Binary F1 with the positive class taken as label==1.

    Matches the convention HF / GLUE uses for MRPC and QQP.
    """
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _matthews(preds: np.ndarray, labels: np.ndarray) -> float:
    """Matthews correlation coefficient for binary classification."""
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom_sq))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return 0.0
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    return float(cov / (sx * sy))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return _safe_pearson(_rankdata(x), _rankdata(y))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank tie handling, like scipy.stats.rankdata."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average across ties.
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


__all__ = [
    "TaskMetricSpec",
    "TASK_METRICS",
    "get_task_spec",
    "supported_tasks",
    "score_evaluation",
    "build_compute_metrics",
]
