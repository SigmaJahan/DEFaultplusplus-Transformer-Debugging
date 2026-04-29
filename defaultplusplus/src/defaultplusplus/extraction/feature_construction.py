"""Build the fixed-length feature vector consumed by the diagnostic model.

Raw training traces are recorded at four granularities:

  layer-internal metrics  per layer per training step (attention
                          entropy, head similarity, FFN output norm,
                          LayerNorm scale + post-norm distribution,
                          residual cosine similarity, QKV alignment,
                          cross-layer CKA).
  gradient metrics        per parameterized component per training
                          step (gradient norm, update ratio, update
                          activity flag).
  behavioral metrics      whole-model per training step (loss, gradient
                          noise scale, step time, peak memory,
                          prediction confidence, output entropy, margin,
                          embedding norm, positional sensitivity, KV
                          cache hidden similarity, cache distribution
                          divergence).
  validation metrics      whole-model at epoch end / validation
                          checkpoints (task accuracy or perplexity,
                          calibration error).

The construction pipeline aggregates these traces in three stages:

  1. Layer aggregation. Layer-internal metrics collapse to five
     statistics per metric: ``early_mean``, ``early_std``, ``mid_mean``,
     ``mid_std``, ``final_layer_value``. The early band covers the
     first third of layers, the mid band covers the second third, and
     ``final_layer_value`` is the value at the last instrumented
     layer. Step-level (gradient, behavioral) metrics skip this stage.

  2. Step-wise feature merge. Within each training epoch, layer
     aggregates and step-level metrics are concatenated into one
     step-level feature vector per training step.

  3. Epoch-level summary. Each metric is reduced within an epoch to
     ``epoch_mean``, ``epoch_std``, and a burst statistic (95th
     percentile or maximum). Validation metrics enter at this stage.

  4. Training-phase summary. The epoch sequence is split into early /
     mid / final thirds. For each metric the pipeline emits five
     statistics: phase means at early, mid, and final, the
     linear-regression slope across phases, and the final-epoch value.

The resulting fixed-length vector is paired with detection / category
/ root-cause labels and passed to the diagnostic model. The CV
filtering and group-assignment steps in
:mod:`defaultplusplus.data.feature_processor` close out the pipeline
inside each cross-validation fold.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

import numpy as np

ARCH_ENCODER = "encoder"
ARCH_DECODER = "decoder"

A_LAYER = 5
A_EPOCH = 3
A_PHASE = 5
C_INT = {ARCH_ENCODER: 15, ARCH_DECODER: 16}
C_OPT = 21
C_TRAIN = {ARCH_ENCODER: 10, ARCH_DECODER: 12}
C_EVAL = 2
EXPECTED_FEATURE_DIMS = {
    arch: A_PHASE * (A_EPOCH * (A_LAYER * C_INT[arch] + C_OPT + C_TRAIN[arch]) + C_EVAL)
    for arch in (ARCH_ENCODER, ARCH_DECODER)
}
STEP_METRIC_COUNTS = {
    arch: C_OPT + C_TRAIN[arch]
    for arch in (ARCH_ENCODER, ARCH_DECODER)
}


# ── Per-metric typed traces ──────────────────────────────────────────────
@dataclass
class LayerInternalTrace:
    """Layer-internal metric over the course of one training run.

    ``values`` has shape ``(n_steps, n_layers)``. Missing layers / steps
    must be encoded as NaN; downstream aggregation ignores NaN cells.
    """
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("LayerInternalTrace.values must be 2-D "
                             "(n_steps, n_layers); got shape "
                             f"{arr.shape}")
        self.values = arr


@dataclass
class StepTrace:
    """Whole-model metric per training step.

    ``values`` has shape ``(n_steps,)``.
    """
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("StepTrace.values must be 1-D (n_steps,); "
                             f"got shape {arr.shape}")
        self.values = arr


@dataclass
class EpochTrace:
    """Whole-model metric per epoch (e.g. validation accuracy)."""
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("EpochTrace.values must be 1-D (n_epochs,); "
                             f"got shape {arr.shape}")
        self.values = arr


@dataclass
class TrainingTrace:
    """Bundle of all metric traces for one fine-tuning run.

    Each dict maps a metric name (e.g. ``"attention_entropy"``,
    ``"grad_norm_attn"``, ``"loss"``, ``"task_accuracy"``) to its
    typed trace. ``epoch_boundaries`` gives the step index at the end
    of each epoch (length ``n_epochs``); the aggregator uses it to
    summarize step-level metrics within each epoch.
    """
    layer_internal: dict[str, LayerInternalTrace]
    step_level: dict[str, StepTrace]
    epoch_level: dict[str, EpochTrace]
    epoch_boundaries: Sequence[int]
    arch: str | None = None


# ── Aggregation ──────────────────────────────────────────────────────────
def _band_indices(n: int) -> tuple[range, range, range]:
    """Split ``range(n)`` into three roughly equal bands (early/mid/late).

    For very small ``n`` (1 or 2) the trailing bands collapse to empty
    ranges so callers receive valid index lists in every case.
    """
    if n <= 0:
        return range(0), range(0), range(0)
    if n == 1:
        # Everything counts as the early band; mid and late are empty.
        return range(0, 1), range(1, 1), range(1, 1)
    if n == 2:
        return range(0, 1), range(1, 2), range(2, 2)
    third = n // 3
    if third == 0:  # already handled above, defensive
        third = 1
    return range(0, third), range(third, 2 * third), range(2 * third, n)


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else 0.0


def _safe_std(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    return float(arr.std()) if arr.size else 0.0


def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _slope(values: Sequence[float]) -> float:
    """Linear-regression slope of ``values`` against integer indices."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]; y = y[mask]
    x_mean = x.mean(); y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def aggregate_layer_internal(trace: LayerInternalTrace) -> dict[str, float]:
    """Reduce a (n_steps, n_layers) trace to five layer-band statistics.

    The reduction is taken per step (across layers): for each step we
    compute the mean of layer values in the early band and the mid
    band, and the value at the final layer. The five output statistics
    are then aggregated across steps by taking the temporal mean and
    standard deviation of the early-band and mid-band sequences, plus
    the temporal mean of the final-layer sequence.
    """
    arr = trace.values  # (n_steps, n_layers)
    n_steps, n_layers = arr.shape
    early, mid, _late = _band_indices(n_layers)

    if n_steps == 0 or n_layers == 0:
        return {"early_mean": 0.0, "early_std": 0.0,
                "mid_mean": 0.0, "mid_std": 0.0,
                "final_layer_value": 0.0}

    with np.errstate(invalid="ignore"):
        early_per_step = np.nanmean(arr[:, list(early)], axis=1) if early else np.zeros(n_steps)
        mid_per_step = np.nanmean(arr[:, list(mid)], axis=1) if mid else np.zeros(n_steps)
        final_per_step = arr[:, n_layers - 1]

    return {
        "early_mean": _safe_mean(early_per_step),
        "early_std":  _safe_std(early_per_step),
        "mid_mean":   _safe_mean(mid_per_step),
        "mid_std":    _safe_std(mid_per_step),
        "final_layer_value": _safe_mean(final_per_step),
    }


def aggregate_layer_internal_by_step(trace: LayerInternalTrace) -> dict[str, np.ndarray]:
    """Reduce layer values to five per-step layer-band trajectories."""
    arr = trace.values
    n_steps, n_layers = arr.shape
    early, mid, _late = _band_indices(n_layers)

    if n_steps == 0 or n_layers == 0:
        zeros = np.zeros(n_steps, dtype=np.float64)
        return {"early_mean": zeros, "early_std": zeros, "mid_mean": zeros,
                "mid_std": zeros, "final_layer_value": zeros}

    with np.errstate(invalid="ignore"):
        early_arr = arr[:, list(early)] if early else np.empty((n_steps, 0))
        mid_arr = arr[:, list(mid)] if mid else np.empty((n_steps, 0))
        early_mean = np.nanmean(early_arr, axis=1) if early else np.zeros(n_steps)
        early_std = np.nanstd(early_arr, axis=1) if early else np.zeros(n_steps)
        mid_mean = np.nanmean(mid_arr, axis=1) if mid else np.zeros(n_steps)
        mid_std = np.nanstd(mid_arr, axis=1) if mid else np.zeros(n_steps)
        final = arr[:, n_layers - 1]

    return {
        "early_mean": np.nan_to_num(early_mean, nan=0.0),
        "early_std": np.nan_to_num(early_std, nan=0.0),
        "mid_mean": np.nan_to_num(mid_mean, nan=0.0),
        "mid_std": np.nan_to_num(mid_std, nan=0.0),
        "final_layer_value": np.nan_to_num(final, nan=0.0),
    }


def aggregate_epoch_summary(values: np.ndarray) -> dict[str, float]:
    """Reduce a within-epoch step-level metric slice to three statistics."""
    return {
        "epoch_mean": _safe_mean(values),
        "epoch_std":  _safe_std(values),
        "epoch_burst_p95": _percentile(values, 95.0),
    }


def aggregate_training_phase(per_epoch: Sequence[float]) -> dict[str, float]:
    """Reduce a per-epoch sequence to five training-phase statistics."""
    arr = np.asarray(per_epoch, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"phase_early_mean": 0.0, "phase_mid_mean": 0.0,
                "phase_final_mean": 0.0, "phase_slope": 0.0,
                "phase_final_value": 0.0}
    early, mid, late = _band_indices(n)
    return {
        "phase_early_mean": _safe_mean(arr[list(early)]) if early else 0.0,
        "phase_mid_mean":   _safe_mean(arr[list(mid)]) if mid else 0.0,
        "phase_final_mean": _safe_mean(arr[list(late)]) if late else 0.0,
        "phase_slope":      _slope([_safe_mean(arr[list(b)])
                                    for b in (early, mid, late) if b]),
        "phase_final_value": float(arr[-1]) if np.isfinite(arr[-1]) else 0.0,
    }


def expected_feature_dim(arch: str) -> int:
    """Return the raw pre-filtering feature dimension from Equation 7.19."""
    _validate_arch(arch)
    return EXPECTED_FEATURE_DIMS[arch]


def assert_feature_dim_invariants(arch: str,
                                  features: dict[str, float] | None = None) -> int:
    """Assert Equation 7.19 constants and, optionally, vector length."""
    _validate_arch(arch)
    expected = EXPECTED_FEATURE_DIMS[arch]
    recomputed = A_PHASE * (
        A_EPOCH * (A_LAYER * C_INT[arch] + C_OPT + C_TRAIN[arch]) + C_EVAL
    )
    if expected != recomputed:
        raise AssertionError(
            f"Equation 7.19 constants inconsistent for {arch}: "
            f"{expected} != {recomputed}"
        )
    if features is not None and len(features) != expected:
        raise AssertionError(
            f"{arch} feature vector has {len(features)} keys; expected {expected}"
        )
    return expected


def _validate_arch(arch: str) -> None:
    if arch not in EXPECTED_FEATURE_DIMS:
        raise ValueError(f"arch must be 'encoder' or 'decoder'; got {arch!r}")


# ── Feature builder ──────────────────────────────────────────────────────
def build_feature_vector(trace: TrainingTrace) -> dict[str, float]:
    """Assemble the fixed-length feature vector from one training trace.

    Output keys follow a structured naming scheme:

        {metric}__{layer_band_stat}              for layer-internal
        {metric}__{epoch_summary}__{phase_stat}  for step-level
        {metric}__epoch__{phase_stat}            for epoch-level

    where ``{layer_band_stat}`` is one of
    ``early_mean / early_std / mid_mean / mid_std / final_layer_value``,
    ``{epoch_summary}`` is one of
    ``epoch_mean / epoch_std / epoch_burst_p95``, and ``{phase_stat}``
    is one of ``phase_early_mean / phase_mid_mean / phase_final_mean /
    phase_slope / phase_final_value``.

    The function is deterministic and order-stable: identical traces
    produce identical key orders so dataset shards can be concatenated
    without a column-alignment step.
    """
    out: dict[str, float] = {}
    boundaries = list(trace.epoch_boundaries)

    # Layer-internal metrics.
    for name, t in sorted(trace.layer_internal.items()):
        for layer_stat, values in aggregate_layer_internal_by_step(t).items():
            per_epoch = _summarize_steps_by_epoch(values, boundaries)
            for epoch_stat in ("epoch_mean", "epoch_std", "epoch_burst_p95"):
                seq = [es[epoch_stat] for es in per_epoch]
                phase = aggregate_training_phase(seq)
                for pstat, pval in phase.items():
                    out[f"{name}__{layer_stat}__{epoch_stat}__{pstat}"] = pval

    # Step-level metrics: epoch summary then training-phase summary.
    for name, t in sorted(trace.step_level.items()):
        per_epoch_summaries = _summarize_steps_by_epoch(t.values, boundaries)
        # Three sequences (mean, std, burst), one entry per epoch.
        for stat in ("epoch_mean", "epoch_std", "epoch_burst_p95"):
            seq = [es[stat] for es in per_epoch_summaries]
            phase = aggregate_training_phase(seq)
            for pstat, pval in phase.items():
                out[f"{name}__{stat}__{pstat}"] = pval

    # Epoch-level metrics: training-phase summary directly.
    for name, t in sorted(trace.epoch_level.items()):
        phase = aggregate_training_phase(t.values)
        for pstat, pval in phase.items():
            out[f"{name}__epoch__{pstat}"] = pval

    if trace.arch is not None:
        assert_feature_dim_invariants(trace.arch, out)
    return out


def _summarize_steps_by_epoch(values: np.ndarray,
                              boundaries: Sequence[int]
                              ) -> list[dict[str, float]]:
    """Split ``values`` at ``boundaries`` and summarize each segment.

    Boundaries are step indices that mark the end of each epoch. The
    returned list has one entry per epoch.
    """
    if len(boundaries) == 0:
        return [aggregate_epoch_summary(values)]
    starts = [0] + list(boundaries[:-1])
    ends = list(boundaries)
    return [aggregate_epoch_summary(values[s:e]) for s, e in zip(starts, ends)]


# ── Convenience for paired clean / faulty traces ─────────────────────────
def build_paired_feature_vector(clean_traces: Iterable[TrainingTrace],
                                faulty_traces: Iterable[TrainingTrace]
                                ) -> dict[str, float]:
    """Build a feature vector from paired (clean, faulty) seed runs.

    Each metric is averaged across seeds within the clean and faulty
    sides separately, then the per-metric delta ``faulty - clean`` is
    used as the input to :func:`build_feature_vector`. Averaging
    across seeds before differencing is robust to seed-to-seed noise
    and matches the way the diagnostic model expects to see signed
    deltas.

    Args:
        clean_traces:  one :class:`TrainingTrace` per matched seed,
                       collected with no fault injected.
        faulty_traces: one :class:`TrainingTrace` per matched seed,
                       collected with the fault active. Must have the
                       same length and ordering as ``clean_traces``.

    Returns:
        Fixed-length feature vector keyed as in
        :func:`build_feature_vector`, computed on faulty − clean.
    """
    clean = list(clean_traces)
    faulty = list(faulty_traces)
    if len(clean) != len(faulty) or len(clean) == 0:
        raise ValueError("clean and faulty trace lists must be equal length and non-empty")
    delta = _delta_trace(clean, faulty)
    return build_feature_vector(delta)


def _delta_trace(clean: list[TrainingTrace],
                 faulty: list[TrainingTrace]) -> TrainingTrace:
    """Average each side across seeds and take the per-metric delta."""

    def _stack_layer(side: list[TrainingTrace], name: str) -> LayerInternalTrace | None:
        arrs = [s.layer_internal[name].values for s in side if name in s.layer_internal]
        if not arrs:
            return None
        # Pad to a common shape with NaN, then average across seeds.
        n_steps = max(a.shape[0] for a in arrs)
        n_layers = max(a.shape[1] for a in arrs)
        pad = np.full((len(arrs), n_steps, n_layers), np.nan)
        for k, a in enumerate(arrs):
            pad[k, : a.shape[0], : a.shape[1]] = a
        return LayerInternalTrace(np.nanmean(pad, axis=0))

    def _stack_step(side: list[TrainingTrace], name: str) -> StepTrace | None:
        arrs = [s.step_level[name].values for s in side if name in s.step_level]
        if not arrs:
            return None
        n = max(a.shape[0] for a in arrs)
        pad = np.full((len(arrs), n), np.nan)
        for k, a in enumerate(arrs):
            pad[k, : a.shape[0]] = a
        return StepTrace(np.nanmean(pad, axis=0))

    def _stack_epoch(side: list[TrainingTrace], name: str) -> EpochTrace | None:
        arrs = [s.epoch_level[name].values for s in side if name in s.epoch_level]
        if not arrs:
            return None
        n = max(a.shape[0] for a in arrs)
        pad = np.full((len(arrs), n), np.nan)
        for k, a in enumerate(arrs):
            pad[k, : a.shape[0]] = a
        return EpochTrace(np.nanmean(pad, axis=0))

    layer_names = set().union(*(s.layer_internal.keys() for s in clean + faulty))
    step_names = set().union(*(s.step_level.keys() for s in clean + faulty))
    epoch_names = set().union(*(s.epoch_level.keys() for s in clean + faulty))

    layer = {}
    for n in sorted(layer_names):
        c = _stack_layer(clean, n)
        f = _stack_layer(faulty, n)
        if c is not None and f is not None:
            # Align shapes by padding with NaN to a common shape.
            n_steps = max(c.values.shape[0], f.values.shape[0])
            n_layers = max(c.values.shape[1], f.values.shape[1])
            c_pad = np.full((n_steps, n_layers), np.nan)
            f_pad = np.full((n_steps, n_layers), np.nan)
            c_pad[: c.values.shape[0], : c.values.shape[1]] = c.values
            f_pad[: f.values.shape[0], : f.values.shape[1]] = f.values
            layer[n] = LayerInternalTrace(f_pad - c_pad)

    step = {}
    for n in sorted(step_names):
        c = _stack_step(clean, n)
        f = _stack_step(faulty, n)
        if c is not None and f is not None:
            n_steps = max(c.values.shape[0], f.values.shape[0])
            c_pad = np.full(n_steps, np.nan)
            f_pad = np.full(n_steps, np.nan)
            c_pad[: c.values.shape[0]] = c.values
            f_pad[: f.values.shape[0]] = f.values
            step[n] = StepTrace(f_pad - c_pad)

    epoch = {}
    for n in sorted(epoch_names):
        c = _stack_epoch(clean, n)
        f = _stack_epoch(faulty, n)
        if c is not None and f is not None:
            n_e = max(c.values.shape[0], f.values.shape[0])
            c_pad = np.full(n_e, np.nan)
            f_pad = np.full(n_e, np.nan)
            c_pad[: c.values.shape[0]] = c.values
            f_pad[: f.values.shape[0]] = f.values
            epoch[n] = EpochTrace(f_pad - c_pad)

    # Use the longest epoch_boundaries from the clean side.
    longest_boundaries: Sequence[int] = ()
    for s in clean + faulty:
        if len(s.epoch_boundaries) > len(longest_boundaries):
            longest_boundaries = s.epoch_boundaries

    return TrainingTrace(layer_internal=layer, step_level=step,
                         epoch_level=epoch,
                         epoch_boundaries=longest_boundaries)
