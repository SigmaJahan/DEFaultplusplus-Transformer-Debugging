"""
Epoch-level aggregation: Welford online statistics, windowed features,
and rolling-window gradient stats.

Ported from statistics.py + running_metrics.py.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Welford online statistics
# ---------------------------------------------------------------------------

@dataclass
class OnlineStatistic:
    """Numerically stable running mean/variance (Welford's algorithm)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float):
        if value is None or math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0


# ---------------------------------------------------------------------------
# Epoch aggregator
# ---------------------------------------------------------------------------

class EpochAggregator:
    """Aggregates online statistics per epoch and stores history."""

    def __init__(self):
        self.current_epoch_stats: Dict[str, OnlineStatistic] = {}
        self.metric_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            stat = self.current_epoch_stats.setdefault(key, OnlineStatistic())
            stat.update(float(value))

    def finalize_epoch(self, epoch_index: int) -> Dict[str, float]:
        summary: Dict[str, float] = {'epoch': epoch_index}
        for key, stat in self.current_epoch_stats.items():
            summary[f'{key}_mean'] = stat.mean
            summary[f'{key}_var'] = stat.variance
            summary[f'{key}_count'] = stat.count
            self.metric_history[key].append((epoch_index + 1, stat.mean))
        self.current_epoch_stats = {}
        return summary

    def reset(self):
        self.current_epoch_stats = {}
        self.metric_history = defaultdict(list)


# ---------------------------------------------------------------------------
# Windowed features
# ---------------------------------------------------------------------------

WINDOW_DEFINITION = {
    'early': (1, 3),
    'mid': (4, 7),
    'late': (8, 10),
}


def _linear_regression_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    xs_list = list(xs)
    ys_list = list(ys)
    if len(xs_list) < 2:
        return 0.0
    x = np.array(xs_list, dtype=np.float64)
    y = np.array(ys_list, dtype=np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite_mask], y[finite_mask]
    if x.size < 2:
        return 0.0
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        denom = x[-1] - x[0]
        return float((y[-1] - y[0]) / denom) if denom != 0 else 0.0


def compute_window_features(
    metric_history: Dict[str, List[Tuple[int, float]]],
    total_epochs: int,
) -> Dict[str, float]:
    """Compress per-epoch means into canonical window features."""
    features: Dict[str, float] = {}
    if total_epochs == 0:
        return features

    for metric, history in metric_history.items():
        if not history:
            continue

        epoch_to_mean = {epoch: value for epoch, value in history}

        for window_name, (start_epoch, end_epoch) in WINDOW_DEFINITION.items():
            epochs = [e for e in range(start_epoch, end_epoch + 1)
                      if e <= total_epochs and e in epoch_to_mean]
            if not epochs:
                continue
            values = [epoch_to_mean[e] for e in epochs]
            finite_pairs = [(e, v) for e, v in zip(epochs, values) if np.isfinite(v)]
            if not finite_pairs:
                continue
            epochs_f, values_f = zip(*finite_pairs)
            features[f'{metric}_{window_name}_mean'] = float(np.mean(values_f))
            features[f'{metric}_{window_name}_slope'] = _linear_regression_slope(epochs_f, values_f)

        last_epoch = max(epoch for epoch, _ in history)
        features[f'{metric}_final'] = epoch_to_mean[last_epoch]

    return features


# ---------------------------------------------------------------------------
# Running metrics (rolling window)
# ---------------------------------------------------------------------------

class RunningMetrics:
    """Rolling-window statistics for high-variance metrics."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, name: str, value: float):
        if value is None:
            return
        self.histories[name].append(float(value))

    def get_variance(self, name: str) -> float:
        history = self.histories.get(name)
        if history is None or len(history) < 2:
            return 0.0
        return float(np.var(history))

    def get_mean(self, name: str) -> float:
        history = self.histories.get(name)
        if history is None or len(history) == 0:
            return 0.0
        return float(np.mean(history))

    def get_noise_scale(self, name: str, eps: float = 1e-8) -> float:
        mean = self.get_mean(name)
        if mean <= eps:
            return 0.0
        return self.get_variance(name) / (mean + eps)

    def reset(self):
        self.histories = defaultdict(lambda: deque(maxlen=self.window_size))
