from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class OnlineStatistic:
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


class EpochAggregator:
    def __init__(self):
        self.current_epoch_stats: Dict[str, OnlineStatistic] = {}
        self.current_epoch_values: Dict[str, List[float]] = defaultdict(list)
        self.metric_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            fval = float(value)
            if math.isnan(fval) or math.isinf(fval):
                continue
            stat = self.current_epoch_stats.setdefault(key, OnlineStatistic())
            stat.update(fval)
            self.current_epoch_values[key].append(fval)

    def finalize_epoch(self, epoch_index: int) -> Dict[str, float]:
        epoch_summary: Dict[str, float] = {'epoch': epoch_index}
        for key, stat in self.current_epoch_stats.items():
            epoch_summary[f'{key}_mean'] = stat.mean
            epoch_summary[f'{key}_var'] = stat.variance
            epoch_summary[f'{key}_count'] = stat.count
            vals = self.current_epoch_values.get(key, [])
            if vals:
                epoch_summary[f'{key}_burst'] = float(np.percentile(vals, 95))
            self.metric_history[key].append((epoch_index + 1, stat.mean))
        self.current_epoch_stats = {}
        self.current_epoch_values = defaultdict(list)
        return epoch_summary

    def reset(self):
        self.current_epoch_stats = {}
        self.current_epoch_values = defaultdict(list)
        self.metric_history = defaultdict(list)


def _linear_regression_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    xs_list = list(xs)
    ys_list = list(ys)
    if len(xs_list) < 2:
        return 0.0
    x = np.array(xs_list, dtype=np.float64)
    y = np.array(ys_list, dtype=np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    if x.size < 2:
        return 0.0
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        denom = x[-1] - x[0]
        if denom == 0:
            return 0.0
        return float((y[-1] - y[0]) / denom)


def _dynamic_windows(total_epochs: int) -> Dict[str, Tuple[int, int]]:
    if total_epochs <= 0:
        return {}
    if total_epochs == 1:
        return {'early': (1, 1), 'mid': (1, 1), 'late': (1, 1)}
    if total_epochs == 2:
        return {'early': (1, 1), 'mid': (2, 2), 'late': (2, 2)}
    base = total_epochs // 3
    remainder = total_epochs % 3
    sizes = [base] * 3
    for i in range(remainder):
        sizes[i] += 1
    early_end = sizes[0]
    mid_end = sizes[0] + sizes[1]
    return {
        'early': (1, early_end),
        'mid': (early_end + 1, mid_end),
        'late': (mid_end + 1, total_epochs),
    }


def compute_window_features(
    metric_history: Dict[str, List[Tuple[int, float]]],
    total_epochs: int
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if total_epochs == 0:
        return features
    windows = _dynamic_windows(total_epochs)
    for metric, history in metric_history.items():
        if not history:
            continue
        epoch_to_mean = {epoch: value for epoch, value in history}
        for window_name, (start_epoch, end_epoch) in windows.items():
            epochs = [
                epoch for epoch in range(start_epoch, end_epoch + 1)
                if epoch <= total_epochs and epoch in epoch_to_mean
            ]
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
