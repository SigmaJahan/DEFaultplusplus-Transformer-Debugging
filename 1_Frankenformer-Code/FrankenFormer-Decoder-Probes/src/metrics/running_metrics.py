from collections import defaultdict, deque
from typing import Dict

import numpy as np


class RunningMetrics:
    """Generic rolling-window statistics for high-variance metrics."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, name: str, value: float):
        """Append a new value to the named history."""
        if value is None:
            return
        self.histories[name].append(float(value))

    def get_variance(self, name: str) -> float:
        """Return variance for the requested history."""
        history = self.histories.get(name)
        if history is None or len(history) < 2:
            return 0.0
        return float(np.var(history))

    def get_mean(self, name: str) -> float:
        """Return mean of the requested history."""
        history = self.histories.get(name)
        if history is None or len(history) == 0:
            return 0.0
        return float(np.mean(history))

    def get_noise_scale(self, name: str, eps: float = 1e-8) -> float:
        """Return variance to mean ratio (gradient noise proxy)."""
        mean = self.get_mean(name)
        if mean <= eps:
            return 0.0
        return self.get_variance(name) / (mean + eps)

    def reset(self):
        """Clear histories."""
        self.histories = defaultdict(lambda: deque(maxlen=self.window_size))
