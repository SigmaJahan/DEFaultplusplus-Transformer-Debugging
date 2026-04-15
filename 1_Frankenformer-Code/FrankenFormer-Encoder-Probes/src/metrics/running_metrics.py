from collections import defaultdict, deque
from typing import Dict

import numpy as np


class RunningMetrics:
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
