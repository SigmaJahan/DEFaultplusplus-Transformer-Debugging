"""
Utilities for reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int):
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Return torch.device based on preference."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_generator(seed: int) -> torch.Generator:
    """Return torch.Generator seeded for DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int):
    """Seed DataLoader worker for deterministic shuffles."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
