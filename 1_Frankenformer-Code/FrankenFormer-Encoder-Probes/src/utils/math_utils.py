"""Safe mathematical operations to prevent division by zero and numerical instability."""

from typing import Union
import math
import torch
import numpy as np
from src.constants import EPSILON_NUMERICAL_STABILITY, MIN_VARIANCE_THRESHOLD


def safe_divide(
    numerator: Union[float, torch.Tensor, np.ndarray],
    denominator: Union[float, torch.Tensor, np.ndarray],
    default: float = 0.0,
    epsilon: float = EPSILON_NUMERICAL_STABILITY
) -> Union[float, torch.Tensor, np.ndarray]:
    if isinstance(denominator, torch.Tensor):
        safe_denom = torch.where(
            torch.abs(denominator) < epsilon,
            torch.ones_like(denominator) * epsilon,
            denominator
        )
        result = numerator / safe_denom
        result = torch.where(
            torch.abs(denominator) < epsilon,
            torch.ones_like(result) * default,
            result
        )
        return result
    elif isinstance(denominator, np.ndarray):
        safe_denom = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
        result = numerator / safe_denom
        return np.where(np.abs(denominator) < epsilon, default, result)
    else:
        if abs(denominator) < epsilon:
            return default
        return numerator / denominator


def safe_mean(
    values: Union[torch.Tensor, np.ndarray, list],
    default: float = 0.0
) -> float:
    if isinstance(values, torch.Tensor):
        if values.numel() == 0:
            return default
        return float(values.mean().item())
    elif isinstance(values, np.ndarray):
        if values.size == 0:
            return default
        return float(np.mean(values))
    elif isinstance(values, list):
        if len(values) == 0:
            return default
        return sum(values) / len(values)
    return default


def safe_std(
    values: Union[torch.Tensor, np.ndarray, list],
    default: float = 0.0,
    min_variance: float = MIN_VARIANCE_THRESHOLD
) -> float:
    if isinstance(values, torch.Tensor):
        if values.numel() < 2:
            return default
        var = float(values.var().item())
        return math.sqrt(var) if var >= min_variance else default
    elif isinstance(values, np.ndarray):
        if values.size < 2:
            return default
        var = float(np.var(values))
        return math.sqrt(var) if var >= min_variance else default
    elif isinstance(values, list):
        if len(values) < 2:
            return default
        mean_val = sum(values) / len(values)
        var = sum((x - mean_val) ** 2 for x in values) / len(values)
        return math.sqrt(var) if var >= min_variance else default
    return default


def safe_entropy(
    probabilities: Union[torch.Tensor, np.ndarray],
    default: float = 0.0,
    epsilon: float = EPSILON_NUMERICAL_STABILITY
) -> float:
    if isinstance(probabilities, torch.Tensor):
        probs = torch.clamp(probabilities, min=epsilon, max=1.0)
        probs = probs / (probs.sum() + epsilon)
        entropy_val = -(probs * torch.log(probs + epsilon)).sum()
        return float(entropy_val.item())
    elif isinstance(probabilities, np.ndarray):
        probs = np.clip(probabilities, epsilon, 1.0)
        probs = probs / (probs.sum() + epsilon)
        return float(-np.sum(probs * np.log(probs + epsilon)))
    return default


def safe_sqrt(
    value: Union[float, torch.Tensor, np.ndarray],
    default: float = 0.0,
    epsilon: float = EPSILON_NUMERICAL_STABILITY
) -> Union[float, torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        safe_val = torch.where(value < epsilon, torch.ones_like(value) * epsilon, value)
        result = torch.sqrt(safe_val)
        return torch.where(value < 0, torch.ones_like(result) * default, result)
    elif isinstance(value, np.ndarray):
        safe_val = np.where(value < epsilon, epsilon, value)
        result = np.sqrt(safe_val)
        return np.where(value < 0, default, result)
    else:
        if value < epsilon:
            return default
        return math.sqrt(value)


def clamp_finite(
    value: Union[float, torch.Tensor, np.ndarray],
    min_val: float = -1e10,
    max_val: float = 1e10
) -> Union[float, torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        result = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        result = torch.where(torch.isinf(result) & (result > 0), torch.ones_like(result) * max_val, result)
        result = torch.where(torch.isinf(result) & (result < 0), torch.ones_like(result) * min_val, result)
        return torch.clamp(result, min=min_val, max=max_val)
    elif isinstance(value, np.ndarray):
        result = np.where(np.isnan(value), 0.0, value)
        result = np.where(np.isinf(result) & (result > 0), max_val, result)
        result = np.where(np.isinf(result) & (result < 0), min_val, result)
        return np.clip(result, min_val, max_val)
    else:
        if math.isnan(value) or value != value:
            return 0.0
        if math.isinf(value):
            return max_val if value > 0 else min_val
        return max(min_val, min(max_val, value))
