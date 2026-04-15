"""
Safe mathematical operations to prevent division by zero and numerical instability.
"""

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
    """
    Safely divide numerator by denominator, avoiding division by zero.

    Args:
        numerator: Value to divide
        denominator: Value to divide by
        default: Value to return if denominator is too small
        epsilon: Minimum threshold for denominator

    Returns:
        Result of division or default if denominator too small
    """
    if isinstance(denominator, torch.Tensor):
        # Avoid division by values close to zero
        safe_denom = torch.where(
            torch.abs(denominator) < epsilon,
            torch.ones_like(denominator) * epsilon,
            denominator
        )
        result = numerator / safe_denom
        # Replace results where original denominator was too small
        result = torch.where(
            torch.abs(denominator) < epsilon,
            torch.ones_like(result) * default,
            result
        )
        return result
    elif isinstance(denominator, np.ndarray):
        # NumPy version
        safe_denom = np.where(
            np.abs(denominator) < epsilon,
            epsilon,
            denominator
        )
        result = numerator / safe_denom
        result = np.where(
            np.abs(denominator) < epsilon,
            default,
            result
        )
        return result
    else:
        # Scalar version
        if abs(denominator) < epsilon:
            return default
        return numerator / denominator


def safe_mean(
    values: Union[torch.Tensor, np.ndarray, list],
    default: float = 0.0
) -> float:
    """
    Safely compute mean, handling empty sequences.

    Args:
        values: Values to average
        default: Value to return if sequence is empty

    Returns:
        Mean of values or default if empty
    """
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
    else:
        return default


def safe_std(
    values: Union[torch.Tensor, np.ndarray, list],
    default: float = 0.0,
    min_variance: float = MIN_VARIANCE_THRESHOLD
) -> float:
    """
    Safely compute standard deviation, handling edge cases.

    Args:
        values: Values to compute std for
        default: Value to return if sequence is too small
        min_variance: Minimum variance threshold

    Returns:
        Standard deviation or default
    """
    if isinstance(values, torch.Tensor):
        if values.numel() < 2:
            return default
        var = float(values.var().item())
        if var < min_variance:
            return default
        return math.sqrt(var)
    elif isinstance(values, np.ndarray):
        if values.size < 2:
            return default
        var = float(np.var(values))
        if var < min_variance:
            return default
        return math.sqrt(var)
    elif isinstance(values, list):
        if len(values) < 2:
            return default
        mean_val = sum(values) / len(values)
        var = sum((x - mean_val) ** 2 for x in values) / len(values)
        if var < min_variance:
            return default
        return math.sqrt(var)
    else:
        return default


def safe_entropy(
    probabilities: Union[torch.Tensor, np.ndarray],
    default: float = 0.0,
    epsilon: float = EPSILON_NUMERICAL_STABILITY
) -> float:
    """
    Safely compute entropy, avoiding log(0).

    Args:
        probabilities: Probability distribution
        default: Value to return if computation fails
        epsilon: Minimum probability value

    Returns:
        Entropy value or default
    """
    if isinstance(probabilities, torch.Tensor):
        # Clamp probabilities to avoid log(0)
        probs = torch.clamp(probabilities, min=epsilon, max=1.0)
        # Normalize to ensure sum = 1
        probs = probs / (probs.sum() + epsilon)
        entropy_val = -(probs * torch.log(probs + epsilon)).sum()
        return float(entropy_val.item())
    elif isinstance(probabilities, np.ndarray):
        # NumPy version
        probs = np.clip(probabilities, epsilon, 1.0)
        probs = probs / (probs.sum() + epsilon)
        entropy_val = -np.sum(probs * np.log(probs + epsilon))
        return float(entropy_val)
    else:
        return default


def safe_sqrt(
    value: Union[float, torch.Tensor, np.ndarray],
    default: float = 0.0,
    epsilon: float = EPSILON_NUMERICAL_STABILITY
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Safely compute square root, handling negative values.

    Args:
        value: Value to take square root of
        default: Value to return for negative inputs
        epsilon: Minimum threshold for taking sqrt

    Returns:
        Square root or default for negative/small values
    """
    if isinstance(value, torch.Tensor):
        # Replace negative values with epsilon
        safe_val = torch.where(value < epsilon, torch.ones_like(value) * epsilon, value)
        result = torch.sqrt(safe_val)
        # Return default for originally negative values
        result = torch.where(value < 0, torch.ones_like(result) * default, result)
        return result
    elif isinstance(value, np.ndarray):
        safe_val = np.where(value < epsilon, epsilon, value)
        result = np.sqrt(safe_val)
        result = np.where(value < 0, default, result)
        return result
    else:
        if value < epsilon:
            return default
        return math.sqrt(value)


def clamp_finite(
    value: Union[float, torch.Tensor, np.ndarray],
    min_val: float = -1e10,
    max_val: float = 1e10
) -> Union[float, torch.Tensor, np.ndarray]:
    """
    Clamp value to finite range, replacing NaN and Inf.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value with NaN/Inf replaced
    """
    if isinstance(value, torch.Tensor):
        # Replace NaN and Inf
        result = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        result = torch.where(torch.isinf(result) & (result > 0), torch.ones_like(result) * max_val, result)
        result = torch.where(torch.isinf(result) & (result < 0), torch.ones_like(result) * min_val, result)
        # Clamp to range
        result = torch.clamp(result, min=min_val, max=max_val)
        return result
    elif isinstance(value, np.ndarray):
        result = np.where(np.isnan(value), 0.0, value)
        result = np.where(np.isinf(result) & (result > 0), max_val, result)
        result = np.where(np.isinf(result) & (result < 0), min_val, result)
        result = np.clip(result, min_val, max_val)
        return result
    else:
        if math.isnan(value) or value != value:  # NaN check
            return 0.0
        if math.isinf(value):
            return max_val if value > 0 else min_val
        return max(min_val, min(max_val, value))
