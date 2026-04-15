"""Statistical Tests for Kill Functions.

Implements the exact permutation test for determining if faults are killed.
Uses all 2^N sign combinations for N seeds to compute exact p-values.
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Any


def exact_permutation_test(
    clean_metrics: List[float],
    faulty_metrics: List[float],
    alpha: float = 0.05
) -> Tuple[bool, float, Dict[str, Any]]:
    if len(clean_metrics) != len(faulty_metrics):
        raise ValueError(
            f"Mismatched lengths: clean={len(clean_metrics)}, faulty={len(faulty_metrics)}"
        )
    n_seeds = len(clean_metrics)
    if n_seeds < 2:
        raise ValueError(f"Need at least 2 seeds, got {n_seeds}")

    differences = [f - c for f, c in zip(faulty_metrics, clean_metrics)]
    mean_diff = np.mean(differences)
    sign_mean = np.sign(mean_diff)

    same_sign_count = sum(1 for d in differences if np.sign(d) == sign_mean or d == 0)
    directional_agreement = same_sign_count >= int(np.ceil(n_seeds * 0.6))

    T_obs = mean_diff
    n_combinations = 2 ** n_seeds
    sign_combinations = list(itertools.product([-1, 1], repeat=n_seeds))

    T_values = []
    for signs in sign_combinations:
        flipped_diffs = [d * s for d, s in zip(differences, signs)]
        T_values.append(np.mean(flipped_diffs))

    if sign_mean < 0:
        extreme_count = sum(1 for T in T_values if T <= T_obs)
    else:
        extreme_count = sum(1 for T in T_values if T >= T_obs)
    p_value = extreme_count / n_combinations

    killed = directional_agreement and (p_value <= alpha)

    details = {
        'n_seeds': n_seeds,
        'differences': differences,
        'mean_difference': mean_diff,
        'std_difference': np.std(differences, ddof=1) if n_seeds > 1 else 0.0,
        'directional_agreement': directional_agreement,
        'same_sign_count': same_sign_count,
        'T_obs': T_obs,
        'n_combinations': n_combinations,
        'extreme_count': extreme_count,
        'p_value': p_value,
        'alpha': alpha,
        'killed': killed
    }
    return killed, p_value, details


def batch_permutation_test(
    clean_metrics: Dict[str, List[float]],
    faulty_metrics: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    results = {}
    common_metrics = set(clean_metrics.keys()) & set(faulty_metrics.keys())
    for metric_name in common_metrics:
        try:
            killed, p_value, details = exact_permutation_test(
                clean_metrics[metric_name],
                faulty_metrics[metric_name],
                alpha=alpha
            )
            results[metric_name] = {'killed': killed, 'p_value': p_value, 'details': details}
        except Exception as e:
            results[metric_name] = {'killed': False, 'p_value': 1.0, 'error': str(e)}
    return results


def summarize_kill_results(
    test_results: Dict[str, Dict[str, Any]],
    required_metrics: List[str] = None
) -> Dict[str, Any]:
    total_metrics = len(test_results)
    killed_metrics = [name for name, result in test_results.items() if result.get('killed', False)]

    if required_metrics:
        overall_killed = all(
            test_results.get(m, {}).get('killed', False) for m in required_metrics
        )
    else:
        overall_killed = len(killed_metrics) > 0

    p_values = [result['p_value'] for result in test_results.values() if 'p_value' in result]

    return {
        'overall_killed': overall_killed,
        'total_metrics': total_metrics,
        'killed_count': len(killed_metrics),
        'killed_metrics': killed_metrics,
        'kill_rate': len(killed_metrics) / total_metrics if total_metrics > 0 else 0.0,
        'min_p_value': min(p_values) if p_values else 1.0,
        'max_p_value': max(p_values) if p_values else 1.0,
        'mean_p_value': np.mean(p_values) if p_values else 1.0,
    }
