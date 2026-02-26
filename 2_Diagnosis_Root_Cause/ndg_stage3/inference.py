from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .mapping import resolve_subsystem

@dataclass
class NDGScoreConfig:
    alpha: float = 0.5  # core evidence weight
    beta: float = 0.5   # subsystem impact weight
    z_max: float = 6.0  # clip for robustness

def _phi(z: float, z_max: float) -> float:
    z = abs(z)
    return z if z <= z_max else z_max

def aggregate_subsystem_anomalies(
    anomaly_scores: Dict[str, float],
    core_map: Optional[Dict[str,str]] = None,
) -> Dict[str, float]:
    """
    Map per-core anomaly scores to subsystem activations via max-abs pooling.
    """
    sub_z: Dict[str,float] = {}
    for core, z in anomaly_scores.items():
        sub = resolve_subsystem(core, core_map)
        sub_z[sub] = max(sub_z.get(sub, 0.0), abs(float(z)))
    return sub_z

def score_families(
    families: List[str],
    shap_weights: Dict[str, float],
    anomaly_scores: Dict[str, float],
    family_impacts: Dict[str, Dict[str, float]],
    core_map: Optional[Dict[str,str]] = None,
    priors: Optional[Dict[str, float]] = None,
    cfg: NDGScoreConfig = NDGScoreConfig(),
) -> List[Tuple[str, float]]:
    """
    Score(f|x) = log prior(f) + alpha * sum_core( w_shap(core) * phi(z_core) )
                          + beta  * sum_sub( impact_f(sub) * phi(z_sub) )

    priors: optional family probability from Stage-2 model (or uniform if None).
    family_impacts: optional subsystem impact weights from mutation-grounded signatures.
                   if not available for a family, its impact term is zero.
    """
    sub_z_raw = aggregate_subsystem_anomalies(anomaly_scores, core_map)
    sub_z = {k: _phi(v, cfg.z_max) for k,v in sub_z_raw.items()}

    scored = []
    for f in families:
        prior = priors.get(f, 1.0/len(families)) if priors else 1.0/len(families)
        score = math.log(max(prior, 1e-12))

        core_sum = 0.0
        for core, w in shap_weights.items():
            if core not in anomaly_scores:
                continue
            core_sum += float(w) * _phi(float(anomaly_scores[core]), cfg.z_max)

        sub_sum = 0.0
        impacts = family_impacts.get(f, {})
        for sub, iw in impacts.items():
            sub_sum += float(iw) * sub_z.get(sub, 0.0)

        score += cfg.alpha * core_sum + cfg.beta * sub_sum
        scored.append((f, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
