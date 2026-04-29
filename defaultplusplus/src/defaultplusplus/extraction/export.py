"""Feature vector export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def export_to_dataframe(
    feature_vector: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Create single-row DataFrame from feature vector."""
    if len(feature_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(feature_vector)} values vs {len(feature_names)} names"
        )
    return pd.DataFrame([feature_vector], columns=feature_names)


def export_to_csv(
    feature_vector: np.ndarray,
    feature_names: List[str],
    path: Union[str, Path],
) -> None:
    """Write feature vector to CSV."""
    df = export_to_dataframe(feature_vector, feature_names)
    df.to_csv(path, index=False)


def export_to_dict(
    feature_vector: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Convert to dict mapping name -> value."""
    if len(feature_vector) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(feature_vector)} values vs {len(feature_names)} names"
        )
    return {name: float(val) for name, val in zip(feature_names, feature_vector)}
