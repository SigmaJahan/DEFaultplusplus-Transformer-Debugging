"""Training-time feature extraction.

Public entry points:

    ModelInspector       auto-discovers transformer structure from any
                         HuggingFace encoder or decoder model.
    MetricCollector      orchestrates per-step / per-epoch metric
                         collection across all metric modules.
    EpochAggregator      Welford-stable running statistics per epoch.
    feature_construction layer / step / epoch / training-phase
                         aggregation that produces the diagnostic
                         model's fixed-length feature vector.
"""

from .aggregator import EpochAggregator, RunningMetrics, compute_window_features
from .collector import MetricCollector
from .feature_construction import (
    EpochTrace,
    LayerInternalTrace,
    StepTrace,
    TrainingTrace,
    assert_feature_dim_invariants,
    build_feature_vector,
    build_paired_feature_vector,
    expected_feature_dim,
)
from .inspector import ModelInspector

__all__ = [
    "ModelInspector",
    "MetricCollector",
    "EpochAggregator",
    "RunningMetrics",
    "compute_window_features",
    "TrainingTrace",
    "LayerInternalTrace",
    "StepTrace",
    "EpochTrace",
    "assert_feature_dim_invariants",
    "expected_feature_dim",
    "build_feature_vector",
    "build_paired_feature_vector",
]
