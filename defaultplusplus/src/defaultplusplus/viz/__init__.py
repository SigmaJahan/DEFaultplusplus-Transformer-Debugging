"""Visualization helpers for DEFault++.

Single-call entry points:

    save_diagnosis_report(diagnosis, features, path)
    save_run_report(features, path)

Plus the underlying figure-returning functions used to compose them:

    plot_diagnosis(diagnosis)
    plot_group_importance(diagnosis)
    plot_per_layer_heatmap(features, metric)
    plot_training_trace(features, keys)
    plot_attention_pattern(features, layer)
    plot_qkv_alignment(features)
    plot_feature_anomaly(features, baseline)

This subpackage requires the ``[viz]`` extra. Importing it without
matplotlib installed raises ``VizDependencyError`` with the install
hint.
"""
from ._deps import VizDependencyError  # noqa: F401
from .plots import (  # noqa: F401
    plot_attention_pattern,
    plot_diagnosis,
    plot_feature_anomaly,
    plot_group_importance,
    plot_per_layer_heatmap,
    plot_qkv_alignment,
    plot_training_trace,
)
from .report import save_diagnosis_report, save_run_report  # noqa: F401

__all__ = [
    "VizDependencyError",
    "plot_attention_pattern",
    "plot_diagnosis",
    "plot_feature_anomaly",
    "plot_group_importance",
    "plot_per_layer_heatmap",
    "plot_qkv_alignment",
    "plot_training_trace",
    "save_diagnosis_report",
    "save_run_report",
]
