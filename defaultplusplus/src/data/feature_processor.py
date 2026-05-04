"""Compatibility shim: research-side import path for FeatureProcessor.

The canonical class lives in :mod:`defaultplusplus.processing.feature_processor`
so the installable wheel can pickle it under a stable module path.
This shim re-exports the symbols so existing research-side imports
(``from src.data.feature_processor import ...``) keep working AND
old pickled checkpoints with ``src.data.feature_processor.FeatureProcessor``
resolve to exactly the same class object as the new canonical name.
"""
from defaultplusplus.processing.feature_processor import (  # noqa: F401
    FeatureProcessor,
    apply_processing_in_fold,
    NAN_DROP_THRESHOLD,
    LOG_VAR_THRESHOLD,
    LOG_RATIO_THRESHOLD,
    CV_THRESHOLD,
    LAYER_RE,
)

__all__ = [
    "FeatureProcessor",
    "apply_processing_in_fold",
    "NAN_DROP_THRESHOLD",
    "LOG_VAR_THRESHOLD",
    "LOG_RATIO_THRESHOLD",
    "CV_THRESHOLD",
    "LAYER_RE",
]
