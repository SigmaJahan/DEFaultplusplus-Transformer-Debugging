"""DEFault++ — hierarchical fault diagnosis for HuggingFace transformers.

Top-level entry points:

    FeatureExtractor          collect training-time features from a
                              fine-tuning run (manual training loop).
    DEFaultPlusCallback       HuggingFace ``Trainer`` callback wrapper
                              around :class:`FeatureExtractor`.
    ExtractionConfig          collection thresholds and sampling cadence.
    build_feature_vector      aggregate a typed training trace into the
                              fixed-length feature vector consumed by
                              the diagnostic model.
    build_paired_feature_vector
                              same, for paired clean / faulty traces
                              (benchmark construction only).

The ``deform`` and ``benchmark`` submodules are imported lazily; see
``defaultplusplus.deform`` and ``defaultplusplus.benchmark``.
"""

from ._version import __version__
from .api import FeatureExtractor
from .config import ExtractionConfig
from .extraction.feature_construction import (
    assert_feature_dim_invariants,
    build_feature_vector,
    build_paired_feature_vector,
    expected_feature_dim,
)


def _hf_callback():
    """Lazy import for the HF callback so missing ``transformers`` is a
    soft failure rather than a hard one at package import time."""
    from .hf_callback import DEFaultPlusCallback
    return DEFaultPlusCallback


def __getattr__(name: str):
    # ``from defaultplusplus import DEFaultPlusCallback`` works only when
    # ``transformers`` is installed; raising ImportError here gives a
    # clearer message than attribute-error spam.
    if name == "DEFaultPlusCallback":
        return _hf_callback()
    raise AttributeError(f"module 'defaultplusplus' has no attribute {name!r}")


__all__ = [
    "FeatureExtractor",
    "DEFaultPlusCallback",
    "ExtractionConfig",
    "build_feature_vector",
    "build_paired_feature_vector",
    "assert_feature_dim_invariants",
    "expected_feature_dim",
    "__version__",
]
