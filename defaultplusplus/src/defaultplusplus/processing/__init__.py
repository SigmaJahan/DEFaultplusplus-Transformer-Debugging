"""Runtime feature processing.

Public entry points:

    RuntimeNormalizer       loads a clean reference and converts a live
                            ``FeatureExtractor.finalize()`` dict into
                            the diagnostic model's input shape.
    RuntimeReference        the underlying ``(schema, median, mad, std)``
                            summary; serialize to ``.npz``.
    fit_reference           build a :class:`RuntimeReference` from a
                            baseline-only feature matrix.
"""
from .normalizer import (
    RuntimeNormalizer,
    RuntimeReference,
    fit_reference,
)

__all__ = [
    "RuntimeNormalizer",
    "RuntimeReference",
    "fit_reference",
]
