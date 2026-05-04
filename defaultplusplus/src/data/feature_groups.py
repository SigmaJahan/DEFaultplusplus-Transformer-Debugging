"""Compatibility shim: research-side import path for feature_groups.

Canonical: :mod:`defaultplusplus.processing.feature_groups`. See
:mod:`defaultplusplus.processing.feature_processor` for the rationale.
"""
from defaultplusplus.processing.feature_groups import (  # noqa: F401
    STRUCTURAL_GROUPS,
    NON_STRUCTURAL_GROUPS,
    SUBSYSTEM_GROUPS,
    assign_feature_to_group,
    build_group_indices,
    get_group_sizes,
)

__all__ = [
    "STRUCTURAL_GROUPS",
    "NON_STRUCTURAL_GROUPS",
    "SUBSYSTEM_GROUPS",
    "assign_feature_to_group",
    "build_group_indices",
    "get_group_sizes",
]
