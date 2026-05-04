"""Compatibility shim: research-side import path for the group encoder.

The canonical implementations live in
:mod:`defaultplusplus.diagnosis._group_encoder` so the installable
wheel can pickle them under a stable module path. This shim re-exports
the symbols so existing research-side imports
(``from src.models.group_encoder import ...``) keep working.
"""
from defaultplusplus.diagnosis._group_encoder import (  # noqa: F401
    FlatEncoder,
    GraphAggregator,
    GroupEncoder,
)

__all__ = [
    "FlatEncoder",
    "GraphAggregator",
    "GroupEncoder",
]
