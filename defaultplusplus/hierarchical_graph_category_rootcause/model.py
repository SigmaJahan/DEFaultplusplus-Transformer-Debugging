"""Compatibility shim: research-side import path for the diagnostic model.

The canonical class lives in :mod:`defaultplusplus.diagnosis.model`
so the installable wheel can pickle it under a stable module path.
This shim re-exports the symbols so existing research-side imports
(``from hierarchical_graph_category_rootcause.model import ...``)
keep working AND old pickled checkpoints whose ``model_kwargs`` field
references the legacy module string still resolve to the same class
object as the new canonical name.
"""
from defaultplusplus.diagnosis.model import (  # noqa: F401
    FlatEncoder,
    HierarchicalDiagnosisModel,
)

__all__ = [
    "FlatEncoder",
    "HierarchicalDiagnosisModel",
]
