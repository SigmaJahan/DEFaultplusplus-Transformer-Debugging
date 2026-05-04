"""Soft import for optional viz dependencies."""
from __future__ import annotations


class VizDependencyError(ImportError):
    """Raised when ``defaultplusplus.viz`` is used without ``[viz]`` extra."""


def require_matplotlib():
    """Return ``matplotlib.pyplot`` or raise :class:`VizDependencyError`."""
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise VizDependencyError(
            "defaultplusplus.viz needs matplotlib. Install with: "
            "pip install 'defaultplusplus[viz]'"
        ) from exc
    return plt
