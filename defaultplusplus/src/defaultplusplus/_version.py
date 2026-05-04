"""Single source of truth for the package version.

The version follows Semantic Versioning 2.0:

    MAJOR.MINOR.PATCH

Bump rules:
  MAJOR   incompatible API changes (e.g. removing a public symbol,
          changing a function signature in a non-additive way).
  MINOR   backwards-compatible additions (new symbols, new optional
          features).
  PATCH   backwards-compatible bug fixes.

The string below is the only place the version lives. ``pyproject.toml``
reads it via ``[tool.setuptools.dynamic]`` and the package re-exports it
as ``defaultplusplus.__version__``.
"""

__version__ = "0.4.0"
