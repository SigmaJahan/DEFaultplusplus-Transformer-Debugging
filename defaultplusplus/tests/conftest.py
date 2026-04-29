"""Ensure the package is importable in tests.

We add two paths to ``sys.path``:

  - the repo root, so the existing ``import src.*`` imports keep working
    (legacy module layout that the rest of the codebase already uses);
  - ``src/``, so ``import defaultplusplus`` works exactly as it would
    after a downstream user runs ``pip install -e .``. This mirrors the
    public API surface of the package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for path in (ROOT, SRC):
    spath = str(path)
    if spath not in sys.path:
        sys.path.insert(0, spath)
