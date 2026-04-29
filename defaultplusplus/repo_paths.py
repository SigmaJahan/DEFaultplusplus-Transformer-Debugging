"""Canonical paths used by the research-side training and evaluation
drivers. Resolves locations based on this file's position rather than
the current working directory, so scripts work no matter where they are
invoked from.
"""

from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent

DATA_ROOT      = REPO_ROOT / "data"
BASELINE_ROOT  = REPO_ROOT / "baselines"
BENCHMARK_ROOT = REPO_ROOT / "realworld_benchmark"
USER_STUDY_ROOT = REPO_ROOT / "user_study"
RESULTS_ROOT   = REPO_ROOT / "results"
CONFIGS_ROOT   = CODE_ROOT / "configs"


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
