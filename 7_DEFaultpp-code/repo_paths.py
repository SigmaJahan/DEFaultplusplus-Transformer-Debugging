"""Canonical paths for the reorganized repository."""

from __future__ import annotations

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent

DATA_ROOT = REPO_ROOT / "3_Mutation-Data-from-Frakenformer"
BASELINE_ROOT = REPO_ROOT / "4_Baseline-comparison_with_defaultpp"
BENCHMARK_ROOT = REPO_ROOT / "5_Benchmarks-realworld_defaultpp"
MANUSCRIPT_ROOT = REPO_ROOT / "2_Frakenformer-DEFaultpp-Manuscript"
RESULTS_ROOT = REPO_ROOT / "results"
CONFIGS_ROOT = CODE_ROOT / "configs"


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
