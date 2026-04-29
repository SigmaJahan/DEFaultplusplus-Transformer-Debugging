"""Phase 0 gate tests — all must pass before starting Phase 1."""

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "defaultplusplus"


def test_all_init_files_exist_and_importable():
    """T0.1 — All 7 __init__.py files exist and are importable."""
    subpackages = [
        "src.defaultplusplus",
        "src.defaultplusplus.extraction",
        "src.defaultplusplus.extraction.metrics",
        "src.defaultplusplus.processing",
        "src.defaultplusplus.diagnosis",
        "src.defaultplusplus.pretrained",
        "src.defaultplusplus.ui",
    ]
    for mod_path in subpackages:
        mod = importlib.import_module(mod_path)
        assert mod is not None, f"Failed to import {mod_path}"


def test_version():
    """T0.2 — __version__ returns '0.2.0'."""
    from src.defaultplusplus import __version__

    assert __version__ == "0.2.0"


def test_directory_structure():
    """T0.3 — Directory structure matches package layout."""
    expected_dirs = [
        PKG,
        PKG / "extraction",
        PKG / "extraction" / "metrics",
        PKG / "processing",
        PKG / "diagnosis",
        PKG / "pretrained",
        PKG / "pretrained" / "weights",
        PKG / "ui",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Missing directory: {d}"


def test_pyproject_toml():
    """T0.4 — pyproject.toml is well-formed and ships defaultplusplus.

    Version is read dynamically from ``defaultplusplus._version`` so
    it is not a literal field in ``[project]``.
    """
    import tomllib

    toml_path = ROOT / "pyproject.toml"
    assert toml_path.exists()
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    project = data["project"]
    assert project["name"] == "defaultplusplus"
    assert "version" in project.get("dynamic", []), \
        "version must be declared dynamic so a single source of truth wins"

    # Core runtime dependencies stay in [project.dependencies]; the
    # viz / hf / baselines extras are checked separately because they
    # are optional. Update this list when new core deps are added.
    deps = project["dependencies"]
    dep_names = [d.split(">=")[0].split("[")[0] for d in deps]
    for req in ["torch", "transformers", "joblib"]:
        assert req in dep_names, f"Missing core dependency: {req}"

    # Visualization moved to the [viz] extra.
    viz_extra = project.get("optional-dependencies", {}).get("viz", [])
    viz_names = [d.split(">=")[0].split("[")[0] for d in viz_extra]
    assert "rich" in viz_names, "expected rich in [viz] extra"


def test_gitignore_pretrained_weights():
    """T0.5 — .gitignore contains pretrained weights exclusion."""
    gitignore = (ROOT / ".gitignore").read_text()
    assert "src/defaultplusplus/pretrained/weights/*.pt" in gitignore
    assert "src/defaultplusplus/pretrained/weights/*.pkl" in gitignore
