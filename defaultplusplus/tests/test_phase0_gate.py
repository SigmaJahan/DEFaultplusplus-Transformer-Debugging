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
    """T0.2 — __version__ returns the current package version."""
    from src.defaultplusplus import __version__

    assert __version__ == "0.4.1"


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
    # tomllib only landed in Python 3.11; on 3.10 we fall back to the
    # tomli package, which is the upstream of tomllib and has the
    # identical API.
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

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


def test_pretrained_weights_are_tracked():
    """T0.5 — Pretrained weights ship in the wheel, so they MUST be
    tracked in git. Earlier versions (pre-0.4.0) ignored ``*.pt`` and
    ``*.pkl`` under ``pretrained/weights/`` on the assumption they
    would be downloaded at runtime; the v0.4.0 release inverted that
    decision and bundles ~5 MB of trained weights directly.

    This test enforces that the inversion stays in place: if someone
    re-adds the gitignore exclusion, CI will refuse the change because
    the encoder/decoder ``.pt`` files would silently disappear from
    the next wheel build.
    """
    import subprocess
    # ``ROOT`` is the ``defaultplusplus/`` package directory, but the
    # git repo root is one level up. ``git ls-files`` paths are
    # relative to the repo root, so query and assert against that
    # form.
    repo_root = ROOT.parent
    result = subprocess.run(
        ["git", "ls-files",
         "defaultplusplus/src/defaultplusplus/pretrained/weights/"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    )
    tracked = set(result.stdout.strip().splitlines())
    must_track = {
        "defaultplusplus/src/defaultplusplus/pretrained/weights/encoder.pt",
        "defaultplusplus/src/defaultplusplus/pretrained/weights/decoder.pt",
        "defaultplusplus/src/defaultplusplus/pretrained/weights/encoder_reference.npz",
        "defaultplusplus/src/defaultplusplus/pretrained/weights/decoder_reference.npz",
    }
    missing = must_track - tracked
    assert not missing, (
        f"these pretrained-weight files must be tracked in git so they "
        f"ship in the wheel: {sorted(missing)}"
    )

    # Negative assertion: the legacy ``*.pt`` gitignore rule must be
    # gone in BOTH the repo-root and the package-level gitignores. If
    # it ever returns, the next person to rebuild the wheel will
    # quietly produce a broken release.
    for gi_path in (repo_root / ".gitignore", ROOT / ".gitignore"):
        if not gi_path.exists():
            continue
        gi = gi_path.read_text()
        assert "pretrained/weights/*.pt" not in gi, (
            f"the legacy 'ignore pretrained weights' rule is back in "
            f"{gi_path.relative_to(repo_root)}; remove it or future "
            "wheel builds will ship without trained checkpoints."
        )
