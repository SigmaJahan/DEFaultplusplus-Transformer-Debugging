"""Wheel-install smoke test.

Builds the wheel from the current source tree, installs it into a
fresh virtual environment, and exercises the public API end-to-end
inside that venv via a subprocess. This catches the failure mode
where a class needed at inference (or the FeatureProcessor pickled in
a checkpoint) lives outside the installable package.

The test is gated behind the ``wheel`` pytest marker so it doesn't
run in normal dev iterations:

    pytest                          # skipped by default
    pytest -m wheel                 # explicit run, ~30–60 s

Run before every publish.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

import pytest


@pytest.mark.wheel
def test_clean_wheel_install_predicts_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Build wheel + sdist into a tmp dist/ so we don't disturb the
    #    real one. ``python -m build`` uses an isolated PEP 517 backend
    #    by default; that's what we want.
    dist_dir = tmp_path / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist_dir), str(repo_root)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, (
        f"`python -m build` failed:\nSTDOUT:\n{build.stdout}\n"
        f"STDERR:\n{build.stderr}"
    )

    wheels = list(dist_dir.glob("defaultplusplus-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel; got {wheels}"
    wheel = wheels[0]

    # 2. Fresh venv with the same Python the test runner uses.
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    venv_python = (venv_dir / "bin" / "python") if os.name != "nt" \
        else (venv_dir / "Scripts" / "python.exe")
    assert venv_python.exists(), f"venv python missing at {venv_python}"

    # 3. Install the just-built wheel **with the [viz] extra** so we
    #    also verify that extra resolves cleanly. Other extras
    #    ([hf], [baselines]) pull HuggingFace and xgboost; we leave
    #    them out of the smoke to keep install time bounded.
    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--quiet",
         f"{wheel}[viz]"],
        capture_output=True, text=True,
    )
    assert install.returncode == 0, (
        f"pip install of {wheel.name} failed:\nSTDOUT:\n{install.stdout}\n"
        f"STDERR:\n{install.stderr}"
    )

    # 4. Subprocess into the venv and exercise the public API. A clean
    #    cache dir keeps the test offline (RuntimeNormalizer doesn't
    #    fetch anything; the bundled reference covers it).
    cache_dir = tmp_path / "cache"
    smoke = textwrap.dedent("""
        import json
        import warnings
        warnings.filterwarnings("ignore")

        import defaultplusplus
        from defaultplusplus.diagnosis import (
            Diagnosis, Predictor, load_pretrained, weights_path,
        )
        from defaultplusplus.diagnosis.model import HierarchicalDiagnosisModel
        from defaultplusplus.processing import RuntimeNormalizer
        from defaultplusplus.viz import plot_diagnosis

        result = {}
        result["version"] = defaultplusplus.__version__

        # Pretrained weights ship inside the wheel.
        ep = weights_path("encoder")
        result["encoder_weights_present"] = ep.exists()

        predictor = load_pretrained("encoder")
        result["n_features"] = len(predictor.feature_names)
        result["n_categories"] = len(predictor.category_names)
        result["has_processor"] = predictor._processor is not None

        norm = RuntimeNormalizer.load("encoder")
        result["reference_n_baseline"] = norm.reference.n_baseline
        result["schemas_aligned"] = (
            norm.reference.schema == list(predictor.feature_names)
        )

        # End-to-end predict on a partial dict (key dropout simulates a
        # runtime that didn't record every metric).
        partial = {k: 0.5 for k in predictor.feature_names[:50]}
        encoded = norm.encode(partial, mode="raw")
        diag = predictor.predict(encoded)
        result["predict_returned_diagnosis"] = isinstance(diag, Diagnosis)
        result["is_faulty_is_bool"] = isinstance(diag.is_faulty, bool)
        result["det_prob_in_range"] = 0.0 <= diag.detection_prob <= 1.0

        # Viz (matplotlib must be available since the wheel pulls it
        # via the [viz] extra; the dependency declaration is what we're
        # also checking here).
        fig = plot_diagnosis(diag)
        result["plot_diagnosis_returned"] = fig is not None

        print("RESULT:" + json.dumps(result))
    """)
    run = subprocess.run(
        [str(venv_python), "-c", smoke],
        env={**os.environ,
             "DEFAULTPP_CACHE_DIR": str(cache_dir),
             "MPLBACKEND": "Agg",
             "PYTHONWARNINGS": "ignore"},
        capture_output=True, text=True,
    )
    assert run.returncode == 0, (
        f"smoke script failed inside the venv:\nSTDOUT:\n{run.stdout}\n"
        f"STDERR:\n{run.stderr}"
    )

    # 5. Parse the JSON the smoke script printed and assert each check.
    line = next(
        (ln for ln in run.stdout.splitlines() if ln.startswith("RESULT:")),
        None,
    )
    assert line is not None, (
        f"smoke script did not print RESULT line:\n{run.stdout}"
    )
    result = json.loads(line[len("RESULT:"):])

    expected = {
        "encoder_weights_present": True,
        "has_processor": True,
        "schemas_aligned": True,
        "predict_returned_diagnosis": True,
        "is_faulty_is_bool": True,
        "det_prob_in_range": True,
        "plot_diagnosis_returned": True,
    }
    for key, want in expected.items():
        assert result.get(key) == want, (
            f"{key} expected {want}, got {result.get(key)}; full: {result}"
        )

    # Sanity: the wheel-installed version matches the source version.
    src_version = (repo_root / "src" / "defaultplusplus" / "_version.py")
    assert f'"{result["version"]}"' in src_version.read_text(), (
        f"installed version {result['version']} does not match source"
    )
