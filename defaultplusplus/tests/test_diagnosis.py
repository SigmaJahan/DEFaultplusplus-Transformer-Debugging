"""Tests for the diagnosis API + training-driver smoke.

Coverage:
  * load_pretrained raises a typed error when no weights exist.
  * The training driver's synthetic mode produces a checkpoint.
  * That checkpoint round-trips through Predictor.predict() and emits
    a sane Diagnosis dataclass.
  * Schema validation rejects unknown keys with strict_schema=True.
  * A second Predictor.validate_feature_names() failure surfaces a
    diff message naming the missing/unexpected entries.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _train_synthetic_checkpoint(tmp_path: Path, *, n_features: int = 16,
                                n_samples: int = 96, epochs: int = 3) -> Path:
    """Run the training driver with --synthetic and return the .pt path."""
    out = tmp_path / "encoder_smoke.pt"
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "train_diagnoser.py"
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(
        [sys.executable, str(script), "--arch", "encoder", "--synthetic",
         "--output", str(out), "--epochs", str(epochs),
         "--n-samples", str(n_samples), "--n-features", str(n_features),
         "--seed", "0"],
        capture_output=True, text=True, env=env, cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"training driver failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert out.exists(), f"checkpoint not written to {out}"
    return out


# ─────────────────────────────────────────────────────────────────────────
# load_pretrained guard rails
# ─────────────────────────────────────────────────────────────────────────
def test_load_pretrained_raises_when_weights_missing(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import (
        PretrainedWeightsMissingError, load_pretrained,
    )
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(PretrainedWeightsMissingError) as exc_info:
        load_pretrained("encoder", weights=missing)
    msg = str(exc_info.value)
    assert "No pretrained weights at" in msg
    assert "train_diagnoser.py" in msg


def test_weights_path_default_under_pretrained_weights() -> None:
    from defaultplusplus.diagnosis import weights_path
    p = weights_path("encoder")
    assert p.parent.name == "weights"
    assert p.name == "encoder.pt"


def test_weights_path_rejects_unknown_arch() -> None:
    from defaultplusplus.diagnosis import weights_path
    with pytest.raises(ValueError, match="unknown arch"):
        weights_path("encoder-decoder")


# ─────────────────────────────────────────────────────────────────────────
# Training-driver synthetic round trip
# ─────────────────────────────────────────────────────────────────────────
def test_synthetic_train_produces_loadable_checkpoint(tmp_path: Path) -> None:
    """End-to-end: synthetic train → save → load → predict."""
    from defaultplusplus.diagnosis import Diagnosis, load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained("encoder", weights=ckpt)

    # Schema bundled with the checkpoint matches what synthetic mode emits.
    assert predictor.arch == "encoder"
    assert len(predictor.feature_names) == 16
    assert all(n.startswith("feat_") for n in predictor.feature_names)
    assert set(predictor.category_names) == {"qkv", "masking"}
    assert "head_interaction" in predictor.rootcause_names["qkv"]

    # Predict on a feature vector from the schema.
    features = {n: 0.5 for n in predictor.feature_names}
    diag = predictor.predict(features)
    assert isinstance(diag, Diagnosis)
    assert isinstance(diag.is_faulty, bool)
    assert 0.0 <= diag.detection_prob <= 1.0
    if diag.is_faulty:
        assert diag.category in {"qkv", "masking"}
        assert 0.0 <= diag.category_prob <= 1.0


def test_predict_returns_root_cause_and_group_importance_when_faulty(tmp_path: Path) -> None:
    """When the detection head fires, stage 2 + 3 must populate."""
    from defaultplusplus.diagnosis import load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained("encoder", weights=ckpt)

    # Push large positive features to drive detection toward faulty.
    features = {n: 5.0 for n in predictor.feature_names}
    diag = predictor.predict(features)
    if not diag.is_faulty:
        pytest.skip("detection head did not fire on synthetic data; "
                    "the synthetic model is too weakly trained — re-run "
                    "with more epochs if this becomes flaky")
    assert diag.category in {"qkv", "masking"}
    assert diag.root_cause is not None
    # Stage 3 group importance has one entry per FPG group.
    assert len(diag.group_importance) == 8


def test_to_dict_serializes_cleanly(tmp_path: Path) -> None:
    import json

    from defaultplusplus.diagnosis import load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained("encoder", weights=ckpt)
    features = {n: 0.0 for n in predictor.feature_names}
    diag = predictor.predict(features)

    payload = diag.to_dict()
    assert set(payload).issuperset({
        "is_faulty", "detection_prob", "category", "category_prob",
        "root_cause", "root_cause_prob", "group_importance",
    })
    # Must round-trip through JSON.
    json.dumps(payload)


# ─────────────────────────────────────────────────────────────────────────
# Schema validation on the predictor side
# ─────────────────────────────────────────────────────────────────────────
def test_strict_schema_rejects_unknown_keys(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained("encoder", weights=ckpt)
    with pytest.raises(ValueError, match="not in the trained schema"):
        predictor.predict({"definitely_unknown_column": 1.0})


def test_non_strict_schema_silently_fills_missing_keys(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained(
        "encoder", weights=ckpt, strict_schema=False,
    )
    # Empty dict — every column gets padded to 0.0; should not raise.
    diag = predictor.predict({})
    assert isinstance(diag.is_faulty, bool)


def test_validate_feature_names_diff_message(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import load_pretrained

    ckpt = _train_synthetic_checkpoint(tmp_path)
    predictor = load_pretrained("encoder", weights=ckpt)

    expected = list(predictor.feature_names) + ["my_extra_column"]
    expected.remove("feat_0000")  # also drop one to test missing branch

    with pytest.raises(ValueError) as exc_info:
        predictor.validate_feature_names(expected)
    msg = str(exc_info.value)
    assert "my_extra_column" in msg
    assert "feat_0000" in msg


# ─────────────────────────────────────────────────────────────────────────
# Format-version guard
# ─────────────────────────────────────────────────────────────────────────
def test_unknown_format_version_raises(tmp_path: Path) -> None:
    """Hand-write a checkpoint with a bogus format_version and confirm
    the loader refuses to silently consume it."""
    import torch

    from defaultplusplus.diagnosis import Predictor

    bad = tmp_path / "bad.pt"
    torch.save({"format_version": "999"}, bad)
    with pytest.raises(ValueError, match="format_version"):
        Predictor.from_checkpoint(bad)
