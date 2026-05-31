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
# FeatureProcessor round-trip (raw input → processed model input)
# ─────────────────────────────────────────────────────────────────────────
def test_predict_applies_feature_processor_when_bundled(tmp_path: Path) -> None:
    """A checkpoint trained with FeatureProcessor turning per-layer columns
    into ``__agg_*`` summaries must accept raw extractor names at predict
    time and replay the processor before scoring.

    We simulate the encoder benchmark shape by hand-rolling a tiny
    training matrix with a per-layer column family. This sidesteps the
    multi-GB benchmark CSV while still exercising the load → vectorize
    → transform → scale path on a non-trivial schema mismatch.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from defaultplusplus.diagnosis import (
        Predictor, load_pretrained, save_checkpoint,
    )
    from data.feature_processor import FeatureProcessor
    from hierarchical_graph_category_rootcause.model import (
        HierarchicalDiagnosisModel,
    )

    rng = np.random.default_rng(0)
    # Per-layer family: 4 layers × the same metric. After Step 3 these
    # collapse to four __agg_{mean,std,min,max} columns; the raw schema
    # length (5) ends up different from the processed schema length (5
    # too in this case but renamed).
    raw_names = [
        "feat_l0_phase",
        "feat_l1_phase",
        "feat_l2_phase",
        "feat_l3_phase",
        "extra_metric",
    ]
    n = 64
    X = rng.normal(size=(n, len(raw_names))).astype(np.float32)
    y_detect = (rng.random(n) < 0.5).astype(np.int64)
    category_names = ["qkv"]
    rootcause_names = {"qkv": ["zero_query", "head_interaction"]}
    y_cat = np.where(y_detect == 1, 0, -1).astype(np.int64)
    y_rc = np.where(y_detect == 1,
                    rng.integers(0, 2, size=n), -1).astype(np.int64)

    proc = FeatureProcessor(arch="encoder")
    X_proc, processed_names, _ = proc.fit_transform(X, raw_names, y=y_detect)
    assert len(processed_names) != len(raw_names) or processed_names != raw_names, (
        "expected layer aggregation to rename or resize columns; "
        f"raw={raw_names}, processed={processed_names}"
    )

    scaler_mean = X_proc.mean(axis=0)
    scaler_scale = X_proc.std(axis=0)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)

    model = HierarchicalDiagnosisModel(
        input_dim=X_proc.shape[1],
        hidden_dim=8,
        embedding_dim=16,
        dropout=0.0,
        mode="flat",
        n_categories=1,
        category_sizes={"qkv": 2},
        group_names=["attention", "qkv_alignment", "ffn_output",
                     "residual_stream", "output", "training_dynamics",
                     "validation_perf", "representation_drift"],
    )
    model.eval()

    ckpt = tmp_path / "fp_smoke.pt"
    save_checkpoint(
        path=ckpt,
        arch="encoder",
        feature_names=processed_names,
        category_names=category_names,
        category_sizes={"qkv": 2},
        rootcause_names=rootcause_names,
        group_names=["attention", "qkv_alignment", "ffn_output",
                     "residual_stream", "output", "training_dynamics",
                     "validation_perf", "representation_drift"],
        model_state_dict=model.state_dict(),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        prototypes={},
        model_kwargs={
            "input_dim": X_proc.shape[1], "hidden_dim": 8,
            "embedding_dim": 16, "dropout": 0.0, "mode": "flat",
            "n_categories": 1, "category_sizes": {"qkv": 2},
            "group_names": ["attention", "qkv_alignment", "ffn_output",
                            "residual_stream", "output", "training_dynamics",
                            "validation_perf", "representation_drift"],
        },
        extra={
            "raw_feature_names": raw_names,
            "feature_processor": proc,
            "group_indices": {},
        },
    )

    predictor = load_pretrained("encoder", weights=ckpt)

    # User-facing schema is the RAW names (what FeatureExtractor emits).
    assert predictor.feature_names == raw_names
    assert predictor._processor is not None

    # Predict using raw-name keys; predictor must replay the processor.
    raw_features = {name: float(X[0, i]) for i, name in enumerate(raw_names)}
    diag = predictor.predict(raw_features)
    assert isinstance(diag.is_faulty, bool)
    assert 0.0 <= diag.detection_prob <= 1.0


def test_legacy_checkpoint_without_extras_still_loads(tmp_path: Path) -> None:
    """v1 checkpoints (no ``extra`` dict) must still round-trip — the
    inference path falls back to direct vectorize → scale."""
    import torch

    from defaultplusplus.diagnosis import Predictor

    ckpt = _train_synthetic_checkpoint(tmp_path)

    # Strip the ``extra`` key to simulate a pre-FeatureProcessor checkpoint.
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload.pop("extra", None)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    predictor = Predictor.from_checkpoint(legacy)
    assert predictor._processor is None
    # The user-facing schema falls back to feature_names at the top level.
    assert predictor.feature_names == predictor._processed_feature_names

    features = {n: 0.0 for n in predictor.feature_names}
    diag = predictor.predict(features)
    assert isinstance(diag.is_faulty, bool)


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


# ─────────────────────────────────────────────────────────────────────────
# Label-space validation against the official taxonomy
# ─────────────────────────────────────────────────────────────────────────
def _import_validate_label_space():
    """Import the training driver's validate_label_space, or skip."""
    repo_root = Path(__file__).resolve().parent.parent
    folder = repo_root / "hierarchical_graph_category_rootcause"
    sys.path.insert(0, str(folder))
    try:
        from train import validate_label_space  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"training driver not importable: {exc}")
    return validate_label_space


def test_label_space_matches_canonical_when_complete() -> None:
    """A full decoder label space reports no missing or unexpected pairs."""
    from defaultplusplus.deform import root_cause_label_space

    validate_label_space = _import_validate_label_space()

    # Build a category_to_rootcauses that mirrors the taxonomy.
    c2rc = {}
    gi = 0
    for comp, rcs in root_cause_label_space("decoder").items():
        c2rc[comp.value] = [(gi + i, rc) for i, rc in enumerate(rcs)]
        gi += len(rcs)

    report = validate_label_space("decoder", c2rc)
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["expected_root_causes"] == 45
    assert report["discovered_root_causes"] == 45


def test_label_space_flags_missing_and_unexpected() -> None:
    """Dropping a taxonomy pair flags missing; adding a bogus one flags drift."""
    from defaultplusplus.deform import root_cause_label_space

    validate_label_space = _import_validate_label_space()

    c2rc = {}
    gi = 0
    for comp, rcs in root_cause_label_space("encoder").items():
        c2rc[comp.value] = [(gi + i, rc) for i, rc in enumerate(rcs)]
        gi += len(rcs)

    # Drop one real root cause and inject one outside the taxonomy.
    some_cat = next(iter(c2rc))
    dropped_rc = c2rc[some_cat].pop()[1]
    c2rc[some_cat].append((999, "not_a_real_root_cause"))

    report = validate_label_space("encoder", c2rc)
    # Missing/unexpected pairs are stored in normalized (lower, underscore) form.
    assert (some_cat.lower(), dropped_rc.lower()) in report["missing"]
    assert any(r == "not_a_real_root_cause" for _, r in report["unexpected"])
    assert report["discovered_root_causes"] == 39  # 40 in the taxonomy minus the dropped one
