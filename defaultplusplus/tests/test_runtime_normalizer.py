"""Tests for ``defaultplusplus.processing.RuntimeNormalizer``.

Covers:

  * fit_reference + RuntimeReference round-trip through ``.npz``
  * encode(mode='raw') fills missing keys with the baseline median
  * encode(mode='anomaly') zeros missing keys and produces z-scores
    on present keys
  * Short-form (``..._l3_...``) and long-form (``..._layer3_...``)
    layer naming both round-trip through encode()
  * ``trace__`` prefix from the live extractor is stripped before
    the schema lookup
  * RuntimeNormalizer.load(arch) finds the shipped reference and the
    schema lines up with the diagnostic-model checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────
# fit_reference + .npz round-trip
# ─────────────────────────────────────────────────────────────────────────
def test_fit_reference_basic_stats() -> None:
    from defaultplusplus.processing import fit_reference

    rng = np.random.default_rng(0)
    X = rng.normal(loc=2.0, scale=1.0, size=(200, 4)).astype(np.float32)
    names = ["a", "b", "c", "d"]
    ref = fit_reference(X, names, arch="encoder")

    assert ref.schema == names
    assert ref.median.shape == (4,)
    # Median of N(2,1) draws is close to 2.
    assert np.allclose(ref.median, 2.0, atol=0.2)
    # MAD-as-sigma should also be near 1 for N(0,1)-ish data.
    assert np.allclose(ref.mad, 1.0, atol=0.3)
    assert ref.n_baseline == 200
    assert ref.arch == "encoder"


def test_reference_save_load_roundtrip(tmp_path: Path) -> None:
    from defaultplusplus.processing import RuntimeReference, fit_reference

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    ref = fit_reference(X, ["x", "y", "z"], arch="decoder")

    out = tmp_path / "ref.npz"
    ref.save(out)
    loaded = RuntimeReference.load(out)

    assert loaded.schema == ref.schema
    assert loaded.arch == ref.arch
    assert loaded.n_baseline == ref.n_baseline
    np.testing.assert_allclose(loaded.median, ref.median)
    np.testing.assert_allclose(loaded.mad, ref.mad)


def test_fit_reference_handles_nans() -> None:
    """NaN cells must be skipped per-column rather than poisoning the
    whole column or raising."""
    from defaultplusplus.processing import fit_reference

    X = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, np.nan], [4.0, 7.0]],
                 dtype=np.float32)
    ref = fit_reference(X, ["a", "b"], arch="encoder")
    assert np.isfinite(ref.median).all()
    # Column b's median over the two non-NaN rows is 6.
    assert ref.median[1] == pytest.approx(6.0)


def test_fit_reference_constant_column_floors_mad() -> None:
    """A constant column has MAD == 0; the floor must keep encode()
    safe from division by zero."""
    from defaultplusplus.processing import fit_reference, RuntimeNormalizer

    X = np.array([[1.0, 7.0], [1.0, 8.0], [1.0, 9.0]], dtype=np.float32)
    ref = fit_reference(X, ["const", "var"], arch="encoder")
    assert ref.mad[0] > 0  # floored
    norm = RuntimeNormalizer(ref)
    out = norm.encode({"const": 1.0, "var": 8.0}, mode="anomaly")
    assert np.isfinite(out["const"]) and np.isfinite(out["var"])


# ─────────────────────────────────────────────────────────────────────────
# encode() behavior
# ─────────────────────────────────────────────────────────────────────────
def _toy_normalizer():
    from defaultplusplus.processing import RuntimeNormalizer, fit_reference

    rng = np.random.default_rng(2)
    schema = ["a", "b", "c", "d"]
    X = rng.normal(loc=10.0, scale=2.0, size=(100, 4)).astype(np.float32)
    ref = fit_reference(X, schema, arch="encoder")
    return RuntimeNormalizer(ref), ref


def test_encode_raw_fills_missing_with_median() -> None:
    norm, ref = _toy_normalizer()
    # User passes only "a"; "b", "c", "d" must be filled with their medians.
    out = norm.encode({"a": 99.0}, mode="raw")
    assert set(out.keys()) == set(ref.schema)
    assert out["a"] == 99.0
    for k in ("b", "c", "d"):
        idx = ref.schema.index(k)
        assert out[k] == pytest.approx(float(ref.median[idx]))


def test_encode_anomaly_zeros_missing_keys() -> None:
    norm, ref = _toy_normalizer()
    out = norm.encode({"a": 99.0}, mode="anomaly")
    # Present key: z-score
    expected_a = (99.0 - ref.median[0]) / ref.mad[0]
    assert out["a"] == pytest.approx(float(expected_a))
    # Absent keys: 0 (definitionally no deviation).
    for k in ("b", "c", "d"):
        assert out[k] == 0.0


def test_encode_rejects_bad_mode() -> None:
    norm, _ = _toy_normalizer()
    with pytest.raises(ValueError, match="mode must be"):
        norm.encode({"a": 1.0}, mode="weird")


# ─────────────────────────────────────────────────────────────────────────
# Layer-name aliasing + trace__ prefix stripping
# ─────────────────────────────────────────────────────────────────────────
def test_short_to_long_layer_aliasing() -> None:
    """A reference that knows only the long-form key ``foo_layer3_bar``
    should still consume a runtime dict with the short-form
    ``foo_l3_bar``."""
    from defaultplusplus.processing import RuntimeNormalizer, fit_reference

    schema = ["foo_layer3_bar"]
    X = np.array([[5.0], [7.0], [9.0]], dtype=np.float32)
    ref = fit_reference(X, schema, arch="encoder")
    norm = RuntimeNormalizer(ref)

    out = norm.encode({"foo_l3_bar": 42.0}, mode="raw")
    assert out["foo_layer3_bar"] == 42.0


def test_long_to_short_layer_aliasing() -> None:
    """Mirror of the above: long-form runtime input, short-form schema."""
    from defaultplusplus.processing import RuntimeNormalizer, fit_reference

    schema = ["foo_l3_bar"]
    X = np.array([[5.0], [7.0], [9.0]], dtype=np.float32)
    ref = fit_reference(X, schema, arch="decoder")
    norm = RuntimeNormalizer(ref)

    out = norm.encode({"foo_layer3_bar": 42.0}, mode="raw")
    assert out["foo_l3_bar"] == 42.0


def test_trace_prefix_stripped() -> None:
    """The live extractor namespaces auxiliary aggregates under
    ``trace__``; the normalizer must drop the prefix when matching."""
    from defaultplusplus.processing import RuntimeNormalizer, fit_reference

    schema = ["loss_mean"]
    ref = fit_reference(np.array([[0.5]], dtype=np.float32),
                        schema, arch="encoder")
    norm = RuntimeNormalizer(ref)
    out = norm.encode({"trace__loss_mean": 1.5}, mode="raw")
    assert out["loss_mean"] == 1.5


# ─────────────────────────────────────────────────────────────────────────
# Coverage helper
# ─────────────────────────────────────────────────────────────────────────
def test_coverage_counts_matched_schema_keys() -> None:
    norm, ref = _toy_normalizer()
    n_match, n_total = norm.coverage({"a": 1.0, "c": 3.0, "extra_key": 9.0})
    assert n_match == 2  # a + c match; extra_key doesn't; b,d missing
    assert n_total == 4


# ─────────────────────────────────────────────────────────────────────────
# Shipped reference loads and aligns with the predictor schema
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("arch", ["encoder", "decoder"])
def test_shipped_reference_matches_predictor_schema(arch: str) -> None:
    from defaultplusplus.diagnosis import load_pretrained, weights_path
    from defaultplusplus.processing import RuntimeNormalizer

    if not weights_path(arch).exists():
        pytest.skip(f"no shipped {arch}.pt; skipping schema-alignment check")

    norm = RuntimeNormalizer.load(arch)
    predictor = load_pretrained(arch)
    # The reference must match the predictor's user-facing schema
    # exactly so encode() output is a drop-in for predict().
    assert norm.reference.schema == list(predictor.feature_names)


def test_load_missing_reference_raises(tmp_path: Path) -> None:
    from defaultplusplus.processing import RuntimeNormalizer

    bogus = tmp_path / "does_not_exist.npz"
    with pytest.raises(FileNotFoundError, match="no runtime reference"):
        RuntimeNormalizer.load("encoder", reference=bogus)
