"""Tests for ``defaultplusplus.viz``.

Renders each plot to PNG via ``Figure.savefig`` and asserts the output
file is non-empty. We don't image-diff — too fragile across matplotlib
versions; non-empty PNG is enough to catch regressions like a function
returning ``None`` or raising during render.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Force a non-interactive backend before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _toy_features() -> dict[str, float]:
    """Hand-rolled feature dict with both per-layer and flat keys plus
    enough phase suffixes to exercise heatmap + trace + qkv + attention
    plots."""
    feats: dict[str, float] = {}
    # Per-layer attention family (long form)
    for li in range(4):
        for ph in ("early_mean", "mid_mean", "final_mean"):
            feats[f"attention_entropy_layer{li}__epoch_mean__phase_{ph}"] = (
                0.5 + 0.05 * li
            )
    # Per-layer QKV cosine families (short form)
    for li in range(4):
        feats[f"qk_cos_l{li}_phase_early_mean"] = 0.7 - 0.05 * li
        feats[f"qv_cos_l{li}_phase_early_mean"] = 0.6 - 0.04 * li
        feats[f"kv_cos_l{li}_phase_early_mean"] = 0.5 - 0.03 * li
    # Flat training-trace keys
    for ph in ("early_mean", "mid_mean", "final_mean"):
        feats[f"accuracy__epoch_mean__phase_{ph}"] = 0.85
        feats[f"loss__epoch_mean__phase_{ph}"] = 0.15
    # Attention shape components for layer 0
    feats["attention_sparsity_layer0__epoch_mean__phase_early_mean"] = 0.3
    feats["attention_mass_future_layer0__epoch_mean__phase_early_mean"] = 0.1
    feats["head_similarity_mean_layer0__epoch_mean__phase_early_mean"] = 0.6
    return feats


def _toy_diagnosis():
    from defaultplusplus.diagnosis import Diagnosis
    return Diagnosis(
        is_faulty=True,
        detection_prob=0.92,
        category="qkv",
        category_prob=0.71,
        root_cause="zero_query",
        root_cause_prob=0.55,
        group_importance={
            "attention": 0.9,
            "qkv_alignment": 1.5,
            "ffn_output": -0.3,
            "training_dynamics": 0.1,
        },
    )


def _save(fig, tmp_path, name) -> Path:
    out = tmp_path / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=80)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Individual plot smoke tests
# ─────────────────────────────────────────────────────────────────────────
def test_plot_diagnosis_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_diagnosis
    out = _save(plot_diagnosis(_toy_diagnosis()), tmp_path, "diag")
    assert out.stat().st_size > 0


def test_plot_diagnosis_handles_clean_run(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import Diagnosis
    from defaultplusplus.viz import plot_diagnosis
    clean = Diagnosis(is_faulty=False, detection_prob=0.05,
                      group_importance={})
    out = _save(plot_diagnosis(clean), tmp_path, "diag_clean")
    assert out.stat().st_size > 0


def test_plot_group_importance_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_group_importance
    out = _save(plot_group_importance(_toy_diagnosis()), tmp_path, "gi")
    assert out.stat().st_size > 0


def test_plot_group_importance_handles_empty(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import Diagnosis
    from defaultplusplus.viz import plot_group_importance
    diag = Diagnosis(is_faulty=False, detection_prob=0.0,
                     group_importance={})
    out = _save(plot_group_importance(diag), tmp_path, "gi_empty")
    assert out.stat().st_size > 0


def test_plot_per_layer_heatmap_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_per_layer_heatmap
    out = _save(plot_per_layer_heatmap(_toy_features(), "attention_entropy"),
                tmp_path, "heatmap")
    assert out.stat().st_size > 0


def test_plot_per_layer_heatmap_no_match(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_per_layer_heatmap
    out = _save(plot_per_layer_heatmap(_toy_features(), "bogus_metric_xyz"),
                tmp_path, "heatmap_miss")
    assert out.stat().st_size > 0


def test_plot_training_trace_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_training_trace
    out = _save(plot_training_trace(_toy_features(), ["accuracy", "loss"]),
                tmp_path, "trace")
    assert out.stat().st_size > 0


def test_plot_attention_pattern_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_attention_pattern
    out = _save(plot_attention_pattern(_toy_features(), layer=0),
                tmp_path, "attn")
    assert out.stat().st_size > 0


def test_plot_qkv_alignment_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_qkv_alignment
    out = _save(plot_qkv_alignment(_toy_features()), tmp_path, "qkv")
    assert out.stat().st_size > 0


def test_plot_feature_anomaly_renders(tmp_path: Path) -> None:
    from defaultplusplus.viz import plot_feature_anomaly
    feats = _toy_features()
    baseline = {k: 0.0 for k in feats}
    out = _save(plot_feature_anomaly(feats, baseline, top_n=15),
                tmp_path, "anomaly")
    assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────
# HTML report writers
# ─────────────────────────────────────────────────────────────────────────
def test_save_diagnosis_report_writes_self_contained_html(tmp_path: Path) -> None:
    from defaultplusplus.viz import save_diagnosis_report
    out = tmp_path / "diag.html"
    save_diagnosis_report(_toy_diagnosis(), _toy_features(), out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    # Must be a complete HTML document
    assert content.startswith("<!doctype html>")
    assert "</html>" in content
    # No external script or link refs (everything embedded)
    assert "<script" not in content.lower()
    assert "src=\"http" not in content
    assert "href=\"http" not in content
    # The plain-language summary lands at the top
    assert "faulty" in content.lower()
    assert "qkv" in content


def test_save_run_report_works_without_diagnosis(tmp_path: Path) -> None:
    from defaultplusplus.viz import save_run_report
    out = tmp_path / "run.html"
    save_run_report(_toy_features(), out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "DEFault++ run report" in content


def test_save_diagnosis_report_clean_run_summary(tmp_path: Path) -> None:
    from defaultplusplus.diagnosis import Diagnosis
    from defaultplusplus.viz import save_diagnosis_report
    clean = Diagnosis(is_faulty=False, detection_prob=0.04,
                      group_importance={})
    out = tmp_path / "clean.html"
    save_diagnosis_report(clean, _toy_features(), out)
    assert "clean" in out.read_text(encoding="utf-8").lower()


# ─────────────────────────────────────────────────────────────────────────
# Diagnosis dict path (for callers that round-trip through JSON)
# ─────────────────────────────────────────────────────────────────────────
def test_plot_diagnosis_accepts_dict_payload(tmp_path: Path) -> None:
    """``Diagnosis.to_dict()`` is the JSON-safe shape; the plot must
    accept it in addition to the dataclass."""
    from defaultplusplus.viz import plot_diagnosis
    payload = _toy_diagnosis().to_dict()
    out = _save(plot_diagnosis(payload), tmp_path, "diag_dict")
    assert out.stat().st_size > 0
