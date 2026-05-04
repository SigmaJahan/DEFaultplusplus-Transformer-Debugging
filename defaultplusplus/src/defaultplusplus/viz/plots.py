"""Figure-returning plot functions.

Each function returns a ``matplotlib.figure.Figure`` so the caller can
drop it into ``fig.savefig(...)`` or embed it in a custom report.

All functions raise :class:`VizDependencyError` if matplotlib is not
installed; otherwise no global state is touched (``plt.figure`` only,
no ``plt.show``).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ._deps import require_matplotlib
from ._features import (
    find_keys_matching,
    parse_layer_key,
    per_layer_families,
    split_phase,
)


def _get_diag_field(diagnosis, name):
    """Read attribute or dict key — Diagnosis is a dataclass but a plain
    dict round-trips through ``Diagnosis.to_dict`` and back."""
    if hasattr(diagnosis, name):
        return getattr(diagnosis, name)
    return diagnosis.get(name)


# ─────────────────────────────────────────────────────────────────────────
# Plot 1 — three-stage verdict
# ─────────────────────────────────────────────────────────────────────────
def plot_diagnosis(diagnosis):
    """Three-stage verdict: detection / category / root-cause probabilities.

    ``diagnosis`` may be a ``Diagnosis`` dataclass or its ``.to_dict()``
    payload (so consumers serializing through JSON still work).
    """
    plt = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    is_faulty = bool(_get_diag_field(diagnosis, "is_faulty"))
    det_p = float(_get_diag_field(diagnosis, "detection_prob") or 0.0)
    cat = _get_diag_field(diagnosis, "category")
    cat_p = float(_get_diag_field(diagnosis, "category_prob") or 0.0)
    rc = _get_diag_field(diagnosis, "root_cause")
    rc_p = float(_get_diag_field(diagnosis, "root_cause_prob") or 0.0)

    # Stage 1: detection
    ax = axes[0]
    color = "#c0392b" if is_faulty else "#27ae60"
    ax.bar(["clean", "faulty"], [1.0 - det_p, det_p],
           color=["#bdc3c7", color])
    ax.set_ylim(0, 1.0)
    ax.set_title("Stage 1: detection")
    ax.set_ylabel("probability")
    ax.text(1.0 if is_faulty else 0.0,
            (det_p if is_faulty else 1.0 - det_p) + 0.02,
            f"{det_p if is_faulty else 1.0 - det_p:.2f}",
            ha="center")

    # Stage 2: category
    ax = axes[1]
    if cat is not None:
        ax.bar([cat], [cat_p], color="#2980b9")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Stage 2: category = {cat}")
    else:
        ax.text(0.5, 0.5, "n/a (clean run)", ha="center", va="center",
                transform=ax.transAxes, color="#7f8c8d")
        ax.set_title("Stage 2: category")
        ax.set_xticks([])
        ax.set_yticks([])

    # Stage 3: root cause
    ax = axes[2]
    if rc is not None:
        ax.bar([rc], [rc_p], color="#8e44ad")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Stage 3: root cause = {rc}")
    else:
        ax.text(0.5, 0.5, "n/a (single root cause)" if cat else "n/a",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_title("Stage 3: root cause")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 2 — group importance bar chart
# ─────────────────────────────────────────────────────────────────────────
def plot_group_importance(diagnosis):
    """Horizontal bar chart of per-group prototype margins.

    Positive margins support the predicted root cause; negative margins
    oppose it. An empty importance dict (e.g. clean-run diagnosis)
    yields an "n/a" placeholder figure.
    """
    plt = require_matplotlib()
    importance = _get_diag_field(diagnosis, "group_importance") or {}
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * max(1, len(importance)))))

    if not importance:
        ax.text(0.5, 0.5, "no group importance available",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#34495e", linewidth=0.8)
    ax.set_xlabel("prototype margin (predicted vs nearest alternative)")
    ax.set_title("Per-group support for the diagnosis")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 3 — per-layer heatmap for one metric family
# ─────────────────────────────────────────────────────────────────────────
def plot_per_layer_heatmap(features: Mapping[str, float], metric: str):
    """Heatmap of one metric family across layers and phase windows.

    ``metric`` is matched as a substring against the per-layer family
    name; the most-populated matching family wins. The y-axis is
    layer index, x-axis is phase window (early/mid/final/slope).
    """
    plt = require_matplotlib()
    families = per_layer_families(features.keys())
    if not families:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.5, 0.5, "no per-layer features in this run",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    needle = metric.lower()
    candidates = [(name, layers) for name, layers in families.items()
                  if needle in name.lower()]
    if not candidates:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.5, 0.5, f"no per-layer family matches {metric!r}",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    # Pick the family with the most layers populated.
    family_name, layers_map = max(candidates, key=lambda c: len(c[1]))
    layer_indices = sorted(layers_map.keys())

    # Build (n_layers, n_phases) matrix, falling back to a single column
    # of values when there's no phase suffix to split on.
    phase_labels: list[str] = []
    grid: list[list[float]] = []
    for li in layer_indices:
        key = layers_map[li]
        v = features.get(key, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float("nan")
        # If this family encodes phase windows, gather the sibling keys.
        stem, phase = split_phase(key)
        if phase is not None:
            row = []
            row_phases = []
            for ph in ("early_mean", "mid_mean", "final_mean",
                       "final_value", "slope"):
                # Try both ``__epoch_mean__phase_*`` and ``__epoch_std__phase_*``
                tried = [
                    f"{stem}__epoch_mean__phase_{ph}",
                    f"{stem}__epoch_std__phase_{ph}",
                ]
                cell = float("nan")
                for sib in tried:
                    if sib in features:
                        try:
                            cell = float(features[sib])
                            break
                        except (TypeError, ValueError):
                            pass
                row.append(cell)
                row_phases.append(ph)
            grid.append(row)
            if not phase_labels:
                phase_labels = row_phases
        else:
            grid.append([v])
            if not phase_labels:
                phase_labels = ["value"]

    arr = np.array(grid, dtype=float)
    fig, ax = plt.subplots(figsize=(1.5 + 0.7 * arr.shape[1],
                                    1.5 + 0.3 * arr.shape[0]))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([f"L{li}" for li in layer_indices])
    ax.set_xticks(range(len(phase_labels)))
    ax.set_xticklabels(phase_labels, rotation=30, ha="right")
    ax.set_title(family_name, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 4 — training trace across phase windows
# ─────────────────────────────────────────────────────────────────────────
def plot_training_trace(features: Mapping[str, float],
                        keys: Iterable[str]):
    """Time series across early / mid / late / final windows.

    For each base key in ``keys`` we look up the four phase values and
    draw a single line. Missing values become NaN gaps.
    """
    plt = require_matplotlib()
    keys = list(keys)
    fig, ax = plt.subplots(figsize=(8, 4))
    phases = ["early_mean", "mid_mean", "final_mean"]
    x = list(range(len(phases)))

    plotted = 0
    for base in keys:
        # If the user passed a fully-qualified key, strip its phase.
        stem, _ = split_phase(base)
        ys = []
        for ph in phases:
            tried = [
                f"{stem}__epoch_mean__phase_{ph}",
                f"{stem}__phase_{ph}",
                f"{base}__phase_{ph}",
            ]
            v = float("nan")
            for sib in tried:
                if sib in features:
                    try:
                        v = float(features[sib])
                        break
                    except (TypeError, ValueError):
                        pass
            ys.append(v)
        if any(not np.isnan(y) for y in ys):
            ax.plot(x, ys, marker="o", label=stem)
            plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "no training-trace keys matched",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    ax.set_xticks(x)
    ax.set_xticklabels(["early", "mid", "final"])
    ax.set_xlabel("training phase")
    ax.set_ylabel("metric value")
    ax.set_title("Training-trace summary")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 5 — attention shape for one layer
# ─────────────────────────────────────────────────────────────────────────
def plot_attention_pattern(features: Mapping[str, float], layer: int):
    """One layer's attention shape (entropy, sparsity, future-mass, head similarity).

    Falls back to whatever attention-family features are present.
    """
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 3.2))

    fields = {
        "entropy": ["attn_entropy", "attention_entropy"],
        "sparsity": ["attn_sparsity", "attention_sparsity"],
        "future-mass": ["mass_future", "attention_mass_future"],
        "head similarity (mean)": ["head_similarity_mean"],
    }

    labels: list[str] = []
    values: list[float] = []
    for label, needles in fields.items():
        # Match: a key for our layer that contains any of the needles
        matched = None
        for k in features:
            kl = k.lower()
            parsed = parse_layer_key(k)
            if parsed is None or parsed[1] != layer:
                continue
            if any(n in kl for n in needles):
                matched = k
                break
        if matched is None:
            continue
        try:
            v = float(features[matched])
        except (TypeError, ValueError):
            v = float("nan")
        if np.isnan(v):
            continue
        labels.append(label)
        values.append(v)

    if not labels:
        ax.text(0.5, 0.5, f"no attention-family features for layer {layer}",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    ax.bar(labels, values, color="#16a085")
    ax.set_title(f"Attention pattern, layer {layer}")
    ax.set_ylabel("value")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 6 — Q-K / Q-V / K-V cosines per layer
# ─────────────────────────────────────────────────────────────────────────
def plot_qkv_alignment(features: Mapping[str, float]):
    """Lineplot of Q-K, Q-V, K-V cosine similarities across layers.

    Reads any per-layer ``qk_cos`` / ``qv_cos`` / ``kv_cos`` family.
    """
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))

    series = {"Q-K": "qk_cos", "Q-V": "qv_cos", "K-V": "kv_cos"}
    plotted = 0
    for label, needle in series.items():
        per_layer: dict[int, float] = {}
        for k, v in features.items():
            if needle not in k.lower():
                continue
            parsed = parse_layer_key(k)
            if parsed is None:
                continue
            try:
                per_layer[parsed[1]] = float(v)
            except (TypeError, ValueError):
                pass
        if not per_layer:
            continue
        xs = sorted(per_layer)
        ys = [per_layer[i] for i in xs]
        ax.plot(xs, ys, marker="o", label=label)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "no QKV cosine features in this run",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Q-K / Q-V / K-V alignment per layer")
    ax.axhline(0, color="#34495e", linewidth=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Plot 7 — top-N feature anomalies vs a baseline
# ─────────────────────────────────────────────────────────────────────────
def plot_feature_anomaly(features: Mapping[str, float],
                         baseline: Mapping[str, float],
                         top_n: int = 20):
    """Top-N features ranked by absolute (run - baseline) difference.

    Both inputs are feature dicts. Keys absent from either side are
    skipped. ``baseline`` can be a literal "clean run" feature dict or
    a per-key average computed offline.
    """
    plt = require_matplotlib()
    diffs: list[tuple[str, float]] = []
    for k, v in features.items():
        b = baseline.get(k)
        if b is None:
            continue
        try:
            d = float(v) - float(b)
        except (TypeError, ValueError):
            continue
        if np.isnan(d):
            continue
        diffs.append((k, d))

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * top_n)))
    if not diffs:
        ax.text(0.5, 0.5, "no overlap between run and baseline keys",
                ha="center", va="center", transform=ax.transAxes,
                color="#7f8c8d")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    diffs.sort(key=lambda kv: -abs(kv[1]))
    diffs = diffs[:top_n][::-1]   # reverse so largest is at the top
    names = [k for k, _ in diffs]
    vals = [v for _, v in diffs]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#34495e", linewidth=0.8)
    ax.set_xlabel("run − baseline")
    ax.set_title(f"Top {len(diffs)} feature anomalies vs baseline")
    fig.tight_layout()
    return fig
