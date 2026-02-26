#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

THIS_DIR = Path(__file__).resolve().parent
PKG_PARENT = THIS_DIR.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

from ndg_graph.mapping import display_name, parse_feature_core_map_md, resolve_to_core

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "results"
STAGE1_DIR = RESULTS_ROOT / "detection"
STAGE2_DIR = RESULTS_ROOT / "categorization"
FEATURE_MAP = THIS_DIR / "feature_core_map.md"
OUT_DIR = RESULTS_ROOT / "paper_figures_stage12_clean_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENC_COLOR = "#6C8397"
DEC_COLOR = "#B49380"
LINK_COLOR = "#C8C2B8"
GRID_COLOR = "#D9D4CD"
TEXT_COLOR = "#2F2F2F"
BG_COLOR = "#FBFAF8"
UP_COLOR = "#7C9B8A"
DOWN_COLOR = "#B08E7B"

MODEL_LABELS = {
    "ElasticNet_LR": "Elastic Net",
    "RBF_SVM": "RBF SVM",
    "XGBoost": "XGBoost",
    "EasyEnsemble": "EasyEnsemble",
}

CLASS_LABELS = {
    "embedding": "Embedding",
    "ffn": "FFN",
    "kernel": "Kernel",
    "kv_cache": "KV Cache",
    "layernorm": "LayerNorm",
    "masking": "Masking",
    "output": "Output",
    "positional": "Positional",
    "qkv": "QKV",
    "residual": "Residual",
    "score": "Score",
    "variant": "Variant",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.75,
        "figure.facecolor": "white",
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, basename: str) -> None:
    fig.savefig(OUT_DIR / f"{basename}.png")
    fig.savefig(OUT_DIR / f"{basename}.pdf")
    plt.close(fig)


def fold_metric(data: dict, model: str, metric_key: str) -> np.ndarray:
    return np.array([r["metrics"][metric_key] for r in data["experiments"][model]["per_fold"]], dtype=float)


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def pretty_class(name: str) -> str:
    return CLASS_LABELS.get(name, name.replace("_", " ").title())


def model_order(enc_data: dict, dec_data: dict, metric_key: str) -> List[str]:
    models = list(enc_data["experiments"].keys())
    scores = []
    for m in models:
        vals = np.concatenate([fold_metric(enc_data, m, metric_key), fold_metric(dec_data, m, metric_key)])
        scores.append((m, float(vals.mean())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scores]


def plot_dumbbell_panels(
    enc_data: dict,
    dec_data: dict,
    metrics: List[Tuple[str, str]],
    out_name: str,
) -> None:
    order = model_order(enc_data, dec_data, metrics[0][0])
    y = np.arange(len(order), dtype=float)

    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 4.2), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, x_label) in zip(axes, metrics):
        all_vals = []
        for i, model in enumerate(order):
            enc = fold_metric(enc_data, model, metric_key)
            dec = fold_metric(dec_data, model, metric_key)
            em, dm = float(enc.mean()), float(dec.mean())
            es, ds = float(enc.std(ddof=0)), float(dec.std(ddof=0))

            ax.plot([em, dm], [i, i], color=LINK_COLOR, lw=1.1, zorder=1)
            ax.errorbar(em, i, xerr=es, fmt="o", color=ENC_COLOR, ecolor=ENC_COLOR, markersize=5.8, capsize=2, elinewidth=0.9, zorder=3)
            ax.errorbar(dm, i, xerr=ds, fmt="s", color=DEC_COLOR, ecolor=DEC_COLOR, markersize=5.5, capsize=2, elinewidth=0.9, zorder=3)
            all_vals.extend(enc.tolist())
            all_vals.extend(dec.tolist())

        pad = 0.012 if max(all_vals) > 0.9 else 0.02
        xmin = max(0.0, min(all_vals) - pad)
        xmax = min(1.0, max(all_vals) + pad)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(x_label)
        ax.grid(axis="x", color=GRID_COLOR, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pretty_model(m) for m in order])
    axes[0].set_ylabel("Classifier")
    axes[0].invert_yaxis()

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ENC_COLOR, markeredgecolor="white", label="Encoder", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DEC_COLOR, markeredgecolor="white", label="Decoder", markersize=7),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID_COLOR,
    )
    fig.subplots_adjust(wspace=0.2, bottom=0.18)
    save_figure(fig, out_name)


def best_model_name(cat_data: dict, metric_key: str) -> str:
    return max(cat_data["experiments"].keys(), key=lambda m: cat_data["experiments"][m]["metrics"][metric_key])


def plot_confusion_clean(enc_cat: dict, dec_cat: dict) -> None:
    enc_model = best_model_name(enc_cat, "macro_f1_mean")
    dec_model = best_model_name(dec_cat, "macro_f1_mean")
    enc_m = enc_cat["experiments"][enc_model]["metrics"]
    dec_m = dec_cat["experiments"][dec_model]["metrics"]

    cmap = LinearSegmentedColormap.from_list("muted_seq", ["#F7F5F2", "#DDE5EA", "#ABC0CE", "#6C8397", "#3F586D"])
    fig = plt.figure(figsize=(10.8, 12.6))
    gs = GridSpec(2, 1, figure=fig, hspace=0.26)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    im = None
    for ax, metrics, arch, model in (
        (ax1, enc_m, "Encoder", pretty_model(enc_model)),
        (ax2, dec_m, "Decoder", pretty_model(dec_model)),
    ):
        cm = np.array(metrics["confusion_matrix_normalized"], dtype=float)
        labels = metrics["confusion_matrix_labels"]
        labels_full = [pretty_class(l) for l in labels]
        im = ax.imshow(cm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        n = len(labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_full, rotation=34, ha="right", fontsize=8.2)
        ax.set_yticklabels(labels_full, fontsize=8.4)
        ax.set_xlabel(f"Predicted ({arch}, {model})")
        ax.set_ylabel("True")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.022, pad=0.02)
    cbar.set_label("Row-normalized recall")
    save_figure(fig, "fig3_categorization_confusion_matrices_clean")


def aggregate_core_importance(top_features: Iterable[list], core_set: set) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for row in top_features:
        if not row or len(row) < 2:
            continue
        feat = str(row[0])
        val = float(row[1])
        core = resolve_to_core(feat, core_set)
        agg[core] = agg.get(core, 0.0) + val
    s = sum(agg.values())
    if s <= 0.0:
        return agg
    return {k: v / s for k, v in agg.items()}


def top_feature_deltas(det_map: Dict[str, float], cat_map: Dict[str, float], n: int = 10) -> List[Tuple[str, float, float, float]]:
    keys = sorted(set(det_map.keys()) | set(cat_map.keys()))
    rows = []
    for k in keys:
        d = det_map.get(k, 0.0)
        c = cat_map.get(k, 0.0)
        rows.append((k, d, c, c - d))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    rows = rows[:n]
    rows.sort(key=lambda r: r[3])
    return rows


def plot_one_delta_panel(
    ax: plt.Axes,
    rows: List[Tuple[str, float, float, float]],
    core_set: set,
    x_limit: float,
    side: str = "left",
    arch_label: str = "",
) -> None:
    y = np.arange(len(rows), dtype=float)
    for i, (feat, _, _, delta) in enumerate(rows):
        dp = delta * 100.0
        color = UP_COLOR if dp >= 0 else DOWN_COLOR
        ax.hlines(i, 0.0, dp, color=color, lw=2.2, zorder=2)
        ax.scatter([dp], [i], s=42, color=color, edgecolors="white", linewidths=0.55, zorder=3)
        off = 0.018 * x_limit
        tx = dp + off if dp >= 0 else dp - off
        ha = "left" if dp >= 0 else "right"
        ax.text(tx, i, f"{dp:+.1f}", fontsize=8, ha=ha, va="center", color="#4D4A46")

    ax.axvline(0.0, color="#8E8A84", lw=0.9, linestyle="--")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_yticks(y)
    ax.set_yticklabels([display_name(f, core_set) for f, _, _, _ in rows], fontsize=8.7)
    if side == "right":
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelright=True, labelleft=False)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.98, arch_label, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#4A4A4A")


def plot_feature_shift_delta(enc_det: dict, dec_det: dict, enc_cat: dict, dec_cat: dict, core_map: Dict[str, str]) -> None:
    core_set = set(core_map.keys())
    enc_det_map = aggregate_core_importance(enc_det["experiments"]["XGBoost"]["top25_features"], core_set)
    dec_det_map = aggregate_core_importance(dec_det["experiments"]["XGBoost"]["top25_features"], core_set)
    enc_cat_map = aggregate_core_importance(enc_cat["experiments"]["XGBoost"]["top25_features"], core_set)
    dec_cat_map = aggregate_core_importance(dec_cat["experiments"]["XGBoost"]["top25_features"], core_set)

    enc_rows = top_feature_deltas(enc_det_map, enc_cat_map, n=9)
    dec_rows = top_feature_deltas(dec_det_map, dec_cat_map, n=9)
    max_abs = max(
        max([abs(r[3] * 100.0) for r in enc_rows], default=1.0),
        max([abs(r[3] * 100.0) for r in dec_rows], default=1.0),
    )
    x_limit = max_abs * 1.35

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharex=True, sharey=False)
    plot_one_delta_panel(axes[0], enc_rows, core_set, x_limit=x_limit, side="left", arch_label="Encoder")
    plot_one_delta_panel(axes[1], dec_rows, core_set, x_limit=x_limit, side="right", arch_label="Decoder")
    axes[0].set_xlabel("Change in importance (Categorization - Detection), percentage points")
    axes[1].set_xlabel("Change in importance (Categorization - Detection), percentage points")

    legend = [
        Line2D([0], [0], color=UP_COLOR, lw=2.2, label="Higher in categorization"),
        Line2D([0], [0], color=DOWN_COLOR, lw=2.2, label="Higher in detection"),
        Line2D([0], [0], color="#8E8A84", lw=0.9, linestyle="--", label="No change"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor=GRID_COLOR,
    )
    fig.subplots_adjust(wspace=0.7, bottom=0.18)
    save_figure(fig, "fig4_feature_importance_shift_delta")


def main() -> None:
    enc_det = load_json(STAGE1_DIR / "enc_detection.json")
    dec_det = load_json(STAGE1_DIR / "dec_detection.json")
    enc_cat = load_json(STAGE2_DIR / "enc_categorization.json")
    dec_cat = load_json(STAGE2_DIR / "dec_categorization.json")
    core_map = parse_feature_core_map_md(FEATURE_MAP)

    plot_dumbbell_panels(
        enc_det,
        dec_det,
        [("auroc", "AUROC (mean ± std, 5 folds)"), ("auprc", "AUPRC (mean ± std, 5 folds)")],
        "fig1_detection_model_comparison_clean",
    )
    plot_dumbbell_panels(
        enc_cat,
        dec_cat,
        [("macro_f1", "Macro-F1 (mean ± std, 5 folds)"), ("balanced_accuracy", "Balanced Accuracy (mean ± std, 5 folds)")],
        "fig2_categorization_model_comparison_clean",
    )
    plot_confusion_clean(enc_cat, dec_cat)
    plot_feature_shift_delta(enc_det, dec_det, enc_cat, dec_cat, core_map)
    print(f"Saved plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
