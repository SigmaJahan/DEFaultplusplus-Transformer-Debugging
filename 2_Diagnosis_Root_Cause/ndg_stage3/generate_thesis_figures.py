#!/usr/bin/env python3
"""Generate all thesis figures for the DEFault++ chapter from Stage 1&2 JSON results.

Produces 10 publication-quality PDF figures:
  1. fig_classifier_comparison.pdf   -- Macro-F1 & Balanced Acc across 4 classifiers
  2. fig_perclass_f1.pdf             -- Cleveland dot plot per-class F1
  3. fig_confusion_matrices.pdf      -- Normalized confusion matrices (enc 11x11, dec 12x12)
  4. fig_arch_comparison.pdf         -- Dumbbell chart enc vs dec F1 gap
  5. fig_perclass_parallel.pdf       -- Parallel coordinates shared families
  6. fig_insight_arch_asymmetry.pdf  -- Scatter + delta-F1 bar chart
  7. fig_insight_confusion_channels.pdf -- Top off-diagonal confusion transitions
  8. fig_insight_xai_shift.pdf       -- SHAP difference + stability/fidelity comparison
  9. fig_insight_support_difficulty.pdf -- F1 vs log support scatter
 10. fig_insight_model_tradeoffs.pdf -- Runtime-accuracy and fold-variability planes

Inputs: 6 JSONs in results/
Output: PDFs in results/thesis_figures/
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# -- paths --
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "thesis_figures"
OUT.mkdir(exist_ok=True)

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

enc_det = load("stage_1_detection/enc_detection.json")
dec_det = load("stage_1_detection/dec_detection.json")
enc_cat = load("stage_2_categorization/enc_categorization.json")
dec_cat = load("stage_2_categorization/dec_categorization.json")
xai_enc = load("stage_2.1_categorization_xai/xai_enc_categorization.json")
xai_dec = load("stage_2.1_categorization_xai/xai_dec_categorization.json")

# -- muted academic palette --
C_ENC = "#5B7EA4"      # muted steel blue
C_DEC = "#C07060"      # muted terracotta
C_ENC_L = "#A3BCD5"    # lighter enc
C_DEC_L = "#DBA99E"    # lighter dec
C_POS = "#5B7EA4"      # positive delta
C_NEG = "#C07060"      # negative delta
C_GRID = "#D0D0D0"
C_ANNOT = "#606060"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.grid": False,
    "grid.color": C_GRID,
    "grid.linewidth": 0.4,
})

MODEL_NAMES = ["ElasticNet_LR", "RBF_SVM", "XGBoost", "EasyEnsemble"]
MODEL_LABELS = ["Elastic Net", "RBF SVM", "XGBoost", "EasyEnsemble"]

FAMILY_ORDER_ENC = enc_cat["label_names"]
FAMILY_ORDER_DEC = dec_cat["label_names"]
SHARED_FAMILIES = [f for f in FAMILY_ORDER_ENC if f in FAMILY_ORDER_DEC]

def pretty(name):
    remap = {"ffn": "FFN", "qkv": "QKV", "kv_cache": "KV Cache", "layernorm": "LayerNorm"}
    return remap.get(name, name.replace("_", " ").title())

def _legend_enc_dec(ax, loc="lower right", **kw):
    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ENC,
               markeredgecolor="#ffffff", markersize=7, label="Encoder"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_DEC,
               markeredgecolor="#ffffff", markersize=7, label="Decoder"),
    ]
    ax.legend(handles=h, loc=loc, framealpha=0.92, edgecolor=C_GRID, **kw)


# ============================================================
# 1. CLASSIFIER COMPARISON
# ============================================================
def fig_classifier_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    metrics_keys = [("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced Accuracy")]

    for idx, (ax, (mkey, mlabel)) in enumerate(zip(axes, metrics_keys)):
        x = np.arange(len(MODEL_NAMES))
        w = 0.34
        enc_vals, dec_vals, enc_stds, dec_stds = [], [], [], []
        for mn in MODEL_NAMES:
            ef = [f["metrics"][mkey] for f in enc_cat["experiments"][mn]["per_fold"]]
            df = [f["metrics"][mkey] for f in dec_cat["experiments"][mn]["per_fold"]]
            enc_vals.append(np.mean(ef)); dec_vals.append(np.mean(df))
            enc_stds.append(np.std(ef));  dec_stds.append(np.std(df))

        ax.bar(x - w/2, enc_vals, w, yerr=enc_stds, color=C_ENC,
               label="Encoder", capsize=3, edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.8, capthick=0.8))
        ax.bar(x + w/2, dec_vals, w, yerr=dec_stds, color=C_DEC,
               label="Decoder", capsize=3, edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.8, capthick=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, rotation=20, ha="right")
        ax.set_ylabel(mlabel if idx == 0 else "")
        ax.set_ylim(0, 1.08)
        ax.axhline(y=0.5, color=C_GRID, linewidth=0.6, linestyle="--")
        ax.set_title(f"({chr(97+idx)}) {mlabel}", fontsize=10)

    axes[0].legend(loc="upper left", framealpha=0.92, edgecolor=C_GRID)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_classifier_comparison.pdf")
    plt.close(fig)
    print("  [1/10] fig_classifier_comparison.pdf")


# ============================================================
# 2. PER-CLASS F1 -- Cleveland dot plot
# ============================================================
def fig_perclass_f1():
    enc_pc = enc_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    dec_pc = dec_cat["experiments"]["XGBoost"]["metrics"]["per_class"]

    all_families = sorted(set(list(enc_pc.keys()) + list(dec_pc.keys())),
                          key=lambda f: enc_pc.get(f, {}).get("f1", 0), reverse=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    y_pos = np.arange(len(all_families))

    for i, fam in enumerate(all_families):
        enc_f1 = enc_pc.get(fam, {}).get("f1", None)
        dec_f1 = dec_pc.get(fam, {}).get("f1", None)

        if enc_f1 is not None and dec_f1 is not None:
            ax.plot([enc_f1, dec_f1], [i, i], color=C_GRID, linewidth=1.2, zorder=1)
        if enc_f1 is not None:
            ax.scatter(enc_f1, i, marker="o", s=45, color=C_ENC, zorder=2,
                       edgecolors="white", linewidths=0.5)
        if dec_f1 is not None:
            ax.scatter(dec_f1, i, marker="s", s=45, color=C_DEC, zorder=2,
                       edgecolors="white", linewidths=0.5)

    # build right-side support column as a second y-axis label
    enc_sups = [enc_pc.get(f, {}).get("support", "") for f in all_families]
    dec_sups = [dec_pc.get(f, {}).get("support", "") for f in all_families]
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    sup_labels = []
    for es, ds, fam in zip(enc_sups, dec_sups, all_families):
        parts = []
        if es: parts.append(str(es))
        else: parts.append("--")
        if ds: parts.append(str(ds))
        else: parts.append("--")
        if fam not in enc_pc: parts.append("*")
        sup_labels.append(" / ".join(parts))
    ax2.set_yticklabels(sup_labels, fontsize=6.5, color=C_ANNOT)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylabel("Support (Enc / Dec)", fontsize=7, color=C_ANNOT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([pretty(f) for f in all_families])
    ax.set_xlabel("Per-Class F1")
    ax.set_xlim(0.5, 1.02)
    ax.axvline(x=0.9, color=C_GRID, linewidth=0.5, linestyle=":")
    _legend_enc_dec(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_perclass_f1.pdf")
    plt.close(fig)
    print("  [2/10] fig_perclass_f1.pdf")


# ============================================================
# 3. CONFUSION MATRICES
# ============================================================
def fig_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [11, 12], "wspace": 0.45})

    for ax, (cat_data, title) in zip(axes, [
        (enc_cat, "Encoder (11 families)"),
        (dec_cat, "Decoder (12 families)")
    ]):
        xgb = cat_data["experiments"]["XGBoost"]["metrics"]
        cm = np.array(xgb["confusion_matrix_normalized"])
        labels = xgb["confusion_matrix_labels"]
        n = len(labels)
        support = [cat_data["class_dist"][l] for l in labels]

        # muted sequential colormap
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("muted_blue",
            ["#F7F7F7", "#D4E3F0", "#A3BCD5", "#5B7EA4", "#2E4A6E"])

        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels([pretty(l) for l in labels], rotation=50, ha="right", fontsize=6.5)
        ax.set_yticks(range(n))
        ylabels = [f"{pretty(l)}  ({support[i]:,})" for i, l in enumerate(labels)]
        ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True  (support)", fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)

        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if val >= 0.01:
                    color = "white" if val > 0.55 else "#333333"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=5 if n > 11 else 5.5, color=color)

    cbar = fig.colorbar(axes[1].images[0], ax=axes, shrink=0.75, pad=0.02,
                        aspect=30)
    cbar.set_label("Row-normalized recall", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "fig_confusion_matrices.pdf")
    plt.close(fig)
    print("  [3/10] fig_confusion_matrices.pdf")


# ============================================================
# 4. ARCHITECTURE COMPARISON -- Dumbbell chart
# ============================================================
def fig_arch_comparison():
    enc_pc = enc_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    dec_pc = dec_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    families = sorted(SHARED_FAMILIES, key=lambda f: enc_pc[f]["f1"], reverse=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    y = np.arange(len(families))

    for i, fam in enumerate(families):
        ef1, df1 = enc_pc[fam]["f1"], dec_pc[fam]["f1"]
        delta = (ef1 - df1) * 100
        ax.plot([ef1, df1], [i, i], color=C_GRID, linewidth=1.2, zorder=1)
        ax.scatter(ef1, i, marker="o", s=50, color=C_ENC, zorder=2, edgecolors="white", linewidths=0.5)
        ax.scatter(df1, i, marker="s", s=50, color=C_DEC, zorder=2, edgecolors="white", linewidths=0.5)

    # right-side axis for delta annotations
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    delta_labels = []
    for fam in families:
        ef1, df1 = enc_pc[fam]["f1"], dec_pc[fam]["f1"]
        d = (ef1 - df1) * 100
        sign = "+" if d > 0 else ""
        delta_labels.append(f"{sign}{d:.1f}%")
    ax2.set_yticklabels(delta_labels, fontsize=7, color=C_ANNOT)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("$\\Delta$F1 (Enc$-$Dec)", fontsize=8, color=C_ANNOT)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty(f) for f in families])
    ax.set_xlabel("Per-Class F1")
    ax.set_xlim(0.60, 1.02)
    _legend_enc_dec(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_arch_comparison.pdf")
    plt.close(fig)
    print("  [4/10] fig_arch_comparison.pdf")


# ============================================================
# 5. PARALLEL COORDINATES -- shared families
# ============================================================
def fig_perclass_parallel():
    enc_pc = enc_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    dec_pc = dec_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    families = sorted(SHARED_FAMILIES, key=lambda f: enc_pc[f]["f1"], reverse=True)
    ef1 = [enc_pc[f]["f1"] for f in families]
    df1 = [dec_pc[f]["f1"] for f in families]

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    x = np.arange(len(families))

    ax.fill_between(x, ef1, df1, alpha=0.10, color="#888888")
    ax.plot(x, ef1, "-o", color=C_ENC, markersize=5.5, label="Encoder", linewidth=1.5, zorder=3)
    ax.plot(x, df1, "-s", color=C_DEC, markersize=5.5, label="Decoder", linewidth=1.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty(f) for f in families], rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Per-Class F1")
    ax.set_ylim(0.58, 1.02)

    ax.legend(loc="lower left", framealpha=0.92, edgecolor=C_GRID)
    fig.tight_layout()
    fig.savefig(OUT / "fig_perclass_parallel.pdf")
    plt.close(fig)
    print("  [5/10] fig_perclass_parallel.pdf")


# ============================================================
# 6. INSIGHT: ARCHITECTURE ASYMMETRY
# ============================================================
def fig_insight_arch_asymmetry():
    enc_pc = enc_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    dec_pc = dec_cat["experiments"]["XGBoost"]["metrics"]["per_class"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) Scatter
    ax = axes[0]
    texts = []
    for fam in SHARED_FAMILIES:
        ef1, df1 = enc_pc[fam]["f1"], dec_pc[fam]["f1"]
        ax.scatter(ef1, df1, s=48, color=C_ENC, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)
        texts.append((ef1, df1, pretty(fam)))

    ax.plot([0.6, 1.02], [0.6, 1.02], "--", color=C_GRID, linewidth=0.8)
    ax.set_xlabel("Encoder F1")
    ax.set_ylabel("Decoder F1")
    ax.set_xlim(0.78, 1.02)
    ax.set_ylim(0.66, 1.02)
    ax.set_title("(a) Encoder vs. Decoder per-class F1", fontsize=10)
    ax.set_aspect("equal")

    # smart label placement using adjustText if available, else manual offsets
    try:
        from adjustText import adjust_text
        txt_objs = [ax.text(x, y, f"  {lbl}", fontsize=6.5, color=C_ANNOT, va="center")
                     for x, y, lbl in texts]
        adjust_text(txt_objs, ax=ax, arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.4),
                    force_text=(0.3, 0.3), force_points=(0.2, 0.2), expand=(1.2, 1.4))
    except ImportError:
        # manual offsets for known cluster collisions
        offsets = {"QKV": (0.005, -0.015), "LayerNorm": (-0.02, 0.015),
                   "Masking": (0.005, 0.015), "Positional": (-0.03, -0.01),
                   "FFN": (0.005, 0.012), "Variant": (-0.005, -0.015)}
        for ex, ey, lbl in texts:
            dx, dy = offsets.get(lbl.replace(" ", ""), (0.005, 0))
            ax.annotate(lbl, (ex, ey), xytext=(ex+dx, ey+dy), fontsize=6.5,
                        color=C_ANNOT, arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.3) if (dx**2+dy**2)>0.0002 else None)

    # (b) Bar chart: delta F1
    ax = axes[1]
    deltas = {f: (enc_pc[f]["f1"] - dec_pc[f]["f1"]) * 100 for f in SHARED_FAMILIES}
    families_sorted = sorted(SHARED_FAMILIES, key=lambda f: deltas[f], reverse=True)
    y = np.arange(len(families_sorted))
    vals = [deltas[f] for f in families_sorted]
    colors = [C_POS if v >= 0 else C_NEG for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.62, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(f) for f in families_sorted], fontsize=7.5)
    ax.set_xlabel("$\\Delta$F1 (Enc $-$ Dec, %)")
    ax.axvline(x=0, color="#888888", linewidth=0.5)
    ax.set_title("(b) Signed $\\Delta$F1", fontsize=10)

    for i, v in enumerate(vals):
        sign = "+" if v > 0 else ""
        offset = 0.5 if v >= 0 else -0.5
        ax.text(v + offset, i, f"{sign}{v:.1f}", va="center", fontsize=6.5, color=C_ANNOT,
                ha="left" if v >= 0 else "right")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_insight_arch_asymmetry.pdf")
    plt.close(fig)
    print("  [6/10] fig_insight_arch_asymmetry.pdf")


# ============================================================
# 7. INSIGHT: CONFUSION CHANNELS
# ============================================================
def fig_insight_confusion_channels():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"wspace": 0.55})

    for ax, (cat_data, title, color) in zip(axes, [
        (enc_cat, "Encoder", C_ENC),
        (dec_cat, "Decoder", C_DEC)
    ]):
        xgb = cat_data["experiments"]["XGBoost"]["metrics"]
        cm = np.array(xgb["confusion_matrix_normalized"])
        labels = xgb["confusion_matrix_labels"]
        n = len(labels)
        support = cat_data["class_dist"]

        transitions = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0.01:
                    transitions.append((labels[i], labels[j], cm[i, j], support[labels[i]]))
        transitions.sort(key=lambda t: t[2], reverse=True)
        top = transitions[:8]

        y = np.arange(len(top))
        vals = [t[2] * 100 for t in top]
        bar_labels = [f"{pretty(t[0])}  {pretty(t[1])}" for t in top]

        ax.barh(y, vals, color=color, edgecolor="white", linewidth=0.4, height=0.62, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(bar_labels, fontsize=7)
        ax.set_xlabel("Off-diagonal recall (%)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()


        for i, (v, t) in enumerate(zip(vals, top)):
            ax.text(v + 0.3, i, f"{v:.1f}%  (n={t[3]:,})", va="center", fontsize=6.5, color=C_ANNOT)

    fig.savefig(OUT / "fig_insight_confusion_channels.pdf")
    plt.close(fig)
    print("  [7/10] fig_insight_confusion_channels.pdf")


# ============================================================
# 8. INSIGHT: XAI SHIFT
# ============================================================
def fig_insight_xai_shift():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.35})

    # (a) SHAP difference
    ax = axes[0]
    enc_shap = {f[0]: f[1] for f in xai_enc["shap"]["top30_core_features"]}
    dec_shap = {f[0]: f[1] for f in xai_dec["shap"]["top30_core_features"]}
    enc_total = sum(enc_shap.values())
    dec_total = sum(dec_shap.values())

    all_feats = set(list(enc_shap.keys())[:15]) | set(list(dec_shap.keys())[:15])
    diffs = {}
    for f in all_feats:
        enc_share = enc_shap.get(f, 0) / enc_total
        dec_share = dec_shap.get(f, 0) / dec_total
        diffs[f] = (enc_share - dec_share) * 100

    feats_sorted = sorted(diffs.keys(), key=lambda f: diffs[f], reverse=True)[:12]
    y = np.arange(len(feats_sorted))
    vals = [diffs[f] for f in feats_sorted]
    colors = [C_POS if v > 0 else C_NEG for v in vals]

    sys.path.insert(0, str(ROOT / "3_A_Diagnosis_Root_Cause"))
    try:
        from ndg_stage3.mapping import display_name
        feat_labels = [display_name(f) for f in feats_sorted]
    except ImportError:
        feat_labels = [f.replace("abs_", "").replace("_", " ").title() for f in feats_sorted]

    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4, height=0.62, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels, fontsize=7)
    ax.set_xlabel("$\\Delta$ SHAP share (Enc $-$ Dec, %)", fontsize=8)
    ax.axvline(x=0, color="#888888", linewidth=0.5)
    ax.set_title("(a) SHAP architecture shift", fontsize=10)
    ax.invert_yaxis()


    # (b) Stability & fidelity
    ax = axes[1]
    metric_names = ["SHAP stability\n(Top-20 Jaccard)", "Rule fidelity\nto XGBoost", "Counterfactual\nsuccess rate"]
    enc_vals = [
        xai_enc["shap"]["stability_jaccard_top20"],
        xai_enc["rules"]["mean_fidelity_to_xgb"],
        xai_enc["counterfactuals"]["total_generated"] / max(xai_enc["counterfactuals"]["total_attempted"], 1)
    ]
    dec_vals = [
        xai_dec["shap"]["stability_jaccard_top20"],
        xai_dec["rules"]["mean_fidelity_to_xgb"],
        xai_dec["counterfactuals"]["total_generated"] / max(xai_dec["counterfactuals"]["total_attempted"], 1)
    ]

    x = np.arange(len(metric_names))
    w = 0.30
    ax.bar(x - w/2, enc_vals, w, color=C_ENC, label="Encoder", edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, dec_vals, w, color=C_DEC, label="Decoder", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, framealpha=0.92, edgecolor=C_GRID, loc="upper right")
    ax.set_title("(b) Explanation quality", fontsize=10)


    for i, (ev, dv) in enumerate(zip(enc_vals, dec_vals)):
        ax.text(i - w/2, ev + 0.025, f"{ev:.2f}", ha="center", fontsize=6.5, color=C_ENC, fontweight="bold")
        ax.text(i + w/2, dv + 0.025, f"{dv:.2f}", ha="center", fontsize=6.5, color=C_DEC, fontweight="bold")

    fig.savefig(OUT / "fig_insight_xai_shift.pdf")
    plt.close(fig)
    print("  [8/10] fig_insight_xai_shift.pdf")


# ============================================================
# 9. INSIGHT: SUPPORT-DIFFICULTY
# ============================================================
def fig_insight_support_difficulty():
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    all_texts = []
    for cat_data, color, marker, label in [
        (enc_cat, C_ENC, "o", "Encoder"),
        (dec_cat, C_DEC, "s", "Decoder")
    ]:
        pc = cat_data["experiments"]["XGBoost"]["metrics"]["per_class"]
        families = list(pc.keys())
        f1s = [pc[f]["f1"] for f in families]
        supports = [pc[f]["support"] for f in families]
        log_supports = [np.log10(s) for s in supports]

        ax.scatter(log_supports, f1s, c=color, marker=marker, s=55, alpha=0.8,
                   edgecolors="white", linewidths=0.5, label=label, zorder=3)

        for f, xv, yv in zip(families, log_supports, f1s):
            all_texts.append((xv, yv, pretty(f), color))

        if len(log_supports) > 2:
            z = np.polyfit(log_supports, f1s, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(log_supports) - 0.05, max(log_supports) + 0.05, 50)
            ax.plot(xs, p(xs), "--", color=color, linewidth=0.8, alpha=0.4)
            corr = np.corrcoef(log_supports, f1s)[0, 1]
            # place r value in upper-left corner (away from legend)
            ax.text(0.03, 0.10 + (0.06 if label == "Encoder" else 0),
                    f"$r$ = {corr:.2f} ({label})", fontsize=7.5, color=color,
                    ha="left", transform=ax.transAxes, style="italic")

    # Use adjustText if available
    try:
        from adjustText import adjust_text
        txt_objs = [ax.text(x, y, f"  {lbl}", fontsize=6, color=c, va="center")
                     for x, y, lbl, c in all_texts]
        adjust_text(txt_objs, ax=ax, arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.3),
                    force_text=(0.4, 0.4), force_points=(0.3, 0.3), expand=(1.3, 1.5))
    except ImportError:
        for x, y, lbl, c in all_texts:
            ax.annotate(f"  {lbl}", (x, y), fontsize=5.5, va="center", color=c, alpha=0.8)

    ax.set_xlabel("log$_{10}$(support)")
    ax.set_ylabel("Per-Class F1")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor=C_GRID)
    ax.set_title("Support-difficulty relationship", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "fig_insight_support_difficulty.pdf")
    plt.close(fig)
    print("  [9/10] fig_insight_support_difficulty.pdf")


# ============================================================
# 10. INSIGHT: MODEL TRADEOFFS
# ============================================================
def fig_insight_model_tradeoffs():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    for panel_idx, (ax, x_fn, x_label, title) in enumerate(zip(axes,
        [lambda exp: exp["time_s"],
         lambda exp: np.std([f["metrics"]["macro_f1"] for f in exp["per_fold"]])],
        ["Runtime (seconds, log scale)", "Fold std(Macro-F1)"],
        ["(a) Runtime-accuracy plane", "(b) Fold-variability plane"]
    )):
        all_pts = []
        for cat_data, color, marker, arch_label in [
            (enc_cat, C_ENC, "o", "Enc"),
            (dec_cat, C_DEC, "s", "Dec")
        ]:
            for mn, ml in zip(MODEL_NAMES, MODEL_LABELS):
                exp = cat_data["experiments"][mn]
                fold_f1s = [f["metrics"]["macro_f1"] for f in exp["per_fold"]]
                mf1 = np.mean(fold_f1s)
                xv = x_fn(exp)
                ax.scatter(xv, mf1, c=color, marker=marker, s=55,
                           edgecolors="white", linewidths=0.5, zorder=3)
                all_pts.append((xv, mf1, f"{ml} ({arch_label})", color))

        if panel_idx == 0:
            ax.set_xscale("log")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel("Macro-F1", fontsize=8)
        ax.set_title(title, fontsize=10)
    

        try:
            from adjustText import adjust_text
            txt_objs = [ax.text(x, y, f"  {lbl}", fontsize=6, color=c, va="center")
                         for x, y, lbl, c in all_pts]
            adjust_text(txt_objs, ax=ax, arrowprops=dict(arrowstyle="-", color=C_GRID, lw=0.3),
                        force_text=(0.3, 0.3), force_points=(0.2, 0.2), expand=(1.2, 1.4))
        except ImportError:
            for xv, yv, lbl, c in all_pts:
                ax.annotate(f"  {lbl}", (xv, yv), fontsize=5, va="center", color=c)

    _legend_enc_dec(axes[0], loc="lower right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_insight_model_tradeoffs.pdf")
    plt.close(fig)
    print("  [10/10] fig_insight_model_tradeoffs.pdf")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Generating thesis figures into: {OUT}")
    fig_classifier_comparison()
    fig_perclass_f1()
    fig_confusion_matrices()
    fig_arch_comparison()
    fig_perclass_parallel()
    fig_insight_arch_asymmetry()
    fig_insight_confusion_channels()
    fig_insight_xai_shift()
    fig_insight_support_difficulty()
    fig_insight_model_tradeoffs()
    print(f"\nDone. {len(list(OUT.glob('*.pdf')))} PDFs generated in {OUT}")
