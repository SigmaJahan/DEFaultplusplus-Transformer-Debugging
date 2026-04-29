"""NeurIPS-quality plotting for hierarchical fault diagnosis.

All plots use a consistent scientific aesthetic:
  - Serif math font (Computer Modern via mathtext)
  - Muted, colorblind-safe palettes
  - Clean axis labels, no chartjunk
  - 300 DPI, tight layout, PDF-ready

Plots generated:
  1. t-SNE / UMAP embedding visualisation (category-colored)
  2. Confusion matrices (Stage 2 category, Stage 3 root-cause)
  3. Per-category root-cause cluster detail (with/without intra-family contrastive loss)
  4. Training curves (loss and val metric across epochs)
  5. Ablation delta barplot (per-category F1 improvement)
  6. Detection ROC curves
  7. Per-category F1 comparison grouped barplot
"""
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".tmp" / "matplotlib"),
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc

try:
    import seaborn as sns
except ImportError:
    class _SeabornCompat:
        _CMAP_ALIASES = {
            "Blues": "Blues",
            "Greys": "Greys",
            "YlOrRd": "YlOrRd",
            "mako": "viridis",
            "vlag": "coolwarm",
        }

        @classmethod
        def color_palette(cls, name, as_cmap=False):
            cmap = plt.get_cmap(cls._CMAP_ALIASES.get(name, name))
            if as_cmap:
                return cmap
            return [cmap(x) for x in np.linspace(0.15, 0.85, 8)]

        @staticmethod
        def heatmap(data, annot=False, fmt=".2f", cmap=None, vmin=None, vmax=None,
                    xticklabels=None, yticklabels=None, ax=None, linewidths=0.0,
                    linecolor="white", cbar=True, cbar_kws=None, annot_kws=None, **_):
            ax = ax or plt.gca()
            matrix = np.asarray(data)
            image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

            if xticklabels is not None:
                ax.set_xticks(np.arange(matrix.shape[1]))
                ax.set_xticklabels(xticklabels)
            if yticklabels is not None:
                ax.set_yticks(np.arange(matrix.shape[0]))
                ax.set_yticklabels(yticklabels)

            if linewidths:
                ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
                ax.grid(which="minor", color=linecolor, linewidth=linewidths)
                ax.tick_params(which="minor", bottom=False, left=False)

            if annot:
                text_kw = {"ha": "center", "va": "center", "fontsize": 6}
                if annot_kws:
                    text_kw.update(annot_kws)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        value = matrix[i, j]
                        if fmt == "d":
                            label = f"{int(round(value))}"
                        else:
                            label = format(value, fmt)
                        ax.text(j, i, label, **text_kw)

            if cbar:
                cbar_kws = cbar_kws or {}
                cbar_obj = ax.figure.colorbar(image, ax=ax, shrink=cbar_kws.get("shrink", 1.0))
                if "label" in cbar_kws:
                    cbar_obj.set_label(cbar_kws["label"])

            return image

    sns = _SeabornCompat()

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Global NeurIPS aesthetic ─────────────────────────────────────────────────

# NeurIPS column width: ~5.5in, full width: ~11in
COLUMN_WIDTH = 5.5
FULL_WIDTH = 11.0
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "patch.linewidth": 0.5,
})

# Colorblind-safe category palette (Wong 2011, extended)
CATEGORY_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
    "#D55E00", "#F0E442", "#999999", "#000000", "#882255",
    "#44AA99", "#332288", "#DDCC77",
]

VARIANT_COLORS = {
    "basic":     "#999999",
    "graph":     "#0072B2",
    "sep":       "#E69F00",
    "graph_sep": "#D55E00",
    "graph+sep": "#D55E00",
}

VARIANT_LABELS = {
    "basic":     "Basic",
    "graph":     "+ FPG message passing",
    "sep":       "+ Separation loss",
    "graph_sep": "Full method",
    "graph+sep": "Full method",
}

# Display order for the group-importance heatmaps. Structural groups first
# (sequenced along the forward pass), then non-structural groups.
GROUP_TOPOLOGY_ORDER = [
    "embedding",
    "positional",
    "qkv_alignment",
    "score",
    "attention",
    "cache",
    "residual_stream",
    "layernorm",
    "ffn_output",
    "output",
    "representation_drift",
    "training_dynamics",
    "validation_perf",
]


def _save(fig, path, close=True):
    """Save figure as both PDF and PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png")
    if close:
        plt.close(fig)


def _topology_sort(group_names):
    rank = {name: i for i, name in enumerate(GROUP_TOPOLOGY_ORDER)}
    return sorted(group_names, key=lambda g: (rank.get(g, len(rank)), g))


def _aggregate_explanations(explanations_by_rc, group_names):
    all_agg = {}
    for rc_idx, expls in explanations_by_rc.items():
        agg = {g: 0.0 for g in group_names}
        for expl in expls:
            for g, v in expl.items():
                if g in agg:
                    agg[g] += v
        for g in agg:
            agg[g] /= max(len(expls), 1)
        all_agg[rc_idx] = agg
    return all_agg


# ── 1. Embedding visualisation (t-SNE / UMAP) ───────────────────────────────

def plot_embedding_tsne(embeddings, labels, label_names, title, path,
                        method="tsne", perplexity=30, alpha=0.5):
    """2D scatter of embeddings colored by category label.

    Args:
        embeddings: (N, D) array
        labels: (N,) integer labels
        label_names: list of label name strings
        title: plot title
        path: output file path (without extension)
        method: "tsne" or "umap"
    """
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1, metric="euclidean")
        coords = reducer.fit_transform(embeddings)
        method_label = "UMAP"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                       learning_rate="auto", init="pca")
        coords = reducer.fit_transform(embeddings)
        method_label = "t-SNE"

    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    colors = CATEGORY_COLORS[:n_labels]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.85))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        name = label_names[lab] if lab < len(label_names) else f"Class {lab}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i % len(colors)], s=6, alpha=alpha,
                   edgecolors="none", label=name, rasterized=True)

    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    if title:
        ax.set_title(title, fontweight="medium")

    # Clean legend
    ncol = 2 if n_labels > 6 else 1
    leg = ax.legend(markerscale=3, frameon=True, fancybox=False,
                    edgecolor="0.8", framealpha=0.95, ncol=ncol,
                    loc="best", handletextpad=0.3, columnspacing=0.8)
    leg.get_frame().set_linewidth(0.5)

    ax.tick_params(labelbottom=False, labelleft=False)
    fig.tight_layout()
    _save(fig, path)


def plot_embedding_comparison(embeddings_dict, labels, label_names, path_prefix):
    """Side-by-side t-SNE for multiple variants (e.g., flat vs graph).

    Args:
        embeddings_dict: {"variant_name": (N, D) array, ...}
        labels: (N,) integer labels (same for all)
        label_names: list of label name strings
        path_prefix: output path prefix
    """
    n_variants = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_variants,
                             figsize=(COLUMN_WIDTH * n_variants / 2, COLUMN_WIDTH * 0.42))
    if n_variants == 1:
        axes = [axes]

    unique_labels = sorted(set(labels))
    colors = CATEGORY_COLORS[:len(unique_labels)]

    all_coords = {}
    for variant_name, emb in embeddings_dict.items():
        tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                    learning_rate="auto", init="pca")
        all_coords[variant_name] = tsne.fit_transform(emb)

    for ax, (variant_name, coords) in zip(axes, all_coords.items()):
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names[lab] if lab < len(label_names) else str(lab)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[i % len(colors)], s=4, alpha=0.4,
                       edgecolors="none", label=name, rasterized=True)
        display_name = VARIANT_LABELS.get(variant_name, variant_name)
        if display_name:
            ax.set_title(display_name, fontweight="medium")
        ax.tick_params(labelbottom=False, labelleft=False)

    # Shared legend from first axis
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=colors[i % len(colors)],
                      markersize=5, label=label_names[lab] if lab < len(label_names) else str(lab))
               for i, lab in enumerate(unique_labels)]
    ncol = min(len(unique_labels), 4)
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               frameon=True, fancybox=False, edgecolor="0.8",
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, path_prefix)


# ── 2. Confusion matrices ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title, path,
                          normalize=True, figsize=None):
    """Publication-quality confusion matrix heatmap.

    Args:
        y_true, y_pred: integer label arrays
        class_names: list of class name strings
        title: plot title
        path: output path
        normalize: if True, show row-normalised percentages
    """
    n = len(class_names)
    if figsize is None:
        side = max(COLUMN_WIDTH * 0.8, n * 0.45)
        figsize = (side, side)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=figsize)

    # Sequential blue colormap (light to dark)
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f" if normalize else "d",
                cmap=cmap, vmin=0, vmax=1 if normalize else None,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Recall" if normalize else "Count"},
                annot_kws={"size": 6})

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title, fontweight="medium")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    _save(fig, path)


# ── 3. Per-category root-cause cluster detail ────────────────────────────────

def plot_rootcause_clusters(embeddings, rc_labels, rc_names, category_name,
                            path, method="tsne"):
    """Zoomed-in t-SNE of root causes within one category.

    Shows how intra-family contrastive loss improves cluster separation.
    """
    if len(embeddings) < 10:
        return

    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1))
        coords = reducer.fit_transform(embeddings)
    else:
        perp = min(30, max(5, len(embeddings) // 5))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp,
                       learning_rate="auto", init="pca")
        coords = reducer.fit_transform(embeddings)

    unique_rc = sorted(set(rc_labels))
    colors = CATEGORY_COLORS[:len(unique_rc)]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.75, COLUMN_WIDTH * 0.65))

    for i, rc in enumerate(unique_rc):
        mask = rc_labels == rc
        name = rc_names[rc] if rc < len(rc_names) else f"RC {rc}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   s=15, alpha=0.6, edgecolors="white", linewidths=0.3,
                   label=name, rasterized=True)

    ax.tick_params(labelbottom=False, labelleft=False)

    leg = ax.legend(markerscale=1.5, frameon=True, fancybox=False,
                    edgecolor="0.8", framealpha=0.95,
                    handletextpad=0.3, fontsize=6)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


def plot_rootcause_comparison(emb_basic, emb_sep, rc_labels, rc_names,
                              category_name, path):
    """Side-by-side root-cause clusters: without vs with the separation loss."""
    if len(emb_basic) < 10:
        return

    perp = min(30, max(5, len(emb_basic) // 5))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.45))

    unique_rc = sorted(set(rc_labels))
    colors = CATEGORY_COLORS[:len(unique_rc)]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for ax, emb, subtitle in [(ax1, emb_basic, "Without separation loss"),
                               (ax2, emb_sep, "With separation loss")]:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                    learning_rate="auto", init="pca")
        coords = tsne.fit_transform(emb)

        for i, rc in enumerate(unique_rc):
            mask = rc_labels == rc
            name = rc_names[rc] if rc < len(rc_names) else f"RC {rc}"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[i % len(colors)],
                       marker=markers[i % len(markers)],
                       s=12, alpha=0.5, edgecolors="white", linewidths=0.2,
                       label=name, rasterized=True)
        ax.tick_params(labelbottom=False, labelleft=False)

    # Shared legend
    handles = [Line2D([0], [0], marker=markers[i % len(markers)], color="w",
                      markerfacecolor=colors[i % len(colors)], markersize=5,
                      label=rc_names[rc] if rc < len(rc_names) else f"RC {rc}")
               for i, rc in enumerate(unique_rc)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(unique_rc), 4),
               frameon=True, fancybox=False, edgecolor="0.8",
               bbox_to_anchor=(0.5, -0.05), handletextpad=0.3, fontsize=6)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, path)


# ── 4. Training curves ──────────────────────────────────────────────────────

def plot_training_curves(curves_dict, path, metric_name="Val metric"):
    """Training dynamics: train/val loss + per-component loss + metrics.

    Three panels:
      Left:   Train vs val total loss (overfitting detection)
      Center: Per-component loss breakdown (detect, category, rootcause, contrastive)
      Right:  Validation metrics (detection F1, category F1)

    Args:
        curves_dict: {"variant_name": {"epochs": [...], "train_loss": [...],
                       "val_loss": [...], "val_metric": [...], ...}}
        path: output path
        metric_name: label for the metric axis
    """
    # Use first (and usually only) variant
    data = list(curves_dict.values())[0]
    epochs = data.get("epochs", [])
    if len(epochs) < 2:
        return

    has_val_loss = "val_loss" in data and len(data["val_loss"]) == len(epochs)
    has_components = all(k in data for k in ["loss_detect", "loss_category", "loss_rootcause"])

    has_stage3 = any(k in data for k in ["val_rc_proto_f1", "val_ce_proto_agreement", "val_proto_margin"])
    n_panels = 1 + int(has_components) + 1 + int(has_stage3)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(COLUMN_WIDTH * n_panels / 2.45, COLUMN_WIDTH * 0.4))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # Panel 1: Train vs val loss
    ax = axes[ax_idx]
    ax.plot(epochs, data["train_loss"], color="#0072B2", label="Train", alpha=0.85)
    if has_val_loss:
        ax.plot(epochs, data["val_loss"], color="#D55E00", label="Val", alpha=0.85,
                linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)
    ax_idx += 1

    # Panel 2: Per-component loss breakdown
    if has_components:
        ax = axes[ax_idx]
        component_colors = {"loss_detect": "#0072B2", "loss_category": "#E69F00",
                            "loss_rootcause": "#009E73", "loss_contrastive": "#CC79A7",
                            "loss_prototype": "#56B4E9"}
        component_labels = {"loss_detect": "Detection", "loss_category": "Category",
                            "loss_rootcause": "Root-cause", "loss_contrastive": "Contrastive",
                            "loss_prototype": "Prototype"}
        for key in ["loss_detect", "loss_category", "loss_rootcause",
                     "loss_contrastive", "loss_prototype"]:
            if key in data and len(data[key]) == len(epochs):
                ax.plot(epochs, data[key], color=component_colors[key],
                        label=component_labels[key], alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss component")
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                        handletextpad=0.3, fontsize=6)
        leg.get_frame().set_linewidth(0.5)
        ax_idx += 1

    # Panel 3: Validation metrics
    ax = axes[ax_idx]
    if "val_det_f1" in data:
        ax.plot(epochs, data["val_det_f1"], color="#0072B2",
                label="Detection F1", alpha=0.85)
    if "val_cat_f1" in data:
        ax.plot(epochs, data["val_cat_f1"], color="#E69F00",
                label="Category F1", alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)

    if has_stage3:
        ax = axes[ax_idx + 1]
        if "val_rc_proto_f1" in data:
            ax.plot(epochs, data["val_rc_proto_f1"], color="#009E73",
                    label="RC proto F1", alpha=0.9)
        if "val_ce_proto_agreement" in data:
            ax.plot(epochs, data["val_ce_proto_agreement"], color="#CC79A7",
                    label="CE-proto agree", alpha=0.9, linestyle="--")
        if "val_proto_margin" in data:
            ax2 = ax.twinx()
            ax2.plot(epochs, data["val_proto_margin"], color="#D55E00",
                     label="Proto margin", alpha=0.85)
            ax2.set_ylabel("Margin")
            ax2.tick_params(labelsize=7)
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            leg = ax.legend(lines_1 + lines_2, labels_1 + labels_2,
                            frameon=True, fancybox=False, edgecolor="0.8",
                            handletextpad=0.3, fontsize=6, loc="lower right")
            leg.get_frame().set_linewidth(0.5)
        else:
            leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                            handletextpad=0.3, fontsize=6)
            leg.get_frame().set_linewidth(0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 / agreement")
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, path)


def plot_prototype_faithfulness_dynamics(curves_dict, path, title=None):
    """Validation-time Stage 3 faithfulness dynamics across epochs."""
    data = list(curves_dict.values())[0]
    epochs = data.get("epochs", [])
    agreement = data.get("val_ce_proto_agreement", [])
    proto_f1 = data.get("val_rc_proto_f1", [])
    margin = data.get("val_proto_margin", [])
    if len(epochs) < 2 or len(agreement) != len(epochs):
        return

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 0.95, COLUMN_WIDTH * 0.38))

    ax = axes[0]
    if len(proto_f1) == len(epochs):
        ax.plot(epochs, proto_f1, color="#009E73", label="RC proto F1", alpha=0.9)
    ax.plot(epochs, agreement, color="#CC79A7", label="CE-proto agreement", alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title, fontweight="medium")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)

    ax = axes[1]
    if len(margin) == len(epochs):
        ax.plot(epochs, margin, color="#D55E00", label="Prototype margin", alpha=0.9)
    if len(proto_f1) == len(epochs):
        ax.plot(epochs, proto_f1, color="#56B4E9", label="RC proto F1", alpha=0.75, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Margin / F1")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


def plot_prototype_margin_dynamics(curves_dict, path, title=None):
    """Compact Stage 3 margin-focused dynamics figure."""
    data = list(curves_dict.values())[0]
    epochs = data.get("epochs", [])
    margin = data.get("val_proto_margin", [])
    agreement = data.get("val_ce_proto_agreement", [])
    if len(epochs) < 2 or len(margin) != len(epochs):
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.58, COLUMN_WIDTH * 0.42))
    ax.plot(epochs, margin, color="#D55E00", label="Prototype margin", alpha=0.9)
    if len(agreement) == len(epochs):
        ax2 = ax.twinx()
        ax2.plot(epochs, agreement, color="#CC79A7", label="CE-proto agreement", alpha=0.8, linestyle="--")
        ax2.set_ylabel("Agreement")
        ax2.set_ylim(0, 1.05)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        leg = ax.legend(lines_1 + lines_2, labels_1 + labels_2,
                        frameon=True, fancybox=False, edgecolor="0.8",
                        handletextpad=0.3, fontsize=6, loc="best")
        leg.get_frame().set_linewidth(0.5)
    else:
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
        leg.get_frame().set_linewidth(0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Margin")
    if title:
        ax.set_title(title, fontweight="medium")
    fig.tight_layout()
    _save(fig, path)


def plot_embedding_trajectory_map(snapshots, label_names, path, title=None, alpha=0.10):
    """Anchored PCA map for validation embedding evolution.

    This is not a prototype trajectory figure. It visualises category centroid
    movement in a fixed 2D basis across saved validation snapshots.
    """
    if len(snapshots) < 2:
        return

    final_snapshot = snapshots[-1]
    final_embeddings = final_snapshot.get("embeddings")
    final_labels = final_snapshot.get("categories")
    if final_embeddings is None or len(final_embeddings) < 10:
        return

    pca = PCA(n_components=2, random_state=42)
    pca.fit(final_embeddings)
    unique_labels = sorted(set(final_labels.tolist()))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.78, COLUMN_WIDTH * 0.68))
    final_coords = pca.transform(final_embeddings)
    ax.scatter(final_coords[:, 0], final_coords[:, 1], s=5, color="0.85",
               alpha=alpha, edgecolors="none", rasterized=True)

    for i, lab in enumerate(unique_labels):
        traj = []
        for snap in snapshots:
            emb = snap.get("embeddings")
            cats = snap.get("categories")
            if emb is None or cats is None:
                continue
            mask = cats == lab
            if mask.sum() == 0:
                continue
            coords = pca.transform(emb[mask])
            traj.append(coords.mean(axis=0))
        if not traj:
            continue
        traj = np.asarray(traj)
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        name = label_names[lab] if lab < len(label_names) else f"Class {lab}"
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.9, linewidth=1.0)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=18, zorder=3, label=name)

    ax.set_xlabel("Anchored PCA 1")
    ax.set_ylabel("Anchored PCA 2")
    if title:
        ax.set_title(title, fontweight="medium")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3, ncol=2 if len(unique_labels) > 6 else 1)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, path)


def plot_prototype_utilization_dynamics(curves_dict, path, title=None):
    """Plot prototype usage entropy and active-prototype count over epochs."""
    data = list(curves_dict.values())[0]
    epochs = data.get("epochs", [])
    entropy = data.get("val_proto_utilization_entropy", [])
    active = data.get("val_proto_active_count", [])
    if len(epochs) < 2 or len(entropy) != len(epochs):
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.62, COLUMN_WIDTH * 0.42))
    ax.plot(epochs, entropy, color="#0072B2", label="Usage entropy", alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.set_ylim(0, 1.05)

    if len(active) == len(epochs):
        ax2 = ax.twinx()
        ax2.plot(epochs, active, color="#E69F00", label="Active prototypes", alpha=0.85, linestyle="--")
        ax2.set_ylabel("Active count")
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        leg = ax.legend(lines_1 + lines_2, labels_1 + labels_2,
                        frameon=True, fancybox=False, edgecolor="0.8",
                        handletextpad=0.3, fontsize=6)
        leg.get_frame().set_linewidth(0.5)
    else:
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
        leg.get_frame().set_linewidth(0.5)

    if title:
        ax.set_title(title, fontweight="medium")
    fig.tight_layout()
    _save(fig, path)


def plot_prototype_drift_dynamics(curves_dict, path, title=None):
    """Plot prototype drift between saved snapshots."""
    data = list(curves_dict.values())[0]
    snaps = data.get("embedding_snapshots", [])
    if len(snaps) < 2:
        return

    epochs = []
    family_drifts = {}
    for prev, cur in zip(snaps[:-1], snaps[1:]):
        if "prototypes" not in prev or "prototypes" not in cur:
            continue
        epochs.append(cur["epoch"])
        families = sorted(set(prev["prototypes"]).intersection(cur["prototypes"]))
        for cat_name in families:
            p_prev = prev["prototypes"][cat_name]
            p_cur = cur["prototypes"][cat_name]
            if p_prev.shape != p_cur.shape:
                continue
            drift = np.sqrt(((p_cur - p_prev) ** 2).sum(axis=1)).mean()
            family_drifts.setdefault(cat_name, []).append(float(drift))
    if not epochs or not family_drifts:
        return

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.72, COLUMN_WIDTH * 0.45))
    for i, (cat_name, values) in enumerate(sorted(family_drifts.items())):
        ax.plot(epochs[:len(values)], values, color=CATEGORY_COLORS[i % len(CATEGORY_COLORS)],
                alpha=0.7, linewidth=1.0, label=cat_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean prototype drift")
    if title:
        ax.set_title(title, fontweight="medium")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3, fontsize=5.5, ncol=2 if len(family_drifts) > 6 else 1)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, path)


def plot_prototype_assignment_heatmap(matrix, rc_names, path, title=None):
    """Rows=true root causes, columns=assigned prototypes."""
    if matrix is None or np.size(matrix) == 0:
        return
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.78, max(COLUMN_WIDTH * 0.36, matrix.shape[0] * 0.32)))
    labels = [rc_names[i] if i < len(rc_names) else f"RC-{i}" for i in range(matrix.shape[0])]
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=sns.color_palette("Blues", as_cmap=True),
                vmin=0, vmax=1, xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Assignment fraction"},
                annot_kws={"size": 6})
    ax.set_xlabel("Assigned prototype")
    ax.set_ylabel("True root cause")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    if title:
        ax.set_title(title, fontweight="medium")
    fig.tight_layout()
    _save(fig, path)


def plot_prototype_separation_matrix(matrix, rc_names, path, title=None):
    """Pairwise prototype distance heatmap."""
    if matrix is None or np.size(matrix) == 0:
        return
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.74, max(COLUMN_WIDTH * 0.34, matrix.shape[0] * 0.32)))
    labels = [rc_names[i] if i < len(rc_names) else f"RC-{i}" for i in range(matrix.shape[0])]
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Prototype distance"},
                annot_kws={"size": 6})
    ax.set_xlabel("Prototype")
    ax.set_ylabel("Prototype")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    if title:
        ax.set_title(title, fontweight="medium")
    fig.tight_layout()
    _save(fig, path)


def plot_group_ablation_faithfulness(ablation_dict, path, title=None):
    """Compare margin drop from top-evidence vs random group ablation."""
    if not ablation_dict:
        return
    families = sorted(ablation_dict)
    top_vals = [ablation_dict[f]["top_drop"] for f in families]
    rand_vals = [ablation_dict[f]["random_drop"] for f in families]
    x = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.95, COLUMN_WIDTH * 0.42))
    ax.bar(x - width / 2, top_vals, width, color="#D55E00", alpha=0.85, label="Top-evidence groups")
    ax.bar(x + width / 2, rand_vals, width, color="#999999", alpha=0.85, label="Random groups")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_ylabel("Margin drop")
    if title:
        ax.set_title(title, fontweight="medium")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8", handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, path)


def plot_prototype_case_panels(cases, group_names, path, title=None, max_cases=6):
    """Representative prototype competition panels."""
    if not cases:
        return
    cases = sorted(cases, key=lambda c: (-c.get("margin", 0.0), c.get("category_name", "")))[:max_cases]
    n = len(cases)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH * 0.78, nrows * 2.4))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, case in zip(axes.flat, cases):
        delta = np.array(case.get("group_margin_delta", []), dtype=float)
        order = np.argsort(-np.abs(delta))[:6]
        names = [group_names[i] for i in order]
        vals = delta[order]
        colors = ["#D55E00" if v >= 0 else "#56B4E9" for v in vals]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="0.5", linewidth=0.5)
        ax.set_xlabel(r"$\Delta_g$")
        header = (
            f"{case['category_name']} | pred={case['pred_root_cause_name']}\n"
            f"true={case['true_root_cause_name']} | margin={case['margin']:.2f}"
        )
        ax.set_title(header, fontsize=7, fontweight="medium")
    if title:
        fig.suptitle(title, fontsize=8, fontweight="medium")
    fig.tight_layout()
    _save(fig, path)


def plot_category_conditioned_fpg_route(route_case, group_names, path, title=None):
    """Three-panel route figure for Stage 2 -> Stage 3 root-cause routing."""
    if not route_case:
        return

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH * 0.92, COLUMN_WIDTH * 0.52),
                             constrained_layout=True)

    # Panel 1: Stage 2 category gate
    ax = axes[0]
    cat_top2 = route_case.get("category_top2", [])
    probs = [c["prob"] for c in cat_top2]
    names = [c["category_name"] for c in cat_top2]
    ax.barh(range(len(names)), probs, color=["#0072B2", "#999999"][:len(names)], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Category probability")

    # Panel 2: Winner vs runner-up group evidence
    ax = axes[1]
    delta = np.array(route_case.get("group_margin_delta", []), dtype=float)
    order = np.argsort(-np.abs(delta))[:8]
    vals = delta[order]
    names = [group_names[i] for i in order]
    colors = ["#D55E00" if v >= 0 else "#56B4E9" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", linewidth=0.5)
    ax.set_xlabel(r"$d_{runnerup,g} - d_{winner,g}$")

    # Panel 3: Topology-ordered route map
    ax = axes[2]
    topo_groups = [g for g in _topology_sort(group_names) if g in group_names]
    group_delta = {group_names[i]: delta[i] for i in range(min(len(group_names), len(delta)))}
    xs = np.arange(len(topo_groups))
    ys = np.zeros(len(topo_groups))
    colors = [group_delta.get(g, 0.0) for g in topo_groups]
    vmax = max(max(abs(v) for v in colors), 1e-6)
    for i in range(len(topo_groups) - 1):
        ax.plot([xs[i], xs[i + 1]], [0, 0], color="0.82", linewidth=1.0, zorder=1)
    sc = ax.scatter(xs, ys, c=colors, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    s=90, edgecolors="white", linewidth=0.6, zorder=2)
    for i, g in enumerate(topo_groups):
        ax.text(xs[i], -0.16, g, rotation=35, ha="right", va="top", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(r"$\Delta_g$")

    if title:
        fig.suptitle(title, fontsize=8, fontweight="medium")
    _save(fig, path)


def plot_prototype_trajectory_map(prototype_snapshots, rc_group_embeddings, rc_group_labels,
                                  rc_names, category_name, path, title=None):
    """Anchored PCA map of prototype trajectories within one family."""
    if len(prototype_snapshots) < 2 or category_name not in rc_group_embeddings:
        return
    valid_snaps = [s for s in prototype_snapshots if "prototypes" in s and category_name in s["prototypes"]]
    if len(valid_snaps) < 2:
        return

    sample_emb = rc_group_embeddings[category_name]
    sample_labels = rc_group_labels[category_name]
    final_protos = valid_snaps[-1]["prototypes"][category_name]
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([sample_emb, final_protos]))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.76, COLUMN_WIDTH * 0.66))
    coords = pca.transform(sample_emb)
    unique_labels = sorted(set(sample_labels.tolist()))
    for i, lab in enumerate(unique_labels):
        mask = sample_labels == lab
        name = rc_names[lab] if lab < len(rc_names) else f"RC-{lab}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.16,
                   color=CATEGORY_COLORS[i % len(CATEGORY_COLORS)], edgecolors="none",
                   rasterized=True, label=name)

    for i in range(final_protos.shape[0]):
        traj = np.asarray([pca.transform(s["prototypes"][category_name])[i] for s in valid_snaps])
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.1, alpha=0.9)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=26, edgecolors="white", linewidth=0.4)

    ax.set_xlabel("Anchored PCA 1")
    ax.set_ylabel("Anchored PCA 2")
    if title:
        ax.set_title(title, fontweight="medium")
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3, fontsize=5.5, ncol=2 if len(unique_labels) > 6 else 1)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, path)


# ── 5. Ablation delta barplot ────────────────────────────────────────────────

def plot_ablation_bars(results_dict, metric_key, path,
                       ylabel="Macro F1", title="Ablation comparison"):
    """Grouped barplot: per-category metric across ablation variants.

    Args:
        results_dict: {"variant": {"per_category": {"cat": value, ...}}}
        metric_key: key into per-category dict
        path: output path
    """
    variants = list(results_dict.keys())
    categories = sorted(set().union(*(
        results_dict[v].get(metric_key, {}).keys() for v in variants
    )))

    if not categories:
        return

    n_cats = len(categories)
    n_vars = len(variants)
    x = np.arange(n_cats)
    width = 0.8 / n_vars

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, COLUMN_WIDTH * 0.5))

    for i, variant in enumerate(variants):
        vals = [results_dict[variant].get(metric_key, {}).get(cat, 0.0)
                for cat in categories]
        color = VARIANT_COLORS.get(variant, CATEGORY_COLORS[i % len(CATEGORY_COLORS)])
        label = VARIANT_LABELS.get(variant, variant)
        ax.bar(x + i * width - (n_vars - 1) * width / 2, vals,
               width * 0.9, color=color, label=label, alpha=0.85,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="medium")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3, loc="lower right")
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


def plot_ablation_delta(results_dict, metric_key, baseline_variant, path,
                        ylabel="$\\Delta$ F1 vs baseline", title="Improvement over baseline"):
    """Barplot showing delta from baseline for each variant.

    Args:
        results_dict: {"variant": {"per_category": {"cat": value, ...}}}
        baseline_variant: name of the baseline variant
        path: output path
    """
    variants = [v for v in results_dict if v != baseline_variant]
    categories = sorted(set().union(*(
        results_dict[v].get(metric_key, {}).keys() for v in results_dict
    )))

    if not categories:
        return

    baseline_vals = {cat: results_dict[baseline_variant].get(metric_key, {}).get(cat, 0.0)
                     for cat in categories}

    n_cats = len(categories)
    n_vars = len(variants)
    x = np.arange(n_cats)
    width = 0.8 / n_vars

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, COLUMN_WIDTH * 0.45))

    for i, variant in enumerate(variants):
        deltas = [results_dict[variant].get(metric_key, {}).get(cat, 0.0) - baseline_vals[cat]
                  for cat in categories]
        color = VARIANT_COLORS.get(variant, CATEGORY_COLORS[i % len(CATEGORY_COLORS)])
        label = VARIANT_LABELS.get(variant, variant)
        ax.bar(x + i * width - (n_vars - 1) * width / 2, deltas,
               width * 0.9, color=color, label=label, alpha=0.85,
               edgecolor="white", linewidth=0.3)

    ax.axhline(0, color="0.5", linewidth=0.5, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="medium")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


# ── 6. ROC curves ───────────────────────────────────────────────────────────

def plot_roc_curves(roc_data_dict, path, title="Stage 1: Detection ROC"):
    """Overlay ROC curves for multiple variants.

    Args:
        roc_data_dict: {"variant": {"y_true": [...], "y_score": [...]}}
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.7, COLUMN_WIDTH * 0.65))

    single_curve = len(roc_data_dict) == 1
    auc_text = None
    for variant, data in roc_data_dict.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_score"])
        roc_auc = auc(fpr, tpr)
        color = VARIANT_COLORS.get(variant, "#333333")
        label = None if single_curve else f"{VARIANT_LABELS.get(variant, variant)} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=color, label=label, alpha=0.85)
        if single_curve:
            auc_text = f"AUC = {roc_auc:.3f}"

    ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    if title:
        ax.set_title(title, fontweight="medium")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    if single_curve:
        ax.text(0.98, 0.04, auc_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "0.8", "linewidth": 0.5, "pad": 2.0})
    else:
        leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                        loc="lower right", handletextpad=0.3)
        leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


# ── 7. Summary table as figure ──────────────────────────────────────────────

def plot_summary_table(rows, col_names, path, title="Ablation results"):
    """Render a comparison table as a clean figure (for paper appendix).

    Args:
        rows: list of dicts with col_names as keys
        col_names: ordered list of column names to display
        path: output path
    """
    cell_text = []
    for row in rows:
        cell_text.append([str(row.get(c, "")) for c in col_names])

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.8, 0.3 + 0.25 * len(rows)))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_names,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    # Style header
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E8E8E8")
        else:
            cell.set_facecolor("white")

    if title:
        ax.set_title(title, fontweight="medium", pad=10)
    fig.tight_layout()
    _save(fig, path)


# ── 8. Stage-wise performance radar/spider chart ────────────────────────────

def plot_stage_comparison(results_dict, path, title="Stage-wise performance"):
    """Grouped barplot showing all 3 stages across variants.

    Stage 3 is reported only under the predicted-category route.

    Args:
        results_dict: {"variant": {"s1_auroc": v, "s1_f1": v, "s2_f1": v,
                                    "s3_pred": v}}
    """
    metrics = ["S1 AUROC", "S1 F1", "S2 Cat-F1", "S3 RC (pred)"]
    metric_keys = ["s1_auroc", "s1_f1", "s2_f1", "s3_pred"]

    variants = list(results_dict.keys())
    n_metrics = len(metrics)
    n_vars = len(variants)
    x = np.arange(n_metrics)
    width = 0.8 / n_vars

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.5))

    for i, variant in enumerate(variants):
        vals = [results_dict[variant].get(k, 0.0) for k in metric_keys]
        color = VARIANT_COLORS.get(variant, CATEGORY_COLORS[i])
        label = VARIANT_LABELS.get(variant, variant)
        bars = ax.bar(x + i * width - (n_vars - 1) * width / 2, vals,
                      width * 0.9, color=color, label=label, alpha=0.85,
                      edgecolor="white", linewidth=0.3)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=5,
                        rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    if title:
        ax.set_title(title, fontweight="medium")

    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.8",
                    handletextpad=0.3, loc="lower left")
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, path)


# ── 9. FPG-based explanation plots ───────────────────────────────────────────

def plot_explanation_barplot(explanations, group_names, category_name,
                            path, title=None, top_k=8):
    """Per-group distance contribution barplot for one category.

    Shows the average fraction of total prototype distance contributed by
    each FPG group. This is the FPG-based explanation for fault diagnosis.

    Args:
        explanations: list of dicts [{group_name: fraction, ...}, ...]
        group_names: ordered list of group names
        category_name: which category
        path: output path
        top_k: show only top-k groups by mean contribution
    """
    if not explanations:
        return

    agg = {g: 0.0 for g in group_names}
    for expl in explanations:
        for g, v in expl.items():
            if g in agg:
                agg[g] += v
    for g in agg:
        agg[g] /= len(explanations)

    sorted_groups = sorted(agg.items(), key=lambda x: -x[1])[:top_k]
    names = [g for g, _ in sorted_groups]
    values = [v for _, v in sorted_groups]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.8, COLUMN_WIDTH * 0.5))

    colors = [CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(names))]
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Fraction of total distance")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=6)

    ax.set_xlim(0, max(values) * 1.2)
    if title:
        ax.set_title(title, fontweight="medium")

    fig.tight_layout()
    _save(fig, path)


def plot_explanation_heatmap(explanations_by_rc, group_names, category_name,
                             rc_names, path, top_k=8, group_order=None,
                             vmin=None, vmax=None, rc_support=None, rc_f1=None):
    """Heatmap: rows = root causes, columns = FPG groups, values = mean contribution.

    Shows how different root causes within the same category are diagnosed
    via different component signatures. This is the key explainability figure.

    Args:
        explanations_by_rc: dict {rc_local_idx: [list of explanation dicts]}
        group_names: ordered list of group names
        category_name: which category
        rc_names: list of root-cause names
        path: output path
        top_k: show only top-k groups
    """
    if not explanations_by_rc:
        return

    all_agg = _aggregate_explanations(explanations_by_rc, group_names)

    ordered_groups = group_order or _topology_sort(group_names)
    group_max = {g: max(all_agg[rc].get(g, 0) for rc in all_agg) for g in ordered_groups}
    top_groups = sorted(group_max, key=lambda g: -group_max[g])[:top_k]
    top_groups = [g for g in ordered_groups if g in top_groups]

    rc_indices = sorted(all_agg.keys())
    matrix = np.zeros((len(rc_indices), len(top_groups)))
    for i, rc in enumerate(rc_indices):
        for j, g in enumerate(top_groups):
            matrix[i, j] = all_agg[rc].get(g, 0)

    rc_labels = [rc_names[rc] if rc < len(rc_names) else f"RC-{rc}" for rc in rc_indices]

    extra_cols = int(rc_support is not None) + int(rc_f1 is not None)
    width_ratios = [max(4.8, len(top_groups) * 0.52)] + [0.7] * extra_cols
    fig, axes = plt.subplots(
        1, 1 + extra_cols,
        figsize=(COLUMN_WIDTH * (0.72 + 0.08 * extra_cols),
                 max(COLUMN_WIDTH * 0.4, len(rc_indices) * 0.35)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=top_groups, yticklabels=rc_labels,
                ax=ax, linewidths=0.3, linecolor="white",
                vmin=vmin, vmax=vmax,
                cbar_kws={"shrink": 0.7, "label": "Distance fraction"},
                annot_kws={"size": 6})

    ax.set_xlabel("FPG component group")
    ax.set_ylabel("Root cause")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    col_idx = 1
    if rc_support is not None:
        support_vals = np.array([[rc_support.get(rc, 0)] for rc in rc_indices], dtype=float)
        sns.heatmap(support_vals, cmap=sns.color_palette("Greys", as_cmap=True),
                    annot=True, fmt=".0f", cbar=False, ax=axes[col_idx],
                    yticklabels=False, xticklabels=["n"],
                    linewidths=0.3, linecolor="white", annot_kws={"size": 6})
        axes[col_idx].tick_params(axis="x", rotation=0)
        col_idx += 1

    if rc_f1 is not None:
        f1_vals = np.array([[rc_f1.get(rc, 0.0)] for rc in rc_indices], dtype=float)
        sns.heatmap(f1_vals, cmap=sns.color_palette("Blues", as_cmap=True),
                    annot=True, fmt=".2f", vmin=0, vmax=1, cbar=False, ax=axes[col_idx],
                    yticklabels=False, xticklabels=["F1"],
                    linewidths=0.3, linecolor="white", annot_kws={"size": 6})
        axes[col_idx].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    _save(fig, path)


def plot_signature_delta_heatmap(explanations_by_rc, group_names, category_name,
                                 rc_names, path, top_k=8, group_order=None,
                                 rc_support=None, rc_f1=None):
    """Heatmap of root-cause signatures as deviation from family mean."""
    if not explanations_by_rc:
        return

    all_agg = _aggregate_explanations(explanations_by_rc, group_names)
    ordered_groups = group_order or _topology_sort(group_names)
    family_mean = {
        g: float(np.mean([all_agg[rc].get(g, 0.0) for rc in all_agg]))
        for g in ordered_groups
    }
    group_delta = {
        g: max(abs(all_agg[rc].get(g, 0.0) - family_mean[g]) for rc in all_agg)
        for g in ordered_groups
    }
    top_groups = sorted(group_delta, key=lambda g: -group_delta[g])[:top_k]
    top_groups = [g for g in ordered_groups if g in top_groups]

    rc_indices = sorted(all_agg.keys())
    matrix = np.zeros((len(rc_indices), len(top_groups)))
    for i, rc in enumerate(rc_indices):
        for j, g in enumerate(top_groups):
            matrix[i, j] = all_agg[rc].get(g, 0.0) - family_mean[g]

    rc_labels = [rc_names[rc] if rc < len(rc_names) else f"RC-{rc}" for rc in rc_indices]
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 1e-6)

    extra_cols = int(rc_support is not None) + int(rc_f1 is not None)
    width_ratios = [max(4.8, len(top_groups) * 0.52)] + [0.7] * extra_cols
    fig, axes = plt.subplots(
        1, 1 + extra_cols,
        figsize=(COLUMN_WIDTH * (0.72 + 0.08 * extra_cols),
                 max(COLUMN_WIDTH * 0.4, len(rc_indices) * 0.35)),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]

    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=top_groups, yticklabels=rc_labels,
                ax=ax, linewidths=0.3, linecolor="white",
                vmin=-vmax, vmax=vmax,
                cbar_kws={"shrink": 0.7, "label": "Delta from family mean"},
                annot_kws={"size": 6})
    ax.set_xlabel("FPG component group")
    ax.set_ylabel("Root cause")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

    col_idx = 1
    if rc_support is not None:
        support_vals = np.array([[rc_support.get(rc, 0)] for rc in rc_indices], dtype=float)
        sns.heatmap(support_vals, cmap=sns.color_palette("Greys", as_cmap=True),
                    annot=True, fmt=".0f", cbar=False, ax=axes[col_idx],
                    yticklabels=False, xticklabels=["n"],
                    linewidths=0.3, linecolor="white", annot_kws={"size": 6})
        axes[col_idx].tick_params(axis="x", rotation=0)
        col_idx += 1

    if rc_f1 is not None:
        f1_vals = np.array([[rc_f1.get(rc, 0.0)] for rc in rc_indices], dtype=float)
        sns.heatmap(f1_vals, cmap=sns.color_palette("Blues", as_cmap=True),
                    annot=True, fmt=".2f", vmin=0, vmax=1, cbar=False, ax=axes[col_idx],
                    yticklabels=False, xticklabels=["F1"],
                    linewidths=0.3, linecolor="white", annot_kws={"size": 6})
        axes[col_idx].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    _save(fig, path)
