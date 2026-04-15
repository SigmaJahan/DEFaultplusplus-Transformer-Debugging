"""Post-hoc analysis: feature and group importance for the trained model.

Analyses:
  1. Per-group permutation importance for Stage 2 (categorization)
     - Shuffle all features in one FPG group, measure Cat-F1 drop
     - Validates: which transformer components matter for fault family identification

  2. Per-group permutation importance for Stage 3 (root-cause, predicted route)
     - Same approach but for root-cause diagnosis after Stage 2 prediction

  3. Top features within each group (ANOVA F-statistic on processed features)
     - Which specific features drive discrimination?

  4. FPG connectivity validation
     - Do structurally connected groups (FPG neighbors) show correlated importance?

Usage:
    python -m hierarchical_graph_category_rootcause.posthoc_analysis --arch encoder
    python -m hierarchical_graph_category_rootcause.posthoc_analysis --arch both
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import CONFIGS_ROOT, RESULTS_ROOT

from train import load_data, build_model, train_one_fold, set_seed, compute_proto_stats
from src.data.feature_processor import apply_processing_in_fold
from src.data.fundamental_fpg import fundamental_to_feature_group_adjacency
from plotting import (plot_explanation_barplot, plot_explanation_heatmap,
                      COLUMN_WIDTH, CATEGORY_COLORS, DPI, _save, sns)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = RESULTS_ROOT / "hierarchical_graph_category_rootcause"


def permutation_importance_groups(model, scaler, X_te, y_cat_te, y_rc_te,
                                  y_det_te, group_indices, category_names,
                                  rootcause_local_labels, category_sizes,
                                  n_repeats=5, seed=42):
    """Compute per-group permutation importance for Stage 2 and Stage 3.

    For each group, shuffle its features across test samples and measure
    the drop in macro-F1. Larger drop = more important group.

    Returns:
        stage2_importance: dict {group_name: mean_f1_drop}
        stage3_importance: dict {group_name: mean_f1_drop}
    """
    device = next(model.parameters()).device
    rng = np.random.RandomState(seed)

    X_te_s = np.nan_to_num(scaler.transform(X_te), nan=0.0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        xte = torch.from_numpy(X_te_s).float().to(device)
        z_te, h_te = model.encode(xte, group_indices)

        # Baseline Stage 2
        faulty = y_det_te == 1
        if faulty.sum() < 10:
            return {}, {}
        cat_logits = model.categorize(z_te[faulty])
        cat_preds = cat_logits.argmax(-1).cpu().numpy()
        base_cat_f1 = f1_score(y_cat_te[faulty], cat_preds, average="macro", zero_division=0)

        base_rc_f1 = compute_proto_stats(
            model, z_te, h_te, y_det_te, y_cat_te, y_rc_te,
            category_names, rootcause_local_labels, category_sizes,
            cat_preds=cat_preds,
        )["macro_f1"]

    # Permute each group
    stage2_imp = {}
    stage3_imp = {}

    for group_name, indices in sorted(group_indices.items()):
        cat_drops = []
        rc_drops = []

        for rep in range(n_repeats):
            X_perm = X_te_s.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm[:, indices] = X_perm[perm_order][:, indices]

            with torch.no_grad():
                xp = torch.from_numpy(X_perm).float().to(device)
                z_p, h_p = model.encode(xp, group_indices)

                # Stage 2
                cat_logits_p = model.categorize(z_p[faulty])
                cat_preds_p = cat_logits_p.argmax(-1).cpu().numpy()
                perm_cat_f1 = f1_score(y_cat_te[faulty], cat_preds_p,
                                       average="macro", zero_division=0)
                cat_drops.append(base_cat_f1 - perm_cat_f1)

                perm_rc_f1 = compute_proto_stats(
                    model, z_p, h_p, y_det_te, y_cat_te, y_rc_te,
                    category_names, rootcause_local_labels, category_sizes,
                    cat_preds=cat_preds_p,
                )["macro_f1"]
                rc_drops.append(base_rc_f1 - perm_rc_f1)

        stage2_imp[group_name] = float(np.mean(cat_drops))
        stage3_imp[group_name] = float(np.mean(rc_drops))

    return stage2_imp, stage3_imp


def anova_feature_importance(X_tr, y_cat_tr, y_det_tr, feature_names, group_indices):
    """ANOVA F-statistic per feature for fault categorization.

    Only uses faulty samples. Returns dict {feature_name: F_statistic}.
    """
    faulty = y_det_tr == 1
    X_f = X_tr[faulty]
    y_f = y_cat_tr[faulty]

    categories = sorted(set(y_f))
    results = {}

    for i, fname in enumerate(feature_names):
        col = X_f[:, i]
        if np.isnan(col).all() or np.std(col) < 1e-12:
            results[fname] = 0.0
            continue
        groups_data = [col[y_f == c] for c in categories if (y_f == c).sum() > 1]
        if len(groups_data) < 2:
            results[fname] = 0.0
            continue
        try:
            f_stat, _ = f_oneway(*groups_data)
            results[fname] = float(f_stat) if np.isfinite(f_stat) else 0.0
        except:
            results[fname] = 0.0

    return results


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_group_importance(importance, path, title, ylabel="F1 drop when shuffled"):
    """Horizontal bar plot of per-group importance."""
    sorted_items = sorted(importance.items(), key=lambda x: -x[1])
    names = [g for g, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.85, COLUMN_WIDTH * 0.55))
    colors = [CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i in range(len(names))]

    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(ylabel, fontsize=8)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=5.5)

    ax.set_title(title, fontweight="medium", fontsize=9)
    ax.axvline(0, color="0.5", linewidth=0.4)
    fig.tight_layout()
    _save(fig, path)


def plot_top_features_per_group(anova_scores, group_indices, feature_names,
                                 path, top_k=5, n_groups=6):
    """Show top-k features within the most important groups."""
    # Get group-level mean ANOVA
    group_means = {}
    for group_name, indices in group_indices.items():
        fscores = [anova_scores.get(feature_names[i], 0.0) for i in indices]
        group_means[group_name] = np.mean(fscores) if fscores else 0.0

    top_groups = sorted(group_means, key=lambda g: -group_means[g])[:n_groups]

    fig, axes = plt.subplots(2, 3, figsize=(COLUMN_WIDTH * 1.5, COLUMN_WIDTH * 0.9))
    axes = axes.flatten()

    for ax_idx, group_name in enumerate(top_groups):
        ax = axes[ax_idx]
        indices = group_indices[group_name]
        feat_scores = [(feature_names[i], anova_scores.get(feature_names[i], 0.0))
                       for i in indices]
        feat_scores.sort(key=lambda x: -x[1])
        top = feat_scores[:top_k]

        names = [f[:20] for f, _ in top]
        vals = [v for _, v in top]

        ax.barh(range(len(names)), vals, color=CATEGORY_COLORS[ax_idx % len(CATEGORY_COLORS)],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5)
        ax.invert_yaxis()
        ax.set_title(group_name, fontsize=7, fontweight="medium")
        ax.tick_params(axis="x", labelsize=5)

    fig.suptitle("Top features per FPG group (ANOVA F-statistic)", fontsize=9, fontweight="medium")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, path)


def plot_fpg_importance_overlay(stage2_imp, arch, path):
    """Show group importance overlaid on FPG connectivity structure.

    Simple adjacency heatmap with importance as node annotation.
    """
    fpg_groups, fpg_adj, _ = fundamental_to_feature_group_adjacency(arch)
    fpg_idx = {g: i for i, g in enumerate(fpg_groups)}

    # Filter to groups that exist in importance dict
    active_groups = [g for g in fpg_groups if g in stage2_imp]
    n = len(active_groups)
    if n < 3:
        return

    adj = np.zeros((n, n))
    for i, gi in enumerate(active_groups):
        for j, gj in enumerate(active_groups):
            if gi in fpg_idx and gj in fpg_idx:
                adj[i, j] = fpg_adj[fpg_idx[gi], fpg_idx[gj]]

    imp_values = [stage2_imp.get(g, 0.0) for g in active_groups]
    # Annotate: group name + importance
    labels = [f"{g}\n({stage2_imp.get(g, 0):.3f})" for g in active_groups]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.9, COLUMN_WIDTH * 0.8))

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(adj, annot=False, cmap=cmap, vmin=0, vmax=1,
                xticklabels=active_groups, yticklabels=labels,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "FPG edge weight"})

    ax.set_title(f"FPG connectivity + group importance ({arch})",
                 fontweight="medium", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    fig.tight_layout()
    _save(fig, path)


def run_posthoc(arch, config):
    """Run full post-hoc analysis for one architecture."""
    print(f"\n{'='*60}")
    print(f"POST-HOC ANALYSIS: {arch.upper()}")
    print(f"{'='*60}")

    data = load_data(arch)
    X = data["X"]
    groups = data["groups"]
    feature_names = data["feature_names"]

    # Use fold 0 for analysis
    gkf = GroupKFold(n_splits=5)
    tr_idx, te_idx = next(iter(gkf.split(X, data["y_detect"], groups)))
    set_seed(42)

    X_tr, X_te, feat_names_proc, g_idx, _ = apply_processing_in_fold(
        X[tr_idx], X[te_idx], feature_names, data["y_detect"][tr_idx], arch)

    g_dims = {g: len(i) for g, i in g_idx.items()}
    g_names = sorted(g_dims)

    # Train model (full method)
    model = build_model(arch, "graph_conditioned", X_tr.shape[1], g_dims, g_names,
                        data["n_categories"], data["category_sizes"],
                        config, use_graph=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    scaler, _ = train_one_fold(
        model, X_tr,
        data["y_detect"][tr_idx], data["y_category"][tr_idx], data["y_rootcause"][tr_idx],
        g_idx, data["category_names"], data["rootcause_local_labels"],
        config, use_sibling=True)

    fig_dir = RESULTS_DIR / "figures" / arch
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Permutation importance
    print("  Computing permutation importance (5 repeats)...")
    t0 = time.time()
    s2_imp, s3_imp = permutation_importance_groups(
        model, scaler, X_te,
        data["y_category"][te_idx], data["y_rootcause"][te_idx],
        data["y_detect"][te_idx], g_idx, data["category_names"],
        data["rootcause_local_labels"], data["category_sizes"],
        n_repeats=5)
    print(f"  Done ({time.time()-t0:.1f}s)")

    plot_group_importance(s2_imp, fig_dir / "importance_stage2",
                          title=None)
    print(f"    importance_stage2")

    plot_group_importance(s3_imp, fig_dir / "importance_stage3",
                          title=None)
    print(f"    importance_stage3")

    # 2. ANOVA feature importance
    print("  Computing ANOVA F-statistics...")
    X_tr_clean = np.nan_to_num(X_tr, nan=0.0).astype(np.float32)
    anova = anova_feature_importance(X_tr_clean, data["y_category"][tr_idx],
                                     data["y_detect"][tr_idx], feat_names_proc, g_idx)

    plot_top_features_per_group(anova, g_idx, feat_names_proc,
                                fig_dir / "top_features_per_group",
                                top_k=5, n_groups=6)
    print(f"    top_features_per_group")

    # 3. FPG validation overlay
    print("  Computing FPG importance overlay...")
    plot_fpg_importance_overlay(s2_imp, arch, fig_dir / "fpg_importance_overlay")
    print(f"    fpg_importance_overlay")

    # Save JSON
    results = {
        "arch": arch,
        "stage2_group_importance": s2_imp,
        "stage3_group_importance": s3_imp,
        "top_anova_per_group": {
            g: sorted([(feat_names_proc[i], anova.get(feat_names_proc[i], 0.0))
                        for i in indices], key=lambda x: -x[1])[:5]
            for g, indices in g_idx.items()
        },
    }
    out = RESULTS_DIR / f"posthoc_{arch}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {out}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Post-hoc feature/group importance analysis")
    ap.add_argument("--arch", choices=["encoder", "decoder", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    import yaml
    config_path = CONFIGS_ROOT / "base.yaml"
    with open(config_path) as f:
        base = yaml.safe_load(f)

    config = {
        **base.get("model", {}),
        **base.get("training", {}),
        "epochs": args.epochs,
        "batch_size": 256,
        "lr": float(base.get("training", {}).get("lr", 1e-3)),
        "patience": 20,
        "alpha": 1.0,
        "lambda_": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
        "sibling_temperature": 0.1,
        "seed": 42,
    }

    archs = ["encoder", "decoder"] if args.arch == "both" else [args.arch]
    for arch in archs:
        run_posthoc(arch, config)


if __name__ == "__main__":
    main()
