"""Run all 4 ablation variants and produce comparison table + plots.

Variants:
  1. basic           -- no graph, no intra-family contrastive loss
  2. graph           -- FPG graph, no intra-family contrastive loss
  3. sibling         -- no graph, intra-family contrastive loss
  4. graph+sibling   -- full method (graph + intra-family contrastive loss)

Usage:
    python -m hierarchical_graph_category_rootcause.evaluate --arch encoder
    python -m hierarchical_graph_category_rootcause.evaluate --arch both
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import CONFIGS_ROOT, RESULTS_ROOT

from train import run_experiment, load_data
from plotting import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_roc_curves,
    plot_explanation_barplot, plot_explanation_heatmap,
    plot_prototype_faithfulness_dynamics,
    plot_prototype_margin_dynamics,
    plot_prototype_utilization_dynamics,
    plot_prototype_drift_dynamics,
    plot_prototype_assignment_heatmap,
    plot_prototype_separation_matrix,
    plot_group_ablation_faithfulness,
    plot_prototype_case_panels,
    plot_category_conditioned_fpg_route,
    plot_prototype_trajectory_map,
    plot_signature_delta_heatmap,
)

DEFAULT_RESULTS_DIR = RESULTS_ROOT / "hierarchical_graph_category_rootcause"


ABLATION_VARIANTS = [
    {"name": "basic",          "use_graph": False, "use_sibling": False},
    {"name": "graph",          "use_graph": True,  "use_sibling": False},
    {"name": "sibling",        "use_graph": False, "use_sibling": True},
    {"name": "graph_sibling",  "use_graph": True,  "use_sibling": True},
]


def _strip_plot_data(summary):
    """Return a JSON-serialisable copy (drop numpy arrays and non-serialisable data)."""
    skip_keys = {"plot_data", "training_curves", "fold_plot_data", "fold_training_curves"}
    out = {}
    for k, v in summary.items():
        if k in skip_keys:
            continue
        out[k] = v
    return out


def run_full_ablation(arch, config=None, results_dir=None):
    """Run all 4 ablation variants for one architecture."""
    config = config or {}
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    results = {}

    for variant in ABLATION_VARIANTS:
        print(f"\n{'#'*70}")
        print(f"# ABLATION: {arch.upper()} -- {variant['name']}")
        print(f"{'#'*70}")

        summary = run_experiment(
            arch,
            use_graph=variant["use_graph"],
            use_sibling=variant["use_sibling"],
            config=config,
        )
        results[variant["name"]] = summary

        # Save individual result (JSON-safe)
        out_file = results_dir / f"{arch}_{variant['name']}.json"
        with open(out_file, "w") as f:
            json.dump(_strip_plot_data(summary), f, indent=2)
        print(f"  Saved -> {out_file}")

        artifact_file = results_dir / f"{arch}_{variant['name']}_artifacts.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved -> {artifact_file}")

    return results


def build_comparison_table(results, arch):
    """Build comparison table across all variants."""
    rows = []
    for variant_name, summary in results.items():
        row = {
            "variant": variant_name,
            "graph": summary.get("use_graph", False),
            "sibling": summary.get("use_sibling", False),
            "stage1_auroc": summary["stage1_detection_auroc"]["mean"],
            "stage1_f1": summary["stage1_detection_f1"]["mean"],
            "stage2_cat_f1": summary["stage2_category_f1"]["mean"],
            "stage3_rc_pred": summary["stage3_rc_pred_cat_macro"]["mean"],
        }
        rows.append(row)
    return rows


def print_table(rows, arch):
    """Pretty-print comparison table."""
    print(f"\n{'='*90}")
    print(f"  ABLATION TABLE -- {arch.upper()}")
    print(f"{'='*90}")
    header = f"{'Variant':<20} {'Graph':>5} {'Sib':>5} {'S1-AUROC':>9} {'S1-F1':>7} {'S2-CatF1':>9} {'S3-RC(pred)':>12}"
    print(header)
    print("-" * 92)
    for r in rows:
        g = "Y" if r["graph"] else "N"
        s = "Y" if r["sibling"] else "N"
        print(f"{r['variant']:<20} {g:>5} {s:>5} "
              f"{r['stage1_auroc']:>9.4f} {r['stage1_f1']:>7.4f} "
              f"{r['stage2_cat_f1']:>9.4f} {r['stage3_rc_pred']:>12.4f}")
    print(f"{'='*90}")

    if len(rows) >= 4:
        basic = next((r for r in rows if r["variant"] == "basic"), None)
        graph = next((r for r in rows if r["variant"] == "graph"), None)
        sibling = next((r for r in rows if r["variant"] == "sibling"), None)
        full = next((r for r in rows if r["variant"] == "graph_sibling"), None)

        if basic and graph:
            delta_cat = graph["stage2_cat_f1"] - basic["stage2_cat_f1"]
            print(f"\n  Q1: Does graph improve categorization?")
            print(f"      Graph vs Basic: {delta_cat:+.4f} Cat-F1")
            print(f"      -> {'YES' if delta_cat > 0.005 else 'MARGINAL' if delta_cat > 0 else 'NO'}")

        if basic and sibling:
            delta_rc = sibling["stage3_rc_pred"] - basic["stage3_rc_pred"]
            print(f"\n  Q2: Does intra-family contrastive loss improve root-cause diagnosis?")
            print(f"      intra-family contrastive vs Basic: {delta_rc:+.4f} RC-F1 (pred cat)")
            print(f"      -> {'YES' if delta_rc > 0.005 else 'MARGINAL' if delta_rc > 0 else 'NO'}")

        if basic and full:
            delta_all = full["stage3_rc_pred"] - basic["stage3_rc_pred"]
            print(f"\n  Q3: Does combining both give best end-to-end RC(pred)?")
            print(f"      Full vs Basic: {delta_all:+.4f} end-to-end RC-F1")
            print(f"      -> {'YES' if delta_all > 0.005 else 'MARGINAL' if delta_all > 0 else 'NO'}")


def generate_all_plots(results, arch, category_names, results_dir=None):
    """Generate the main-method figures only.

    Ablation comparisons stay in tables or explicit ablation studies.
    The default figure set should focus on the final method and avoid
    extra "basic vs full" or t-SNE comparison plots.
    """
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    fig_dir = results_dir / "figures" / arch
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating plots -> {fig_dir}/")

    # ── Collect fold-0 data from each variant ────────────────────────────
    fold0_data = {}
    for vname, summary in results.items():
        if "fold_plot_data" in summary and len(summary["fold_plot_data"]) > 0:
            fold0_data[vname] = summary["fold_plot_data"][0]

    proposed = "graph_sibling" if "graph_sibling" in fold0_data else list(fold0_data.keys())[-1]
    data_meta = load_data(arch)
    rootcause_local_labels = data_meta["rootcause_local_labels"]
    category_to_rootcauses = data_meta["category_to_rootcauses"]

    def rc_names_for(cat_name):
        local_map = rootcause_local_labels.get(cat_name, {})
        local_to_name = {}
        for gi, rc_name in category_to_rootcauses.get(cat_name, []):
            li = local_map.get(gi)
            if li is not None:
                local_to_name[li] = rc_name
        if not local_to_name:
            return []
        return [local_to_name.get(i, f"RC-{i}") for i in range(max(local_to_name) + 1)]

    # ── 1. Training dynamics (proposed method) ───────────────────────────
    #    Train vs val loss + per-component loss breakdown
    if proposed in results:
        summary_p = results[proposed]
        curves_list = summary_p.get("fold_training_curves", [])
        if curves_list and curves_list[0]:
            tc = curves_list[0]
            epochs = tc.get("epochs", [])
            if len(epochs) > 2:
                # 1a. Train vs val loss (overfitting check)
                plot_training_curves(
                    {proposed: tc}, fig_dir / "train_val_loss",
                    metric_name="Val metric")
                print(f"    train_val_loss")
                plot_prototype_faithfulness_dynamics(
                    {proposed: tc}, fig_dir / "prototype_faithfulness_dynamics")
                print(f"    prototype_faithfulness_dynamics")
                plot_prototype_margin_dynamics(
                    {proposed: tc}, fig_dir / "prototype_margin_dynamics")
                print(f"    prototype_margin_dynamics")
                plot_prototype_utilization_dynamics(
                    {proposed: tc}, fig_dir / "prototype_utilization_dynamics")
                print(f"    prototype_utilization_dynamics")
                plot_prototype_drift_dynamics(
                    {proposed: tc}, fig_dir / "prototype_drift_dynamics")
                print(f"    prototype_drift_dynamics")

    # ── 2. Confusion matrix (proposed method) ────────────────────────────
    if proposed in fold0_data:
        pd_ = fold0_data[proposed]
        faulty = pd_["faulty_mask"]
        if faulty.sum() > 50:
            y_true = pd_["y_category_true"][faulty]
            y_pred = pd_["y_category_pred"][faulty]
            valid = y_pred >= 0
            if valid.sum() > 50:
                plot_confusion_matrix(
                    y_true[valid], y_pred[valid], category_names,
                    title=None,
                    path=fig_dir / "confusion_category",
                    normalize=True)
                print(f"    confusion_category")

    # ── 3. ROC curve (proposed method) ───────────────────────────────────
    if proposed in fold0_data:
        pd_ = fold0_data[proposed]
        if "y_detect_true" in pd_ and "y_detect_score" in pd_:
            if len(np.unique(pd_["y_detect_true"])) > 1:
                roc_data = {proposed: {
                    "y_true": pd_["y_detect_true"],
                    "y_score": pd_["y_detect_score"],
                }}
                plot_roc_curves(roc_data, fig_dir / "roc_detection",
                                title=None)
                print(f"    roc_detection")

    # ── 4. FPG-based explanation plots (proposed method) ─────────────────
    #    This is the key contribution: per-component diagnosis signatures
    if proposed in fold0_data:
        pd_ = fold0_data[proposed]
        group_names = pd_.get("group_names", [])
        explanations = pd_.get("explanations", {})
        rc_labels_data = pd_.get("rc_labels", {})
        rc_group_embeddings = pd_.get("rc_group_embeddings", {})
        rc_group_labels = pd_.get("rc_group_labels", {})
        prototype_snapshots = results[proposed].get("fold_training_curves", [{}])[0].get("embedding_snapshots", [])

        if explanations and group_names:
            for cat_name in explanations:
                expls = explanations[cat_name]
                if not expls or len(expls) < 5:
                    continue

                # Per-category explanation barplot
                plot_explanation_barplot(
                    expls, group_names, cat_name,
                    path=fig_dir / f"explain_{cat_name}",
                    top_k=8)
                print(f"    explain_{cat_name}")

                # Per-root-cause explanation heatmap (the key figure)
                if cat_name in rc_labels_data:
                    rc_labs = rc_labels_data[cat_name]
                    expls_by_rc = {}
                    for expl, lab in zip(expls, rc_labs):
                        expls_by_rc.setdefault(int(lab), []).append(expl)
                    if len(expls_by_rc) >= 2:
                        rc_names_list = rc_names_for(cat_name)
                        if not rc_names_list:
                            rc_names_list = [f"RC-{i}" for i in range(max(rc_labs) + 1)]
                        plot_explanation_heatmap(
                            expls_by_rc, group_names, cat_name, rc_names_list,
                            path=fig_dir / f"explain_heatmap_{cat_name}",
                            top_k=8)
                        print(f"    explain_heatmap_{cat_name}")
                        plot_signature_delta_heatmap(
                            expls_by_rc, group_names, cat_name, rc_names_list,
                            path=fig_dir / f"signature_delta_heatmap_{cat_name}",
                            top_k=8)
                        print(f"    signature_delta_heatmap_{cat_name}")

                if cat_name in rc_group_embeddings and cat_name in rc_group_labels:
                    rc_names_list = rc_names_for(cat_name)
                    plot_prototype_trajectory_map(
                        prototype_snapshots,
                        rc_group_embeddings,
                        rc_group_labels,
                        rc_names_list,
                        cat_name,
                        path=fig_dir / f"prototype_trajectory_map_{cat_name}",
                    )
                    print(f"    prototype_trajectory_map_{cat_name}")

        assignment = pd_.get("assignment_matrices", {})
        for cat_name, matrix in assignment.items():
            rc_names_list = rc_names_for(cat_name)
            plot_prototype_assignment_heatmap(
                matrix, rc_names_list, fig_dir / f"prototype_assignment_heatmap_{cat_name}")
            print(f"    prototype_assignment_heatmap_{cat_name}")

        separation = pd_.get("separation_matrices", {})
        for cat_name, matrix in separation.items():
            rc_names_list = rc_names_for(cat_name)
            plot_prototype_separation_matrix(
                matrix, rc_names_list, fig_dir / f"prototype_separation_matrix_{cat_name}")
            print(f"    prototype_separation_matrix_{cat_name}")

        ablation = pd_.get("group_ablation_faithfulness", {})
        if ablation:
            plot_group_ablation_faithfulness(
                ablation, fig_dir / "group_ablation_faithfulness")
            print(f"    group_ablation_faithfulness")

        cases = pd_.get("case_panels", [])
        if cases:
            plot_prototype_case_panels(cases, group_names, fig_dir / "prototype_case_panels")
            print(f"    prototype_case_panels")

        routes = pd_.get("category_conditioned_routes", {})
        for cat_name, route_case in routes.items():
            plot_category_conditioned_fpg_route(
                route_case, group_names, fig_dir / f"category_conditioned_fpg_route_{cat_name}")
            print(f"    category_conditioned_fpg_route_{cat_name}")


def main():
    ap = argparse.ArgumentParser(description="Run full ablation study")
    ap.add_argument("--arch", choices=["encoder", "decoder", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", default=None, help="Override output directory")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = ap.parse_args()

    import yaml
    config_path = CONFIGS_ROOT / "base.yaml"
    with open(config_path) as f:
        base = yaml.safe_load(f)

    merged_training = base.get("training", {})
    if "lr" in merged_training:
        merged_training["lr"] = float(merged_training["lr"])
    if "lambda_rootcause" in merged_training and "lambda_" not in merged_training:
        merged_training["lambda_"] = merged_training["lambda_rootcause"]
    if "beta_sibling" in merged_training and "beta" not in merged_training:
        merged_training["beta"] = merged_training["beta_sibling"]

    config = {
        **base.get("model", {}),
        **merged_training,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": float(merged_training.get("lr", 1e-3)),
        "patience": 20,
        "alpha": float(merged_training.get("alpha", 1.0)),
        "lambda_": float(merged_training.get("lambda_", 1.0)),
        "beta": float(merged_training.get("beta", 0.5)),
        "gamma": 0.3,
        "sibling_temperature": 0.1,
        "seed": 42,
    }

    results_dir = Path(args.output) if args.output else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    archs = ["encoder", "decoder"] if args.arch == "both" else [args.arch]
    all_tables = {}

    for arch in archs:
        results = run_full_ablation(arch, config, results_dir=results_dir)

        # Extract category names from first variant
        first_summary = next(iter(results.values()))
        category_names = list(first_summary.get("stage2_per_category_f1", {}).keys())

        table = build_comparison_table(results, arch)
        all_tables[arch] = table
        print_table(table, arch)

        # Generate plots
        if not args.no_plots:
            try:
                generate_all_plots(results, arch, category_names, results_dir=results_dir)
            except Exception as exc:
                err_file = results_dir / f"{arch}_plot_error.txt"
                with open(err_file, "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n")
                print(f"\n  Plot generation failed for {arch}: {exc}")
                print(f"  Error saved -> {err_file}")

    # Save combined ablation table (JSON-safe)
    combined = {
        arch: {
            "table": table,
            "timestamp": datetime.now().isoformat(),
        }
        for arch, table in all_tables.items()
    }
    combined_file = results_dir / "full_ablation_table.json"
    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Full ablation table saved -> {combined_file}")


if __name__ == "__main__":
    main()
