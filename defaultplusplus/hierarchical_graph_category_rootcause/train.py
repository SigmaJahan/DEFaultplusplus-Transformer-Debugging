"""Training driver for the hierarchical fault-diagnosis model.

Trains the three-level model with the combined loss

    L = L_detect + alpha * L_cat + lambda_rc * L_rc + L_sep
    L_sep = beta * L_ctr + gamma * L_pm

Training pipeline per fold:

  - 80/20 stratified train/val split inside the CV training portion for
    early stopping.
  - Detection uses all samples with inverse-frequency class weights and
    minority oversampling.
  - Categorization uses faulty samples only.
  - Root-cause cross-entropy uses the true category (teacher forcing).
  - Separation loss (contrastive + prototype matching) is restricted to
    faulty samples within their ground-truth category.
  - Final root-cause evaluation reports both oracle-category and
    predicted-category routes; misclassified-category samples count as
    errors in the predicted route.

Variant flags:

  --no-graph     disable FPG message passing
  --no-sep       disable the separation loss (sets beta = gamma = 0)

Usage:
    python -m hierarchical_graph_category_rootcause.train --arch encoder
    python -m hierarchical_graph_category_rootcause.train --arch both
"""
import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                      StratifiedShuffleSplit)
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_paths import CONFIGS_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data.loader import prepare_dataset_from_csv
from src.data.feature_processor import apply_processing_in_fold
from src.data.feature_groups import build_group_indices, get_group_sizes
from src.data.fundamental_fpg import fundamental_to_feature_group_adjacency
from defaultplusplus.deform import root_cause_label_space

from model import HierarchicalDiagnosisModel
from losses import hierarchical_loss, compute_detection_weights

DATA_DIR = DATA_ROOT
RESULTS_DIR = RESULTS_ROOT / "hierarchical_graph_category_rootcause"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def validate_label_space(arch, category_to_rootcauses):
    """Check the discovered label space against the official taxonomy.

    The official Level-3 label space is 40 root causes across 11
    categories for encoders and 45 across 12 for decoders
    (``deform.root_cause_label_space``). The benchmark CSV may contain a
    subset when not every operator has been run yet. We warn on the two
    kinds of mismatch rather than crash, so partial data still trains:

      - missing: a taxonomy (category, root cause) absent from the data,
        which usually means the benchmark has not been fully regenerated.
      - unexpected: a (category, root cause) in the data that is not in
        the taxonomy, which indicates taxonomy drift to investigate.
    """
    def _norm(s):
        # Normalize surface form so "KV Cache"/"kv_cache" and
        # "Parameter Initialization"/"parameter_initialization" compare
        # equal across the writer's official names and any display-name
        # variant used in the processed CSV.
        return str(s).strip().lower().replace(" ", "_").replace("-", "_")

    canonical = {
        comp.value: set(rcs)
        for comp, rcs in root_cause_label_space(arch).items()
    }
    discovered = {
        cat: {rc for _, rc in pairs}
        for cat, pairs in category_to_rootcauses.items()
    }

    canon_pairs = {(_norm(c), _norm(r)) for c, rcs in canonical.items() for r in rcs}
    disc_pairs = {(_norm(c), _norm(r)) for c, rcs in discovered.items() for r in rcs}

    missing = sorted(canon_pairs - disc_pairs)
    unexpected = sorted(disc_pairs - canon_pairs)

    n_canon_rc = len(canon_pairs)
    n_disc_rc = len(disc_pairs & canon_pairs)
    print(f"  Label space ({arch}): {n_disc_rc}/{n_canon_rc} taxonomy "
          f"root causes present across {len(discovered)} categories "
          f"(taxonomy: {len(canonical)})")

    if missing:
        print(f"  [warn] {len(missing)} taxonomy root cause(s) absent from "
              f"the data (benchmark not fully regenerated?): "
              f"{', '.join(f'{c}/{r}' for c, r in missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    if unexpected:
        print(f"  [warn] {len(unexpected)} root cause(s) in the data are "
              f"outside the taxonomy (drift?): "
              f"{', '.join(f'{c}/{r}' for c, r in unexpected[:8])}"
              + (" ..." if len(unexpected) > 8 else ""))

    return {
        "expected_root_causes": n_canon_rc,
        "discovered_root_causes": n_disc_rc,
        "missing": missing,
        "unexpected": unexpected,
    }


def load_data(arch):
    """Load CSV data and extract all label hierarchies.

    Detection labels are derived from the mutation testing 'killed' column
    in the origin dataset:
      - killed=1 (mutation detected by tests) -> faulty (1)
      - killed=0 (mutation survived) -> clean (0)
      - baseline (no mutation injected) -> clean (0)

    Category and root-cause labels only apply to killed (faulty) samples.
    """
    csv = DATA_DIR / f"{arch}_dataset.csv"
    df = pd.read_csv(csv)

    meta_cols = ["Identifier", "arch", "model_name", "dataset_name", "seed",
                 "is_faulty", "fault_category", "fault_subcategory", "layer_idx",
                 "severity_params", "label", "killed"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float32)
    feature_names = feature_cols
    groups = (df["model_name"].astype(str) + "__" +
              df["dataset_name"].astype(str)).values

    # Detection labels: killed=1 -> faulty, killed=0 or baseline -> clean
    # baseline rows have killed=NaN, treat as clean
    is_killed = df["killed"].fillna(0).astype(int) == 1
    y_detect = is_killed.astype(np.int64).values

    n_clean = (y_detect == 0).sum()
    n_faulty = (y_detect == 1).sum()
    print(f"  Labels: {n_clean} clean ({100*n_clean/len(df):.1f}%), "
          f"{n_faulty} faulty/killed ({100*n_faulty/len(df):.1f}%)")

    # Category labels: only for killed (faulty) samples; clean = -1
    faulty_categories = sorted(df.loc[is_killed, "fault_category"].unique())
    cat2idx = {c: i for i, c in enumerate(faulty_categories)}
    y_category = np.full(len(df), -1, dtype=np.int64)
    for i in range(len(df)):
        if is_killed.iloc[i]:
            y_category[i] = cat2idx[df.iloc[i]["fault_category"]]

    # Root-cause labels (fault_subcategory) — only meaningful for killed samples
    # Sentinel -1 for clean/invalid (never collides with valid RC index 0..N-1)
    killed_subcats = sorted(df.loc[is_killed, "fault_subcategory"].dropna().unique())
    rc2idx = {rc: i for i, rc in enumerate(killed_subcats)}
    y_rootcause = np.full(len(df), -1, dtype=np.int64)
    for i in range(len(df)):
        sc = df.iloc[i].get("fault_subcategory")
        if is_killed.iloc[i] and pd.notna(sc) and sc in rc2idx:
            y_rootcause[i] = rc2idx[sc]

    # Build category -> root-cause mapping (killed samples only)
    category_to_rootcauses = {}
    rootcause_local_labels = {}
    for cat_name in faulty_categories:
        mask = is_killed & (df["fault_category"] == cat_name)
        subcats = sorted(df.loc[mask, "fault_subcategory"].dropna().unique())
        global_idxs = [rc2idx[sc] for sc in subcats]
        category_to_rootcauses[cat_name] = list(zip(global_idxs, subcats))
        rootcause_local_labels[cat_name] = {gi: li for li, gi in enumerate(global_idxs)}

    label_space_report = validate_label_space(arch, category_to_rootcauses)

    return {
        "X": X,
        "groups": groups,
        "feature_names": feature_names,
        "y_detect": y_detect,
        "y_category": y_category,
        "y_rootcause": y_rootcause,
        "category_names": faulty_categories,
        "rootcause_names": killed_subcats,
        "category_to_rootcauses": category_to_rootcauses,
        "rootcause_local_labels": rootcause_local_labels,
        "n_categories": len(faulty_categories),
        "category_sizes": {cat: len(rcs) for cat, rcs in category_to_rootcauses.items()},
        "label_space_report": label_space_report,
    }


def build_adjacency(arch, group_names):
    """Build aligned FPG adjacency for the active feature groups."""
    fpg_groups, fpg_adj, _ = fundamental_to_feature_group_adjacency(arch)
    fpg_idx = {g: i for i, g in enumerate(fpg_groups)}
    n = len(group_names)
    adj = np.zeros((n, n), dtype=np.float32)
    for i, gi in enumerate(group_names):
        for j, gj in enumerate(group_names):
            if gi in fpg_idx and gj in fpg_idx:
                adj[i, j] = fpg_adj[fpg_idx[gi], fpg_idx[gj]]
    np.fill_diagonal(adj, 1.0)
    return adj


def build_model(arch, mode, input_dim, group_dims, group_names,
                n_categories, category_sizes, config, use_graph=True):
    """Construct HierarchicalDiagnosisModel."""
    adjacency = None
    if use_graph and mode == "graph_conditioned":
        adjacency = build_adjacency(arch, group_names)

    return HierarchicalDiagnosisModel(
        input_dim=input_dim if mode == "flat" else None,
        group_dims=group_dims,
        adjacency=adjacency,
        hidden_dim=config.get("group_hidden_dim", 32),
        embedding_dim=config.get("embedding_dim", 64),
        n_message_passing=config.get("n_message_passing", 2),
        dropout=config.get("dropout", 0.1),
        mode=mode,
        n_categories=n_categories,
        category_sizes=category_sizes,
        group_names=group_names,
    )


def refresh_prototypes(model, X_scaled, y_detect, y_category, y_rootcause,
                       group_indices, category_names, rootcause_local_labels):
    """Recompute prototype tensors from a scaled training split."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_scaled).float().to(device)
        _, h_groups = model.encode(xt, group_indices)
        model._prototypes = {}

        for cat_idx, cat_name in enumerate(category_names):
            cat_mask = (y_detect == 1) & (y_category == cat_idx)
            if cat_mask.sum() < 2 or cat_name not in rootcause_local_labels:
                continue

            local_map = rootcause_local_labels[cat_name]
            h_cat = h_groups[cat_mask]
            y_rc_cat = y_rootcause[cat_mask]
            valid = np.array([int(y_rc_cat[i]) in local_map for i in range(len(y_rc_cat))])
            if valid.sum() < 2:
                continue

            h_valid = h_cat[valid]
            y_local = torch.tensor(
                [local_map[int(y_rc_cat[i])] for i in range(len(y_rc_cat))
                 if int(y_rc_cat[i]) in local_map],
                dtype=torch.long,
                device=device,
            )
            model.compute_prototypes(h_valid, y_local, cat_name)


def compute_proto_stats(model, z, h_groups, y_detect, y_category, y_rootcause,
                        category_names, rootcause_local_labels, category_sizes,
                        cat_preds=None):
    """Evaluate Stage 3 with prototype matching and compute CE/prototype agreement."""
    faulty_idx = np.where(y_detect == 1)[0]
    if len(faulty_idx) == 0:
        return {
            "macro_f1": 0.0,
            "by_family": {},
            "acc_by_family": {},
            "f1w_by_family": {},
            "f1mi_by_family": {},
            "prec_by_family": {},
            "rec_by_family": {},
            "ce_proto_agreement": 0.0,
            "mean_margin": 0.0,
        }

    y_true_local = np.full(len(faulty_idx), -1, dtype=np.int64)
    y_pred_local = np.full(len(faulty_idx), -1, dtype=np.int64)
    ce_preds_local = np.full(len(faulty_idx), -1, dtype=np.int64)
    margins = []

    for i, fi in enumerate(faulty_idx):
        true_cat_idx = int(y_category[fi])
        true_cat_name = category_names[true_cat_idx]
        local_map = rootcause_local_labels.get(true_cat_name, {})

        rc_global = int(y_rootcause[fi])
        if rc_global in local_map:
            y_true_local[i] = local_map[rc_global]

        eval_cat_idx = true_cat_idx if cat_preds is None else int(cat_preds[i])
        if eval_cat_idx < 0 or eval_cat_idx >= len(category_names):
            continue
        if cat_preds is not None and eval_cat_idx != true_cat_idx:
            continue

        eval_cat_name = category_names[eval_cat_idx]
        proto_pred, distances, _ = model.diagnose_proto(h_groups[fi:fi + 1], eval_cat_name)
        if proto_pred is not None:
            y_pred_local[i] = int(proto_pred.item())
            if distances is not None and distances.shape[1] >= 2:
                d_sorted = torch.sort(distances[0]).values
                margins.append(float((d_sorted[1] - d_sorted[0]).item()))

        ce_logits = model.diagnose(z[fi:fi + 1], eval_cat_name)
        if ce_logits is not None:
            ce_preds_local[i] = int(ce_logits.argmax(-1).item())

    by_family = {}
    acc_by_family = {}
    f1w_by_family = {}
    f1mi_by_family = {}
    prec_by_family = {}
    rec_by_family = {}
    agreements = []
    for ci, cat_name in enumerate(category_names):
        true_cat_mask = np.array([y_category[fi] == ci for fi in faulty_idx])
        valid_mask = (y_true_local != -1) & true_cat_mask
        if valid_mask.sum() < 2 or len(np.unique(y_true_local[valid_mask])) < 2:
            continue

        valid_labels = list(range(category_sizes.get(cat_name, 0)))
        y_true_sub = y_true_local[valid_mask]
        y_pred_sub = y_pred_local[valid_mask]

        by_family[cat_name] = float(f1_score(
            y_true_sub,
            y_pred_sub,
            average="macro",
            zero_division=0,
            labels=valid_labels,
        ))
        acc_by_family[cat_name] = float(accuracy_score(y_true_sub, y_pred_sub))
        f1w_by_family[cat_name] = float(f1_score(y_true_sub, y_pred_sub, average="weighted", zero_division=0))
        f1mi_by_family[cat_name] = float(f1_score(y_true_sub, y_pred_sub, average="micro", zero_division=0))
        prec_by_family[cat_name] = float(precision_score(
            y_true_sub, y_pred_sub, average="macro", zero_division=0, labels=valid_labels))
        rec_by_family[cat_name] = float(recall_score(
            y_true_sub, y_pred_sub, average="macro", zero_division=0, labels=valid_labels))

        agree_mask = valid_mask & (ce_preds_local != -1) & (y_pred_local != -1)
        if agree_mask.sum() > 0:
            agreements.append(float(np.mean(ce_preds_local[agree_mask] == y_pred_local[agree_mask])))

    macro_f1 = float(np.mean(list(by_family.values()))) if by_family else 0.0
    ce_proto_agreement = float(np.mean(agreements)) if agreements else 0.0
    mean_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "macro_f1": macro_f1,
        "by_family": by_family,
        "acc_by_family": acc_by_family,
        "f1w_by_family": f1w_by_family,
        "f1mi_by_family": f1mi_by_family,
        "prec_by_family": prec_by_family,
        "rec_by_family": rec_by_family,
        "macro_acc": float(np.mean(list(acc_by_family.values()))) if acc_by_family else 0.0,
        "macro_f1_weighted": float(np.mean(list(f1w_by_family.values()))) if f1w_by_family else 0.0,
        "macro_f1_micro": float(np.mean(list(f1mi_by_family.values()))) if f1mi_by_family else 0.0,
        "macro_precision": float(np.mean(list(prec_by_family.values()))) if prec_by_family else 0.0,
        "macro_recall": float(np.mean(list(rec_by_family.values()))) if rec_by_family else 0.0,
        "ce_proto_agreement": ce_proto_agreement,
        "mean_margin": mean_margin,
    }


def compute_proto_utilization(model, h_groups, y_detect, y_category, y_rootcause,
                              category_names, rootcause_local_labels, category_sizes,
                              cat_preds=None):
    """Compute prototype usage entropy and active-count diagnostics."""
    faulty_idx = np.where(y_detect == 1)[0]
    entropies = []
    active_counts = []

    for ci, cat_name in enumerate(category_names):
        if cat_name not in rootcause_local_labels or cat_name not in model._prototypes:
            continue
        local_map = rootcause_local_labels[cat_name]
        valid_samples = []
        for i, fi in enumerate(faulty_idx):
            if int(y_category[fi]) != ci:
                continue
            if int(y_rootcause[fi]) not in local_map:
                continue
            if cat_preds is not None:
                eval_cat_idx = int(cat_preds[i])
                if eval_cat_idx != ci:
                    continue
            valid_samples.append(fi)
        if len(valid_samples) < 2:
            continue

        h_valid = h_groups[valid_samples]
        proto_pred, _, _ = model.diagnose_proto(h_valid, cat_name)
        if proto_pred is None:
            continue
        proto_pred = proto_pred.cpu().numpy()
        n_rc = int(category_sizes.get(cat_name, 0))
        if n_rc < 2:
            continue
        counts = np.bincount(proto_pred, minlength=n_rc).astype(np.float64)
        probs = counts / max(counts.sum(), 1.0)
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum() / np.log(n_rc))
        entropies.append(entropy)
        active_counts.append(float((counts > 0).sum()))

    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_active_count": float(np.mean(active_counts)) if active_counts else 0.0,
    }


def snapshot_prototypes(model):
    """Return flattened prototype tensors for plotting snapshots."""
    out = {}
    for cat_name, protos in model._prototypes.items():
        out[cat_name] = protos.detach().cpu().reshape(protos.shape[0], -1).numpy()
    return out


def collect_proto_eval_artifacts(model, h_groups, y_detect, y_category, y_rootcause,
                                 category_names, rootcause_local_labels, category_sizes,
                                 category_to_rootcauses, cat_preds, cat_probs=None):
    """Collect prototype-specific evaluation artifacts for plotting."""
    faulty_idx = np.where(y_detect == 1)[0]
    rng = np.random.RandomState(42)

    artifacts = {
        "assignment_matrices": {},
        "separation_matrices": {},
        "utilization_by_family": {},
        "group_ablation_faithfulness": {},
        "case_panels": [],
        "category_conditioned_routes": {},
        "rc_group_embeddings": {},
        "rc_group_labels": {},
        "prototype_snapshots_final": snapshot_prototypes(model),
    }

    for ci, cat_name in enumerate(category_names):
        if cat_name not in rootcause_local_labels or cat_name not in model._prototypes:
            continue
        local_map = rootcause_local_labels[cat_name]
        local_to_name = {}
        for gi, rc_name in category_to_rootcauses.get(cat_name, []):
            li = local_map.get(gi)
            if li is not None:
                local_to_name[li] = rc_name

        valid_pairs = []
        for i, fi in enumerate(faulty_idx):
            if int(y_category[fi]) != ci:
                continue
            if int(y_rootcause[fi]) not in local_map:
                continue
            if int(cat_preds[i]) != ci:
                continue
            valid_pairs.append((i, fi))
        if len(valid_pairs) < 2:
            continue

        faulty_positions = [i for i, _ in valid_pairs]
        sample_indices = [fi for _, fi in valid_pairs]
        h_valid = h_groups[sample_indices]
        y_local = np.array([local_map[int(y_rootcause[fi])] for fi in sample_indices], dtype=np.int64)
        proto_pred, distances, group_dists = model.diagnose_proto(h_valid, cat_name)
        if proto_pred is None or distances is None or group_dists is None:
            continue

        proto_pred_np = proto_pred.cpu().numpy()
        dist_np = distances.cpu().numpy()
        group_np = group_dists.cpu().numpy()
        n_rc = int(category_sizes.get(cat_name, 0))

        assignment = np.zeros((n_rc, n_rc), dtype=np.float32)
        for t, p in zip(y_local, proto_pred_np):
            assignment[t, p] += 1
        row_sums = assignment.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        artifacts["assignment_matrices"][cat_name] = assignment / row_sums

        protos = model._prototypes[cat_name].detach().cpu().reshape(n_rc, -1).numpy()
        diff = protos[:, None, :] - protos[None, :, :]
        artifacts["separation_matrices"][cat_name] = np.sqrt((diff ** 2).sum(axis=-1))

        counts = np.bincount(proto_pred_np, minlength=n_rc).astype(np.float64)
        probs = counts / max(counts.sum(), 1.0)
        nz = probs > 0
        entropy = float(-(probs[nz] * np.log(probs[nz])).sum() / np.log(max(n_rc, 2)))
        artifacts["utilization_by_family"][cat_name] = {
            "entropy": entropy,
            "active_count": float((counts > 0).sum()),
            "counts": counts.tolist(),
        }

        artifacts["rc_group_embeddings"][cat_name] = h_valid.detach().cpu().reshape(len(sample_indices), -1).numpy()
        artifacts["rc_group_labels"][cat_name] = y_local

        margins = []
        top_drops = []
        random_drops = []
        best_case = None
        n_groups = group_np.shape[-1]

        for sample_pos, faulty_pos in enumerate(faulty_positions):
            order = np.argsort(dist_np[sample_pos])
            winner = int(order[0])
            runnerup = int(order[1]) if len(order) > 1 else winner
            margin = float(dist_np[sample_pos, runnerup] - dist_np[sample_pos, winner]) if runnerup != winner else 0.0
            margins.append(margin)

            delta_g = group_np[sample_pos, runnerup] - group_np[sample_pos, winner]
            top_idx = np.argsort(-delta_g)[:min(2, n_groups)]
            rand_idx = rng.choice(np.arange(n_groups), size=len(top_idx), replace=False)

            h_top = h_valid[sample_pos:sample_pos + 1].clone()
            h_top[:, top_idx, :] = 0.0
            _, dist_top, _ = model.diagnose_proto(h_top, cat_name)
            if dist_top is not None and dist_top.shape[1] >= 2:
                d_sorted = torch.sort(dist_top[0]).values
                top_drops.append(margin - float((d_sorted[1] - d_sorted[0]).item()))

            h_rand = h_valid[sample_pos:sample_pos + 1].clone()
            h_rand[:, rand_idx, :] = 0.0
            _, dist_rand, _ = model.diagnose_proto(h_rand, cat_name)
            if dist_rand is not None and dist_rand.shape[1] >= 2:
                d_sorted = torch.sort(dist_rand[0]).values
                random_drops.append(margin - float((d_sorted[1] - d_sorted[0]).item()))

            prefer = (proto_pred_np[sample_pos] == y_local[sample_pos], margin)
            if best_case is None or prefer > best_case["score"]:
                contrib = group_np[sample_pos, winner]
                denom = max(float(contrib.sum()), 1e-12)
                contrib_frac = (contrib / denom).tolist()
                top3 = []
                for idx in order[:3]:
                    top3.append({
                        "root_cause_idx": int(idx),
                        "root_cause_name": local_to_name.get(int(idx), f"rc_{int(idx)}"),
                        "distance": float(dist_np[sample_pos, idx]),
                    })
                cat_top2 = []
                if cat_probs is not None and len(cat_probs) > faulty_pos:
                    probs_cat = cat_probs[faulty_pos]
                    cat_order = np.argsort(-probs_cat)[:2]
                    cat_top2 = [{
                        "category_idx": int(idx),
                        "category_name": category_names[int(idx)],
                        "prob": float(probs_cat[idx]),
                    } for idx in cat_order]
                best_case = {
                    "score": prefer,
                    "category_name": cat_name,
                    "sample_index": int(sample_indices[sample_pos]),
                    "true_root_cause_idx": int(y_local[sample_pos]),
                    "true_root_cause_name": local_to_name.get(int(y_local[sample_pos]), f"rc_{int(y_local[sample_pos])}"),
                    "pred_root_cause_idx": winner,
                    "pred_root_cause_name": local_to_name.get(winner, f"rc_{winner}"),
                    "runnerup_root_cause_idx": runnerup,
                    "runnerup_root_cause_name": local_to_name.get(runnerup, f"rc_{runnerup}"),
                    "margin": margin,
                    "top3": top3,
                    "category_top2": cat_top2,
                    "group_contributions": contrib_frac,
                    "group_margin_delta": delta_g.tolist(),
                }

        artifacts["group_ablation_faithfulness"][cat_name] = {
            "top_drop": float(np.mean(top_drops)) if top_drops else 0.0,
            "random_drop": float(np.mean(random_drops)) if random_drops else 0.0,
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
        }
        if best_case is not None:
            best_case.pop("score", None)
            artifacts["case_panels"].append(best_case)
            artifacts["category_conditioned_routes"][cat_name] = best_case

    return artifacts


def oversample_minority(X, y_detect, y_category, y_rootcause, seed=42):
    """Oversample clean (minority) class to ~50% of faulty class size.

    This brings the ratio from ~1:99 to roughly 1:2, enough for the
    detection head to learn meaningful decision boundaries without
    drowning in the majority class.
    """
    rng = np.random.RandomState(seed)
    clean_idx = np.where(y_detect == 0)[0]
    faulty_idx = np.where(y_detect == 1)[0]

    if len(clean_idx) == 0:
        return X, y_detect, y_category, y_rootcause

    # Target: clean samples = 50% of faulty count
    target_clean = max(len(faulty_idx) // 2, len(clean_idx))
    n_extra = target_clean - len(clean_idx)

    if n_extra > 0:
        extra_idx = rng.choice(clean_idx, size=n_extra, replace=True)
        all_idx = np.concatenate([np.arange(len(X)), extra_idx])
    else:
        all_idx = np.arange(len(X))

    # Shuffle
    rng.shuffle(all_idx)
    return X[all_idx], y_detect[all_idx], y_category[all_idx], y_rootcause[all_idx]


def split_train_val(X, y_detect, y_category, y_rootcause, val_fraction=0.2, seed=42):
    """Stratified train/val split within a CV fold.

    Stratifies on y_category (with clean as -1) to ensure all categories
    appear in both train and val.
    """
    # Use y_category for stratification (-1 = clean, 0..n = faulty categories)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, val_idx = next(sss.split(X, y_category))
    return (X[tr_idx], y_detect[tr_idx], y_category[tr_idx], y_rootcause[tr_idx],
            X[val_idx], y_detect[val_idx], y_category[val_idx], y_rootcause[val_idx])


def train_one_fold(model, X_tr, y_det_tr, y_cat_tr, y_rc_tr,
                   group_indices, category_names, rootcause_local_labels,
                   config, use_sep=True, val_split=None):
    """Train one fold with a train/val split and early stopping.

    When ``val_split`` is given as a ``(train_idx, val_idx)`` pair of index
    arrays into ``X_tr``, that split provides the early-stopping validation
    set. Otherwise an internal stratified 80/20 split is used.

    Returns (scaler fitted on full training data, training_curves dict).
    """
    device = next(model.parameters()).device

    # --- Resolve the train/val split for early stopping -------------------
    if val_split is not None:
        in_idx, val_idx = val_split
        X_t_raw, y_det_t_raw = X_tr[in_idx], y_det_tr[in_idx]
        y_cat_t_raw, y_rc_t_raw = y_cat_tr[in_idx], y_rc_tr[in_idx]
        X_v, y_det_v = X_tr[val_idx], y_det_tr[val_idx]
        y_cat_v, y_rc_v = y_cat_tr[val_idx], y_rc_tr[val_idx]
    else:
        (X_t_raw, y_det_t_raw, y_cat_t_raw, y_rc_t_raw,
         X_v, y_det_v, y_cat_v, y_rc_v) = split_train_val(
            X_tr, y_det_tr, y_cat_tr, y_rc_tr,
            val_fraction=0.2, seed=config.get("seed", 42))

    # --- Fit scaler on training portion, transform both -------------------
    scaler = StandardScaler()
    X_t_s_base = np.nan_to_num(scaler.fit_transform(X_t_raw), nan=0.0).astype(np.float32)
    X_v_s = np.nan_to_num(scaler.transform(X_v), nan=0.0).astype(np.float32)

    # --- Oversample clean class in TRAINING portion only ------------------
    X_t_s, y_det_t, y_cat_t, y_rc_t = oversample_minority(
        X_t_s_base, y_det_t_raw, y_cat_t_raw, y_rc_t_raw, seed=config.get("seed", 42))

    # --- Compute detection class weights from training portion ------------
    det_weights = compute_detection_weights(y_det_t)
    if det_weights is not None:
        det_weights = det_weights.to(device)

    # --- Training setup ---------------------------------------------------
    lr = config.get("lr", 1e-3)
    if isinstance(lr, str):
        lr = float(lr)

    opt = optim.Adam(model.parameters(), lr=lr,
                     weight_decay=config.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    epochs = config.get("epochs", 150)
    bs = config.get("batch_size", 256)
    patience = config.get("patience", 20)
    alpha = config.get("alpha", 1.0)
    lambda_rc = config.get("lambda_rc", 1.0)
    beta = config.get("beta", 0.5) if use_sep else 0.0
    gamma = config.get("gamma", 0.3) if use_sep else 0.0
    temperature = config.get("temperature", 0.1)

    best_metric, best_state, patience_ctr = -1.0, None, 0
    training_curves = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "val_metric": [], "val_det_f1": [], "val_cat_f1": [],
        "val_rc_proto_f1": [], "val_ce_proto_agreement": [], "val_proto_margin": [],
        "val_proto_utilization_entropy": [], "val_proto_active_count": [],
        "loss_detect": [], "loss_category": [], "loss_rootcause": [],
        "loss_contrastive": [], "loss_prototype": [], "loss_separation": [],
    }
    # Snapshot embeddings at intervals for t-SNE evolution plot
    embedding_snapshots = []
    snapshot_interval = max(epochs // 5, 10)

    for epoch in range(epochs):
        # --- Train step ---------------------------------------------------
        model.train()
        perm = np.random.permutation(len(X_t_s))
        ep_loss = 0.0
        ep_losses = {"detection": 0.0, "category": 0.0, "rootcause": 0.0,
                     "contrastive": 0.0, "prototype": 0.0, "separation": 0.0}
        n_batches = 0
        for start in range(0, len(X_t_s), bs):
            idx = perm[start:start + bs]
            xb = torch.from_numpy(X_t_s[idx]).float().to(device)
            y_det = torch.from_numpy(y_det_t[idx]).long().to(device)
            y_cat = torch.from_numpy(y_cat_t[idx]).long().to(device)
            y_rc = torch.from_numpy(y_rc_t[idx]).long().to(device)

            z, h_g = model.encode(xb, group_indices)
            loss, loss_dict = hierarchical_loss(
                model, z, h_g, y_det, y_cat, y_rc,
                category_names, rootcause_local_labels,
                group_indices=group_indices,
                alpha=alpha, lambda_rc=lambda_rc, beta=beta, gamma=gamma,
                temperature=temperature,
                detection_weights=det_weights,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += loss.item()
            for k in ep_losses:
                ep_losses[k] += loss_dict.get(k, 0.0)
            n_batches += 1

        # --- Validate on held-out val set every 5 epochs ------------------
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_v_s).float().to(device)
                z_v, h_v = model.encode(xv, group_indices)

                # Val loss (for overfitting detection)
                val_loss_val, val_loss_dict = hierarchical_loss(
                    model, z_v, h_v,
                    torch.from_numpy(y_det_v).long().to(device),
                    torch.from_numpy(y_cat_v).long().to(device),
                    torch.from_numpy(y_rc_v).long().to(device),
                    category_names, rootcause_local_labels,
                    alpha=alpha, lambda_rc=lambda_rc, beta=beta, gamma=gamma,
                    temperature=temperature,
                    detection_weights=det_weights,
                )

                # Detection F1 on val
                det_logits = model.detect(z_v)
                det_preds = det_logits.argmax(-1).cpu().numpy()
                det_f1 = f1_score(y_det_v, det_preds, average="binary",
                                  pos_label=1, zero_division=0)

                # Category F1 on val (faulty only)
                faulty_v = y_det_v == 1
                if faulty_v.sum() > 0:
                    cat_logits = model.categorize(z_v[faulty_v])
                    cat_preds = cat_logits.argmax(-1).cpu().numpy()
                    cat_f1 = f1_score(y_cat_v[faulty_v], cat_preds, average="macro")
                else:
                    cat_f1 = 0.0

                refresh_prototypes(
                    model, X_t_s_base, y_det_t_raw, y_cat_t_raw, y_rc_t_raw,
                    group_indices, category_names, rootcause_local_labels)
                proto_stats = compute_proto_stats(
                    model, z_v, h_v, y_det_v, y_cat_v, y_rc_v,
                    category_names, rootcause_local_labels, model.category_sizes,
                    cat_preds=cat_preds if faulty_v.sum() > 0 else None)
                proto_usage = compute_proto_utilization(
                    model, h_v, y_det_v, y_cat_v, y_rc_v,
                    category_names, rootcause_local_labels, model.category_sizes,
                    cat_preds=cat_preds if faulty_v.sum() > 0 else None)
                val_rc_proto_f1 = proto_stats["macro_f1"]
                val_ce_proto_agreement = proto_stats["ce_proto_agreement"]
                val_proto_margin = proto_stats["mean_margin"]
                val_proto_entropy = proto_usage["mean_entropy"]
                val_proto_active = proto_usage["mean_active_count"]

                val_metric = 0.2 * det_f1 + 0.3 * cat_f1 + 0.5 * val_rc_proto_f1

            # Log training curves (train vs val for overfitting detection)
            training_curves["epochs"].append(epoch + 1)
            training_curves["train_loss"].append(ep_loss / max(n_batches, 1))
            training_curves["val_loss"].append(val_loss_val.item())
            training_curves["val_metric"].append(val_metric)
            training_curves["val_det_f1"].append(det_f1)
            training_curves["val_cat_f1"].append(cat_f1)
            training_curves["val_rc_proto_f1"].append(val_rc_proto_f1)
            training_curves["val_ce_proto_agreement"].append(val_ce_proto_agreement)
            training_curves["val_proto_margin"].append(val_proto_margin)
            training_curves["val_proto_utilization_entropy"].append(val_proto_entropy)
            training_curves["val_proto_active_count"].append(val_proto_active)
            training_curves["loss_detect"].append(ep_losses["detection"] / max(n_batches, 1))
            training_curves["loss_category"].append(ep_losses["category"] / max(n_batches, 1))
            training_curves["loss_rootcause"].append(ep_losses["rootcause"] / max(n_batches, 1))
            training_curves["loss_contrastive"].append(ep_losses["contrastive"] / max(n_batches, 1))
            training_curves["loss_prototype"].append(ep_losses["prototype"] / max(n_batches, 1))
            training_curves["loss_separation"].append(ep_losses["separation"] / max(n_batches, 1))

            # Snapshot embeddings for t-SNE evolution
            if (epoch + 1) % snapshot_interval == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    # Use val set for consistent snapshots
                    faulty_mask_v = y_det_v == 1
                    if faulty_mask_v.sum() > 20:
                        snapshot = {
                            "epoch": epoch + 1,
                            "embeddings": z_v[faulty_mask_v].cpu().numpy(),
                            "categories": y_cat_v[faulty_mask_v],
                            "prototypes": snapshot_prototypes(model),
                        }
                        embedding_snapshots.append(snapshot)

            scheduler.step(-val_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= patience // 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Refit scaler on FULL training data (for test-time transform)
    scaler_full = StandardScaler()
    scaler_full.fit(np.nan_to_num(X_tr, nan=0.0))

    # Compute prototypes from training data for explainability
    X_tr_full = np.nan_to_num(scaler_full.transform(X_tr), nan=0.0).astype(np.float32)
    refresh_prototypes(
        model, X_tr_full, y_det_tr, y_cat_tr, y_rc_tr,
        group_indices, category_names, rootcause_local_labels)

    training_curves["embedding_snapshots"] = embedding_snapshots
    training_curves["best_val_metric"] = float(best_metric)
    return scaler_full, training_curves


def evaluate_one_fold(model, scaler, X_tr, X_te,
                      y_det_tr, y_det_te, y_cat_tr, y_cat_te,
                      y_rc_tr, y_rc_te, group_indices,
                      category_names, rootcause_local_labels, category_sizes,
                      category_to_rootcauses=None):
    """Full 3-stage evaluation on one fold.

    RC(pred) evaluation: samples whose predicted category is wrong are
    counted as ERRORS (F1=0 for those samples), not silently dropped.
    """
    device = next(model.parameters()).device

    X_te_s = np.nan_to_num(scaler.transform(X_te), nan=0.0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        xte = torch.from_numpy(X_te_s).float().to(device)
        z_te, h_groups_te = model.encode(xte, group_indices)

        # -- Stage 1: Detection --------------------------------------------
        det_logits = model.detect(z_te)
        det_probs = torch.softmax(det_logits, dim=-1)[:, 1].cpu().numpy()
        det_preds = det_logits.argmax(-1).cpu().numpy()

        auroc = float(roc_auc_score(y_det_te, det_probs)) if len(np.unique(y_det_te)) > 1 else 0.0
        det_f1 = float(f1_score(y_det_te, det_preds, average="binary", pos_label=1,
                                zero_division=0))
        det_acc = float(accuracy_score(y_det_te, det_preds))
        det_f1_weighted = float(f1_score(y_det_te, det_preds, average="weighted"))
        det_f1_micro = float(f1_score(y_det_te, det_preds, average="micro"))
        det_precision = float(precision_score(y_det_te, det_preds, average="binary",
                                              pos_label=1, zero_division=0))
        det_recall = float(recall_score(y_det_te, det_preds, average="binary",
                                        pos_label=1, zero_division=0))

        # -- Stage 2: Categorization (faulty samples only) -----------------
        faulty_te = y_det_te == 1
        if faulty_te.sum() > 0:
            cat_logits = model.categorize(z_te[faulty_te])
            cat_preds_te = cat_logits.argmax(-1).cpu().numpy()
            cat_probs_te = torch.softmax(cat_logits, dim=-1).cpu().numpy()
            y_cat_true_faulty = y_cat_te[faulty_te]
            cat_f1 = float(f1_score(y_cat_true_faulty, cat_preds_te,
                                     average="macro", zero_division=0))
            cat_acc = float(accuracy_score(y_cat_true_faulty, cat_preds_te))
            cat_f1_weighted = float(f1_score(y_cat_true_faulty, cat_preds_te,
                                              average="weighted", zero_division=0))
            cat_f1_micro = float(f1_score(y_cat_true_faulty, cat_preds_te,
                                           average="micro", zero_division=0))
            cat_precision = float(precision_score(y_cat_true_faulty, cat_preds_te,
                                                   average="macro", zero_division=0))
            cat_recall = float(recall_score(y_cat_true_faulty, cat_preds_te,
                                             average="macro", zero_division=0))
            per_cat_f1 = {}
            per_cat_prec = {}
            per_cat_rec = {}
            for ci, cn in enumerate(category_names):
                mask = y_cat_true_faulty == ci
                if mask.sum() > 0:
                    y_bin_true = (y_cat_true_faulty == ci).astype(int)
                    y_bin_pred = (cat_preds_te == ci).astype(int)
                    per_cat_f1[cn] = float(f1_score(
                        y_bin_true, y_bin_pred,
                        average="binary", zero_division=0))
                    per_cat_prec[cn] = float(precision_score(
                        y_bin_true, y_bin_pred,
                        average="binary", zero_division=0))
                    per_cat_rec[cn] = float(recall_score(
                        y_bin_true, y_bin_pred,
                        average="binary", zero_division=0))
        else:
            cat_f1 = 0.0
            cat_acc = 0.0
            cat_f1_weighted = 0.0
            cat_f1_micro = 0.0
            cat_precision = 0.0
            cat_recall = 0.0
            cat_preds_te = np.array([])
            cat_probs_te = np.zeros((0, len(category_names)), dtype=np.float32)
            per_cat_f1 = {}
            per_cat_prec = {}
            per_cat_rec = {}

        # -- Stage 3: Root-cause diagnosis with prototype head -------------
        proto_pred = compute_proto_stats(
            model, z_te, h_groups_te, y_det_te, y_cat_te, y_rc_te,
            category_names, rootcause_local_labels, category_sizes,
            cat_preds=cat_preds_te if faulty_te.sum() > 0 else None)

        rc_f1_pred_cat = proto_pred["by_family"]
        rc_acc_pred_cat = proto_pred["acc_by_family"]
        rc_f1w_pred_cat = proto_pred["f1w_by_family"]
        rc_f1mi_pred_cat = proto_pred["f1mi_by_family"]
        rc_prec_pred_cat = proto_pred["prec_by_family"]
        rc_rec_pred_cat = proto_pred["rec_by_family"]
        rc_f1_pred_macro = proto_pred["macro_f1"]
        rc_acc_pred_macro = proto_pred["macro_acc"]
        rc_f1w_pred_macro = proto_pred["macro_f1_weighted"]
        rc_f1mi_pred_macro = proto_pred["macro_f1_micro"]
        rc_prec_pred_macro = proto_pred["macro_precision"]
        rc_rec_pred_macro = proto_pred["macro_recall"]

        # Per-root-cause metrics under the predicted-category route
        per_rc_predicted_route_metrics = {}
        if faulty_te.sum() > 0:
            faulty_indices = np.where(faulty_te)[0]
        else:
            faulty_indices = np.array([], dtype=np.int64)
        for ci, cat_name in enumerate(category_names):
            true_cat_mask = np.array([y_cat_te[fi] == ci for fi in faulty_indices])
            if true_cat_mask.sum() < 2 or cat_name not in rootcause_local_labels:
                continue
            local_map = rootcause_local_labels[cat_name]
            valid_indices = []
            for local_pos, fi in enumerate(faulty_indices):
                if not true_cat_mask[local_pos]:
                    continue
                if cat_preds_te[local_pos] != ci:
                    continue
                if int(y_rc_te[fi]) not in local_map:
                    continue
                valid_indices.append(fi)
            if len(valid_indices) < 2:
                continue

            h_valid = h_groups_te[valid_indices]
            y_local = np.array([local_map[int(y_rc_te[fi])] for fi in valid_indices])
            proto_pred_local, _, _ = model.diagnose_proto(h_valid, cat_name)
            if proto_pred_local is None:
                continue
            rc_preds = proto_pred_local.cpu().numpy()
            n_rc_classes = category_sizes.get(cat_name, 0)
            valid_labels = list(range(n_rc_classes))
            if len(np.unique(y_local)) < 2:
                continue

            per_rc_f1 = f1_score(y_local, rc_preds, average=None,
                                 zero_division=0, labels=valid_labels)
            per_rc_prec = precision_score(y_local, rc_preds, average=None,
                                          zero_division=0, labels=valid_labels)
            per_rc_rec = recall_score(y_local, rc_preds, average=None,
                                      zero_division=0, labels=valid_labels)
            local_to_name = {}
            for gi, sname in category_to_rootcauses.get(cat_name, []):
                li = local_map.get(gi)
                if li is not None:
                    local_to_name[li] = sname
            for local_idx in valid_labels:
                rc_name = local_to_name.get(local_idx, f"rc_{local_idx}")
                rc_key = f"{cat_name}/{rc_name}"
                per_rc_predicted_route_metrics[rc_key] = {
                    "family": cat_name,
                    "root_cause": rc_name,
                    "f1": float(per_rc_f1[local_idx]),
                    "precision": float(per_rc_prec[local_idx]),
                    "recall": float(per_rc_rec[local_idx]),
                }

    # -- Collect plot artifacts + explanations ────────────────────────────
    plot_data = {
        "embeddings": z_te.cpu().numpy(),
        "y_detect_true": y_det_te,
        "y_detect_pred": det_preds,
        "y_detect_score": det_probs,
        "y_category_true": y_cat_te,
        "y_category_pred": np.full(len(y_cat_te), -1, dtype=np.int64),
        "faulty_mask": faulty_te,
        "group_names": model.group_names,
        "stage3_ce_proto_agreement_predicted": proto_pred["ce_proto_agreement"],
        "stage3_proto_margin_predicted": proto_pred["mean_margin"],
    }
    if faulty_te.sum() > 0:
        faulty_idx_arr = np.where(faulty_te)[0]
        for i, fi in enumerate(faulty_idx_arr):
            plot_data["y_category_pred"][fi] = cat_preds_te[i]

        proto_artifacts = collect_proto_eval_artifacts(
            model, h_groups_te, y_det_te, y_cat_te, y_rc_te,
            category_names, rootcause_local_labels, category_sizes,
            category_to_rootcauses or {}, cat_preds_te, cat_probs=cat_probs_te)
        plot_data.update(proto_artifacts)

        # Per-category root-cause embeddings + explanations
        plot_data["rc_embeddings"] = {}
        plot_data["rc_labels"] = {}
        plot_data["explanations"] = {}
        for ci, cat_name in enumerate(category_names):
            cat_mask = faulty_te & (y_cat_te == ci)
            if cat_mask.sum() < 5:
                continue
            if cat_name not in rootcause_local_labels:
                continue
            local_map = rootcause_local_labels[cat_name]
            z_cat = z_te[cat_mask].cpu().numpy()
            h_cat = h_groups_te[cat_mask]
            y_rc_cat = y_rc_te[cat_mask]
            valid = np.array([int(y_rc_cat[j]) in local_map for j in range(len(y_rc_cat))])
            if valid.sum() < 5:
                continue
            y_local = np.array([local_map[int(y_rc_cat[j])]
                                for j in range(len(y_rc_cat)) if int(y_rc_cat[j]) in local_map])
            plot_data["rc_embeddings"][cat_name] = z_cat[valid]
            plot_data["rc_labels"][cat_name] = y_local

            # Generate explanations via prototype distance decomposition
            h_valid = h_cat[valid]
            expls = model.explain_diagnosis(h_valid, cat_name)
            if expls is not None:
                plot_data["explanations"][cat_name] = expls

    return {
        "detection_auroc": round(auroc, 4),
        "detection_f1": round(det_f1, 4),
        "detection_acc": round(det_acc, 4),
        "detection_f1_weighted": round(det_f1_weighted, 4),
        "detection_f1_micro": round(det_f1_micro, 4),
        "detection_precision": round(det_precision, 4),
        "detection_recall": round(det_recall, 4),
        "category_f1": round(cat_f1, 4),
        "category_acc": round(cat_acc, 4),
        "category_f1_weighted": round(cat_f1_weighted, 4),
        "category_f1_micro": round(cat_f1_micro, 4),
        "category_precision": round(cat_precision, 4),
        "category_recall": round(cat_recall, 4),
        "per_category_f1": {k: round(v, 4) for k, v in per_cat_f1.items()},
        "per_category_precision": {k: round(v, 4) for k, v in per_cat_prec.items()},
        "per_category_recall": {k: round(v, 4) for k, v in per_cat_rec.items()},
        "per_rootcause_predicted_route_metrics": {
            k: {mk: round(mv, 4) if isinstance(mv, float) else mv for mk, mv in v.items()}
            for k, v in per_rc_predicted_route_metrics.items()
        },
        "rc_f1_predicted_category": {k: round(v, 4) for k, v in rc_f1_pred_cat.items()},
        "rc_f1_predicted_macro": round(rc_f1_pred_macro, 4),
        "rc_acc_predicted_macro": round(rc_acc_pred_macro, 4),
        "rc_f1w_predicted_macro": round(rc_f1w_pred_macro, 4),
        "rc_f1mi_predicted_macro": round(rc_f1mi_pred_macro, 4),
        "rc_prec_predicted_macro": round(rc_prec_pred_macro, 4),
        "rc_rec_predicted_macro": round(rc_rec_pred_macro, 4),
        "rc_acc_predicted_by_family": {k: round(v, 4) for k, v in rc_acc_pred_cat.items()},
        "rc_f1w_predicted_by_family": {k: round(v, 4) for k, v in rc_f1w_pred_cat.items()},
        "rc_f1mi_predicted_by_family": {k: round(v, 4) for k, v in rc_f1mi_pred_cat.items()},
        "rc_prec_predicted_by_family": {k: round(v, 4) for k, v in rc_prec_pred_cat.items()},
        "rc_rec_predicted_by_family": {k: round(v, 4) for k, v in rc_rec_pred_cat.items()},
        "rc_ce_proto_agreement_predicted_macro": round(proto_pred["ce_proto_agreement"], 4),
        "rc_proto_margin_predicted_macro": round(proto_pred["mean_margin"], 4),
        "plot_data": plot_data,
    }


def run_experiment(arch, use_graph=True, use_sep=True, config=None):
    """Run a full CV experiment for one architecture.

    Args:
        arch:      "encoder" or "decoder".
        use_graph: if False, disable FPG message passing (flat encoder).
        use_sep:   if False, disable the separation loss (beta = gamma = 0).
        config:    training and model config dict.
    """
    config = config or {}
    variant = []
    if use_graph:
        variant.append("graph")
    if use_sep:
        variant.append("sep")
    variant_str = "+".join(variant) if variant else "basic"
    mode = "graph_conditioned" if use_graph else "flat"

    print(f"\n{'='*70}")
    print(f"HIERARCHICAL DIAGNOSIS: {arch.upper()} | variant={variant_str} | mode={mode}")
    print(f"{'='*70}")

    data = load_data(arch)
    X = data["X"]
    groups = data["groups"]
    feature_names = data["feature_names"]
    y_detect = data["y_detect"]
    y_category = data["y_category"]
    y_rootcause = data["y_rootcause"]

    n_outer = config.get("n_outer_folds", 5)
    n_inner = config.get("n_inner_folds", 5)
    outer_cv = GroupKFold(n_splits=n_outer)
    fold_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_model(X_tr_in, y_det_in, y_cat_in, y_rc_in, g_dims, g_names, g_idx,
                   val_split=None):
        """Build and train one model on the given training portion."""
        model = build_model(
            arch, mode, X_tr_in.shape[1], g_dims, g_names,
            data["n_categories"], data["category_sizes"],
            config, use_graph=use_graph).to(device)
        scaler, curves = train_one_fold(
            model, X_tr_in, y_det_in, y_cat_in, y_rc_in,
            g_idx, data["category_names"], data["rootcause_local_labels"],
            config, use_sep=use_sep, val_split=val_split)
        return model, scaler, curves

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y_detect, groups)):
        set_seed(42 + fold_idx)
        t0 = time.time()

        # Feature processing is fit on the outer training portion only.
        X_tr, X_te, feat_names_proc, g_idx, proc_log = apply_processing_in_fold(
            X[tr_idx], X[te_idx], feature_names, y_detect[tr_idx], arch)

        g_dims = {g: len(i) for g, i in g_idx.items()}
        g_names = sorted(g_dims)

        y_det_tr, y_cat_tr, y_rc_tr = (y_detect[tr_idx], y_category[tr_idx],
                                       y_rootcause[tr_idx])
        inner_groups = groups[tr_idx]

        # --- Inner loop: StratifiedGroupKFold for model selection ----------
        inner_cv = StratifiedGroupKFold(n_splits=n_inner)
        # Stratify on category with clean folded into a single stratum so
        # every category appears across inner folds.
        strat = np.where(y_det_tr == 1, y_cat_tr + 1, 0)
        best_inner = None
        for inner_idx, (in_tr, in_val) in enumerate(
                inner_cv.split(X_tr, strat, inner_groups)):
            model, scaler, curves = _fit_model(
                X_tr, y_det_tr, y_cat_tr, y_rc_tr,
                g_dims, g_names, g_idx, val_split=(in_tr, in_val))
            score = curves.get("best_val_metric", -1.0)
            if best_inner is None or score > best_inner["score"]:
                best_inner = {"score": score, "model": model,
                              "scaler": scaler, "curves": curves}

        # Selected model from inner model selection, scored on the outer
        # held-out model--task pairs.
        model = best_inner["model"]
        scaler = best_inner["scaler"]
        fold_curves = best_inner["curves"]

        fold_res = evaluate_one_fold(
            model, scaler, X_tr, X_te,
            y_det_tr, y_detect[te_idx],
            y_cat_tr, y_category[te_idx],
            y_rc_tr, y_rootcause[te_idx],
            g_idx, data["category_names"],
            data["rootcause_local_labels"], data["category_sizes"],
            data["category_to_rootcauses"])

        fold_res["training_curves"] = fold_curves
        fold_results.append(fold_res)
        elapsed = time.time() - t0
        print(f"  Fold {fold_idx+1}/{n_outer}: "
              f"Det={fold_res['detection_f1']:.4f}  "
              f"Cat={fold_res['category_f1']:.4f}  "
              f"RC(pred)={fold_res['rc_f1_predicted_macro']:.4f}  "
              f"[{elapsed:.1f}s]")

    # Aggregate
    def agg(key):
        vals = [r[key] for r in fold_results]
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)}

    per_cat_agg = {}
    per_cat_prec_agg = {}
    per_cat_rec_agg = {}
    for cn in data["category_names"]:
        vals = [r["per_category_f1"].get(cn, 0.0) for r in fold_results]
        per_cat_agg[cn] = round(float(np.mean(vals)), 4)
        vals_p = [r["per_category_precision"].get(cn, 0.0) for r in fold_results]
        per_cat_prec_agg[cn] = round(float(np.mean(vals_p)), 4)
        vals_r = [r["per_category_recall"].get(cn, 0.0) for r in fold_results]
        per_cat_rec_agg[cn] = round(float(np.mean(vals_r)), 4)

    rc_pred_agg = {}
    for cn in data["category_names"]:
        vals_p = [r["rc_f1_predicted_category"].get(cn, 0.0) for r in fold_results]
        rc_pred_agg[cn] = round(float(np.mean(vals_p)), 4)

    # Collect per-fold plot data and training curves (for plotting)
    fold_plot_data = [r.get("plot_data") for r in fold_results]
    fold_training_curves = [r.get("training_curves") for r in fold_results]

    # Aggregate Stage 3 additional metrics by family
    rc_pred_acc_agg, rc_pred_f1w_agg, rc_pred_f1mi_agg = {}, {}, {}
    rc_pred_prec_agg, rc_pred_rec_agg = {}, {}
    for cn in data["category_names"]:
        for d, key in [(rc_pred_acc_agg, "rc_acc_predicted_by_family"),
                       (rc_pred_f1w_agg, "rc_f1w_predicted_by_family"),
                       (rc_pred_f1mi_agg, "rc_f1mi_predicted_by_family"),
                       (rc_pred_prec_agg, "rc_prec_predicted_by_family"),
                       (rc_pred_rec_agg, "rc_rec_predicted_by_family")]:
            vals = [r.get(key, {}).get(cn, 0.0) for r in fold_results]
            d[cn] = round(float(np.mean(vals)), 4)

    # Aggregate per-root-cause metrics across folds
    all_rc_keys = set()
    for r in fold_results:
        all_rc_keys.update(r.get("per_rootcause_predicted_route_metrics", {}).keys())
    per_rc_agg = {}
    for rc_key in sorted(all_rc_keys):
        f1_vals = [r.get("per_rootcause_predicted_route_metrics", {}).get(rc_key, {}).get("f1", 0.0)
                    for r in fold_results]
        prec_vals = [r.get("per_rootcause_predicted_route_metrics", {}).get(rc_key, {}).get("precision", 0.0)
                      for r in fold_results]
        rec_vals = [r.get("per_rootcause_predicted_route_metrics", {}).get(rc_key, {}).get("recall", 0.0)
                     for r in fold_results]
        sample_entry = None
        for r in fold_results:
            entry = r.get("per_rootcause_predicted_route_metrics", {}).get(rc_key)
            if entry:
                sample_entry = entry
                break
        per_rc_agg[rc_key] = {
            "family": sample_entry.get("family", "") if sample_entry else "",
            "root_cause": sample_entry.get("root_cause", "") if sample_entry else "",
            "f1": round(float(np.mean(f1_vals)), 4),
            "precision": round(float(np.mean(prec_vals)), 4),
            "recall": round(float(np.mean(rec_vals)), 4),
        }

    summary = {
        "arch": arch,
        "variant": variant_str,
        "mode": mode,
        "use_graph": use_graph,
        "use_sep": use_sep,
        "n_folds": 5,
        # Stage 1
        "stage1_detection_auroc": agg("detection_auroc"),
        "stage1_detection_f1": agg("detection_f1"),
        "stage1_detection_acc": agg("detection_acc"),
        "stage1_detection_f1_weighted": agg("detection_f1_weighted"),
        "stage1_detection_f1_micro": agg("detection_f1_micro"),
        "stage1_detection_precision": agg("detection_precision"),
        "stage1_detection_recall": agg("detection_recall"),
        # Stage 2
        "stage2_category_f1": agg("category_f1"),
        "stage2_category_acc": agg("category_acc"),
        "stage2_category_f1_weighted": agg("category_f1_weighted"),
        "stage2_category_f1_micro": agg("category_f1_micro"),
        "stage2_category_precision": agg("category_precision"),
        "stage2_category_recall": agg("category_recall"),
        "stage2_per_category_f1": per_cat_agg,
        "stage2_per_category_precision": per_cat_prec_agg,
        "stage2_per_category_recall": per_cat_rec_agg,
        # Stage 3 predicted-category route
        "stage3_rc_pred_cat_macro": agg("rc_f1_predicted_macro"),
        "stage3_rc_pred_cat_acc": agg("rc_acc_predicted_macro"),
        "stage3_rc_pred_cat_f1_weighted": agg("rc_f1w_predicted_macro"),
        "stage3_rc_pred_cat_f1_micro": agg("rc_f1mi_predicted_macro"),
        "stage3_rc_pred_cat_precision": agg("rc_prec_predicted_macro"),
        "stage3_rc_pred_cat_recall": agg("rc_rec_predicted_macro"),
        "stage3_rc_pred_cat_by_family": rc_pred_agg,
        "stage3_rc_pred_acc_by_family": rc_pred_acc_agg,
        "stage3_rc_pred_f1w_by_family": rc_pred_f1w_agg,
        "stage3_rc_pred_f1mi_by_family": rc_pred_f1mi_agg,
        "stage3_rc_pred_prec_by_family": rc_pred_prec_agg,
        "stage3_rc_pred_rec_by_family": rc_pred_rec_agg,
        # Per-root-cause (subcategory) breakdown under predicted-category routing
        "stage3_per_rootcause_predicted_route": per_rc_agg,
        # Meta
        "fold_plot_data": fold_plot_data,
        "fold_training_curves": fold_training_curves,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  SUMMARY {arch.upper()} [{variant_str}]:")
    print(f"    Stage 1 AUROC       = {summary['stage1_detection_auroc']['mean']:.4f} "
          f"+/- {summary['stage1_detection_auroc']['std']:.4f}")
    print(f"    Stage 1 F1          = {summary['stage1_detection_f1']['mean']:.4f} "
          f"+/- {summary['stage1_detection_f1']['std']:.4f}")
    print(f"    Stage 2 Cat F1      = {summary['stage2_category_f1']['mean']:.4f} "
          f"+/- {summary['stage2_category_f1']['std']:.4f}")
    print(f"    Stage 3 RC (pred)   = {summary['stage3_rc_pred_cat_macro']['mean']:.4f} "
          f"+/- {summary['stage3_rc_pred_cat_macro']['std']:.4f}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Hierarchical Fault Diagnosis Training")
    ap.add_argument("--arch", choices=["encoder", "decoder", "both"], default="both")
    ap.add_argument("--no-graph", action="store_true",
                    help="Disable FPG message passing")
    ap.add_argument("--no-sep", action="store_true",
                    help="Disable the separation loss (sets beta = gamma = 0)")
    ap.add_argument("--output", default=None, help="Override output directory")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="Category loss weight")
    ap.add_argument("--lambda-rc", type=float, default=1.0,
                    help="Root-cause cross-entropy loss weight")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Contrastive term weight inside L_sep")
    ap.add_argument("--gamma", type=float, default=0.3,
                    help="Prototype-matching term weight inside L_sep")
    args = ap.parse_args()

    import yaml
    config_path = CONFIGS_ROOT / "base.yaml"
    with open(config_path) as f:
        base = yaml.safe_load(f)
    training_cfg = base.get("training", {}).copy()

    config = {
        **base.get("model", {}),
        **training_cfg,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": training_cfg.get("patience", 20),
        "alpha": args.alpha,
        "lambda_rc": args.lambda_rc,
        "beta": args.beta,
        "gamma": args.gamma,
        "temperature": training_cfg.get("temperature",
                                        base.get("model", {}).get("temperature", 0.1)),
        "seed": 42,
    }

    use_graph = not args.no_graph
    use_sep = not args.no_sep

    variant_parts = []
    if use_graph:
        variant_parts.append("graph")
    if use_sep:
        variant_parts.append("sep")
    variant_name = "_".join(variant_parts) if variant_parts else "basic"

    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    archs = ["encoder", "decoder"] if args.arch == "both" else [args.arch]
    all_results = {}

    for arch in archs:
        summary = run_experiment(arch, use_graph=use_graph,
                                 use_sep=use_sep, config=config)
        all_results[arch] = summary

        # Strip non-serializable data (numpy arrays in plot_data)
        skip_keys = {"plot_data", "training_curves", "fold_plot_data", "fold_training_curves"}
        summary_json = {k: v for k, v in summary.items() if k not in skip_keys}
        out_file = output_dir / f"{arch}_{variant_name}.json"
        with open(out_file, "w") as f:
            json.dump(summary_json, f, indent=2)
        print(f"  Saved -> {out_file}")

    # Update status
    status_path = RESULTS_ROOT / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["hierarchical_experiment"] = {
        "variant": variant_name,
        "complete": True,
        "results": {
            arch: {
                "stage1_auroc": r["stage1_detection_auroc"]["mean"],
                "stage2_cat_f1": r["stage2_category_f1"]["mean"],
                "stage3_rc_pred": r["stage3_rc_pred_cat_macro"]["mean"],
            }
            for arch, r in all_results.items()
        },
        "timestamp": datetime.now().isoformat(),
    }
    status_path.write_text(json.dumps(status, indent=2))
    print("Status updated.")


if __name__ == "__main__":
    main()
