"""Baseline comparison for hierarchical fault diagnosis.

Implements adapted versions of prior techniques on the same data and CV splits:

Learning-based techniques (evaluated on all 3 stages):
  1. DEFault-style   — Hierarchical Random Forest (detection → per-family binary
                        RF → per-family root-cause RF)
  2. DeepFD-style    — Flat KNN + Decision Tree + Random Forest with majority voting

Rule-based techniques (evaluated on Stage 1 detection only):
  3. AutoTrainer-style — 5 threshold rules for training symptoms
  4. DeepDiagnosis-style — 8 symptom detection rules

All baselines use the same GroupKFold(5) splits as the proposed method.

Usage:
    python baselines/run_baselines.py --arch both
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
_CODE_ROOT = _ROOT / "defaultplusplus"
sys.path.insert(0, str(_CODE_ROOT))
from src.data.feature_processor import apply_processing_in_fold  # noqa: E402
from repo_paths import DATA_ROOT, RESULTS_ROOT  # noqa: E402

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = DATA_ROOT
ORIGIN_DIR = DATA_ROOT
RESULTS_DIR = RESULTS_ROOT / "baselines"


def _default_feature_mask(feature_names):
    """Select features matching DEFault's 23 dynamic feature categories.

    DEFault monitors: loss, val_loss, train_acc, val_acc, weight stats
    (large_weight, cons_mean/std_weight, nan_weight), accuracy/loss trends,
    activation stats (dying_relu, saturated_activation), gradient stats
    (vanish, explode, nan_gradients), learning rate, hardware utilisation.

    We select our features that correspond to these categories.
    Transformer-specific features (attention patterns, positional encoding,
    logit distribution, residual stream, cache diagnostics) are excluded.
    """
    patterns = [
        "loss", "accuracy", "acc", "ece",           # training metrics
        "grad",                                       # gradient statistics
        "update_ratio",                               # weight update dynamics
        "step_time", "peak_mem",                      # hardware/timing
    ]
    mask = []
    for fn in feature_names:
        fn_lower = fn.lower()
        mask.append(any(p in fn_lower for p in patterns))
    return np.array(mask)


def _deepfd_feature_mask(feature_names):
    """Select features matching DeepFD's 20 runtime data traces.

    DeepFD monitors: loss, acc, loss_val, acc_val, nan counts, large_weight,
    decrease_acc, increase_loss, cons_mean/std_weight, gap_train_test,
    test_turn_bad, slow_converge, oscillating_loss, dying_relu,
    gradient_vanish, gradient_explosion.

    Processed through 8 statistical operators (max, min, median, mean,
    var, std, skew, sem). We select our features that correspond to these.
    """
    patterns = [
        "loss", "accuracy", "acc",                    # loss, accuracy
        "grad",                                       # gradient statistics
        "update_ratio",                               # weight dynamics proxy
        "step_time", "peak_mem",                      # timing/hardware
    ]
    mask = []
    for fn in feature_names:
        fn_lower = fn.lower()
        mask.append(any(p in fn_lower for p in patterns))
    return np.array(mask)


def load_data(arch):
    """Load data — identical to hierarchical experiment for fair comparison."""
    csv = DATA_DIR / f"{arch}_v1_killed_binary.csv"
    df = pd.read_csv(csv)
    origin_csv = ORIGIN_DIR / f"{arch}_absolute_filled_labeled.csv"
    df_origin = pd.read_csv(origin_csv, usecols=["Identifier", "killed"])
    df = df.merge(df_origin, on="Identifier", how="left")

    meta_cols = ["Identifier", "arch", "model_name", "dataset_name", "seed",
                 "is_faulty", "fault_category", "fault_subcategory", "layer_idx",
                 "severity_params", "label", "killed"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values.astype(np.float32)
    feature_names = feature_cols
    groups = (df["model_name"].astype(str) + "__" +
              df["dataset_name"].astype(str) + "__" +
              df["seed"].astype(str)).values

    is_killed = df["killed"].fillna(0).astype(int) == 1
    y_detect = is_killed.astype(np.int64).values

    faulty_categories = sorted(df.loc[is_killed, "fault_category"].unique())
    cat2idx = {c: i for i, c in enumerate(faulty_categories)}
    y_category = np.full(len(df), -1, dtype=np.int64)
    for i in range(len(df)):
        if is_killed.iloc[i]:
            y_category[i] = cat2idx[df.iloc[i]["fault_category"]]

    killed_subcats = sorted(df.loc[is_killed, "fault_subcategory"].dropna().unique())
    rc2idx = {rc: i for i, rc in enumerate(killed_subcats)}
    y_rootcause = np.full(len(df), -1, dtype=np.int64)
    for i in range(len(df)):
        sc = df.iloc[i].get("fault_subcategory")
        if is_killed.iloc[i] and pd.notna(sc) and sc in rc2idx:
            y_rootcause[i] = rc2idx[sc]

    category_to_rootcauses = {}
    rootcause_local_labels = {}
    for cat_name in faulty_categories:
        mask = is_killed & (df["fault_category"] == cat_name)
        subcats = sorted(df.loc[mask, "fault_subcategory"].dropna().unique())
        global_idxs = [rc2idx[sc] for sc in subcats]
        category_to_rootcauses[cat_name] = list(zip(global_idxs, subcats))
        rootcause_local_labels[cat_name] = {gi: li for li, gi in enumerate(global_idxs)}

    print(f"  {arch}: {len(df)} samples, {(y_detect==1).sum()} faulty, "
          f"{len(faulty_categories)} families, {len(killed_subcats)} root causes")

    return {
        "X": X, "groups": groups, "feature_names": feature_names,
        "y_detect": y_detect, "y_category": y_category,
        "y_rootcause": y_rootcause,
        "category_names": faulty_categories,
        "rootcause_local_labels": rootcause_local_labels,
        "category_sizes": {cat: len(rcs) for cat, rcs in category_to_rootcauses.items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# RULE-BASED TECHNIQUES (Stage 1 detection only)
# ═══════════════════════════════════════════════════════════════════════

def _find_feature(feature_names, *patterns):
    """Find feature index matching any of the patterns."""
    for p in patterns:
        for i, fn in enumerate(feature_names):
            if p in fn.lower():
                return i
    return None


def autotrainer_style_detection(X, feature_names):
    """AutoTrainer-style: 5 threshold rules for training symptoms.

    Adapted rules using published thresholds mapped to available features:
      VG: gradient magnitude < threshold → vanishing gradient
      EG: gradient magnitude > threshold or NaN → exploding gradient
      DR: gradient zero ratio > 0.7 → dying ReLU
      OL: loss slope oscillates (changes sign) → oscillating loss
      SC: accuracy slope near zero → slow convergence
    """
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)  # 0=clean, 1=faulty

    # Find relevant feature indices
    fi_grad_max = _find_feature(feature_names, "grad_abs_max_final", "grad_abs_max_mid")
    fi_grad_zero = _find_feature(feature_names, "grad_zero_ratio_early_mean",
                                  "grad_zero_ratio_final")
    fi_loss_slope = _find_feature(feature_names, "loss_mid_slope", "loss_early_slope")
    fi_acc_slope = _find_feature(feature_names, "accuracy_mid_slope",
                                  "accuracy_early_slope")
    fi_acc = _find_feature(feature_names, "accuracy_final", "accuracy_mid_mean")

    for i in range(n):
        detected = False

        # VG: vanishing gradient — gradient magnitude very small
        if fi_grad_max is not None:
            val = X[i, fi_grad_max]
            if not np.isnan(val) and val < 1e-4:
                detected = True

        # EG: exploding gradient — gradient magnitude very large or NaN
        if fi_grad_max is not None:
            val = X[i, fi_grad_max]
            if np.isnan(val) or val > 70:
                detected = True

        # DR: dying ReLU — high fraction of zero gradients
        if fi_grad_zero is not None:
            val = X[i, fi_grad_zero]
            if not np.isnan(val) and val > 0.7:
                detected = True

        # OL: oscillating loss — loss slope changes sign (proxy: large magnitude)
        if fi_loss_slope is not None:
            val = X[i, fi_loss_slope]
            if not np.isnan(val) and abs(val) > 0.5:
                detected = True

        # SC: slow convergence — accuracy slope near zero while accuracy is low
        if fi_acc_slope is not None and fi_acc is not None:
            slope = X[i, fi_acc_slope]
            acc = X[i, fi_acc]
            if (not np.isnan(slope) and not np.isnan(acc) and
                    abs(slope) < 0.01 and acc < 0.6):
                detected = True

        preds[i] = 1 if detected else 0

    return preds


def deepdiagnosis_style_detection(X, feature_names):
    """DeepDiagnosis-style: 8 symptom detection rules.

    Adapted rules using published thresholds mapped to available features:
      S1: Numerical errors — NaN in any feature
      S2: Unchanged weight — update ratio near zero
      S3: Saturated activation — (no direct feature, skip)
      S4: Dead node — gradient zero ratio > 0.7
      S5: Out of range — (no direct feature, skip)
      S6: Loss not decreasing — loss slope >= 0
      S7: Accuracy not increasing — accuracy slope <= 0
      S8: Vanishing gradient — gradient magnitude < 1e-7
    """
    n = X.shape[0]
    preds = np.zeros(n, dtype=np.int64)

    fi_grad_max = _find_feature(feature_names, "grad_abs_max_final", "grad_abs_max_mid")
    fi_grad_zero = _find_feature(feature_names, "grad_zero_ratio_early_mean",
                                  "grad_zero_ratio_final")
    fi_loss_slope = _find_feature(feature_names, "loss_mid_slope", "loss_early_slope")
    fi_acc_slope = _find_feature(feature_names, "accuracy_mid_slope",
                                  "accuracy_early_slope")
    fi_update = _find_feature(feature_names, "update_ratio_total_final",
                               "update_ratio_emb_final",
                               "update_ratio_classifier_final")

    for i in range(n):
        detected = False

        # S1: Numerical errors — any NaN in the row
        if np.any(np.isnan(X[i])):
            detected = True

        # S2: Unchanged weight — update ratio near zero
        if fi_update is not None:
            val = X[i, fi_update]
            if not np.isnan(val) and abs(val) < 1e-6:
                detected = True

        # S4: Dead node — gradient zero ratio high
        if fi_grad_zero is not None:
            val = X[i, fi_grad_zero]
            if not np.isnan(val) and val > 0.7:
                detected = True

        # S6: Loss not decreasing
        if fi_loss_slope is not None:
            val = X[i, fi_loss_slope]
            if not np.isnan(val) and val >= 0:
                detected = True

        # S7: Accuracy not increasing
        if fi_acc_slope is not None:
            val = X[i, fi_acc_slope]
            if not np.isnan(val) and val <= 0:
                detected = True

        # S8: Vanishing gradient
        if fi_grad_max is not None:
            val = X[i, fi_grad_max]
            if not np.isnan(val) and val < 1e-7:
                detected = True

        preds[i] = 1 if detected else 0

    return preds


def evaluate_rule_based(name, X_te, y_det_te, feature_names):
    """Evaluate a rule-based technique on detection only."""
    if name == "autotrainer":
        det_preds = autotrainer_style_detection(X_te, feature_names)
    else:
        det_preds = deepdiagnosis_style_detection(X_te, feature_names)

    return {
        "detection_acc": float(accuracy_score(y_det_te, det_preds)),
        "detection_f1": float(f1_score(y_det_te, det_preds, average="macro")),
        "detection_f1_weighted": float(f1_score(y_det_te, det_preds, average="weighted")),
        "detection_f1_micro": float(f1_score(y_det_te, det_preds, average="micro")),
        "detection_precision": float(precision_score(y_det_te, det_preds,
                                                      average="macro", zero_division=0)),
        "detection_recall": float(recall_score(y_det_te, det_preds,
                                                average="macro", zero_division=0)),
    }


# ═══════════════════════════════════════════════════════════════════════
# SUPPORTED FAMILY MAPPINGS
# ═══════════════════════════════════════════════════════════════════════

# DEFault supports 7 fault categories: Hyperparameter, Loss, Activation,
# Optimization, Layer, Weights, Regularization.
#
# "Layer" in DEFault means structural faults: wrong layer count, wrong
# filter size, wrong neuron count, wrong pooling — these are CNN/FFNN
# architectural issues, NOT Transformer sublayer component faults.
#
# None of DEFault's 7 categories semantically map to our Transformer
# component families (embedding, ffn, qkv, layernorm, positional,
# residual, score, masking, kernel, variant, kv_cache, output).
#
# All families score F1=0 — DEFault cannot classify any of them.
DEFAULT_SUPPORTED_FAMILIES = set()  # none supported

# DeepFD supports 5 fault types: Loss function, Optimizer, Activation
# function, Insufficient iterations, Learning rate.
# These are all training-configuration faults, not component-level faults.
# None map to our Transformer component families.
DEEPFD_SUPPORTED_FAMILIES = set()  # none supported


# ═══════════════════════════════════════════════════════════════════════
# LEARNING-BASED TECHNIQUES (all 3 stages)
# ═══════════════════════════════════════════════════════════════════════

def _all_metrics(y_true, y_pred, average="macro", labels=None):
    """Compute all standard metrics."""
    kw = {"zero_division": 0}
    if labels is not None:
        kw["labels"] = labels
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", **kw)),
        "precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
    }


def _learned_baseline(X_tr, X_te, y_det_tr, y_det_te,
                       y_cat_tr, y_cat_te, y_rc_tr, y_rc_te,
                       category_names, rootcause_local_labels,
                       category_sizes, supported_families, make_classifiers):
    """Generic learned baseline with supported-family restriction.

    For families not in supported_families, the technique cannot make a
    prediction and those samples are counted as errors (F1=0 contribution
    to the macro average). This mirrors how DEFault evaluated baselines
    that did not support certain fault categories.

    Args:
        supported_families: set of family names this technique can classify
        make_classifiers: callable(purpose) returning (list_of_classifiers, predict_fn)
    """
    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    Xte = np.nan_to_num(scaler.transform(X_te), nan=0.0)

    # ── Stage 1: Detection ─────────────────────────────────────────
    det_clfs, det_predict = make_classifiers("detection")
    for clf in det_clfs:
        clf.fit(Xtr, y_det_tr)
    det_preds = det_predict(Xte)

    # AUROC from last classifier's probabilities
    det_probs = det_clfs[-1].predict_proba(Xte)
    det_probs_pos = det_probs[:, 1] if det_probs.shape[1] == 2 else det_probs[:, 0]
    auroc = float(roc_auc_score(y_det_te, det_probs_pos)) if len(np.unique(y_det_te)) > 1 else 0.0
    det_metrics = _all_metrics(y_det_te, det_preds)
    det_metrics["auroc"] = auroc

    # ── Stage 2: Categorization ────────────────────────────────────
    faulty_tr = y_det_tr == 1
    faulty_te = y_det_te == 1

    # Build mapping: which category indices are supported?
    supported_idx = {ci for ci, cn in enumerate(category_names) if cn in supported_families}
    n_cats = len(category_names)

    if faulty_te.sum() > 0:
        # Train only on samples from supported families
        sup_mask_tr = np.array([y_cat_tr[i] in supported_idx
                                 for i in range(len(y_cat_tr))]) & faulty_tr

        if sup_mask_tr.sum() >= 2:
            cat_clfs, cat_predict = make_classifiers("category")
            for clf in cat_clfs:
                clf.fit(Xtr[sup_mask_tr], y_cat_tr[sup_mask_tr])

            # Predict for ALL faulty test samples
            raw_cat_preds = cat_predict(Xte[faulty_te])

            # For test samples whose true category is unsupported,
            # the baseline was never trained on them and can't predict them.
            # For test samples whose true category IS supported but the
            # baseline predicts an unsupported category, that's still wrong.
            # We keep raw predictions as-is — the F1 computation naturally
            # penalizes wrong predictions.
            cat_preds = raw_cat_preds
        else:
            cat_preds = np.full(faulty_te.sum(), -1, dtype=np.int64)

        cat_metrics = _all_metrics(y_cat_te[faulty_te], cat_preds)

        per_cat_f1 = {}
        for ci, cn in enumerate(category_names):
            mask = y_cat_te[faulty_te] == ci
            if mask.sum() > 0:
                if cn in supported_families:
                    per_cat_f1[cn] = float(f1_score(
                        (y_cat_te[faulty_te] == ci).astype(int),
                        (cat_preds == ci).astype(int),
                        average="binary", zero_division=0))
                else:
                    # Unsupported family: technique cannot predict → F1=0
                    per_cat_f1[cn] = 0.0
    else:
        cat_metrics = {k: 0.0 for k in ["acc", "f1_macro", "f1_weighted",
                                          "f1_micro", "precision", "recall"]}
        cat_preds = np.array([])
        per_cat_f1 = {}

    # ── Stage 3: Root cause ────────────────────────────────────────
    rc_true_cat = {}
    rc_pred_cat = {}

    for ci, cat_name in enumerate(category_names):
        if cat_name not in supported_families:
            # Unsupported → F1=0
            rc_true_cat[cat_name] = 0.0
            rc_pred_cat[cat_name] = 0.0
            continue

        # Oracle RC
        cat_mask_tr = faulty_tr & (y_cat_tr == ci)
        cat_mask_te = faulty_te & (y_cat_te == ci)
        if cat_mask_tr.sum() < 5 or cat_mask_te.sum() < 2:
            continue
        if cat_name not in rootcause_local_labels:
            continue
        local_map = rootcause_local_labels[cat_name]

        valid_tr = np.array([int(y_rc_tr[j]) in local_map for j in np.where(cat_mask_tr)[0]])
        valid_te = np.array([int(y_rc_te[j]) in local_map for j in np.where(cat_mask_te)[0]])
        if valid_tr.sum() < 5 or valid_te.sum() < 2:
            continue

        tr_idx = np.where(cat_mask_tr)[0][valid_tr]
        te_idx = np.where(cat_mask_te)[0][valid_te]
        y_local_tr = np.array([local_map[int(y_rc_tr[j])] for j in tr_idx])
        y_local_te = np.array([local_map[int(y_rc_te[j])] for j in te_idx])

        if len(np.unique(y_local_tr)) < 2 or len(np.unique(y_local_te)) < 2:
            continue

        rc_clfs, rc_predict = make_classifiers("rootcause")
        for clf in rc_clfs:
            clf.fit(Xtr[tr_idx], y_local_tr)
        rc_preds = rc_predict(Xte[te_idx])

        n_rc = category_sizes.get(cat_name, 0)
        valid_labels = list(range(n_rc))
        rc_true_cat[cat_name] = float(f1_score(y_local_te, rc_preds,
                                                average="macro", zero_division=0,
                                                labels=valid_labels))

        # E2E RC
        if faulty_te.sum() > 0 and len(cat_preds) > 0:
            faulty_indices = np.where(faulty_te)[0]
            true_cat_mask = np.array([y_cat_te[fi] == ci for fi in faulty_indices])
            pred_cat_mask = np.array([cat_preds[j] == ci for j in range(len(faulty_indices))])
            valid_mask = true_cat_mask & np.array(
                [int(y_rc_te[fi]) in local_map for fi in faulty_indices])
            if valid_mask.sum() >= 2:
                y_true_sub = np.array([local_map[int(y_rc_te[faulty_indices[j]])]
                                        for j in np.where(valid_mask)[0]])
                y_pred_sub = np.full_like(y_true_sub, -1)
                for j_idx, j in enumerate(np.where(valid_mask)[0]):
                    if pred_cat_mask[j]:
                        fi = faulty_indices[j]
                        y_pred_sub[j_idx] = rc_predict(Xte[fi:fi+1].reshape(1, -1))[0]

                if len(np.unique(y_true_sub)) >= 2:
                    rc_pred_cat[cat_name] = float(f1_score(
                        y_true_sub, y_pred_sub, average="macro", zero_division=0,
                        labels=valid_labels))

    # Macro averages include 0s for unsupported families
    all_rc_true = [rc_true_cat.get(cn, 0.0) for cn in category_names]
    all_rc_pred = [rc_pred_cat.get(cn, 0.0) for cn in category_names]
    rc_true_macro = float(np.mean(all_rc_true))
    rc_pred_macro = float(np.mean(all_rc_pred))

    return {
        "detection": det_metrics,
        "category": cat_metrics,
        "per_category_f1": per_cat_f1,
        "rc_true_by_family": rc_true_cat,
        "rc_true_macro": rc_true_macro,
        "rc_pred_by_family": rc_pred_cat,
        "rc_pred_macro": rc_pred_macro,
        "supported_families": sorted(supported_families),
        "n_supported": len(supported_families & set(category_names)),
        "n_total_families": len(category_names),
    }


def default_style_baseline(X_tr, X_te, y_det_tr, y_det_te,
                            y_cat_tr, y_cat_te, y_rc_tr, y_rc_te,
                            category_names, rootcause_local_labels,
                            category_sizes):
    """DEFault-style hierarchical Random Forest.

    Stage 1: Binary RF for detection
    Stage 2: Multi-class RF trained only on supported families
    Stage 3: Per-family RF for root cause (supported families only)
    Unsupported families score F1=0 in macro averages.
    """
    def make_classifiers(purpose):
        rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     random_state=42, n_jobs=-1,
                                     class_weight="balanced")
        return [rf], lambda X: rf.predict(X)

    return _learned_baseline(
        X_tr, X_te, y_det_tr, y_det_te, y_cat_tr, y_cat_te,
        y_rc_tr, y_rc_te, category_names, rootcause_local_labels,
        category_sizes, DEFAULT_SUPPORTED_FAMILIES, make_classifiers)


def deepfd_style_baseline(X_tr, X_te, y_det_tr, y_det_te,
                           y_cat_tr, y_cat_te, y_rc_tr, y_rc_te,
                           category_names, rootcause_local_labels,
                           category_sizes):
    """DeepFD-style flat KNN + DT + RF ensemble with majority voting.

    Adapted from multi-label to single-label since our data is single-label.
    Uses majority voting across 3 classifiers.
    Only supported families are classified; unsupported score F1=0.
    """
    def make_classifiers(purpose):
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        dt = DecisionTreeClassifier(max_depth=20, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     random_state=42, n_jobs=-1)

        def predict(X):
            p1 = knn.predict(X)
            p2 = dt.predict(X)
            p3 = rf.predict(X)
            n_classes = max(p1.max(), p2.max(), p3.max()) + 1 if len(p1) > 0 else 1
            return np.array([np.bincount([p1[i], p2[i], p3[i]],
                                          minlength=n_classes).argmax()
                              for i in range(len(X))])

        return [knn, dt, rf], predict

    return _learned_baseline(
        X_tr, X_te, y_det_tr, y_det_te, y_cat_tr, y_cat_te,
        y_rc_tr, y_rc_te, category_names, rootcause_local_labels,
        category_sizes, DEEPFD_SUPPORTED_FAMILIES, make_classifiers)


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP

# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_baselines(arch):
    """Run all baselines for one architecture using same CV splits."""
    print(f"\n{'='*70}")
    print(f"BASELINES: {arch.upper()}")
    print(f"{'='*70}")

    data = load_data(arch)
    X = data["X"]
    groups = data["groups"]
    feature_names = data["feature_names"]
    y_detect = data["y_detect"]
    y_category = data["y_category"]
    y_rootcause = data["y_rootcause"]

    gkf = GroupKFold(n_splits=5)

    # Accumulators
    rule_results = {"autotrainer": [], "deepdiagnosis": []}
    learned_results = {"default_style": [], "deepfd_style": []}

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y_detect, groups)):
        np.random.seed(42 + fold_idx)
        t0 = time.time()

        # Feature processing (same as main experiment)
        X_tr, X_te, feat_names_proc, g_idx, proc_log = apply_processing_in_fold(
            X[tr_idx], X[te_idx], feature_names, y_detect[tr_idx], arch)

        proc_feat_names = feat_names_proc

        # ── Rule-based (detection only, on processed features) ─────
        for name in ["autotrainer", "deepdiagnosis"]:
            res = evaluate_rule_based(name, X_te, y_detect[te_idx], proc_feat_names)
            rule_results[name].append(res)

        # ── Learning-based (all 3 stages) ──────────────────────────
        # Restrict features to match each baseline's original feature scope

        # DEFault-style: only DEFault-matching features
        def_mask = _default_feature_mask(proc_feat_names)
        X_tr_def = X_tr[:, def_mask]
        X_te_def = X_te[:, def_mask]
        n_def = def_mask.sum()

        # DeepFD-style: only DeepFD-matching features
        dfd_mask = _deepfd_feature_mask(proc_feat_names)
        X_tr_dfd = X_tr[:, dfd_mask]
        X_te_dfd = X_te[:, dfd_mask]
        n_dfd = dfd_mask.sum()

        if fold_idx == 0:
            print(f"  Feature counts: DEFault-style={n_def}, "
                  f"DeepFD-style={n_dfd}, Ours(full)={X_tr.shape[1]}")

        t1 = time.time()
        def_res = default_style_baseline(
            X_tr_def, X_te_def,
            y_detect[tr_idx], y_detect[te_idx],
            y_category[tr_idx], y_category[te_idx],
            y_rootcause[tr_idx], y_rootcause[te_idx],
            data["category_names"], data["rootcause_local_labels"],
            data["category_sizes"])
        learned_results["default_style"].append(def_res)
        t2 = time.time()

        dfd_res = deepfd_style_baseline(
            X_tr_dfd, X_te_dfd,
            y_detect[tr_idx], y_detect[te_idx],
            y_category[tr_idx], y_category[te_idx],
            y_rootcause[tr_idx], y_rootcause[te_idx],
            data["category_names"], data["rootcause_local_labels"],
            data["category_sizes"])
        learned_results["deepfd_style"].append(dfd_res)
        t3 = time.time()

        elapsed = time.time() - t0
        print(f"  Fold {fold_idx+1}/5: "
              f"DEFault-style Cat={def_res['category']['f1_macro']:.4f} "
              f"DeepFD-style Cat={dfd_res['category']['f1_macro']:.4f} "
              f"[{elapsed:.1f}s]")

    # ── Aggregate results ──────────────────────────────────────────
    def agg_metric(results, key):
        vals = [r[key] for r in results]
        return {"mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)}

    def agg_detection(results):
        return {
            metric: agg_metric(results, f"detection_{metric}")
            for metric in ["acc", "f1", "f1_weighted", "f1_micro",
                          "precision", "recall"]
        }

    def agg_learned(results, category_names):
        det = {}
        for m in ["acc", "f1_macro", "f1_weighted", "f1_micro", "precision", "recall", "auroc"]:
            vals = [r["detection"].get(m, 0.0) for r in results]
            det[m] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4)}

        cat = {}
        for m in ["acc", "f1_macro", "f1_weighted", "f1_micro", "precision", "recall"]:
            vals = [r["category"].get(m, 0.0) for r in results]
            cat[m] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4)}

        per_cat = {}
        for cn in category_names:
            vals = [r["per_category_f1"].get(cn, 0.0) for r in results]
            per_cat[cn] = round(float(np.mean(vals)), 4)

        rc_true = {}
        for cn in category_names:
            vals = [r["rc_true_by_family"].get(cn, 0.0) for r in results]
            rc_true[cn] = round(float(np.mean(vals)), 4)
        rc_true_macro_vals = [r["rc_true_macro"] for r in results]
        rc_true_macro = {"mean": round(float(np.mean(rc_true_macro_vals)), 4),
                         "std": round(float(np.std(rc_true_macro_vals)), 4)}

        rc_pred = {}
        for cn in category_names:
            vals = [r["rc_pred_by_family"].get(cn, 0.0) for r in results]
            rc_pred[cn] = round(float(np.mean(vals)), 4)
        rc_pred_macro_vals = [r["rc_pred_macro"] for r in results]
        rc_pred_macro = {"mean": round(float(np.mean(rc_pred_macro_vals)), 4),
                         "std": round(float(np.std(rc_pred_macro_vals)), 4)}

        return {
            "detection": det,
            "category": cat,
            "per_category_f1": per_cat,
            "rc_true_by_family": rc_true,
            "rc_true_macro": rc_true_macro,
            "rc_pred_by_family": rc_pred,
            "rc_pred_macro": rc_pred_macro,
        }

    # Rule-based aggregation
    rule_summary = {}
    for name in ["autotrainer", "deepdiagnosis"]:
        rule_summary[name] = agg_detection(rule_results[name])

    # Learned aggregation
    learned_summary = {}
    for name in ["default_style", "deepfd_style"]:
        learned_summary[name] = agg_learned(
            learned_results[name], data["category_names"])

    # Print summary
    print(f"\n  Rule-based detection ({arch}):")
    for name in ["autotrainer", "deepdiagnosis"]:
        s = rule_summary[name]
        print(f"    {name:20s}: F1={s['f1']['mean']:.4f}±{s['f1']['std']:.4f}  "
              f"Acc={s['acc']['mean']:.4f}")

    print(f"\n  Learning-based ({arch}):")
    for name in ["default_style", "deepfd_style"]:
        s = learned_summary[name]
        print(f"    {name:20s}: Det={s['detection']['f1_macro']['mean']:.4f}  "
              f"Cat={s['category']['f1_macro']['mean']:.4f}  "
              f"RC(true)={s['rc_true_macro']['mean']:.4f}  "
              f"RC(pred)={s['rc_pred_macro']['mean']:.4f}")

    all_results = {
        "arch": arch,
        "rule_based": rule_summary,
        "learning_based": learned_summary,
        "n_folds": 5,
        "timestamp": datetime.now().isoformat(),
    }

    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["encoder", "decoder", "both"], default="both")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    archs = ["encoder", "decoder"] if args.arch == "both" else [args.arch]

    for arch in archs:
        results = run_baselines(arch)
        out_file = RESULTS_DIR / f"{arch}_baselines.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {out_file}")


if __name__ == "__main__":
    main()
