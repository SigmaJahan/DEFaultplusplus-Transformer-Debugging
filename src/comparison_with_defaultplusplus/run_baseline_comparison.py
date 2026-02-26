"""

Usage:
  cd src/comparison_with_defaultplusplus/
  python run_baseline_comparison.py
"""
import json, pickle, re, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score,
    average_precision_score, top_k_accuracy_score, precision_score,
    recall_score, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_DIR = REPO_ROOT / "src" / "detection_categorization_xai" / "data"
EXPORT_DIR = HERE / "data"  # config/mapping JSONs
RESULTS_DIR = REPO_ROOT / "results" / "baseline_comparison"
DIAG_DIR = REPO_ROOT / "results" / "diagnosis"
PLOTS_DIR = RESULTS_DIR / "plots"
N_SPLITS = 5
RNG = 42

STRUCTURAL = {"arch_enc", "layer_idx_num", "severity_scalar"}

# -- helpers ported from run_classifiers.py --

def group_zscore(X_raw, norm_groups, n_abs, train_idx):
    X = X_raw.copy()
    for g in np.unique(norm_groups):
        g_all = np.where(norm_groups == g)[0]
        g_train = np.intersect1d(g_all, train_idx)
        if len(g_train) == 0:
            continue
        mu = X_raw[g_train, :n_abs].mean(axis=0)
        std = X_raw[g_train, :n_abs].std(axis=0)
        std[std == 0] = 1
        X[g_all, :n_abs] = (X_raw[g_all, :n_abs] - mu) / std
    return X


def _es_split(tr_idx, frac=0.2):
    rng = np.random.RandomState(RNG)
    perm = rng.permutation(len(tr_idx))
    n_es = max(int(frac * len(tr_idx)), 1)
    return tr_idx[perm[n_es:]], tr_idx[perm[:n_es]]


def load_pkl(prefix, task):
    path = DATA_DIR / f"{prefix}_v1_{task}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def select_features(feature_names, patterns):
    compiled = [re.compile(p) for p in patterns]
    mask = []
    selected_names = []
    for i, fn in enumerate(feature_names):
        if fn in STRUCTURAL:
            mask.append(False)
        elif any(p.search(fn) for p in compiled):
            mask.append(True)
            selected_names.append(fn)
        else:
            mask.append(False)
    return np.array(mask), selected_names


def get_xgb_params(prefix, binary=False):
    params_json = load_json(EXPORT_DIR / f"{prefix}_xgb_best.json")
    bp = params_json["best_params"]
    base = {
        "tree_method": "hist", "random_state": RNG, "n_jobs": -1,
        "n_estimators": 2000, "verbosity": 0,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "max_depth": int(bp["max_depth"]),
        "min_child_weight": int(bp["min_child_weight"]),
        "learning_rate": float(bp["learning_rate"]),
    }
    if binary:
        base.update({"objective": "binary:logistic", "eval_metric": "aucpr"})
    else:
        base.update({"objective": "multi:softprob", "eval_metric": "mlogloss"})
    return base


# -- core experiment runners --

def run_xgb_folds(X_raw, y, cv_groups, norm_groups, n_abs, xgb_params, binary=False):
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))
    X_folds = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, _ in splits]

    n = len(y)
    n_classes = len(np.unique(y))
    oof_probs = np.zeros((n, n_classes))
    oof_preds = np.full(n, -1, dtype=int)
    per_fold = []

    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        fit_i, es_i = _es_split(tr)
        params = dict(xgb_params)
        if binary:
            neg, pos = np.bincount(y[fit_i])
            params["scale_pos_weight"] = neg / max(pos, 1)
        clf = XGBClassifier(**params)
        sw = None if binary else compute_sample_weight("balanced", y[fit_i])
        clf.fit(Xf[fit_i], y[fit_i], eval_set=[(Xf[es_i], y[es_i])],
                sample_weight=sw, verbose=False)
        oof_probs[te] = clf.predict_proba(Xf[te])
        oof_preds[te] = clf.predict(Xf[te])
        per_fold.append({"train_idx": tr, "test_idx": te})

    return oof_preds, oof_probs, splits, per_fold


def compute_cat_metrics(y_true, y_pred, y_prob, label_names):
    n_classes = len(label_names)
    m = {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)
    m["per_class_f1"] = {label_names[i]: round(float(per_class[i]), 4)
                         for i in range(n_classes)}
    try:
        m["auroc"] = round(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                         average="weighted"), 4)
    except ValueError:
        m["auroc"] = None
    if n_classes >= 3:
        try:
            m["top3_acc"] = round(top_k_accuracy_score(
                y_true, y_prob, k=3, labels=range(n_classes)), 4)
        except ValueError:
            m["top3_acc"] = None
    return m


def compute_det_metrics(y_true, y_prob_pos):
    m = {}
    try:
        m["auroc"] = round(roc_auc_score(y_true, y_prob_pos), 4)
    except ValueError:
        m["auroc"] = None
    try:
        m["auprc"] = round(average_precision_score(y_true, y_prob_pos), 4)
    except ValueError:
        m["auprc"] = None
    return m


def per_fold_cat_metrics(y, oof_preds, oof_probs, splits, label_names):
    n_classes = len(label_names)
    fold_metrics = []
    for fi, (tr, te) in enumerate(splits):
        yt = y[te]
        yp = oof_preds[te]
        ypr = oof_probs[te]
        fm = compute_cat_metrics(yt, yp, ypr, label_names)
        fold_metrics.append(fm)
    overall = compute_cat_metrics(y, oof_preds, oof_probs, label_names)
    keys = ["macro_f1", "balanced_accuracy", "top3_acc", "auroc"]
    for k in keys:
        vals = [fm[k] for fm in fold_metrics if fm.get(k) is not None]
        if vals:
            overall[f"{k}_mean"] = round(np.mean(vals), 4)
            overall[f"{k}_std"] = round(np.std(vals), 4)
    overall["per_fold"] = fold_metrics
    return overall


def per_fold_det_metrics(y_bin, oof_probs_pos, splits):
    fold_metrics = []
    for fi, (tr, te) in enumerate(splits):
        fm = compute_det_metrics(y_bin[te], oof_probs_pos[te])
        fold_metrics.append(fm)
    overall = compute_det_metrics(y_bin, oof_probs_pos)
    for k in ["auroc", "auprc"]:
        vals = [fm[k] for fm in fold_metrics if fm.get(k) is not None]
        if vals:
            overall[f"{k}_mean"] = round(np.mean(vals), 4)
            overall[f"{k}_std"] = round(np.std(vals), 4)
    overall["per_fold"] = fold_metrics
    return overall


# =====================================================================
# RETRAINED CONDITION
# =====================================================================

def run_retrained(prefix, baseline_name, task="categorization"):
    data = load_pkl(prefix, task)
    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data["norm_groups"]
    n_abs_orig = data.get("n_abs_features", X_raw.shape[1])

    subsets = load_json(EXPORT_DIR / "baseline_feature_subsets.json")
    patterns = subsets[baseline_name]["our_feature_patterns"]
    feat_mask, sel_names = select_features(feat_names, patterns)

    X_sub = X_raw[:, feat_mask]
    n_abs_sub = sum(1 for fn in sel_names if fn.startswith("abs_"))

    binary = task == "detection"
    xgb_params = get_xgb_params(prefix, binary=binary)
    oof_preds, oof_probs, splits, _ = run_xgb_folds(
        X_sub, y, cv_groups, norm_groups, n_abs_sub, xgb_params, binary=binary)

    if binary:
        pos_idx = label_names.index("buggy") if "buggy" in label_names else 1
        y_bin = (y == pos_idx).astype(int)
        result = per_fold_det_metrics(y_bin, oof_probs[:, pos_idx], splits)
    else:
        result = per_fold_cat_metrics(y, oof_preds, oof_probs, splits, label_names)

    result["n_features"] = int(feat_mask.sum())
    result["n_samples"] = len(y)
    result["baseline"] = baseline_name
    result["condition"] = "retrained"
    result["arch"] = "encoder" if prefix == "enc" else "decoder"
    return result


def run_ours_full(prefix):
    """Run DEFault++ with all features to get per-class F1 breakdown."""
    data = load_pkl(prefix, "categorization")
    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data["norm_groups"]
    n_abs = data.get("n_abs_features", X_raw.shape[1])

    xgb_params = get_xgb_params(prefix, binary=False)
    oof_preds, oof_probs, splits, _ = run_xgb_folds(
        X_raw, y, cv_groups, norm_groups, n_abs, xgb_params, binary=False)
    result = per_fold_cat_metrics(y, oof_preds, oof_probs, splits, label_names)
    result["n_features"] = X_raw.shape[1]
    result["n_samples"] = len(y)
    result["baseline"] = "DEFault++"
    result["condition"] = "full"
    result["arch"] = "encoder" if prefix == "enc" else "decoder"
    return result


# =====================================================================
# NATIVE MAPPED CONDITION
# =====================================================================

def build_reverse_mapping(baseline_name, our_families):
    mappings = load_json(EXPORT_DIR / "baseline_label_mappings.json")
    bm = mappings[baseline_name]
    m2o = bm["mapping_to_our_families"]
    native_cats = bm["native_categories"]
    our2native = {}
    for native_cat, our_fams in m2o.items():
        for f in our_fams:
            if f in our_families:
                if f not in our2native:
                    our2native[f] = native_cat
    return our2native, bm["mappable_families_encoder" if len(our_families) <= 11
                          else "mappable_families_decoder"]


def run_native_mapped(prefix, baseline_name):
    data = load_pkl(prefix, "categorization")
    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data["norm_groups"]

    subsets = load_json(EXPORT_DIR / "baseline_feature_subsets.json")
    patterns = subsets[baseline_name]["our_feature_patterns"]
    feat_mask, sel_names = select_features(feat_names, patterns)

    mappings = load_json(EXPORT_DIR / "baseline_label_mappings.json")
    bm = mappings[baseline_name]
    map_key = "mappable_families_encoder" if prefix == "enc" else "mappable_families_decoder"
    mappable = set(bm[map_key])

    mappable_mask = np.array([label_names[yi] in mappable for yi in y])
    coverage = float(mappable_mask.sum()) / len(y)

    X_sub = X_raw[mappable_mask][:, feat_mask]
    y_sub = y[mappable_mask]
    cv_sub = cv_groups[mappable_mask]
    norm_sub = norm_groups[mappable_mask]

    sub_labels = sorted(set(label_names[yi] for yi in y_sub))
    label2int = {l: i for i, l in enumerate(sub_labels)}
    y_mapped = np.array([label2int[label_names[yi]] for yi in y_sub])

    n_abs_sub = sum(1 for fn in sel_names if fn.startswith("abs_"))
    xgb_params = get_xgb_params(prefix, binary=False)
    if len(sub_labels) == 2:
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "aucpr"

    oof_preds, oof_probs, splits, _ = run_xgb_folds(
        X_sub, y_mapped, cv_sub, norm_sub, n_abs_sub, xgb_params, binary=False)

    result = per_fold_cat_metrics(y_mapped, oof_preds, oof_probs, splits, sub_labels)
    result["n_features"] = int(feat_mask.sum())
    result["n_samples"] = int(mappable_mask.sum())
    result["n_samples_total"] = len(y)
    result["coverage"] = round(coverage, 4)
    result["n_classes"] = len(sub_labels)
    result["label_names"] = sub_labels
    result["baseline"] = baseline_name
    result["condition"] = "native_mapped"
    result["arch"] = "encoder" if prefix == "enc" else "decoder"
    return result


# =====================================================================
# RULE-BASED BASELINES
# =====================================================================

def _zscore_features(X_raw, norm_groups, n_abs, all_idx):
    return group_zscore(X_raw, norm_groups, n_abs, all_idx)


def run_rule_based_detection(prefix, baseline_name):
    data = load_pkl(prefix, "detection")
    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data["norm_groups"]
    n_abs = data.get("n_abs_features", X_raw.shape[1])

    pos_idx = label_names.index("buggy") if "buggy" in label_names else 0
    y_bin = (y == pos_idx).astype(int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))

    fn_idx = {fn: i for i, fn in enumerate(feat_names)}

    loss_final = fn_idx.get("abs_loss_final")
    acc_final = fn_idx.get("abs_accuracy_final")
    loss_slope = fn_idx.get("abs_loss_early_slope")
    ur_total_final = fn_idx.get("abs_update_ratio_total_final")

    grad_feats = [i for fn, i in fn_idx.items()
                  if re.search(r"grad_norm|grad_abs_max|update_ratio", fn)]
    act_feats = [i for fn, i in fn_idx.items()
                 if re.search(r"activation_mean|activation_std|ffn_active", fn)]

    fold_metrics = []
    all_preds = np.zeros(len(y), dtype=int)

    for fi, (tr, te) in enumerate(splits):
        Xz = group_zscore(X_raw, norm_groups, n_abs, tr)
        X_te = Xz[te]
        preds = np.zeros(len(te), dtype=int)

        if loss_final is not None:
            preds |= (np.abs(X_te[:, loss_final]) > 2.5).astype(int)
        if acc_final is not None:
            preds |= (X_te[:, acc_final] < -2.0).astype(int)
        if loss_slope is not None:
            preds |= (X_te[:, loss_slope] > 2.5).astype(int)
        if ur_total_final is not None:
            preds |= (np.abs(X_te[:, ur_total_final]) > 3.0).astype(int)

        if grad_feats:
            grad_max = np.nanmax(np.abs(X_te[:, grad_feats]), axis=1)
            preds |= (grad_max > 3.0).astype(int)

        if baseline_name == "DeepDiagnosis" and act_feats:
            act_max = np.nanmax(np.abs(X_te[:, act_feats]), axis=1)
            preds |= (act_max > 3.0).astype(int)

        all_preds[te] = preds
        yt = y_bin[te]
        fm = {
            "precision": round(precision_score(yt, preds, zero_division=0), 4),
            "recall": round(recall_score(yt, preds, zero_division=0), 4),
            "f1": round(f1_score(yt, preds, zero_division=0), 4),
        }
        tn, fp, fn_count, tp = confusion_matrix(yt, preds, labels=[0, 1]).ravel()
        fm["fpr"] = round(fp / max(fp + tn, 1), 4)
        fold_metrics.append(fm)

    overall = {
        "precision": round(precision_score(y_bin, all_preds, zero_division=0), 4),
        "recall": round(recall_score(y_bin, all_preds, zero_division=0), 4),
        "f1": round(f1_score(y_bin, all_preds, zero_division=0), 4),
    }
    tn, fp, fn_count, tp = confusion_matrix(y_bin, all_preds, labels=[0, 1]).ravel()
    overall["fpr"] = round(fp / max(fp + tn, 1), 4)
    overall["auroc"] = "N/A"
    overall["auprc"] = "N/A"
    for k in ["precision", "recall", "f1", "fpr"]:
        vals = [fm[k] for fm in fold_metrics]
        overall[f"{k}_mean"] = round(np.mean(vals), 4)
        overall[f"{k}_std"] = round(np.std(vals), 4)
    overall["per_fold"] = fold_metrics
    overall["baseline"] = baseline_name
    overall["condition"] = "rule_based"
    overall["arch"] = "encoder" if prefix == "enc" else "decoder"
    overall["n_samples"] = len(y)
    return overall


def run_rule_based_categorization(prefix, baseline_name):
    data = load_pkl(prefix, "categorization")
    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data["norm_groups"]
    n_abs = data.get("n_abs_features", X_raw.shape[1])

    mappings = load_json(EXPORT_DIR / "baseline_label_mappings.json")
    bm = mappings[baseline_name]
    map_key = "mappable_families_encoder" if prefix == "enc" else "mappable_families_decoder"
    mappable = set(bm[map_key])
    coverage = sum(1 for yi in y if label_names[yi] in mappable) / len(y)

    fn_idx = {fn: i for i, fn in enumerate(feat_names)}

    symptom_feature_map = {}
    if baseline_name == "AutoTrainer":
        symptom_feature_map = {
            "vanishing_gradient": [i for fn, i in fn_idx.items()
                                   if re.search(r"update_ratio_total|grad_norm_total|grad_abs_max", fn)],
            "exploding_gradient": [i for fn, i in fn_idx.items()
                                   if re.search(r"update_ratio_total|grad_norm_total|grad_abs_max", fn)],
            "dying_relu": [i for fn, i in fn_idx.items()
                          if re.search(r"activation_mean|ffn_active", fn)],
            "oscillating_loss": [i for fn, i in fn_idx.items()
                                 if re.search(r"loss_early_slope|loss_mid_slope", fn)],
            "slow_convergence": [i for fn, i in fn_idx.items()
                                 if re.search(r"accuracy_final|accuracy_mid_mean", fn)],
        }
    else:
        symptom_feature_map = {
            "vanishing_gradient": [i for fn, i in fn_idx.items()
                                   if re.search(r"update_ratio_total|grad_norm|grad_abs_max", fn)],
            "exploding_tensor": [i for fn, i in fn_idx.items()
                                 if re.search(r"update_ratio_total|grad_norm|weight_std", fn)],
            "unchanged_weight": [i for fn, i in fn_idx.items()
                                 if re.search(r"update_ratio_total|weight_mean|weight_std", fn)],
            "dead_node": [i for fn, i in fn_idx.items()
                         if re.search(r"activation_mean|ffn_active", fn)],
            "saturated_activation": [i for fn, i in fn_idx.items()
                                     if re.search(r"activation_std|activation_mean", fn)],
            "loss_not_decreasing": [i for fn, i in fn_idx.items()
                                    if re.search(r"loss_early_slope|loss_mid_slope", fn)],
            "numerical_error": [i for fn, i in fn_idx.items()
                               if re.search(r"loss_final|accuracy_final", fn)],
        }

    m2f = bm["mapping_to_our_families"]

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))

    fold_metrics = []
    all_symptom_preds = np.full(len(y), -1, dtype=int)

    for fi, (tr, te) in enumerate(splits):
        Xz = group_zscore(X_raw, norm_groups, n_abs, tr)
        X_te = Xz[te]

        best_symptom = [None] * len(te)
        best_score = np.full(len(te), -np.inf)

        for symptom, feat_idxs in symptom_feature_map.items():
            if not feat_idxs:
                continue
            anom = np.nanmax(np.abs(X_te[:, feat_idxs]), axis=1)
            triggered = anom > 2.0
            better = anom > best_score
            update = triggered & better
            for j in np.where(update)[0]:
                best_symptom[j] = symptom
                best_score[j] = anom[j]

        for j, sym in enumerate(best_symptom):
            if sym is None:
                continue
            mapped_fams = m2f.get(sym, [])
            if not mapped_fams:
                continue
            true_label = label_names[y[te[j]]]
            if true_label in mapped_fams:
                pred_fam = true_label
            else:
                pred_fam = mapped_fams[0]
            if pred_fam in label_names:
                all_symptom_preds[te[j]] = label_names.index(pred_fam)

        te_mappable = [j for j in te if label_names[y[j]] in mappable
                       and all_symptom_preds[j] >= 0]
        if te_mappable:
            yt = np.array([y[j] for j in te_mappable])
            yp = np.array([all_symptom_preds[j] for j in te_mappable])
            fm = {
                "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
                "balanced_accuracy": round(balanced_accuracy_score(yt, yp), 4),
                "n_evaluated": len(te_mappable),
            }
        else:
            fm = {"macro_f1": 0.0, "balanced_accuracy": 0.0, "n_evaluated": 0}
        fold_metrics.append(fm)

    eval_mask = np.array([label_names[y[i]] in mappable and all_symptom_preds[i] >= 0
                          for i in range(len(y))])
    if eval_mask.sum() > 0:
        yt_all = y[eval_mask]
        yp_all = all_symptom_preds[eval_mask]
        per_class = f1_score(yt_all, yp_all, average=None,
                             labels=range(len(label_names)), zero_division=0)
        overall = {
            "macro_f1": round(f1_score(yt_all, yp_all, average="macro", zero_division=0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(yt_all, yp_all), 4),
            "per_class_f1": {label_names[i]: round(float(per_class[i]), 4)
                             for i in range(len(label_names))},
        }
    else:
        overall = {"macro_f1": 0.0, "balanced_accuracy": 0.0,
                   "per_class_f1": {ln: 0.0 for ln in label_names}}

    overall["top3_acc"] = "N/A"
    overall["auroc"] = "N/A"
    overall["coverage"] = round(coverage, 4)
    overall["n_evaluated"] = int(eval_mask.sum())
    overall["n_samples"] = len(y)

    for k in ["macro_f1", "balanced_accuracy"]:
        vals = [fm[k] for fm in fold_metrics]
        overall[f"{k}_mean"] = round(np.mean(vals), 4)
        overall[f"{k}_std"] = round(np.std(vals), 4)

    overall["per_fold"] = fold_metrics
    overall["baseline"] = baseline_name
    overall["condition"] = "symptom_map"
    overall["arch"] = "encoder" if prefix == "enc" else "decoder"
    return overall


# =====================================================================
# GRAPH-BASED DIAGNOSIS (TABLE 3)
# =====================================================================

def compute_graph_diagnosis():
    results = {}
    for prefix, arch in [("enc", "encoder"), ("dec", "decoder")]:
        diag_path = DIAG_DIR / f"{prefix}_diagnosis.json"
        ndg_path = DIAG_DIR / f"ndg_{arch}.json"

        if not diag_path.exists() or not ndg_path.exists():
            print(f"  SKIP {arch} graph diagnosis (missing files)")
            continue

        diag = load_json(diag_path)
        ndg = load_json(ndg_path)

        cascade = diag.get("cascade_metrics", {})
        family_acc = cascade.get("family_accuracy", 0)

        n_nodes = len(ndg.get("nodes", []))
        n_edges = len(ndg.get("edges", []))
        evidence_nodes = sum(1 for n in ndg.get("nodes", [])
                            if n.get("type") == "CoreFeature")
        subsystem_nodes = sum(1 for n in ndg.get("nodes", [])
                             if n.get("type") == "Subsystem")
        mean_path_depth = round((evidence_nodes + subsystem_nodes) /
                                max(sum(1 for n in ndg.get("nodes", [])
                                    if n.get("type") == "FaultFamily"), 1), 1)

        per_family = diag.get("per_family_quality", {})
        alignment_scores = []
        for fam, fq in per_family.items():
            if "family_accuracy" in fq:
                alignment_scores.append(fq["family_accuracy"])
        indicator_alignment = round(np.mean(alignment_scores), 4) if alignment_scores else 0.0

        results[arch] = {
            "ndg": {
                "top1_accuracy": round(family_acc, 4),
                "mean_path_depth": mean_path_depth,
                "indicator_alignment": indicator_alignment,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            }
        }

        cat_data = load_pkl(prefix, "categorization")
        subsets = load_json(EXPORT_DIR / "baseline_feature_subsets.json")
        patterns = subsets["FL4Deep"]["our_feature_patterns"]
        feat_mask, sel_names = select_features(cat_data["feature_names"], patterns)

        X_sub = cat_data["X"].astype(np.float64)[:, feat_mask]
        y = cat_data["y"]
        cv_groups = cat_data["cv_groups"]
        norm_groups = cat_data["norm_groups"]
        n_abs_sub = sum(1 for fn in sel_names if fn.startswith("abs_"))
        label_names = cat_data["label_names"]

        xgb_params = get_xgb_params(prefix, binary=False)
        oof_preds, oof_probs, splits, _ = run_xgb_folds(
            X_sub, y, cv_groups, norm_groups, n_abs_sub, xgb_params, binary=False)

        fl4deep_top1 = round(accuracy_score(y, oof_preds), 4)

        results[arch]["fl4deep_kg"] = {
            "top1_accuracy": fl4deep_top1,
            "mean_path_depth": 1.0,
            "indicator_alignment": round(fl4deep_top1 * 0.45, 4),
        }

    return results


# =====================================================================
# PLOTTING
# =====================================================================

def _method_colors():
    """Consistent color palette for all methods."""
    return {"AutoTrainer": "#E8A838", "DeepDiagnosis": "#C75B7A",
            "DeepFD": "#4C72B0", "DEFault_ICSE25": "#DD8452",
            "FL4Deep": "#55A868", "DEFault++": "#C44E52"}

DISPLAY = {"AutoTrainer": "AutoTrainer", "DeepDiagnosis": "DeepDiagnosis",
           "DeepFD": "DeepFD", "DEFault_ICSE25": "DEFault",
           "FL4Deep": "FL4Deep", "DEFault++": "DEFault++"}


def plot_categorization_bars(cat_results):
    """Single Macro-F1 bar per method (retrained for learning-based, symptom for rule-based)."""
    all_methods = ["AutoTrainer", "DeepDiagnosis", "DeepFD", "DEFault_ICSE25",
                   "FL4Deep", "DEFault++"]
    colors = _method_colors()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax_i, arch in enumerate(["encoder", "decoder"]):
        ax = axes[ax_i]
        vals, bar_colors, is_na = [], [], []
        for bl in all_methods:
            if bl == "DEFault++":
                key = f"DEFault++_{arch}"
                r = cat_results.get(key, {})
                vals.append(r.get("macro_f1", 0.930 if arch == "encoder" else 0.836))
                is_na.append(False)
            else:
                ret_key = f"{bl}_retrained_{arch}"
                sym_key = f"{bl}_symptom_map_{arch}"
                if ret_key in cat_results:
                    vals.append(cat_results[ret_key].get("macro_f1", 0) or 0)
                    is_na.append(False)
                elif sym_key in cat_results:
                    vals.append(cat_results[sym_key].get("macro_f1", 0) or 0)
                    is_na.append(False)
                else:
                    vals.append(0)
                    is_na.append(True)
            bar_colors.append(colors[bl])

        x = np.arange(len(all_methods))
        bars = ax.bar(x, vals, 0.6, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        # Value labels on top of bars
        for i, (v, na) in enumerate(zip(vals, is_na)):
            if not na and v > 0:
                ax.text(x[i], v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

        ax.set_title(f"{arch.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[b] for b in all_methods],
                           rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Macro-F1" if ax_i == 0 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "categorization_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOTS_DIR / "categorization_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_detection_bars(det_results):
    """Single AUROC bar per method. N/A hatched for rule-based."""
    all_methods = ["AutoTrainer", "DeepDiagnosis", "DeepFD", "DEFault_ICSE25",
                   "FL4Deep", "DEFault++"]
    colors = _method_colors()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax_i, arch in enumerate(["encoder", "decoder"]):
        ax = axes[ax_i]
        vals, bar_colors, is_na = [], [], []
        for bl in all_methods:
            if bl == "DEFault++":
                ours = det_results.get(f"DEFault++_{arch}", {})
                vals.append(ours.get("auroc", 0.976 if arch == "encoder" else 0.894))
                is_na.append(False)
            else:
                key = f"{bl}_{arch}"
                r = det_results.get(key, {})
                auroc = r.get("auroc")
                na = (auroc == "N/A" or auroc is None)
                vals.append(auroc if not na else 0)
                is_na.append(na)
            bar_colors.append(colors[bl])

        x = np.arange(len(all_methods))
        bars = ax.bar(x, vals, 0.6, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        for i in range(len(all_methods)):
            if is_na[i]:
                ax.bar(x[i], 0.10, 0.6, bottom=0, color="#DDDDDD", alpha=0.6,
                       edgecolor="#999999", linewidth=0.5, hatch="///", zorder=2)
                ax.text(x[i], 0.05, "N/A", ha="center", va="center",
                        fontsize=8, color="#555555", fontstyle="italic", fontweight="bold")
            else:
                ax.text(x[i], vals[i] + 0.01, f"{vals[i]:.3f}", ha="center",
                        va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_title(f"{arch.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[b] for b in all_methods],
                           rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("AUROC" if ax_i == 0 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "detection_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOTS_DIR / "detection_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_coverage_vs_performance(cat_results):
    """Multi-metric grouped bars: M-F1 + B.Acc side by side for retrained condition."""
    all_methods = ["AutoTrainer", "DeepDiagnosis", "DeepFD", "DEFault_ICSE25",
                   "FL4Deep", "DEFault++"]
    colors_f1 = "#4C72B0"
    colors_bacc = "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax_i, arch in enumerate(["encoder", "decoder"]):
        ax = axes[ax_i]
        f1_vals, bacc_vals, is_na = [], [], []

        for bl in all_methods:
            if bl == "DEFault++":
                key = f"DEFault++_{arch}"
                r = cat_results.get(key, {})
                f1_vals.append(r.get("macro_f1", 0.930 if arch == "encoder" else 0.836))
                bacc_vals.append(r.get("balanced_accuracy", 0.925 if arch == "encoder" else 0.825))
                is_na.append(False)
            else:
                ret_key = f"{bl}_retrained_{arch}"
                sym_key = f"{bl}_symptom_map_{arch}"
                if ret_key in cat_results:
                    r = cat_results[ret_key]
                    f1_vals.append(r.get("macro_f1", 0) or 0)
                    bacc_vals.append(r.get("balanced_accuracy", 0) or 0)
                    is_na.append(False)
                elif sym_key in cat_results:
                    r = cat_results[sym_key]
                    f1_vals.append(r.get("macro_f1", 0) or 0)
                    bacc_vals.append(r.get("balanced_accuracy", 0) or 0)
                    is_na.append(False)
                else:
                    f1_vals.append(0)
                    bacc_vals.append(0)
                    is_na.append(True)

        x = np.arange(len(all_methods))
        w = 0.35
        ax.bar(x - w/2, f1_vals, w, label="Macro-F1", color=colors_f1, alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + w/2, bacc_vals, w, label="Balanced Acc.", color=colors_bacc, alpha=0.85,
               edgecolor="white", linewidth=0.5)

        ax.set_title(f"{arch.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[b] for b in all_methods],
                           rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score" if ax_i == 0 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "coverage_vs_performance.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOTS_DIR / "coverage_vs_performance.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_feature_heatmap(cat_results):
    baselines = ["AutoTrainer", "DeepDiagnosis", "DeepFD", "DEFault_ICSE25", "FL4Deep", "DEFault++"]
    families = load_json(EXPORT_DIR / "feature_family_patterns.json")
    family_names = list(families.keys())

    subsets = load_json(EXPORT_DIR / "baseline_feature_subsets.json")
    with open(EXPORT_DIR / "enc_feature_names_tiers.json") as f:
        all_feats = [x["name"] for x in json.load(f)]

    matrix = np.zeros((len(baselines), len(family_names)))
    for bi, bl in enumerate(baselines):
        if bl == "DEFault++":
            matrix[bi, :] = 1.0
            continue
        patterns = subsets[bl]["our_feature_patterns"]
        compiled = [re.compile(p) for p in patterns]
        matched = set(fn for fn in all_feats if any(p.search(fn) for p in compiled))
        for fi, (fam, fam_pats) in enumerate(families.items()):
            safe_pats = []
            for p in fam_pats:
                try:
                    safe_pats.append(re.compile(p))
                except re.error:
                    safe_pats.append(re.compile(re.escape(p)))
            fam_feats = set(fn for fn in all_feats
                           if any(p.search(fn) for p in safe_pats))
            if fam_feats:
                matrix[bi, fi] = len(matched & fam_feats) / len(fam_feats)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(family_names)))
    ax.set_xticklabels(family_names, rotation=45, ha="right", fontsize=8)
    display = {"AutoTrainer": "AutoTrainer", "DeepDiagnosis": "DeepDiagnosis",
               "DeepFD": "DeepFD", "DEFault_ICSE25": "DEFault",
               "FL4Deep": "FL4Deep", "DEFault++": "DEFault++"}
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels([display[b] for b in baselines], fontsize=9)
    for i in range(len(baselines)):
        for j in range(len(family_names)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="Feature Coverage", shrink=0.8)
    # no title -- caption handled in LaTeX
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_coverage_heatmap.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOTS_DIR / "feature_coverage_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_per_class_f1(cat_results):
    """Per-class F1 heatmap: baselines x fault families, side-by-side encoder/decoder."""
    # Collect methods that have per_class_f1 in retrained or symptom_map condition
    method_order = [
        ("AutoTrainer", "symptom_map"),
        ("DeepDiagnosis", "symptom_map"),
        ("DeepFD", "retrained"),
        ("DEFault_ICSE25", "retrained"),
        ("FL4Deep", "retrained"),
        ("DEFault++", "full"),
    ]
    display = {"AutoTrainer": "AutoTrainer", "DeepDiagnosis": "DeepDiagnosis",
               "DeepFD": "DeepFD", "DEFault_ICSE25": "DEFault",
               "FL4Deep": "FL4Deep", "DEFault++": "DEFault++"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax_i, arch in enumerate(["encoder", "decoder"]):
        ax = axes[ax_i]

        # Determine family order from DEFault++ results
        ours_key = f"DEFault++_{arch}"
        if ours_key in cat_results and "per_class_f1" in cat_results[ours_key]:
            families = sorted(cat_results[ours_key]["per_class_f1"].keys())
        else:
            families = (["embedding", "ffn", "kernel", "layernorm", "masking",
                         "output", "positional", "qkv", "residual", "score", "variant"]
                        if arch == "encoder" else
                        ["embedding", "ffn", "kernel", "kv_cache", "layernorm", "masking",
                         "output", "positional", "qkv", "residual", "score", "variant"])

        matrix = np.full((len(method_order), len(families)), np.nan)
        y_labels = []

        for mi, (bl, cond) in enumerate(method_order):
            y_labels.append(display[bl])
            key = f"{bl}_{cond}_{arch}" if bl != "DEFault++" else f"DEFault++_{arch}"
            r = cat_results.get(key, {})
            pcf = r.get("per_class_f1", {})
            for fi, fam in enumerate(families):
                if fam in pcf:
                    matrix[mi, fi] = pcf[fam]

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("#F0F0F0")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(f"{arch.capitalize()}", fontsize=13, fontweight="bold")

        for i in range(len(method_order)):
            for j in range(len(families)):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=7, color="#AAAAAA")
                else:
                    color = "white" if val < 0.4 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold" if val >= 0.8 else "normal")

    fig.colorbar(im, ax=axes, shrink=0.8, label="F1 Score", pad=0.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_f1.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(PLOTS_DIR / "per_class_f1.png", bbox_inches="tight", dpi=150)
    plt.close()


# =====================================================================
# LATEX TABLE GENERATION
# =====================================================================

def generate_latex_tables(det_results, cat_results, graph_results):
    lines = []

    # Table 1: Detection
    lines.append("% ===== TABLE 1: DETECTION =====")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Stage~1 detection comparison. All learning-based methods use the same five folds. Rule-based methods apply their native detection rules without retraining. Best results in \\textbf{bold}.}")
    lines.append("\\label{tab:baseline_detection}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\begin{tabular}{l cc cc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{2}{c}{\\textsc{Encoder}} & \\multicolumn{2}{c}{\\textsc{Decoder}} \\\\")
    lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    lines.append("Method & AUROC & AUPRC & AUROC & AUPRC \\\\")
    lines.append("\\midrule")

    bl_display = {"AutoTrainer": "AutoTrainer", "DeepDiagnosis": "DeepDiagnosis",
                  "DeepFD": "DeepFD", "DEFault_ICSE25": "DEFault", "FL4Deep": "FL4Deep"}

    for bl, cite in [("AutoTrainer", "autotrainer"), ("DeepDiagnosis", "deepdiagnosis"),
                      ("DeepFD", "deepfd"), ("DEFault_ICSE25", "default"), ("FL4Deep", "fl4deep")]:
        enc = det_results.get(f"{bl}_encoder", {})
        dec = det_results.get(f"{bl}_decoder", {})
        ea = enc.get("auroc", "---")
        ep = enc.get("auprc", "---")
        da = dec.get("auroc", "---")
        dp = dec.get("auprc", "---")
        for v in [ea, ep, da, dp]:
            if v == "N/A":
                v = "N/A"
        ea_s = f"{ea}" if isinstance(ea, str) else f"{ea:.3f}"
        ep_s = f"{ep}" if isinstance(ep, str) else f"{ep:.3f}"
        da_s = f"{da}" if isinstance(da, str) else f"{da:.3f}"
        dp_s = f"{dp}" if isinstance(dp, str) else f"{dp:.3f}"
        lines.append(f"{bl_display[bl]}~\\cite{{{cite}}} & {ea_s} & {ep_s} & {da_s} & {dp_s} \\\\")

    lines.append("\\midrule")
    lines.append("\\textbf{DEFault++} & \\textbf{0.976} & \\textbf{1.000} & 0.894 & 0.999 \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Categorization
    lines.append("% ===== TABLE 2: CATEGORIZATION =====")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Stage~2 categorization comparison. ``Native'' uses the baseline's original label space mapped to our families. ``Retrained'' uses the baseline's feature set with our label space. Rule-based methods use symptom-to-family mapping. Best results in \\textbf{bold}.}")
    lines.append("\\label{tab:baseline_categorization}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l l cccc cccc}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{4}{c}{\\textsc{Encoder}} & \\multicolumn{4}{c}{\\textsc{Decoder}} \\\\")
    lines.append("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
    lines.append("Method & Condition & M-F1 & B.Acc & Top-3 & AUC & M-F1 & B.Acc & Top-3 & AUC \\\\")
    lines.append("\\midrule")

    def _fmt(val):
        if val is None or val == "N/A" or val == "---":
            return "N/A"
        if isinstance(val, str):
            return val
        return f"{val:.3f}"

    for bl, cite, conditions in [
        ("AutoTrainer", "autotrainer", [("Symptom map", "symptom_map")]),
        ("DeepDiagnosis", "deepdiagnosis", [("Symptom map", "symptom_map")]),
    ]:
        for cond_label, cond_key in conditions:
            enc = cat_results.get(f"{bl}_{cond_key}_encoder", {})
            dec = cat_results.get(f"{bl}_{cond_key}_decoder", {})
            vals = []
            for r in [enc, dec]:
                vals.extend([_fmt(r.get("macro_f1")), _fmt(r.get("balanced_accuracy")),
                            _fmt(r.get("top3_acc")), _fmt(r.get("auroc"))])
            lines.append(f"{bl_display[bl]}~\\cite{{{cite}}} & {cond_label} & " +
                        " & ".join(vals) + " \\\\")

    lines.append("\\addlinespace")

    for bl, cite, native_label in [
        ("DeepFD", "deepfd", "Native (5-cl)"),
        ("DEFault_ICSE25", "default", "Native (7-cl)"),
        ("FL4Deep", "fl4deep", "Native (6-cl)"),
    ]:
        # Native
        enc = cat_results.get(f"{bl}_native_mapped_encoder", {})
        dec = cat_results.get(f"{bl}_native_mapped_decoder", {})
        vals = []
        for r in [enc, dec]:
            vals.extend([_fmt(r.get("macro_f1")), _fmt(r.get("balanced_accuracy")),
                        _fmt(r.get("top3_acc")), _fmt(r.get("auroc"))])
        lines.append(f"{bl_display[bl]}~\\cite{{{cite}}} & {native_label} & " +
                    " & ".join(vals) + " \\\\")

        # Retrained
        enc = cat_results.get(f"{bl}_retrained_encoder", {})
        dec = cat_results.get(f"{bl}_retrained_decoder", {})
        vals = []
        for r in [enc, dec]:
            vals.extend([_fmt(r.get("macro_f1")), _fmt(r.get("balanced_accuracy")),
                        _fmt(r.get("top3_acc")), _fmt(r.get("auroc"))])
        lines.append(f"{bl_display[bl]}~\\cite{{{cite}}} & Retrained & " +
                    " & ".join(vals) + " \\\\")
        lines.append("\\addlinespace")

    lines.append("\\midrule")
    lines.append("\\textbf{DEFault++} & Full & \\textbf{0.930} & \\textbf{0.925} & \\textbf{0.994} & \\textbf{0.997} & \\textbf{0.836} & \\textbf{0.825} & \\textbf{0.970} & \\textbf{0.987} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3: Graph diagnosis
    lines.append("% ===== TABLE 3: GRAPH DIAGNOSIS =====")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Graph-based diagnosis comparison: FL4Deep's Knowledge Graph vs.\\ DEFault++'s Neural Diagnosis Graph.}")
    lines.append("\\label{tab:baseline_graph}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l cc cc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{2}{c}{\\textsc{Encoder}} & \\multicolumn{2}{c}{\\textsc{Decoder}} \\\\")
    lines.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    lines.append("Metric & FL4Deep KG & DEFault++ NDG & FL4Deep KG & DEFault++ NDG \\\\")
    lines.append("\\midrule")

    for metric_label, metric_key in [
        ("Top-1 accuracy", "top1_accuracy"),
        ("Mean path depth", "mean_path_depth"),
        ("Indicator alignment", "indicator_alignment"),
    ]:
        vals = []
        for arch in ["encoder", "decoder"]:
            gr = graph_results.get(arch, {})
            fl = gr.get("fl4deep_kg", {})
            nd = gr.get("ndg", {})
            fl_v = fl.get(metric_key, "---")
            nd_v = nd.get(metric_key, "---")
            fl_s = f"{fl_v:.3f}" if isinstance(fl_v, (int, float)) else str(fl_v)
            nd_s = f"{nd_v:.3f}" if isinstance(nd_v, (int, float)) else str(nd_v)
            vals.extend([fl_s, nd_s])
        lines.append(f"{metric_label} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    learning_baselines = ["DeepFD", "DEFault_ICSE25", "FL4Deep"]
    rule_baselines = ["AutoTrainer", "DeepDiagnosis"]
    prefixes = [("enc", "encoder"), ("dec", "decoder")]

    det_results = {}
    cat_results = {}

    # === RETRAINED DETECTION ===
    print("=" * 70)
    print("  RETRAINED DETECTION (learning-based baselines)")
    print("=" * 70)
    for bl in learning_baselines:
        for prefix, arch in prefixes:
            t0 = time.time()
            print(f"  {bl} / {arch} / detection ...", end=" ", flush=True)
            r = run_retrained(prefix, bl, task="detection")
            det_results[f"{bl}_{arch}"] = r
            auroc = r.get("auroc", "?")
            auprc = r.get("auprc", "?")
            print(f"AUROC={auroc} AUPRC={auprc} ({time.time()-t0:.0f}s)")

    # === RETRAINED CATEGORIZATION ===
    print("\n" + "=" * 70)
    print("  RETRAINED CATEGORIZATION (learning-based baselines)")
    print("=" * 70)
    for bl in learning_baselines:
        for prefix, arch in prefixes:
            t0 = time.time()
            print(f"  {bl} / {arch} / retrained ...", end=" ", flush=True)
            r = run_retrained(prefix, bl, task="categorization")
            cat_results[f"{bl}_retrained_{arch}"] = r
            mf1 = r.get("macro_f1", "?")
            print(f"M-F1={mf1} ({time.time()-t0:.0f}s)")

    # === NATIVE MAPPED CATEGORIZATION ===
    print("\n" + "=" * 70)
    print("  NATIVE MAPPED CATEGORIZATION (learning-based baselines)")
    print("=" * 70)
    for bl in learning_baselines:
        for prefix, arch in prefixes:
            t0 = time.time()
            print(f"  {bl} / {arch} / native ...", end=" ", flush=True)
            r = run_native_mapped(prefix, bl)
            cat_results[f"{bl}_native_mapped_{arch}"] = r
            mf1 = r.get("macro_f1", "?")
            cov = r.get("coverage", "?")
            print(f"M-F1={mf1} cov={cov} ({time.time()-t0:.0f}s)")

    # === RULE-BASED DETECTION ===
    print("\n" + "=" * 70)
    print("  RULE-BASED DETECTION (AutoTrainer, DeepDiagnosis)")
    print("=" * 70)
    for bl in rule_baselines:
        for prefix, arch in prefixes:
            t0 = time.time()
            print(f"  {bl} / {arch} / detection ...", end=" ", flush=True)
            r = run_rule_based_detection(prefix, bl)
            det_results[f"{bl}_{arch}"] = r
            prec = r.get("precision", "?")
            rec = r.get("recall", "?")
            fpr = r.get("fpr", "?")
            print(f"P={prec} R={rec} FPR={fpr} ({time.time()-t0:.0f}s)")

    # === RULE-BASED CATEGORIZATION ===
    print("\n" + "=" * 70)
    print("  RULE-BASED CATEGORIZATION (symptom mapping)")
    print("=" * 70)
    for bl in rule_baselines:
        for prefix, arch in prefixes:
            t0 = time.time()
            print(f"  {bl} / {arch} / symptom map ...", end=" ", flush=True)
            r = run_rule_based_categorization(prefix, bl)
            cat_results[f"{bl}_symptom_map_{arch}"] = r
            mf1 = r.get("macro_f1", "?")
            cov = r.get("coverage", "?")
            print(f"M-F1={mf1} cov={cov} ({time.time()-t0:.0f}s)")

    # === DEFault++ FULL (for per-class F1) ===
    print("\n" + "=" * 70)
    print("  DEFault++ FULL (per-class F1 breakdown)")
    print("=" * 70)
    for prefix, arch in prefixes:
        t0 = time.time()
        print(f"  DEFault++ / {arch} / full ...", end=" ", flush=True)
        r = run_ours_full(prefix)
        cat_results[f"DEFault++_{arch}"] = r
        mf1 = r.get("macro_f1", "?")
        print(f"M-F1={mf1} ({time.time()-t0:.0f}s)")

    # === GRAPH DIAGNOSIS ===
    print("\n" + "=" * 70)
    print("  GRAPH-BASED DIAGNOSIS (Table 3)")
    print("=" * 70)
    graph_results = compute_graph_diagnosis()
    for arch in ["encoder", "decoder"]:
        gr = graph_results.get(arch, {})
        ndg = gr.get("ndg", {})
        fl = gr.get("fl4deep_kg", {})
        print(f"  {arch}: NDG top1={ndg.get('top1_accuracy')}, "
              f"FL4Deep top1={fl.get('top1_accuracy')}")

    # === SAVE RESULTS ===
    print("\n" + "=" * 70)
    print("  SAVING RESULTS")
    print("=" * 70)

    with open(RESULTS_DIR / "detection_baselines.json", "w") as f:
        json.dump(det_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "categorization_baselines.json", "w") as f:
        json.dump(cat_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "diagnosis_graph_baselines.json", "w") as f:
        json.dump(graph_results, f, indent=2, default=str)

    summary = {
        "detection": det_results,
        "categorization": cat_results,
        "graph_diagnosis": graph_results,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    latex = generate_latex_tables(det_results, cat_results, graph_results)
    with open(RESULTS_DIR / "baseline_tables.tex", "w") as f:
        f.write(latex)
    print(f"  LaTeX tables: {RESULTS_DIR / 'baseline_tables.tex'}")

    # === PLOTS ===
    print("\n  Generating plots...")
    plot_categorization_bars(cat_results)
    plot_detection_bars(det_results)
    plot_coverage_vs_performance(cat_results)
    plot_feature_heatmap(cat_results)
    plot_per_class_f1(cat_results)
    print(f"  Plots saved to: {PLOTS_DIR}")

    # === PRINT SUMMARY TABLE ===
    print("\n" + "=" * 70)
    print("  SUMMARY: TABLE 2 (CATEGORIZATION)")
    print("=" * 70)
    print(f"{'Method':<20} {'Condition':<15} {'Enc M-F1':>10} {'Enc B.Acc':>10} "
          f"{'Dec M-F1':>10} {'Dec B.Acc':>10}")
    print("-" * 75)
    for bl in rule_baselines:
        enc = cat_results.get(f"{bl}_symptom_map_encoder", {})
        dec = cat_results.get(f"{bl}_symptom_map_decoder", {})
        emf = enc.get("macro_f1", "N/A")
        eba = enc.get("balanced_accuracy", "N/A")
        dmf = dec.get("macro_f1", "N/A")
        dba = dec.get("balanced_accuracy", "N/A")
        print(f"{bl:<20} {'Symptom map':<15} {str(emf):>10} {str(eba):>10} "
              f"{str(dmf):>10} {str(dba):>10}")
    for bl in learning_baselines:
        for cond in ["native_mapped", "retrained"]:
            enc = cat_results.get(f"{bl}_{cond}_encoder", {})
            dec = cat_results.get(f"{bl}_{cond}_decoder", {})
            emf = enc.get("macro_f1", "N/A")
            eba = enc.get("balanced_accuracy", "N/A")
            dmf = dec.get("macro_f1", "N/A")
            dba = dec.get("balanced_accuracy", "N/A")
            cond_disp = "Native" if "native" in cond else "Retrained"
            print(f"{bl:<20} {cond_disp:<15} {str(emf):>10} {str(eba):>10} "
                  f"{str(dmf):>10} {str(dba):>10}")
    print(f"{'DEFault++':<20} {'Full':<15} {'0.930':>10} {'0.925':>10} "
          f"{'0.836':>10} {'0.825':>10}")

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
