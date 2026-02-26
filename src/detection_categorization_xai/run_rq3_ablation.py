"""RQ3: Feature family ablation for fault categorization.

Leave-one-family-out: retrain XGBoost with each feature family removed,
report delta Macro-F1 and identify the most affected fault class.

Self-contained version for Compute Canada.
  python run_rq3_ablation.py --arch both
"""
import argparse, json, pickle, re, time, warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PKG = Path(__file__).resolve().parent
DATA_DIR = PKG / "data"
REPO_ROOT = PKG.parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "rq3_ablation"
N_SPLITS = 5
RNG = 42

FEATURE_FAMILIES = {
    "Attention statistics": [
        r"attn_entropy", r"attn_cross_example", r"attn_mass", r"attn_max",
        r"attn_score_skew", r"attn_score_var", r"attn_sparsity",
        r"attn_weight_magnitude", r"head_similarity", r"pos_recv",
        r"presoftmax", r"mass_leak", r"mass_pad", r"cross_example_attn",
    ],
    "QKV alignment": [
        r"update_ratio.*qkv", r"update_active.*qkv", r"grad_norm.*qkv",
    ],
    "Score statistics": [
        r"presoftmax_kurt", r"presoftmax_mean", r"presoftmax_skew",
        r"presoftmax_var", r"attn_score",
    ],
    "Positional discrim.": [
        r"pos_acc", r"pos_loss", r"pos_margin", r"val_pos_inv",
        r"val_pos_margin", r"val_pos_acc",
    ],
    "FFN output magnitude": [
        r"ffn_delta", r"ffn_out_skew", r"ffn_var_ratio", r"ffn_active",
        r"h1_delta_norm", r"activation_mean", r"activation_std",
    ],
    "LayerNorm statistics": [
        r"(?:^|_)ln_", r"layernorm", r"update_ratio.*layernorm",
        r"update_active.*layernorm", r"grad_norm.*layernorm",
    ],
    "Residual stream sim.": [
        r"residual_cos",
    ],
    "Representation drift": [
        r"(?:^|_)repr_", r"(?:^|_)repr$",
    ],
    "Embedding statistics": [
        r"emb_norm", r"update_ratio_emb", r"update_active_emb",
        r"grad_norm_emb",
    ],
    "Training dynamics": [
        r"(?:^|_)loss(?:_|$)", r"val_loss", r"grad_norm_total", r"grad_abs_max",
        r"grad_zero_ratio", r"update_ratio_total", r"update_ratio_classifier",
        r"weight_mean", r"weight_std",
    ],
    "Task metrics": [
        r"accuracy", r"precision(?!_target)", r"recall(?!_at)", r"f1_score",
        r"val_primary_metric", r"val_accuracy", r"val_ece", r"val_edge_case",
        r"val_f1", r"val_nll", r"val_recall", r"val_precision",
        r"perplexity", r"(?:^|_)ece(?:_|$)", r"(?:^|_)nll(?:_|$)",
    ],
    "Kernel timing": [
        r"peak_mem", r"step_time",
    ],
    "Cache diagnostics": [
        r"cache_hidden", r"cache_nll",
    ],
}


def _family_match(feat_name, patterns):
    for pat in patterns:
        if re.search(pat, feat_name):
            return True
    return False


def assign_feature_families(feature_names):
    assignments = {fam: [] for fam in FEATURE_FAMILIES}
    unassigned = []
    for i, fn in enumerate(feature_names):
        if fn in ("arch_enc", "layer_idx_num", "severity_scalar"):
            continue
        matched = False
        for fam, pats in FEATURE_FAMILIES.items():
            if _family_match(fn, pats):
                assignments[fam].append(i)
                matched = True
                break
        if not matched:
            unassigned.append((i, fn))
    for i, fn in unassigned:
        if re.search(r"logit", fn):
            assignments["Task metrics"].append(i)
        elif re.search(r"grad|update", fn):
            assignments["Training dynamics"].append(i)
    return assignments


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


def run_xgb_categorization(X_folds, y, splits, label_names, best_params):
    n = len(y)
    n_classes = len(label_names)
    oof_preds = np.zeros(n, dtype=int)
    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        fit_i, es_i = _es_split(tr)
        clf = XGBClassifier(**best_params)
        sw = compute_sample_weight("balanced", y[fit_i])
        clf.fit(Xf[fit_i], y[fit_i], eval_set=[(Xf[es_i], y[es_i])],
                sample_weight=sw, verbose=False)
        oof_preds[te] = clf.predict(Xf[te])
    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(y, oof_preds, average=None, zero_division=0,
                            labels=range(n_classes))
    return macro_f1, per_class_f1


def run_ablation(arch_prefix):
    pkl_path = DATA_DIR / f"{arch_prefix}_v1_categorization.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X_raw = data["X"].astype(np.float64)
    y = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data.get("norm_groups")
    n_abs = data.get("n_abs_features", X_raw.shape[1])
    n_classes = len(label_names)

    arch_name = "encoder" if arch_prefix == "enc" else "decoder"
    results_json = DATA_DIR / f"{arch_prefix}_categorization.json"
    with open(results_json) as f:
        existing = json.load(f)
    xgb_saved = existing["experiments"]["XGBoost"]["best_params"]
    xgb_base = {
        "tree_method": "hist", "random_state": RNG, "n_jobs": -1,
        "n_estimators": 2000, "verbosity": 0, "subsample": 0.8,
        "colsample_bytree": 0.8, "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "max_depth": int(xgb_saved["max_depth"]),
        "min_child_weight": int(xgb_saved["min_child_weight"]),
        "learning_rate": float(xgb_saved["learning_rate"]),
    }

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))

    if norm_groups is not None:
        X_folds_full = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, te in splits]
    else:
        X_folds_full = [X_raw] * N_SPLITS

    print(f"\n{'='*60}")
    print(f"  RQ3 ABLATION: {arch_name.upper()}")
    print(f"  {len(y)} samples, {X_raw.shape[1]} features, {n_classes} classes")
    print(f"{'='*60}")

    t0 = time.time()
    baseline_f1, baseline_per_class = run_xgb_categorization(
        X_folds_full, y, splits, label_names, xgb_base)
    print(f"\n  Baseline Macro-F1: {baseline_f1:.4f} ({time.time()-t0:.1f}s)")

    family_idx = assign_feature_families(feat_names)
    print(f"\n  Feature family assignments:")
    for fam, idxs in sorted(family_idx.items()):
        print(f"    {fam:25s}: {len(idxs):4d} columns")

    ablation_results = []
    for fam_name, drop_idxs in sorted(family_idx.items()):
        if len(drop_idxs) == 0:
            print(f"  SKIP {fam_name} (0 columns)")
            continue

        keep_mask = np.ones(X_raw.shape[1], dtype=bool)
        keep_mask[drop_idxs] = False
        X_ablated = X_raw[:, keep_mask]
        n_abs_abl = min(n_abs, keep_mask[:n_abs].sum())

        if norm_groups is not None:
            X_folds_abl = [group_zscore(X_ablated, norm_groups, n_abs_abl, tr)
                           for tr, te in splits]
        else:
            X_folds_abl = [X_ablated] * N_SPLITS

        t1 = time.time()
        abl_f1, abl_per_class = run_xgb_categorization(
            X_folds_abl, y, splits, label_names, xgb_base)
        dt = time.time() - t1

        delta = abl_f1 - baseline_f1
        per_class_delta = abl_per_class - baseline_per_class
        most_affected_idx = int(np.argmin(per_class_delta))
        most_affected_class = label_names[most_affected_idx]
        most_affected_delta = float(per_class_delta[most_affected_idx])

        ablation_results.append({
            "feature_family": fam_name,
            "n_columns_removed": len(drop_idxs),
            "n_columns_remaining": int(keep_mask.sum()),
            "macro_f1": round(float(abl_f1), 4),
            "delta_macro_f1": round(float(delta), 4),
            "most_affected_class": most_affected_class,
            "most_affected_delta_f1": round(float(most_affected_delta), 4),
            "per_class_f1": {label_names[c]: round(float(abl_per_class[c]), 4)
                            for c in range(n_classes)},
            "per_class_delta_f1": {label_names[c]: round(float(per_class_delta[c]), 4)
                                  for c in range(n_classes)},
        })

        sign = "+" if delta >= 0 else ""
        print(f"  -{fam_name:25s}: F1={abl_f1:.4f} ({sign}{delta:.4f}), "
              f"worst={most_affected_class}({most_affected_delta:+.4f}), "
              f"dropped {len(drop_idxs)} cols ({dt:.1f}s)")

    ablation_results.sort(key=lambda x: x["delta_macro_f1"])

    output = {
        "architecture": arch_name,
        "baseline_macro_f1": round(float(baseline_f1), 4),
        "baseline_per_class_f1": {label_names[c]: round(float(baseline_per_class[c]), 4)
                                  for c in range(n_classes)},
        "n_samples": len(y),
        "n_features_total": X_raw.shape[1],
        "n_classes": n_classes,
        "label_names": label_names,
        "xgb_params": {k: v for k, v in xgb_base.items()
                       if isinstance(v, (int, float, str, bool))},
        "ablation_results": ablation_results,
    }
    return output


def main():
    p = argparse.ArgumentParser(description="RQ3: Feature family ablation")
    p.add_argument("--arch", choices=["enc", "dec", "both"], default="both")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    archs = ["enc", "dec"] if args.arch == "both" else [args.arch]
    for arch in archs:
        t0 = time.time()
        result = run_ablation(arch)
        out_path = RESULTS_DIR / f"{arch}_ablation.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
