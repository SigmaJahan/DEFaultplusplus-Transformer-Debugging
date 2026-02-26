"""RQ5: Within-family root cause diagnosis via signature matching.

Evaluates Stage 3's ability to identify root-cause subcategories within each
fault family using behavioral indicator representation and fault signature
profiles. Reports per-family Macro-F1 under oracle gating (ground-truth family)
and predicted gating (Stage 2 XGBoost output), plus calibration analysis.

Self-contained version for Compute Canada.
  python run_rq5_signature_matching.py --arch both
"""
import argparse, json, pickle, re, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PKG = Path(__file__).resolve().parent
DATA_DIR = PKG / "data"
REPO_ROOT = PKG.parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "stage_3_diagnosis"
N_SPLITS = 5
RNG = 42

_TOKEN_RULES = [
    (r"(?:^|_)kv_cache(?:_|$)|(?:^|_)cache_|cache_hidden|cache_nll", "kv_cache"),
    (r"peak_mem|mem_|step_time|latency|runtime|kernel", "runtime"),
    (r"(?:^|_)ffn(?:_|$)|mlp|activation", "ffn"),
    (r"(?:^|_)ln(?:_|$)|layernorm", "layernorm"),
    (r"residual", "residual"),
    (r"(?:^|_)pos(?:_|$)|position|positional", "positional"),
    (r"(?:^|_)qkv(?:_|$)|presoftmax|head_similarity|attn|mass_|entropy", "attention"),
    (r"logit|margin|ece|calib", "output_logits"),
    (r"loss|accuracy|f1|perplexity|nll|precision|recall", "performance"),
    (r"grad|update_ratio|weight_|gns", "gradient"),
    (r"(?:^|_)repr(?:_|$)|h1_|drift|cos", "representation"),
    (r"(?:^|_)emb(?:_|$)|embedding", "embedding"),
    (r"severity|layer_idx|arch_", "structural"),
]


def feature_to_subsystem(feat_name):
    c = feat_name.lower()
    for pat, sub in _TOKEN_RULES:
        if re.search(pat, c):
            return sub
    return "other"


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


_STAT_SUFFIXES = [
    "_early_mean", "_early_slope", "_mid_mean", "_mid_slope", "_final",
    "_finalwin", "_mean_finalwin",
]


def _core_name(feat):
    f = feat
    for suf in _STAT_SUFFIXES:
        if f.endswith(suf):
            f = f[:-len(suf)]
            break
    f = re.sub(r"_l\d+$", "", f)
    return f


def build_indicator_vectors(X, feat_names, baseline_stats, active_indicators=None):
    med, mad = baseline_stats["median"], baseline_stats["mad"]
    X_z = (X - med) / np.maximum(mad, 1e-8)
    X_z = np.clip(X_z, -6, 6)

    groups = defaultdict(list)
    for i, fn in enumerate(feat_names):
        if fn in ("arch_enc", "layer_idx_num", "severity_scalar"):
            continue
        cn = _core_name(fn)
        sub = feature_to_subsystem(fn)
        key = f"{cn}__{sub}"
        groups[key].append(i)

    indicator_names = sorted(groups.keys())
    if active_indicators is not None:
        indicator_names = [n for n in indicator_names if n in active_indicators]

    Z = np.zeros((X.shape[0], len(indicator_names)), dtype=np.float64)
    for j, name in enumerate(indicator_names):
        cols = groups[name]
        if len(cols) == 1:
            Z[:, j] = X_z[:, cols[0]]
        else:
            block = X_z[:, cols]
            abs_block = np.abs(block)
            max_idx = np.argmax(abs_block, axis=1)
            Z[:, j] = block[np.arange(len(block)), max_idx]
    return Z, indicator_names


def estimate_signatures(Z_train, y_train_sc, indicator_names, subcategory_names):
    family_mean = np.nanmean(Z_train, axis=0)
    signatures = {}
    for sc in subcategory_names:
        mask = y_train_sc == sc
        n_sc = mask.sum()
        if n_sc < 3:
            continue
        sc_mean = np.nanmean(Z_train[mask], axis=0)
        if n_sc >= 20:
            signatures[sc] = sc_mean - family_mean
        else:
            signatures[sc] = sc_mean
    return signatures


def signature_match_score(z_instance, signatures, subcategory_names):
    scores = np.zeros(len(subcategory_names))
    for i, sc in enumerate(subcategory_names):
        if sc not in signatures:
            scores[i] = -1e6
            continue
        sig = signatures[sc]
        weights = np.abs(sig)
        weights = weights / (np.sum(weights) + 1e-8)
        scores[i] = np.sum(weights * np.sign(sig) * z_instance)
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / (np.sum(exp_scores) + 1e-12)
    return scores, probs


def compute_ece(y_true_bin, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true_bin[mask].mean()
        ece += mask.sum() * abs(avg_acc - avg_conf)
    return ece / len(y_true_bin)


def run_rq5(arch_prefix):
    arch_name = "encoder" if arch_prefix == "enc" else "decoder"
    print(f"\n{'='*70}")
    print(f"  RQ5: ROOT CAUSE DIAGNOSIS -- {arch_name.upper()}")
    print(f"{'='*70}")

    cat_pkl = DATA_DIR / f"{arch_prefix}_v1_categorization.pkl"
    with open(cat_pkl, "rb") as f:
        data = pickle.load(f)

    X_raw = data["X"].astype(np.float64)
    y_cat = data["y"]
    feat_names = data["feature_names"]
    label_names = data["label_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data.get("norm_groups")
    n_abs = data.get("n_abs_features", X_raw.shape[1])
    n_classes = len(label_names)
    meta = data["meta"]

    orig_csv = DATA_DIR / f"{arch_name}_absolute_filled_labeled.csv"
    orig_df = pd.read_csv(orig_csv, low_memory=False)
    faulty_df = orig_df[orig_df["is_faulty"] == 1].reset_index(drop=True)
    subcategories = faulty_df["fault_subcategory"].values
    categories_str = faulty_df["fault_category"].values

    cat_json = DATA_DIR / f"{arch_prefix}_categorization.json"
    with open(cat_json) as f:
        cat_results = json.load(f)
    xgb_saved = cat_results["experiments"]["XGBoost"]["best_params"]
    xgb_cat_params = {
        "tree_method": "hist", "random_state": RNG, "n_jobs": -1,
        "n_estimators": 2000, "verbosity": 0, "subsample": 0.8,
        "colsample_bytree": 0.8, "objective": "multi:softprob",
        "eval_metric": "mlogloss", "num_class": n_classes,
        "max_depth": int(xgb_saved["max_depth"]),
        "min_child_weight": int(xgb_saved["min_child_weight"]),
        "learning_rate": float(xgb_saved["learning_rate"]),
    }

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y_cat, cv_groups))

    if norm_groups is not None:
        X_folds = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, te in splits]
    else:
        X_folds = [X_raw] * N_SPLITS

    print(f"  {len(y_cat)} samples, {X_raw.shape[1]} features, {n_classes} families")
    unique_sc = sorted(set(subcategories))
    print(f"  {len(unique_sc)} unique subcategories")

    family_subcats = {}
    for fam in label_names:
        mask = categories_str == fam
        scs = sorted(set(subcategories[mask]))
        family_subcats[fam] = scs
        print(f"    {fam:15s}: {sum(mask):5d} samples, {len(scs)} subcats: {scs}")

    n = len(y_cat)
    oof_cat_preds = np.full(n, -1, dtype=int)
    oof_cat_probs = np.zeros((n, n_classes))
    oof_rc_oracle = np.full(n, "", dtype=object)
    oof_rc_predicted = np.full(n, "", dtype=object)
    oof_rc_oracle_prob = np.zeros(n)
    oof_rc_predicted_prob = np.zeros(n)
    oof_rc_oracle_scores = {}
    oof_true_sc = subcategories.copy()

    oof_rc_supervised_oracle = np.full(n, "", dtype=object)
    oof_rc_supervised_predicted = np.full(n, "", dtype=object)

    all_signatures = {}
    all_indicator_names = {}

    for fi, (tr, te) in enumerate(splits):
        print(f"\n  Fold {fi}: {len(te)} test instances...", flush=True)
        Xf = X_folds[fi]
        t0 = time.time()

        fit_c, es_c = _es_split(tr)
        clf_cat = XGBClassifier(**xgb_cat_params)
        sw = compute_sample_weight("balanced", y_cat[fit_c])
        clf_cat.fit(Xf[fit_c], y_cat[fit_c],
                    eval_set=[(Xf[es_c], y_cat[es_c])],
                    sample_weight=sw, verbose=False)
        cat_preds = clf_cat.predict(Xf[te])
        cat_probs = clf_cat.predict_proba(Xf[te])
        if cat_probs.shape[1] < n_classes:
            full = np.zeros((len(te), n_classes))
            for ci, c in enumerate(clf_cat.classes_):
                full[:, c] = cat_probs[:, ci]
            cat_probs = full
        oof_cat_preds[te] = cat_preds
        oof_cat_probs[te] = cat_probs

        scaler = StandardScaler()
        scaler.fit(Xf[tr])
        baseline_stats = {
            "median": np.nanmedian(Xf[tr], axis=0),
            "mad": np.nanmedian(np.abs(Xf[tr] - np.nanmedian(Xf[tr], axis=0)), axis=0),
        }

        for fam_idx, fam in enumerate(label_names):
            scs = family_subcats[fam]
            if len(scs) < 2:
                for i in range(len(te)):
                    idx = te[i]
                    if categories_str[idx] == fam:
                        oof_rc_oracle[idx] = scs[0]
                        oof_rc_oracle_prob[idx] = 1.0
                    if label_names[cat_preds[i]] == fam:
                        oof_rc_predicted[idx] = scs[0]
                        oof_rc_predicted_prob[idx] = 1.0
                        oof_rc_supervised_predicted[idx] = scs[0]
                    if categories_str[idx] == fam:
                        oof_rc_supervised_oracle[idx] = scs[0]
                continue

            fam_tr_mask = categories_str[tr] == fam
            if fam_tr_mask.sum() < 10:
                continue
            fam_tr_indices = tr[fam_tr_mask]
            X_fam_tr = Xf[fam_tr_indices]
            sc_fam_tr = subcategories[fam_tr_indices]

            Z_tr, ind_names = build_indicator_vectors(
                X_fam_tr, feat_names, baseline_stats)

            sigs = estimate_signatures(Z_tr, sc_fam_tr, ind_names, scs)

            if fam not in all_signatures:
                all_signatures[fam] = {sc: [] for sc in scs}
                all_indicator_names[fam] = ind_names
            for sc, sig_vec in sigs.items():
                all_signatures[fam][sc].append(sig_vec.tolist())

            le_sc = LabelEncoder()
            y_rc_tr = le_sc.fit_transform(sc_fam_tr)
            rc_classes = le_sc.classes_.tolist()
            n_rc = len(rc_classes)
            if n_rc >= 2 and len(X_fam_tr) >= 20:
                xgb_rc = {
                    "tree_method": "hist", "random_state": RNG, "n_jobs": -1,
                    "n_estimators": 1000, "verbosity": 0, "subsample": 0.8,
                    "colsample_bytree": 0.8, "objective": "multi:softprob",
                    "eval_metric": "mlogloss", "num_class": n_rc,
                    "max_depth": 3, "min_child_weight": 5, "learning_rate": 0.1,
                }
                fit_r, es_r = _es_split(np.arange(len(X_fam_tr)))
                clf_rc = XGBClassifier(**xgb_rc)
                fit_ok = len(fit_r) >= 5 and len(es_r) >= 2
                fit_ok = fit_ok and len(np.unique(y_rc_tr[fit_r])) >= 2
                if fit_ok:
                    clf_rc.fit(X_fam_tr[fit_r], y_rc_tr[fit_r],
                               eval_set=[(X_fam_tr[es_r], y_rc_tr[es_r])],
                               sample_weight=compute_sample_weight("balanced", y_rc_tr[fit_r]),
                               verbose=False)
                else:
                    clf_rc.fit(X_fam_tr, y_rc_tr,
                               sample_weight=compute_sample_weight("balanced", y_rc_tr),
                               verbose=False)
                has_supervised = True
            else:
                has_supervised = False

            for i in range(len(te)):
                idx = te[i]
                true_fam = categories_str[idx]
                pred_fam = label_names[cat_preds[i]]
                x_test = Xf[idx:idx+1]

                z_inst, _ = build_indicator_vectors(
                    x_test, feat_names, baseline_stats)
                z_vec = z_inst[0] if z_inst.shape[0] > 0 else np.zeros(len(ind_names))

                if true_fam == fam:
                    scores, probs = signature_match_score(z_vec, sigs, scs)
                    best_sc = scs[np.argmax(probs)]
                    oof_rc_oracle[idx] = best_sc
                    oof_rc_oracle_prob[idx] = float(np.max(probs))
                    oof_rc_oracle_scores[idx] = {
                        "family": fam, "true_sc": subcategories[idx],
                        "pred_sc": best_sc, "prob": float(np.max(probs)),
                        "all_probs": {sc: round(float(p), 4) for sc, p in zip(scs, probs)},
                        "indicator_names": ind_names[:10],
                        "indicator_values": z_vec[:10].tolist(),
                    }
                    if has_supervised:
                        sup_pred = int(clf_rc.predict(x_test).ravel()[0])
                        oof_rc_supervised_oracle[idx] = rc_classes[sup_pred]

                if pred_fam == fam:
                    scores, probs = signature_match_score(z_vec, sigs, scs)
                    best_sc = scs[np.argmax(probs)]
                    oof_rc_predicted[idx] = best_sc
                    oof_rc_predicted_prob[idx] = float(np.max(probs))
                    if has_supervised:
                        sup_pred = int(clf_rc.predict(x_test).ravel()[0])
                        oof_rc_supervised_predicted[idx] = rc_classes[sup_pred]

        print(f"    done ({time.time()-t0:.1f}s)")

    print(f"\n  {'='*50}")
    print(f"  RESULTS ({arch_name.upper()})")
    print(f"  {'='*50}")

    per_family_results = {}
    for fam in label_names:
        fam_mask = categories_str == fam
        scs = family_subcats[fam]
        n_fam = fam_mask.sum()
        if n_fam == 0 or len(scs) < 2:
            per_family_results[fam] = {
                "n_samples": int(n_fam),
                "n_subcategories": len(scs),
                "subcategory_names": scs,
                "oracle": {"macro_f1": 1.0 if len(scs) == 1 else None,
                           "balanced_accuracy": 1.0 if len(scs) == 1 else None},
                "predicted": {"macro_f1": None, "balanced_accuracy": None},
                "note": "single subcategory" if len(scs) == 1 else "insufficient data",
            }
            continue

        true_sc_fam = oof_true_sc[fam_mask]

        pred_oracle_fam = oof_rc_oracle[fam_mask]
        valid_oracle = pred_oracle_fam != ""
        if valid_oracle.sum() > 0:
            oracle_f1 = f1_score(true_sc_fam[valid_oracle],
                                 pred_oracle_fam[valid_oracle],
                                 average="macro", zero_division=0)
            oracle_ba = balanced_accuracy_score(true_sc_fam[valid_oracle],
                                               pred_oracle_fam[valid_oracle])
            oracle_acc = accuracy_score(true_sc_fam[valid_oracle],
                                        pred_oracle_fam[valid_oracle])
        else:
            oracle_f1 = oracle_ba = oracle_acc = None

        pred_sup_oracle = oof_rc_supervised_oracle[fam_mask]
        valid_sup = pred_sup_oracle != ""
        if valid_sup.sum() > 0:
            sup_oracle_f1 = f1_score(true_sc_fam[valid_sup],
                                     pred_sup_oracle[valid_sup],
                                     average="macro", zero_division=0)
            sup_oracle_ba = balanced_accuracy_score(true_sc_fam[valid_sup],
                                                    pred_sup_oracle[valid_sup])
        else:
            sup_oracle_f1 = sup_oracle_ba = None

        pred_mask = np.array([label_names[p] == fam for p in oof_cat_preds])
        pred_gated = pred_mask & fam_mask
        pred_rc_vals = oof_rc_predicted[pred_gated]
        valid_pred = pred_rc_vals != ""
        if valid_pred.sum() > 0:
            pred_f1 = f1_score(oof_true_sc[pred_gated][valid_pred],
                               pred_rc_vals[valid_pred],
                               average="macro", zero_division=0)
            pred_ba = balanced_accuracy_score(oof_true_sc[pred_gated][valid_pred],
                                              pred_rc_vals[valid_pred])
        else:
            pred_f1 = pred_ba = None

        per_sc = {}
        if valid_oracle.sum() > 0:
            for sc in scs:
                sc_mask = true_sc_fam[valid_oracle] == sc
                if sc_mask.sum() == 0:
                    continue
                sc_correct = (pred_oracle_fam[valid_oracle][sc_mask] == sc).sum()
                per_sc[sc] = {
                    "support": int(sc_mask.sum()),
                    "accuracy": round(float(sc_correct / sc_mask.sum()), 4),
                }

        pred_sup_vals = oof_rc_supervised_predicted[pred_gated]
        valid_sup_pred = pred_sup_vals != ""
        if valid_sup_pred.sum() > 0:
            sup_pred_f1 = f1_score(oof_true_sc[pred_gated][valid_sup_pred],
                                    pred_sup_vals[valid_sup_pred],
                                    average="macro", zero_division=0)
        else:
            sup_pred_f1 = None

        per_family_results[fam] = {
            "n_samples": int(n_fam),
            "n_subcategories": len(scs),
            "subcategory_names": scs,
            "oracle": {
                "macro_f1": round(float(oracle_f1), 4) if oracle_f1 is not None else None,
                "balanced_accuracy": round(float(oracle_ba), 4) if oracle_ba is not None else None,
                "accuracy": round(float(oracle_acc), 4) if oracle_acc is not None else None,
                "n_evaluated": int(valid_oracle.sum()),
            },
            "supervised_oracle": {
                "macro_f1": round(float(sup_oracle_f1), 4) if sup_oracle_f1 is not None else None,
                "balanced_accuracy": round(float(sup_oracle_ba), 4) if sup_oracle_ba is not None else None,
            },
            "predicted": {
                "macro_f1": round(float(pred_f1), 4) if pred_f1 is not None else None,
                "balanced_accuracy": round(float(pred_ba), 4) if pred_ba is not None else None,
                "n_correctly_gated": int(pred_gated.sum()),
                "n_evaluated": int(valid_pred.sum()),
            },
            "supervised_predicted": {
                "macro_f1": round(float(sup_pred_f1), 4) if sup_pred_f1 is not None else None,
            },
            "per_subcategory": per_sc,
        }

        print(f"    {fam:15s}: oracle F1={oracle_f1:.4f}" if oracle_f1 else f"    {fam:15s}: oracle F1=N/A", end="")
        if sup_oracle_f1 is not None:
            print(f", sup F1={sup_oracle_f1:.4f}", end="")
        if pred_f1 is not None:
            print(f", pred F1={pred_f1:.4f}", end="")
        print(f"  ({n_fam} samples, {len(scs)} subcats)")

    oracle_f1s, oracle_weights = [], []
    sup_f1s, sup_weights = [], []
    pred_f1s, pred_weights = [], []
    for fam, res in per_family_results.items():
        if res["oracle"].get("macro_f1") is not None:
            oracle_f1s.append(res["oracle"]["macro_f1"])
            oracle_weights.append(res["n_samples"])
        if res.get("supervised_oracle", {}).get("macro_f1") is not None:
            sup_f1s.append(res["supervised_oracle"]["macro_f1"])
            sup_weights.append(res["n_samples"])
        if res["predicted"].get("macro_f1") is not None:
            pred_f1s.append(res["predicted"]["macro_f1"])
            pred_weights.append(res["n_samples"])

    agg_oracle_f1 = np.average(oracle_f1s, weights=oracle_weights) if oracle_f1s else None
    agg_sup_f1 = np.average(sup_f1s, weights=sup_weights) if sup_f1s else None
    agg_pred_f1 = np.average(pred_f1s, weights=pred_weights) if pred_f1s else None

    macro_oracle_f1 = np.mean(oracle_f1s) if oracle_f1s else None
    macro_sup_f1 = np.mean(sup_f1s) if sup_f1s else None
    macro_pred_f1 = np.mean(pred_f1s) if pred_f1s else None

    print(f"\n  Aggregate (weighted):")
    print(f"    Oracle gating:     F1 = {agg_oracle_f1:.4f}" if agg_oracle_f1 else "    Oracle gating:     N/A")
    print(f"    Supervised oracle: F1 = {agg_sup_f1:.4f}" if agg_sup_f1 else "    Supervised oracle: N/A")
    print(f"    Predicted gating:  F1 = {agg_pred_f1:.4f}" if agg_pred_f1 else "    Predicted gating:  N/A")
    print(f"\n  Macro-average (unweighted across families):")
    print(f"    Oracle:     {macro_oracle_f1:.4f}" if macro_oracle_f1 else "    Oracle:     N/A")
    print(f"    Supervised: {macro_sup_f1:.4f}" if macro_sup_f1 else "    Supervised: N/A")
    print(f"    Predicted:  {macro_pred_f1:.4f}" if macro_pred_f1 else "    Predicted:  N/A")

    valid_all = oof_rc_oracle != ""
    if valid_all.sum() > 0:
        correct = (oof_rc_oracle[valid_all] == oof_true_sc[valid_all]).astype(int)
        probs_all = oof_rc_oracle_prob[valid_all]
        ece = compute_ece(correct, probs_all)
        try:
            frac_pos, mean_pred = calibration_curve(correct, probs_all, n_bins=10,
                                                     strategy="uniform")
            reliability = [{"mean_predicted": round(float(mp), 4),
                           "fraction_correct": round(float(fp), 4)}
                          for mp, fp in zip(mean_pred, frac_pos)]
        except ValueError:
            reliability = []

        conf_buckets = {"high": {"correct": 0, "total": 0},
                        "moderate": {"correct": 0, "total": 0},
                        "low": {"correct": 0, "total": 0}}
        for prob, corr in zip(probs_all, correct):
            if prob >= 0.8:
                lvl = "high"
            elif prob >= 0.5:
                lvl = "moderate"
            else:
                lvl = "low"
            conf_buckets[lvl]["total"] += 1
            conf_buckets[lvl]["correct"] += int(corr)
        for lvl in conf_buckets:
            t = conf_buckets[lvl]["total"]
            conf_buckets[lvl]["accuracy"] = round(conf_buckets[lvl]["correct"] / t, 4) if t > 0 else 0.0

        print(f"\n  Calibration (oracle gating):")
        print(f"    ECE: {ece:.4f}")
        for lvl in ["high", "moderate", "low"]:
            b = conf_buckets[lvl]
            print(f"    {lvl}: {b['correct']}/{b['total']} = {b['accuracy']:.3f}")
    else:
        ece = None
        reliability = []
        conf_buckets = {}

    cat_f1 = f1_score(y_cat, oof_cat_preds, average="macro", zero_division=0)
    cascade_e2e = cat_f1 * (macro_oracle_f1 if macro_oracle_f1 else 0)

    print(f"\n  Cascade E2E: cat_f1={cat_f1:.4f} x rc_f1={macro_oracle_f1:.4f} = {cascade_e2e:.4f}"
          if macro_oracle_f1 else f"\n  Cascade E2E: N/A")

    signature_heatmaps = {}
    for fam, sc_sigs in all_signatures.items():
        ind_nm = all_indicator_names.get(fam, [])
        fam_data = {"indicator_names": ind_nm, "subcategories": {}}
        for sc, sig_list in sc_sigs.items():
            if not sig_list:
                continue
            arr = np.array(sig_list)
            fam_data["subcategories"][sc] = {
                "mean_signature": np.nanmean(arr, axis=0).tolist(),
                "n_folds": len(sig_list),
                "n_indicators_nonzero": int((np.abs(np.nanmean(arr, axis=0)) > 0.01).sum()),
            }
        signature_heatmaps[fam] = fam_data

    examples = {"correct_high_conf": None, "misdiagnosed": None, "low_confidence": None}
    for idx, info in sorted(oof_rc_oracle_scores.items()):
        is_correct = info["true_sc"] == info["pred_sc"]
        prob = info["prob"]
        if is_correct and prob >= 0.8 and examples["correct_high_conf"] is None:
            examples["correct_high_conf"] = info
        elif not is_correct and examples["misdiagnosed"] is None:
            examples["misdiagnosed"] = info
        elif prob < 0.5 and examples["low_confidence"] is None:
            examples["low_confidence"] = info
        if all(v is not None for v in examples.values()):
            break

    output = {
        "architecture": arch_name,
        "n_samples": len(y_cat),
        "n_families": n_classes,
        "n_subcategories_total": len(unique_sc),
        "aggregate": {
            "oracle_gating": {
                "weighted_macro_f1": round(float(agg_oracle_f1), 4) if agg_oracle_f1 else None,
                "macro_avg_f1": round(float(macro_oracle_f1), 4) if macro_oracle_f1 else None,
            },
            "supervised_oracle": {
                "weighted_macro_f1": round(float(agg_sup_f1), 4) if agg_sup_f1 else None,
                "macro_avg_f1": round(float(macro_sup_f1), 4) if macro_sup_f1 else None,
            },
            "predicted_gating": {
                "weighted_macro_f1": round(float(agg_pred_f1), 4) if agg_pred_f1 else None,
                "macro_avg_f1": round(float(macro_pred_f1), 4) if macro_pred_f1 else None,
            },
            "cascade_e2e": round(float(cascade_e2e), 4),
            "categorization_macro_f1": round(float(cat_f1), 4),
        },
        "calibration": {
            "ece": round(float(ece), 4) if ece is not None else None,
            "reliability_diagram": reliability,
            "confidence_levels": conf_buckets,
        },
        "per_family": per_family_results,
        "signature_heatmaps": signature_heatmaps,
        "example_reports": examples,
    }
    return output


def main():
    p = argparse.ArgumentParser(description="RQ5: Root cause signature matching evaluation")
    p.add_argument("--arch", choices=["enc", "dec", "both"], default="both")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    archs = ["enc", "dec"] if args.arch == "both" else [args.arch]
    for arch in archs:
        t0 = time.time()
        result = run_rq5(arch)
        out_path = RESULTS_DIR / f"{arch}_rq5_signature_matching.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else
                      round(float(o), 6) if isinstance(o, (np.floating,)) else
                      o.tolist() if isinstance(o, np.ndarray) else str(o))
        print(f"\n  Saved: {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
