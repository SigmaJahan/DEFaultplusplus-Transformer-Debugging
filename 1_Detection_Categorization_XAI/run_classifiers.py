"""
4-model classifier pipeline for FrankenFormer detection/categorization.
Models: ElasticNet LR, Calibrated RBF SVM, XGBoost, EasyEnsembleClassifier.
Per-group z-score applied per-fold (fit on train only) -- no preprocessing leak.
Usage:
  python run_classifiers.py --data data/enc_v1_detection.pkl --out results/enc_det.json
  python run_classifiers.py --data data/dec_v1_categorization.pkl
"""
import argparse, json, pickle, time, warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score,
    average_precision_score, top_k_accuracy_score, confusion_matrix,
    precision_recall_curve, precision_score, recall_score)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.ensemble import EasyEnsembleClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

N_SPLITS = 5
RNG = 42


def group_zscore(X_raw, norm_groups, n_abs, train_idx):
    """Per-(model,dataset) group z-score: fit stats on train_idx only, apply to all.
    Only the first n_abs columns (abs_ features) are normalized; structural columns are untouched."""
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


def build_models(binary):
    models = {}
    models["ElasticNet_LR"] = {
        "base": {"penalty": "elasticnet", "solver": "saga", "max_iter": 5000,
                 "class_weight": "balanced", "random_state": RNG},
        "grid": {"lr__C": [0.1, 1.0, 10.0], "lr__l1_ratio": [0.0, 0.5, 1.0]},
    }
    models["RBF_SVM"] = {
        "base": {"kernel": "rbf", "class_weight": "balanced", "random_state": RNG},
        "grid": {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1, 1.0]},
    }
    xgb_base = {"tree_method": "hist", "random_state": RNG, "n_jobs": -1,
                "n_estimators": 2000, "verbosity": 0,
                "subsample": 0.8, "colsample_bytree": 0.8}
    if binary:
        xgb_base.update({"objective": "binary:logistic", "eval_metric": "aucpr"})
    else:
        xgb_base.update({"objective": "multi:softprob", "eval_metric": "mlogloss"})
    models["XGBoost"] = {
        "base": xgb_base,
        "grid_combos": [{"max_depth": d, "min_child_weight": w, "learning_rate": lr}
                        for d in [2, 3, 4] for w in [5, 10] for lr in [0.03, 0.1]],
    }
    models["EasyEnsemble"] = {
        "base": {"random_state": 0, "n_jobs": -1,
                 "estimator": AdaBoostClassifier(
                     estimator=DecisionTreeClassifier(max_depth=1),
                     n_estimators=200, learning_rate=0.5, random_state=0)},
        "grid": {"n_estimators": [10, 20]},
    }
    return models


def _pos_idx(label_names):
    if "correct" in label_names:
        return 1 - label_names.index("correct")
    return 0


def find_threshold(y_bin, prob_pos, target=0.95, fallback=0.90):
    prec, rec, thr = precision_recall_curve(y_bin, prob_pos)
    for t in [target, fallback]:
        valid = prec[:-1] >= t
        if valid.any():
            idxs = np.where(valid)[0]
            best = idxs[np.argmax(rec[idxs])]
            return float(thr[best]), t, float(rec[best])
    return 0.5, None, None


def balanced_weights(y):
    return compute_sample_weight("balanced", y)


def _es_split(tr_idx, frac=0.2):
    rng = np.random.RandomState(RNG)
    perm = rng.permutation(len(tr_idx))
    n_es = max(int(frac * len(tr_idx)), 1)
    return tr_idx[perm[n_es:]], tr_idx[perm[:n_es]]


def run_elasticnet(X_folds, y, splits, cfg, label_names, binary):
    base_pipe = Pipeline([("scaler", StandardScaler()),
                          ("lr", LogisticRegression(**cfg["base"]))])
    gs = GridSearchCV(base_pipe, cfg["grid"], cv=3, scoring="f1_weighted",
                      n_jobs=-1, refit=False)
    gs.fit(X_folds[0][splits[0][0]], y[splits[0][0]])
    best_params = dict(cfg["base"])
    for k, v in gs.best_params_.items():
        best_params[k.replace("lr__", "")] = v

    folds = []
    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(**best_params))])
        pipe.fit(Xf[tr], y[tr])
        folds.append({"y_true": y[te], "y_pred": pipe.predict(Xf[te]),
                       "y_prob": pipe.predict_proba(Xf[te])})

    Xf = X_folds[-1]
    pipe_f = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(**best_params))])
    pipe_f.fit(Xf[splits[-1][0]], y[splits[-1][0]])
    perm = permutation_importance(pipe_f, Xf[splits[-1][1]], y[splits[-1][1]],
                                  n_repeats=10, random_state=RNG, n_jobs=-1,
                                  scoring="f1_weighted")
    return gs.best_params_, folds, perm.importances_mean


def run_svm(X_folds, y, splits, cfg, label_names, binary):
    Xf0 = X_folds[0]
    sc0 = StandardScaler().fit(Xf0[splits[0][0]])
    gs = GridSearchCV(SVC(**cfg["base"]), cfg["grid"], cv=3,
                      scoring="f1_weighted", n_jobs=-1, refit=False)
    gs.fit(sc0.transform(Xf0[splits[0][0]]), y[splits[0][0]])
    best_svc = {**cfg["base"], **gs.best_params_}

    folds = []
    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        sc = StandardScaler().fit(Xf[tr])
        cal = CalibratedClassifierCV(SVC(**best_svc), method="sigmoid", cv=3)
        cal.fit(sc.transform(Xf[tr]), y[tr])
        folds.append({"y_true": y[te], "y_pred": cal.predict(sc.transform(Xf[te])),
                       "y_prob": cal.predict_proba(sc.transform(Xf[te]))})

    Xf = X_folds[-1]
    sc_f = StandardScaler().fit(Xf[splits[-1][0]])
    cal_f = CalibratedClassifierCV(SVC(**best_svc), method="sigmoid", cv=3)
    cal_f.fit(sc_f.transform(Xf[splits[-1][0]]), y[splits[-1][0]])
    perm = permutation_importance(cal_f, sc_f.transform(Xf[splits[-1][1]]), y[splits[-1][1]],
                                  n_repeats=10, random_state=RNG, n_jobs=-1,
                                  scoring="f1_weighted")
    return gs.best_params_, folds, perm.importances_mean


def run_xgboost(X_folds, y, splits, cfg, label_names, binary):
    base = dict(cfg["base"])
    if binary:
        neg, pos_c = np.bincount(y)
        base["scale_pos_weight"] = neg / max(pos_c, 1)

    Xf0 = X_folds[0]
    fit0, es0 = _es_split(splits[0][0])
    best_score, best_combo = -1, {}
    for combo in cfg["grid_combos"]:
        clf = XGBClassifier(**{**base, **combo})
        sw = None if binary else balanced_weights(y[fit0])
        clf.fit(Xf0[fit0], y[fit0], eval_set=[(Xf0[es0], y[es0])],
                sample_weight=sw, verbose=False)
        sc = f1_score(y[es0], clf.predict(Xf0[es0]),
                      average="weighted", zero_division=0)
        if sc > best_score:
            best_score, best_combo = sc, combo
    best_params = {**base, **best_combo}

    folds = []
    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        fit_i, es_i = _es_split(tr)
        clf = XGBClassifier(**best_params)
        sw = None if binary else balanced_weights(y[fit_i])
        clf.fit(Xf[fit_i], y[fit_i], eval_set=[(Xf[es_i], y[es_i])],
                sample_weight=sw, verbose=False)
        folds.append({"y_true": y[te], "y_pred": clf.predict(Xf[te]),
                       "y_prob": clf.predict_proba(Xf[te])})

    Xf = X_folds[-1]
    fit_f, es_f = _es_split(splits[-1][0])
    clf_f = XGBClassifier(**best_params)
    sw_f = None if binary else balanced_weights(y[fit_f])
    clf_f.fit(Xf[fit_f], y[fit_f], eval_set=[(Xf[es_f], y[es_f])],
              sample_weight=sw_f, verbose=False)
    perm = permutation_importance(clf_f, Xf[splits[-1][1]], y[splits[-1][1]],
                                  n_repeats=10, random_state=RNG, n_jobs=-1,
                                  scoring="f1_weighted")
    clean = dict(best_combo)
    if binary:
        clean["scale_pos_weight"] = round(best_params["scale_pos_weight"], 3)
    return clean, folds, perm.importances_mean


def run_easyensemble(X_folds, y, splits, cfg, label_names, binary):
    Xf0 = X_folds[0]
    gs = GridSearchCV(EasyEnsembleClassifier(**cfg["base"]), cfg["grid"],
                      cv=3, scoring="f1_weighted", n_jobs=-1, refit=False)
    gs.fit(Xf0[splits[0][0]], y[splits[0][0]])
    best_params = {**cfg["base"], **gs.best_params_}

    folds = []
    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        clf = EasyEnsembleClassifier(**best_params)
        clf.fit(Xf[tr], y[tr])
        folds.append({"y_true": y[te], "y_pred": clf.predict(Xf[te]),
                       "y_prob": clf.predict_proba(Xf[te])})

    Xf = X_folds[-1]
    clf_f = EasyEnsembleClassifier(**best_params)
    clf_f.fit(Xf[splits[-1][0]], y[splits[-1][0]])
    perm = permutation_importance(clf_f, Xf[splits[-1][1]], y[splits[-1][1]],
                                  n_repeats=10, random_state=RNG, n_jobs=-1,
                                  scoring="f1_weighted")
    return gs.best_params_, folds, perm.importances_mean


RUNNERS = {"ElasticNet_LR": run_elasticnet, "RBF_SVM": run_svm,
           "XGBoost": run_xgboost, "EasyEnsemble": run_easyensemble}


def detection_fold_metrics(y_true, y_prob, pos, threshold, prec_target):
    yt_bin = (y_true == pos).astype(int)
    m = {}
    try:
        m["auroc"] = round(roc_auc_score(yt_bin, y_prob[:, pos]), 4)
        m["auprc"] = round(average_precision_score(yt_bin, y_prob[:, pos]), 4)
    except ValueError:
        m["auroc"] = m["auprc"] = None
    y_thr = (y_prob[:, pos] >= threshold).astype(int)
    m["threshold"] = round(threshold, 4)
    m["precision_target"] = prec_target
    m["recall_at_thr"] = round(recall_score(yt_bin, y_thr, zero_division=0), 4)
    m["precision_at_thr"] = round(precision_score(yt_bin, y_thr, zero_division=0), 4)
    m["f1_at_thr"] = round(f1_score(yt_bin, y_thr, zero_division=0), 4)
    cm = confusion_matrix(yt_bin, y_thr, labels=[0, 1])
    m["cm_at_thr"] = cm.tolist()
    return m


def categorization_fold_metrics(y_true, y_pred, y_prob, n_classes, label_names):
    m = {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }
    try:
        m["auroc"] = round(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                         average="weighted"), 4)
    except ValueError:
        m["auroc"] = None
    for k in [3, 5]:
        if k <= n_classes:
            try:
                m[f"top{k}_acc"] = round(top_k_accuracy_score(
                    y_true, y_prob, k=k, labels=range(n_classes)), 4)
            except ValueError:
                m[f"top{k}_acc"] = None
    return m


def aggregate_results(fold_results, label_names, binary, splits, y):
    n_classes = len(label_names)
    pos = _pos_idx(label_names) if binary else None
    n = len(y)

    oof_probs = np.zeros((n, n_classes))
    oof_preds = np.zeros(n, dtype=int)
    for fr, (tr, te) in zip(fold_results, splits):
        oof_probs[te] = fr["y_prob"]
        oof_preds[te] = fr["y_pred"]

    per_fold = []
    for i, (fr, (tr, te)) in enumerate(zip(fold_results, splits)):
        yt, yprob = fr["y_true"], fr["y_prob"]
        fold_info = {"fold": i, "n_test": len(yt),
                     "class_dist_test": {label_names[c]: int(v) for c, v in
                                         enumerate(np.bincount(yt, minlength=n_classes))},
                     "class_dist_train": {label_names[c]: int(v) for c, v in
                                          enumerate(np.bincount(y[tr], minlength=n_classes))}}
        if binary:
            cal_y_bin = (y[tr] == pos).astype(int)
            cal_prob_pos = oof_probs[tr, pos]
            thr, ptgt, _ = find_threshold(cal_y_bin, cal_prob_pos)
            fold_info["metrics"] = detection_fold_metrics(yt, yprob, pos, thr, ptgt)
        else:
            fold_info["metrics"] = categorization_fold_metrics(yt, fr["y_pred"], yprob, n_classes, label_names)
        per_fold.append(fold_info)

    if binary:
        yt_bin = (y == pos).astype(int)
        overall = {
            "f1_macro": round(f1_score(y, oof_preds, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(y, oof_preds, average="weighted", zero_division=0), 4),
            "positive_class": label_names[pos],
            "base_rate": round(yt_bin.mean(), 4),
        }
        try:
            overall["auroc"] = round(roc_auc_score(yt_bin, oof_probs[:, pos]), 4)
            overall["auprc"] = round(average_precision_score(yt_bin, oof_probs[:, pos]), 4)
        except ValueError:
            overall["auroc"] = overall["auprc"] = None
        thresholds = [f["metrics"]["threshold"] for f in per_fold if f["metrics"].get("threshold")]
        if thresholds:
            agg_thr = float(np.median(thresholds))
            y_thr_all = (oof_probs[:, pos] >= agg_thr).astype(int)
            overall["agg_threshold"] = round(agg_thr, 4)
            overall["recall_at_agg_thr"] = round(recall_score(yt_bin, y_thr_all, zero_division=0), 4)
            overall["precision_at_agg_thr"] = round(precision_score(yt_bin, y_thr_all, zero_division=0), 4)
            overall["f1_at_agg_thr"] = round(f1_score(yt_bin, y_thr_all, zero_division=0), 4)
            overall["cm_at_agg_thr"] = confusion_matrix(yt_bin, y_thr_all, labels=[0, 1]).tolist()
        aurocs = [f["metrics"]["auroc"] for f in per_fold if f["metrics"].get("auroc") is not None]
        if aurocs:
            overall["auroc_mean"] = round(np.mean(aurocs), 4)
            overall["auroc_std"] = round(np.std(aurocs), 4)
    else:
        overall = {
            "macro_f1": round(f1_score(y, oof_preds, average="macro", zero_division=0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y, oof_preds), 4),
            "accuracy": round(accuracy_score(y, oof_preds), 4),
        }
        try:
            overall["auroc"] = round(roc_auc_score(y, oof_probs,
                                                   multi_class="ovr", average="weighted"), 4)
        except ValueError:
            overall["auroc"] = None
        for k in [3, 5]:
            if k <= n_classes:
                try:
                    overall[f"top{k}_acc"] = round(top_k_accuracy_score(
                        y, oof_probs, k=k, labels=range(n_classes)), 4)
                except ValueError:
                    overall[f"top{k}_acc"] = None
        cm = confusion_matrix(y, oof_preds, labels=range(n_classes))
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        overall["confusion_matrix_raw"] = cm.tolist()
        overall["confusion_matrix_normalized"] = np.round(cm_norm, 4).tolist()
        overall["confusion_matrix_labels"] = label_names
        mf1s = [f["metrics"]["macro_f1"] for f in per_fold]
        overall["macro_f1_mean"] = round(np.mean(mf1s), 4)
        overall["macro_f1_std"] = round(np.std(mf1s), 4)

        per_class_f1 = f1_score(y, oof_preds, average=None, zero_division=0, labels=range(n_classes))
        per_class_prec = precision_score(y, oof_preds, average=None, zero_division=0, labels=range(n_classes))
        per_class_rec = recall_score(y, oof_preds, average=None, zero_division=0, labels=range(n_classes))
        support = np.bincount(y, minlength=n_classes)
        overall["per_class"] = {
            label_names[c]: {
                "f1": round(float(per_class_f1[c]), 4),
                "precision": round(float(per_class_prec[c]), 4),
                "recall": round(float(per_class_rec[c]), 4),
                "support": int(support[c]),
            } for c in range(n_classes)
        }

    return {"overall": overall, "per_fold": per_fold}


def run_pipeline(data):
    X, y = data["X"], data["y"]
    feat_names = data["feature_names"]
    tier_map = data["tier_map"]
    cv_groups = data["cv_groups"]
    norm_groups = data.get("norm_groups")
    n_abs = data.get("n_abs_features", X.shape[1])
    label_names = data["label_names"]
    binary = len(label_names) == 2

    has_nan = data.get("has_nan", False)
    X_raw = X.astype(np.float64)

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))

    if norm_groups is not None:
        X_folds = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, te in splits]
        print(f"  Per-fold group z-score: {len(np.unique(norm_groups))} norm groups, {n_abs} abs features")
    else:
        X_folds = [X_raw] * N_SPLITS

    if has_nan:
        X_folds_clean = [np.nan_to_num(Xf, nan=0.0) for Xf in X_folds]
        print(f"  NaN fraction: {data['nan_fraction']:.3f} (preserved for XGBoost, 0-filled for others)")
    else:
        X_folds_clean = X_folds

    task_type = "detection" if binary else "categorization"
    class_dist = {label_names[i]: int(c) for i, c in enumerate(np.bincount(y))}
    print(f"Task: {task_type}, classes={label_names}, n={len(y)}, p={X.shape[1]}")
    print(f"Class dist: {class_dist}")
    print(f"Folds: {N_SPLITS}, groups: {len(np.unique(cv_groups))}")

    fold_dists = []
    for i, (tr, te) in enumerate(splits):
        fd = {"fold": i,
              "train": {label_names[c]: int(v) for c, v in enumerate(np.bincount(y[tr], minlength=len(label_names)))},
              "test": {label_names[c]: int(v) for c, v in enumerate(np.bincount(y[te], minlength=len(label_names)))}}
        fold_dists.append(fd)
        print(f"  Fold {i}: train={fd['train']}, test={fd['test']}")

    model_cfgs = build_models(binary)
    results = {
        "task_type": task_type, "label_names": label_names,
        "n_samples": len(y), "n_features": X.shape[1],
        "class_dist": class_dist, "n_folds": N_SPLITS,
        "fold_distributions": fold_dists,
        "cv_note": "GroupKFold by (model, dataset, seed); per-fold group z-score (train-only); grid search on fold-0 train with 3-fold inner CV; NaN preserved for XGBoost (union mode), 0-filled for LR/SVM/EasyEnsemble",
        "experiments": {},
    }

    for model_name, cfg in model_cfgs.items():
        runner = RUNNERS[model_name]
        print(f"\n  {model_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            xf = X_folds if model_name == "XGBoost" else X_folds_clean
            best_params, fold_results, feat_imp = runner(xf, y, splits, cfg, label_names, binary)
            agg = aggregate_results(fold_results, label_names, binary, splits, y)
            dt = round(time.time() - t0, 1)

            top_idx = np.argsort(feat_imp)[::-1][:25]
            top_features = [(feat_names[i], round(float(feat_imp[i]), 5),
                             tier_map.get(feat_names[i], "?")) for i in top_idx if i < len(feat_names)]

            bp_clean = {}
            for k, v in best_params.items():
                if isinstance(v, (int, float, bool, str, type(None))):
                    bp_clean[k] = v
                else:
                    bp_clean[k] = str(v)

            results["experiments"][model_name] = {
                "metrics": agg["overall"],
                "per_fold": agg["per_fold"],
                "best_params": bp_clean,
                "top25_features": top_features,
                "time_s": dt,
            }
            key_metric = agg["overall"].get("macro_f1") or agg["overall"].get("f1_macro", "?")
            auroc = agg["overall"].get("auroc", "N/A")
            print(f"F1m={key_metric} AUROC={auroc} ({dt}s)")
        except Exception as e:
            dt = round(time.time() - t0, 1)
            results["experiments"][model_name] = {"error": str(e), "time_s": dt}
            print(f"ERROR: {e} ({dt}s)")

    return results


def main():
    p = argparse.ArgumentParser(description="4-model classifier pipeline")
    p.add_argument("--data", type=str, required=True, help="Preprocessed .pkl file")
    p.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = p.parse_args()

    data_path = Path(args.data)
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    out_name = args.out or str(data_path).replace(".pkl", "_results.json")
    print(f"Loading {data_path.name}: X={data['X'].shape}, labels={data['label_names']}")

    results = run_pipeline(data)

    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
