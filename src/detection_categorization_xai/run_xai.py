"""
XAI pipeline for FrankenFormer fault detection/categorization.
Three explanation types:
  1. SHAP TreeExplainer -- aggregated to core features, per-layer patterns
  2. Counterfactual (DiCE) -- minimal changes for misclassified/low-margin instances
  3. Rule extraction -- surrogate tree on quantile-discretized features
All explanations computed on held-out test folds only.
Usage:
  python run_xai.py --data data/enc_v1_categorization.pkl --out results/xai_enc_cat.json
  python run_xai.py --data data/enc_v1_categorization.pkl --results results/enc_v1_categorization_results.json
"""
import argparse, json, os, pickle, re, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from run_classifiers import group_zscore, _es_split, N_SPLITS, RNG

try:
    import shap
except ImportError:
    sys.exit("pip install shap")
try:
    import dice_ml
except ImportError:
    dice_ml = None
    print("WARNING: dice-ml not installed, counterfactual analysis skipped")

MAX_CF_INSTANCES = 30
N_CFS = 3
RULE_DEPTH = 4
N_BINS = 5
IMMUTABLE = {"arch_enc", "layer_idx_num", "severity_scalar"}

_LAYER_RE = re.compile(r'_l\d+')
_TIME_RE = re.compile(r'_(early_mean|early_slope|mid_mean|mid_slope|final|finalwin)$')
_INTERVAL_RE = re.compile(r'_\d+_\d+$')


def core_name(fname):
    if not fname.startswith("abs_"):
        return fname
    c = _LAYER_RE.sub('', fname)
    c = _TIME_RE.sub('', c)
    c = _INTERVAL_RE.sub('', c)
    return c


def build_core_map(feature_names):
    m = {}
    for i, f in enumerate(feature_names):
        m.setdefault(core_name(f), []).append(i)
    return m


def aggregate_shap(sv, core_map):
    if sv.ndim == 3:
        flat = np.abs(sv).mean(axis=2)
    else:
        flat = np.abs(sv)
    return {c: float(flat[:, cols].sum(axis=1).mean()) for c, cols in core_map.items()}


def layer_pattern(sv, feature_names):
    lr = re.compile(r'_l(\d+)')
    if sv.ndim == 3:
        flat = np.abs(sv).mean(axis=2)
    else:
        flat = np.abs(sv)
    mean_sv = flat.mean(axis=0)
    layer_imp = {}
    for i, f in enumerate(feature_names):
        m = lr.search(f)
        if m:
            layer_imp.setdefault(int(m.group(1)), 0)
            layer_imp[int(m.group(1))] += mean_sv[i]
    if not layer_imp:
        return {}
    total = sum(layer_imp.values())
    if total < 1e-12:
        return {}
    n_layers = max(layer_imp.keys()) + 1
    third = max(n_layers // 3, 1)
    early = sum(v for k, v in layer_imp.items() if k < third)
    mid = sum(v for k, v in layer_imp.items() if third <= k < 2 * third)
    late = sum(v for k, v in layer_imp.items() if k >= 2 * third)
    return {
        "early_layers_frac": round(early / total, 4),
        "mid_layers_frac": round(mid / total, 4),
        "late_layers_frac": round(late / total, 4),
        "per_layer": {f"L{k}": round(v / total, 4) for k, v in sorted(layer_imp.items())},
    }


def shap_fold(model, X_test, feature_names, core_map):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    core_imp = aggregate_shap(sv, core_map)
    lp = layer_pattern(sv, feature_names)
    return sv, core_imp, lp


def counterfactual_fold(model, X_train, X_test, y_test, y_pred,
                        feature_names, label_names, has_nan):
    if dice_ml is None:
        return {"skipped": "dice-ml not installed"}
    probs = model.predict_proba(X_test)
    margins = probs.max(axis=1) - np.partition(probs, -2, axis=1)[:, -2]
    misclass = np.where(y_test != y_pred)[0]
    low_margin = np.argsort(margins)[:MAX_CF_INSTANCES]
    selected = np.unique(np.concatenate([misclass[:MAX_CF_INSTANCES], low_margin]))
    if len(selected) > MAX_CF_INSTANCES:
        selected = selected[:MAX_CF_INSTANCES]
    if len(selected) == 0:
        return {"skipped": "no candidates"}

    if has_nan:
        valid = [i for i in range(X_train.shape[1])
                 if np.isnan(X_train[:, i]).mean() < 0.5]
    else:
        valid = list(range(X_train.shape[1]))
    vnames = [feature_names[i] for i in valid]
    mutable = [f for f in vnames if f not in IMMUTABLE]

    X_tr_v = np.nan_to_num(X_train[:, valid], nan=0.0)
    df_tr = pd.DataFrame(X_tr_v, columns=vnames)
    tr_pred = model.predict(X_train)
    df_tr["__label__"] = tr_pred

    try:
        d = dice_ml.Data(dataframe=df_tr, continuous_features=mutable,
                         outcome_name="__label__")

        class _Wrap:
            def __init__(self, mdl, cols, n_total):
                self.mdl, self.cols, self.n = mdl, cols, n_total
                self.classes_ = mdl.classes_
            def predict(self, X):
                a = np.zeros((len(X), self.n))
                a[:, self.cols] = X.values if hasattr(X, 'values') else X
                return self.mdl.predict(a)
            def predict_proba(self, X):
                a = np.zeros((len(X), self.n))
                a[:, self.cols] = X.values if hasattr(X, 'values') else X
                return self.mdl.predict_proba(a)

        wrap = _Wrap(model, valid, X_train.shape[1]) if has_nan else model
        dm = dice_ml.Model(model=wrap, backend="sklearn", model_type="classifier")
        exp = dice_ml.Dice(d, dm, method="random")

        results, change_counter = [], Counter()
        for idx in selected[:20]:
            qv = np.nan_to_num(X_test[idx:idx + 1, valid], nan=0.0)
            query = pd.DataFrame(qv, columns=vnames)
            try:
                pred_cls = int(y_pred[idx])
                if len(label_names) == 2:
                    desired = "opposite"
                else:
                    true_cls = int(y_test[idx])
                    desired = true_cls if true_cls != pred_cls else (pred_cls + 1) % len(label_names)
                cf = exp.generate_counterfactuals(query, total_CFs=N_CFS,
                                                   desired_class=desired,
                                                   features_to_vary=mutable,
                                                   verbose=False)
                if cf.cf_examples_list and cf.cf_examples_list[0].final_cfs_df is not None:
                    cf_df = cf.cf_examples_list[0].final_cfs_df
                    changes = {}
                    for col in vnames:
                        if col in cf_df.columns:
                            orig = query[col].values[0]
                            diffs = np.abs(cf_df[col].values - orig)
                            if diffs.max() > 1e-6:
                                changes[col] = round(float(diffs.mean()), 4)
                                change_counter[core_name(col)] += 1
                    results.append({
                        "true": label_names[y_test[idx]],
                        "pred": label_names[y_pred[idx]],
                        "margin": round(float(margins[idx]), 4),
                        "n_changed": len(changes),
                        "top_changes": dict(sorted(changes.items(), key=lambda x: -x[1])[:5]),
                    })
            except Exception:
                continue

        return {
            "n_generated": len(results),
            "top_changed_core_features": change_counter.most_common(15),
            "examples": results[:5],
        }
    except Exception as e:
        return {"error": str(e)}


def rule_fold(model, X_train, X_test, y_train, y_test, feature_names, label_names):
    X_tr = np.nan_to_num(X_train, nan=0.0)
    X_te = np.nan_to_num(X_test, nan=0.0)

    try:
        disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile",
                                subsample=None)
        X_tr_d = disc.fit_transform(X_tr)
        X_te_d = disc.transform(X_te)
    except Exception:
        disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="uniform",
                                subsample=None)
        X_tr_d = disc.fit_transform(X_tr)
        X_te_d = disc.transform(X_te)

    surr = DecisionTreeClassifier(max_depth=RULE_DEPTH, random_state=RNG,
                                  class_weight="balanced")
    surr.fit(X_tr_d, y_train)

    xgb_pred = model.predict(X_test)
    surr_pred = surr.predict(X_te_d)
    fidelity = float((surr_pred == xgb_pred).mean())
    accuracy = float((surr_pred == y_test).mean())

    core_names = [core_name(f) for f in feature_names]
    tree_text = export_text(surr, feature_names=core_names, max_depth=RULE_DEPTH)

    leaf_ids = surr.apply(X_te_d)
    tree_ = surr.tree_
    rules = []
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        n = int(mask.sum())
        if n < 3:
            continue
        pred_cls = int(tree_.value[leaf][0].argmax())
        prec = float((y_test[mask] == pred_cls).mean())
        cov = n / len(y_test)
        sample_path = surr.decision_path(X_te_d[mask][:1]).toarray()[0]
        path_nodes = np.where(sample_path)[0]
        conds = []
        for j in range(len(path_nodes) - 1):
            node = path_nodes[j]
            nxt = path_nodes[j + 1]
            feat = core_names[tree_.feature[node]]
            thr = int(tree_.threshold[node])
            if nxt == tree_.children_left[node]:
                conds.append(f"{feat} <= Q{thr}")
            else:
                conds.append(f"{feat} > Q{thr}")
        rules.append({
            "conditions": conds,
            "class": label_names[pred_cls] if pred_cls < len(label_names) else str(pred_cls),
            "precision": round(prec, 4),
            "coverage": round(cov, 4),
            "n": n,
        })
    rules.sort(key=lambda r: r["precision"] * r["coverage"], reverse=True)

    return {
        "fidelity_to_xgb": round(fidelity, 4),
        "surrogate_accuracy": round(accuracy, 4),
        "n_rules": len(rules),
        "top_rules": rules[:20],
        "tree_text": tree_text[:3000],
        "disc_note": f"{N_BINS} quantile bins, fit on training fold only",
    }


def run_xai(data, xgb_params=None):
    X, y = data["X"], data["y"]
    feat_names = data["feature_names"]
    cv_groups = data["cv_groups"]
    norm_groups = data.get("norm_groups")
    n_abs = data.get("n_abs_features", X.shape[1])
    label_names = data["label_names"]
    binary = len(label_names) == 2
    has_nan = data.get("has_nan", False)
    core_map = build_core_map(feat_names)

    X_raw = X.astype(np.float64)

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y, cv_groups))

    if norm_groups is not None:
        X_folds = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, te in splits]
    else:
        X_folds = [X_raw] * N_SPLITS

    if xgb_params is None:
        xgb_params = {}
    base = {"tree_method": "hist", "random_state": RNG, "n_jobs": -1,
            "n_estimators": 2000, "verbosity": 0,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "max_depth": xgb_params.get("max_depth", 3),
            "min_child_weight": xgb_params.get("min_child_weight", 5),
            "learning_rate": xgb_params.get("learning_rate", 0.1)}
    if binary:
        neg, pos = np.bincount(y)
        base.update({"objective": "binary:logistic", "eval_metric": "aucpr",
                      "scale_pos_weight": neg / max(pos, 1)})
    else:
        base.update({"objective": "multi:softprob", "eval_metric": "mlogloss"})

    print(f"XAI pipeline: {len(y)} samples, {X.shape[1]} features, "
          f"{len(label_names)} classes, {N_SPLITS} folds")
    print(f"Core features: {len(core_map)} (from {len(feat_names)} columns)")

    all_core_imp, all_layer_pat = [], []
    all_rules, all_cf = [], []
    fold_top_cores = []

    for fi, (tr, te) in enumerate(splits):
        Xf = X_folds[fi]
        fit_i, es_i = _es_split(tr)
        sw = None if binary else compute_sample_weight("balanced", y[fit_i])

        clf = XGBClassifier(**base)
        clf.fit(Xf[fit_i], y[fit_i], eval_set=[(Xf[es_i], y[es_i])],
                sample_weight=sw, verbose=False)
        y_pred = clf.predict(Xf[te])
        acc = float((y_pred == y[te]).mean())
        print(f"\n  Fold {fi}: acc={acc:.3f}, test={len(te)}", flush=True)

        print("    SHAP...", end=" ", flush=True)
        t0 = time.time()
        sv, ci, lp = shap_fold(clf, Xf[te], feat_names, core_map)
        all_core_imp.append(ci)
        all_layer_pat.append(lp)
        top_k = sorted(ci, key=ci.get, reverse=True)[:20]
        fold_top_cores.append(set(top_k))
        print(f"{time.time()-t0:.1f}s")

        print("    Counterfactuals...", end=" ", flush=True)
        t0 = time.time()
        cf = counterfactual_fold(clf, Xf[tr], Xf[te], y[te], y_pred,
                                 feat_names, label_names, has_nan)
        all_cf.append(cf)
        print(f"{time.time()-t0:.1f}s")

        print("    Rules...", end=" ", flush=True)
        t0 = time.time()
        rl = rule_fold(clf, Xf[tr], Xf[te], y[tr], y[te], feat_names, label_names)
        all_rules.append(rl)
        print(f"fidelity={rl['fidelity_to_xgb']:.3f} ({time.time()-t0:.1f}s)")

    # Aggregate across folds
    agg_imp = {}
    for ci in all_core_imp:
        for c, v in ci.items():
            agg_imp[c] = agg_imp.get(c, 0) + v / N_SPLITS
    top_cores = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)[:30]

    jaccard_pairs = []
    for i in range(N_SPLITS):
        for j in range(i + 1, N_SPLITS):
            a, b = fold_top_cores[i], fold_top_cores[j]
            jaccard_pairs.append(len(a & b) / len(a | b) if a | b else 1.0)
    stability = round(np.mean(jaccard_pairs), 4) if jaccard_pairs else 1.0

    agg_lp = {}
    valid_lps = [lp for lp in all_layer_pat if lp]
    if valid_lps:
        for key in ["early_layers_frac", "mid_layers_frac", "late_layers_frac"]:
            vals = [lp[key] for lp in valid_lps if key in lp]
            if vals:
                agg_lp[key] = round(np.mean(vals), 4)

    cf_agg_changes = Counter()
    cf_total = 0
    for cf in all_cf:
        if "top_changed_core_features" in cf:
            for feat, cnt in cf["top_changed_core_features"]:
                cf_agg_changes[feat] += cnt
            cf_total += cf.get("n_generated", 0)

    fidelities = [r["fidelity_to_xgb"] for r in all_rules]
    rule_hash = Counter()
    for r in all_rules:
        for rule in r.get("top_rules", []):
            key = " AND ".join(rule["conditions"]) + " => " + rule["class"]
            rule_hash[key] += 1
    stable_rules = [(r, c) for r, c in rule_hash.most_common(20) if c >= 2]

    results = {
        "shap": {
            "top30_core_features": [(c, round(v, 5)) for c, v in top_cores],
            "stability_jaccard_top20": stability,
            "layer_pattern": agg_lp,
            "per_fold_top5": [sorted(ci.items(), key=lambda x: -x[1])[:5]
                              for ci in all_core_imp],
            "note": "mean(|SHAP|) aggregated to core features across time-window and per-layer expansions",
        },
        "counterfactuals": {
            "total_generated": cf_total,
            "top_changed_core_features": cf_agg_changes.most_common(15),
            "per_fold": [{k: v for k, v in cf.items() if k != "examples"} for cf in all_cf],
            "examples": [ex for cf in all_cf for ex in cf.get("examples", [])][:10],
            "note": "DiCE random method; structural features immutable; "
                    "selected misclassified + low-margin instances",
        },
        "rules": {
            "mean_fidelity_to_xgb": round(np.mean(fidelities), 4),
            "std_fidelity": round(np.std(fidelities), 4),
            "stable_rules_across_folds": stable_rules,
            "per_fold": [{k: v for k, v in r.items() if k != "tree_text"} for r in all_rules],
            "sample_tree": all_rules[0].get("tree_text", ""),
            "note": f"{N_BINS} quantile bins fit on training fold only; "
                    f"surrogate DecisionTree(depth={RULE_DEPTH}); "
                    f"precision and coverage evaluated on held-out test fold",
        },
        "meta": {
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_core_features": len(core_map),
            "label_names": label_names,
            "binary": binary,
            "has_nan": has_nan,
            "n_folds": N_SPLITS,
            "xgb_params": {k: v for k, v in base.items()
                           if isinstance(v, (int, float, str, bool))},
        },
    }
    return results


def main():
    p = argparse.ArgumentParser(description="XAI pipeline")
    p.add_argument("--data", required=True, help="Preprocessed .pkl file")
    p.add_argument("--results", default=None, help="Classifier results .json (for XGBoost params)")
    p.add_argument("--out", default=None, help="Output .json path")
    args = p.parse_args()

    data_path = Path(args.data)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded: {data_path.name}, X={data['X'].shape}, labels={data['label_names']}")

    xgb_params = None
    if args.results:
        with open(args.results) as f:
            res = json.load(f)
        xgb_exp = res.get("experiments", {}).get("XGBoost", {})
        bp = xgb_exp.get("best_params", {})
        if bp:
            xgb_params = {k: v for k, v in bp.items()
                          if k in ("max_depth", "min_child_weight", "learning_rate")}
            print(f"Using XGBoost params from results: {xgb_params}")

    results = run_xai(data, xgb_params)

    out_name = args.out or str(data_path).replace(".pkl", "_xai.json")
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    print("\n=== SHAP Top 10 Core Features ===")
    for feat, imp in results["shap"]["top30_core_features"][:10]:
        print(f"  {imp:.5f}  {feat}")
    print(f"  Stability (Jaccard top-20): {results['shap']['stability_jaccard_top20']}")
    if results["shap"]["layer_pattern"]:
        lp = results["shap"]["layer_pattern"]
        print(f"  Layer pattern: early={lp.get('early_layers_frac','?')} "
              f"mid={lp.get('mid_layers_frac','?')} late={lp.get('late_layers_frac','?')}")
    print(f"\n=== Counterfactuals: {results['counterfactuals']['total_generated']} generated ===")
    for feat, cnt in results["counterfactuals"]["top_changed_core_features"][:5]:
        print(f"  {cnt}x changed: {feat}")
    print(f"\n=== Rules: fidelity={results['rules']['mean_fidelity_to_xgb']:.3f} "
          f"+/- {results['rules']['std_fidelity']:.3f} ===")
    for rule, count in results["rules"]["stable_rules_across_folds"][:5]:
        print(f"  [{count}/{N_SPLITS} folds] {rule}")


if __name__ == "__main__":
    main()
