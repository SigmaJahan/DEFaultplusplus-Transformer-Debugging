"""
Reusable preprocessor for FrankenFormer detection/categorization.
Usage:
  python preprocess.py --enc data/encoder_v1_killed_binary.csv --dec data/decoder_v1_killed_binary.csv --mode union --out data/ready.pkl
  python preprocess.py --enc data/encoder_v1_killed_binary.csv --mode full --out data/ready.pkl
"""
import argparse, json, pickle, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# 86 shared abs_ features (encoder-decoder intersection)
SHARED_FEATURES = [
    "abs_accuracy_early_mean", "abs_accuracy_early_slope", "abs_accuracy_final",
    "abs_accuracy_mid_mean", "abs_accuracy_mid_slope",
    "abs_ece_early_mean", "abs_ece_early_slope", "abs_ece_final",
    "abs_ece_mid_mean", "abs_ece_mid_slope",
    "abs_logit_conf_early_mean", "abs_logit_conf_early_slope", "abs_logit_conf_final",
    "abs_logit_conf_mid_mean", "abs_logit_conf_mid_slope",
    "abs_logit_entropy_early_mean", "abs_logit_entropy_early_slope", "abs_logit_entropy_final",
    "abs_logit_entropy_mid_mean", "abs_logit_entropy_mid_slope",
    "abs_logit_kl_uniform_early_mean", "abs_logit_kl_uniform_early_slope", "abs_logit_kl_uniform_final",
    "abs_logit_kl_uniform_mid_mean", "abs_logit_kl_uniform_mid_slope",
    "abs_logit_margin_mean_early_mean", "abs_logit_margin_mean_early_slope", "abs_logit_margin_mean_final",
    "abs_logit_margin_mean_mid_mean", "abs_logit_margin_mean_mid_slope",
    "abs_logit_margin_min_early_mean", "abs_logit_margin_min_early_slope", "abs_logit_margin_min_final",
    "abs_logit_margin_min_mid_mean", "abs_logit_margin_min_mid_slope",
    "abs_logit_margin_p25_early_mean", "abs_logit_margin_p25_early_slope", "abs_logit_margin_p25_final",
    "abs_logit_margin_p25_mid_mean", "abs_logit_margin_p25_mid_slope",
    "abs_logit_margin_p50_early_mean", "abs_logit_margin_p50_early_slope", "abs_logit_margin_p50_final",
    "abs_logit_margin_p50_mid_mean", "abs_logit_margin_p50_mid_slope",
    "abs_logit_margin_p75_early_mean", "abs_logit_margin_p75_early_slope", "abs_logit_margin_p75_final",
    "abs_logit_margin_p75_mid_mean", "abs_logit_margin_p75_mid_slope",
    "abs_logit_margin_var_early_mean", "abs_logit_margin_var_early_slope", "abs_logit_margin_var_final",
    "abs_logit_margin_var_mid_mean", "abs_logit_margin_var_mid_slope",
    "abs_loss_early_mean", "abs_loss_early_slope", "abs_loss_final",
    "abs_loss_mid_mean", "abs_loss_mid_slope",
    "abs_peak_mem_alloc_mb", "abs_peak_mem_reserved_mb", "abs_step_time_mean_finalwin",
    "abs_update_ratio_agg_attn_final", "abs_update_ratio_agg_ffn_final",
    "abs_update_ratio_agg_layernorm_final", "abs_update_ratio_agg_qkv_final",
    "abs_update_ratio_emb_early_mean", "abs_update_ratio_emb_early_slope",
    "abs_update_ratio_emb_final", "abs_update_ratio_emb_mid_mean",
    "abs_update_ratio_total_early_mean", "abs_update_ratio_total_early_slope",
    "abs_update_ratio_total_final", "abs_update_ratio_total_mid_mean",
    "abs_update_ratio_total_mid_slope",
    "abs_val_loss_early_mean", "abs_val_loss_early_slope", "abs_val_loss_final",
    "abs_val_loss_mid_mean", "abs_val_loss_mid_slope",
    "abs_val_pos_inv_early_mean", "abs_val_pos_inv_early_slope", "abs_val_pos_inv_final",
    "abs_val_pos_inv_mid_mean", "abs_val_pos_inv_mid_slope",
]

TIER_P = {f for f in SHARED_FEATURES if any(
    k in f for k in ["accuracy", "loss", "val_loss", "ece", "val_pos_inv"])}
TIER_D = {f for f in SHARED_FEATURES if any(
    k in f for k in ["logit_conf", "logit_entropy", "logit_kl_uniform", "logit_margin"])}
TIER_I = set(SHARED_FEATURES) - TIER_P - TIER_D

META_COLS = ["Identifier", "arch", "model_name", "dataset_name", "seed",
             "fault_category", "fault_subcategory", "layer_idx", "severity_params"]

_LAYER_RE = re.compile(r"^(abs_.+)_l(\d+)_(.+)$")
_REPR_RE = re.compile(r"^(abs_repr)_l(\d+)_(.+)$")


def parse_severity_scalar(s):
    if pd.isna(s):
        return np.nan
    try:
        d = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.nan
    nums = [abs(float(v)) for v in d.values() if isinstance(v, (int, float)) or
            (isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).isdigit())]
    return max(nums) if nums else np.nan


def aggregate_per_layer_features(df):
    abs_cols = [c for c in df.columns if c.startswith("abs_")]
    layer_groups = {}
    for c in abs_cols:
        m = _LAYER_RE.match(c) or _REPR_RE.match(c)
        if m:
            key = (m.group(1), m.group(3))
            layer_groups.setdefault(key, []).append(c)

    new_cols = {}
    drop_cols = []
    for (prefix, suffix), cols in layer_groups.items():
        if len(cols) < 2:
            continue
        drop_cols.extend(cols)
        base = f"{prefix}_agg_{suffix}"
        vals = df[cols].values
        new_cols[f"{base}_mean"] = np.nanmean(vals, axis=1)
        new_cols[f"{base}_std"] = np.nanstd(vals, axis=1)
        new_cols[f"{base}_max"] = np.nanmax(vals, axis=1)

    df_out = df.drop(columns=drop_cols, errors="ignore")
    for name, vals in new_cols.items():
        df_out[name] = vals
    return df_out, list(new_cols.keys())


def rank_normalize_within_group(df, cols, group_col):
    for g in df[group_col].unique():
        mask = df[group_col] == g
        for c in cols:
            vals = df.loc[mask, c]
            valid = vals.notna()
            if valid.sum() > 1:
                ranks = rankdata(vals[valid], method="average")
                df.loc[mask & valid, c] = (ranks - 1) / (len(ranks) - 1)
    return df


def load_and_merge(enc_path=None, dec_path=None, mode="shared"):
    frames = []
    for path in [enc_path, dec_path]:
        if path is None:
            continue
        df = pd.read_csv(path)
        if "arch" not in df.columns:
            df["arch"] = "encoder" if "encoder" in str(path) else "decoder"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if mode == "shared":
        feat_cols = [c for c in SHARED_FEATURES if c in df.columns]
    elif mode == "union":
        feat_cols = sorted([c for c in df.columns if c.startswith("abs_")])
    elif mode == "semantic":
        df, agg_cols = aggregate_per_layer_features(df)
        group_key = df["model_name"].astype(str) + "__" + df["dataset_name"].astype(str)
        df["_norm_group"] = group_key
        abs_feats = [c for c in df.columns if c.startswith("abs_")]
        df = rank_normalize_within_group(df, abs_feats, "_norm_group")
        df = df.drop(columns=["_norm_group"])
        feat_cols = sorted([c for c in df.columns if c.startswith("abs_")])
    else:  # full
        feat_cols = [c for c in df.columns if c.startswith("abs_")]

    tier_map = {}
    for f in feat_cols:
        if f in TIER_P:
            tier_map[f] = "P"
        elif f in TIER_D:
            tier_map[f] = "D"
        else:
            tier_map[f] = "I"

    return df, feat_cols, tier_map


def preprocess(df, feat_cols, tier_map, mode="shared", no_structural=False):
    preserve_nan = mode in ("union", "semantic")

    labels = df["label"].astype(str)
    label_names = sorted(labels.unique())
    label2int = {l: i for i, l in enumerate(label_names)}
    y = labels.map(label2int).values

    df = df.copy()
    df["arch_enc"] = (df["arch"] == "encoder").astype(int)
    if no_structural:
        structural = ["arch_enc"]
    else:
        df["layer_idx_num"] = df["layer_idx"].fillna(-1).astype(float)
        df["severity_scalar"] = df["severity_params"].apply(parse_severity_scalar)
        structural = ["arch_enc", "layer_idx_num", "severity_scalar"]

    X_feat = df[feat_cols].copy()
    norm_groups = (df["model_name"].astype(str) + "__" + df["dataset_name"].astype(str)).values

    X_struct = df[structural].copy()
    if "severity_scalar" in X_struct.columns:
        X_struct["severity_scalar"] = X_struct["severity_scalar"].fillna(0)
    X = pd.concat([X_feat.reset_index(drop=True), X_struct.reset_index(drop=True)], axis=1)
    if not preserve_nan:
        X = X.fillna(0)
    X = X.values.astype(np.float32)

    all_feature_names = feat_cols + structural
    full_tier_map = dict(tier_map)
    for s in structural:
        full_tier_map[s] = "S"

    cv_groups = (df["model_name"].astype(str) + "__" +
                 df["dataset_name"].astype(str) + "__" +
                 df["seed"].astype(str)).values

    meta = df[["Identifier", "arch", "model_name", "dataset_name",
               "fault_category"]].copy().reset_index(drop=True)

    nan_frac = np.isnan(X).mean() if preserve_nan else 0.0

    return {
        "X": X, "y": y,
        "feature_names": all_feature_names,
        "tier_map": full_tier_map,
        "cv_groups": cv_groups,
        "norm_groups": norm_groups,
        "n_abs_features": len(feat_cols),
        "label_names": label_names,
        "label2int": label2int,
        "meta": meta,
        "has_nan": preserve_nan,
        "nan_fraction": float(nan_frac),
        "mode": mode,
    }


def _save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _make_detection_and_cat(csv_path, mode, prefix, data_dir):
    """Build detection + categorization pkl pair from a single CSV."""
    df_raw = pd.read_csv(csv_path, low_memory=False)
    if "arch" not in df_raw.columns:
        df_raw["arch"] = "encoder" if "encoder" in str(csv_path) else "decoder"
    feat_cols = sorted([c for c in df_raw.columns if c.startswith("abs_")])
    tier_map = {}
    for f in feat_cols:
        if f in TIER_P: tier_map[f] = "P"
        elif f in TIER_D: tier_map[f] = "D"
        else: tier_map[f] = "I"

    data_det = preprocess(df_raw, feat_cols, tier_map, mode=mode)
    det_out = data_dir / f"{prefix}_detection.pkl"
    _save_pkl(data_det, det_out)
    print(f"  {det_out.name}: X={data_det['X'].shape}, labels={data_det['label_names']}")

    df_cat = df_raw[df_raw["label"] != "correct"].copy()
    df_cat["label"] = df_cat["fault_category"].astype(str)
    data_cat = preprocess(df_cat, feat_cols, {f: tier_map[f] for f in feat_cols}, mode=mode)
    cat_out = data_dir / f"{prefix}_categorization.pkl"
    _save_pkl(data_cat, cat_out)
    print(f"  {cat_out.name}: X={data_cat['X'].shape}, labels={data_cat['label_names']}")


def batch_regenerate(data_dir):
    """Regenerate all 6 pkl files (enc/dec/cross x detection/categorization) from CSVs."""
    enc_csv = data_dir / "encoder_v1_killed_binary.csv"
    dec_csv = data_dir / "decoder_v1_killed_binary.csv"

    print("=== Encoder (full features) ===")
    _make_detection_and_cat(enc_csv, "full", "enc_v1", data_dir)

    print("\n=== Decoder (full features) ===")
    _make_detection_and_cat(dec_csv, "full", "dec_v1", data_dir)

    print("\n=== Cross (union features) ===")
    df_cross, feat_cols, tier_map = load_and_merge(enc_csv, dec_csv, mode="union")

    data_det = preprocess(df_cross, feat_cols, tier_map, mode="union")
    _save_pkl(data_det, data_dir / "cross_v1_detection.pkl")
    print(f"  cross_v1_detection.pkl: X={data_det['X'].shape}, nan={data_det['nan_fraction']:.3f}")

    df_cat = df_cross[df_cross["label"] != "correct"].copy()
    df_cat["label"] = df_cat["fault_category"].astype(str)
    feat_cols_cat = [c for c in feat_cols if c in df_cat.columns]
    tier_map_cat = {f: tier_map[f] for f in feat_cols_cat}
    data_cat = preprocess(df_cat, feat_cols_cat, tier_map_cat, mode="union")
    _save_pkl(data_cat, data_dir / "cross_v1_categorization.pkl")
    print(f"  cross_v1_categorization.pkl: X={data_cat['X'].shape}, nan={data_cat['nan_fraction']:.3f}")


def main():
    p = argparse.ArgumentParser(description="Preprocess FrankenFormer datasets")
    p.add_argument("--enc", type=str, default=None, help="Encoder CSV path")
    p.add_argument("--dec", type=str, default=None, help="Decoder CSV path")
    p.add_argument("--mode", choices=["shared", "full", "union", "semantic"], default="shared")
    p.add_argument("--out", type=str, default="ready.pkl")
    p.add_argument("--no-structural", action="store_true",
                   help="Drop layer_idx and severity_scalar (keep only arch_enc)")
    p.add_argument("--batch", action="store_true",
                   help="Regenerate all 6 pkl files from CSVs in data/")
    args = p.parse_args()

    base = Path(__file__).parent

    if args.batch:
        batch_regenerate(base / "data")
        return

    assert args.enc or args.dec, "Provide at least --enc or --dec"

    enc_path = base / args.enc if args.enc else None
    dec_path = base / args.dec if args.dec else None

    df, feat_cols, tier_map = load_and_merge(enc_path, dec_path, args.mode)

    print(f"Loaded: {len(df)} rows, {len(feat_cols)} {args.mode} features")
    print(f"  Tier P: {sum(1 for v in tier_map.values() if v=='P')}")
    print(f"  Tier D: {sum(1 for v in tier_map.values() if v=='D')}")
    print(f"  Tier I: {sum(1 for v in tier_map.values() if v=='I')}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    no_struct = getattr(args, 'no_structural', False)
    if no_struct:
        print("  Structural: layer_idx + severity DROPPED (--no-structural)")
    data = preprocess(df, feat_cols, tier_map, mode=args.mode, no_structural=no_struct)
    print(f"  X shape: {data['X'].shape}")
    print(f"  Groups: {len(np.unique(data['cv_groups']))}")
    if data.get("has_nan"):
        print(f"  NaN fraction: {data['nan_fraction']:.3f} (preserved for NaN-native models)")

    out_path = base / args.out
    _save_pkl(data, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
