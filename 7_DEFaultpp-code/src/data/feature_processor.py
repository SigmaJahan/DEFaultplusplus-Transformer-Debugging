"""
Feature Processing Pipeline — src/data/feature_processor.py

Implements the 6-step pipeline specified in FEATURE_PROCESSING.md.
All design decisions are architecture-grounded: grouping is determined by
FPG node membership, not by statistical properties of any specific dataset.

Steps:
  1. Structural drop    — remove columns with NaN rate > 0.40 (MNAR, not MAR)
  2. Scale normalise    — log1p transform explosive-variance columns
  3. Layer aggregation  — collapse per-layer expansions to {mean,std,min,max}
                          (encoder only; decoder features are already aggregated)
  4. Median imputation  — fill remaining NaN with within-fold reference median
  5. CV filter          — drop columns with coefficient of variation < 0.01
  6. Group alignment    — split training_dynamics into FPG-aligned subgroups

Usage:
    processor = FeatureProcessor(arch="encoder")
    processor.fit(X_train, feature_names, y_train)
    X_clean, new_names, group_indices = processor.transform(X_test)

    # Or combined:
    X_clean, new_names, group_indices = processor.fit_transform(
        X_train, feature_names, y_train
    )
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

NAN_DROP_THRESHOLD  = 0.40   # Step 1: drop column if NaN rate > this
LOG_VAR_THRESHOLD   = 1e6    # Step 2: log-transform if variance exceeds this
LOG_RATIO_THRESHOLD = 1000   # Step 2: also log-transform if max/median > this
CV_THRESHOLD        = 0.01   # Step 5: drop if CV (std/|mean|) < this
CV_TINY_MEAN        = 1e-9   # Step 5: fallback to abs std if |mean| < this
CV_FALLBACK_STD     = 1e-6   # Step 5: fallback threshold when mean ≈ 0

LAYER_RE = re.compile(r"^(.*?)_l(\d+)_(.*)$")   # matches ffn_delta_l3_mean_final

# ── Step 6: training_dynamics subgroup tokens ─────────────────────────────────
# These map feature name tokens to FPG node types.
# Applied to both encoder and decoder identically.
TD_LOGIT_TOKENS   = ("logit_conf", "logit_entropy", "logit_kl", "logit_margin")
TD_GRADIENT_TOKENS = ("grad_norm", "grad_abs", "grad_zero", "grad_noise")
TD_UPDATE_TOKENS   = ("update_ratio", "update_active", "weight_mean", "weight_std")

# ── Group assignment (mirrors feature_groups.py, used in Step 6) ─────────────
COMP_RE = re.compile(
    r"(?:grad_norm_(?:agg_)?(?:l\d+_)?|update_ratio_(?:agg_)?(?:l\d+_)?|"
    r"update_active_(?:agg_)?(?:l\d+_)?)(attn|ffn|layernorm|qkv|emb)"
)
COMP_MAP = {"attn": "attention", "ffn": "ffn", "layernorm": "layernorm",
            "qkv": "qkv", "emb": "embedding"}

TOKEN_RULES = [
    (["attn_entropy","attn_pad","attn_weight","attn_cross","attn_mass",
      "attn_max","attn_sparsity","head_similarity","head_util",
      "mass_leak","mass_pad","cross_example"], "attention"),
    (["qk_cos","qv_cos","kv_cos","qkv_"], "qkv"),
    (["score_mean","score_var","score_max","score_std","score_skew",
      "attn_score","presoftmax"], "score"),
    (["pos_discrim","pos_acc","positional","pos_recv","pos_inv",
      "pos_loss","pos_margin"], "positional"),
    (["ffn_delta","ffn_norm","ffn_out","ffn_var","ffn_active",
      "ffn_","activation_"], "ffn"),
    (["ln_gamma","ln_post","layernorm","ln_"], "layernorm"),
    (["res_cos","res_sim","residual"], "residual"),
    (["cka_","repr_drift","representation","repr_l","h1_delta"], "representation"),
    (["emb_norm","emb_var","embedding"], "embedding"),
    (["cache_","kv_cache"], "cache_diagnostics"),
    (["step_time","peak_mem","kernel_time"], "kernel_timing"),
    (["accuracy","loss","perplexity","ece","nll","f1_score",
      "precision","recall","primary_metric","edge_case",
      "margin_gap","margin_neg","margin_pos"], "task_metrics"),
]


def _assign_group(fname: str) -> str:
    """Assign a feature to its FPG node group by name."""
    f = fname.lower()
    for pfx in ("abs_", "delta_", "rank_"):
        if f.startswith(pfx):
            f = f[len(pfx):]
    m = COMP_RE.match(f)
    if m:
        return COMP_MAP[m.group(1)]
    for tokens, grp in TOKEN_RULES:
        for t in tokens:
            if t in f:
                return grp
    return "training_dynamics"


def _td_subgroup(fname: str) -> str:
    """Return the training_dynamics subgroup for a column (Step 6)."""
    f = fname.lower()
    for t in TD_GRADIENT_TOKENS:
        if t in f:
            return "gradient_flow"
    for t in TD_UPDATE_TOKENS:
        if t in f:
            return "update_dynamics"
    for t in TD_LOGIT_TOKENS:
        if t in f:
            return "logit_distribution"
    return "logit_distribution"   # default for remaining td features



class FeatureProcessor:
    """
    Stateful feature processing pipeline.

    Call fit() on training data inside each CV fold, then transform() on
    both train and test splits. Never fit on test data.

    Attributes (set after fit):
        feature_names_out_  : list[str]  processed feature names
        group_indices_out_  : dict[str, list[int]]  group → column indices
        processing_log_     : dict  per-step record of what was dropped/changed
    """

    def __init__(self, arch: str):
        assert arch in ("encoder", "decoder"), f"arch must be 'encoder' or 'decoder', got {arch}"
        self.arch = arch
        self._fitted = False

        # Learned from fit()
        self._drop_step1: list[str] = []       # cols dropped: NaN rate > threshold
        self._log_cols: list[str] = []          # cols log-transformed (step 2)
        self._layer_families: dict = {}         # base → [(layer_idx, col)] (step 3)
        self._impute_values: dict = {}          # col → median value (step 4)
        self._drop_step5: list[str] = []        # cols dropped: CV < threshold
        self.feature_names_out_: list[str] = []
        self.group_indices_out_: dict[str, list[int]] = {}
        self.processing_log_: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self,
            X: np.ndarray,
            feature_names: list[str],
            y: Optional[np.ndarray] = None) -> "FeatureProcessor":
        """Fit the processor on training data. y is used for reference imputation."""
        assert X.shape[1] == len(feature_names), (
            f"X has {X.shape[1]} columns but {len(feature_names)} names"
        )
        df = pd.DataFrame(X, columns=feature_names)
        df = self._step1_fit(df)
        df = self._step2_fit(df)
        df = self._step3_fit(df)
        df = self._step4_fit(df, y)
        df = self._step5_fit(df)
        names_after, group_idx = self._step6_assign(list(df.columns))
        self.feature_names_out_ = names_after
        self.group_indices_out_ = group_idx
        self._fitted = True
        self._build_log()
        return self

    def transform(self,
                  X: np.ndarray,
                  feature_names: list[str]) -> tuple[np.ndarray, list[str], dict]:
        """Transform data using fitted parameters. Returns (X_clean, names, group_indices)."""
        assert self._fitted, "Call fit() before transform()"
        assert X.shape[1] == len(feature_names), (
            f"X has {X.shape[1]} columns but {len(feature_names)} names"
        )
        df = pd.DataFrame(X, columns=feature_names)
        df = self._step1_transform(df)
        df = self._step2_transform(df)
        df = self._step3_transform(df)
        df = self._step4_transform(df)
        df = self._step5_transform(df)
        return df.values.astype(np.float32), self.feature_names_out_, self.group_indices_out_

    def fit_transform(self,
                      X: np.ndarray,
                      feature_names: list[str],
                      y: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, list[str], dict]:
        """Fit and transform in one call. Equivalent to fit().transform()."""
        self.fit(X, feature_names, y)
        return self.transform(X, feature_names)


    # ── Step 1: Structural drop (MNAR columns) ────────────────────────────────

    def _step1_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        nan_rates = df.isna().mean()
        self._drop_step1 = nan_rates[nan_rates > NAN_DROP_THRESHOLD].index.tolist()
        return df.drop(columns=self._drop_step1)

    def _step1_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self._drop_step1 if c in df.columns]
        return df.drop(columns=present)

    # ── Step 2: Scale normalisation (log1p for explosive variance) ────────────

    def _step2_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_cols = []
        for col in df.columns:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            # Use numpy for variance to avoid pandas overflow on extreme values
            # (e.g. perplexity ~1e67). np.float64 overflows to inf rather than
            # raising; treat inf variance as needing log-transform.
            with np.errstate(over="ignore", invalid="ignore"):
                arr = vals.values.astype(np.float64)
                var = float(np.var(arr))
                med = float(np.median(arr))
                mx  = float(np.max(np.abs(arr)))
            needs_log = (
                not np.isfinite(var)                              # overflow → log
                or var > LOG_VAR_THRESHOLD
                or (med != 0 and mx / abs(med) > LOG_RATIO_THRESHOLD)
            )
            if needs_log:
                self._log_cols.append(col)
        return self._apply_log(df)

    def _step2_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply_log(df)

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._log_cols:
            if col in df.columns:
                v = df[col].values.astype(np.float64)
                df[col] = np.sign(v) * np.log1p(np.abs(v))
        return df

    # ── Step 3: Layer aggregation (encoder only) ──────────────────────────────

    def _step3_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.arch != "encoder":
            self._layer_families = {}
            return df
        families: dict[str, list] = {}
        for col in df.columns:
            m = LAYER_RE.match(col.lower())
            if m:
                base = f"{m.group(1)}__{m.group(3)}"
                families.setdefault(base, []).append((int(m.group(2)), col))
        self._layer_families = {b: sorted(v) for b, v in families.items() if len(v) > 1}
        return self._apply_layer_agg(df)

    def _step3_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.arch != "encoder" or not self._layer_families:
            return df
        return self._apply_layer_agg(df)

    def _apply_layer_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols, new_data = [], {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for base, pairs in self._layer_families.items():
                layer_cols = [c for _, c in pairs if c in df.columns]
                if len(layer_cols) < 2:
                    continue
                sub = df[layer_cols].values.astype(np.float64)
                # Rows where all layer values are NaN (model doesn't have those layers)
                # produce NaN aggregates — these are caught by Step 4 imputation.
                new_data[f"{base}__agg_mean"] = np.nanmean(sub, axis=1)
                new_data[f"{base}__agg_std"]  = np.nanstd(sub,  axis=1)
                new_data[f"{base}__agg_min"]  = np.nanmin(sub,  axis=1)
                new_data[f"{base}__agg_max"]  = np.nanmax(sub,  axis=1)
                drop_cols.extend(layer_cols)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        for name, vals in new_data.items():
            df[name] = vals
        return df


    # ── Step 4: Median imputation within reference distribution ───────────────

    def _step4_fit(self, df: pd.DataFrame, y: Optional[np.ndarray]) -> pd.DataFrame:
        self._impute_values = {}
        nan_cols = [c for c in df.columns if df[c].isna().any()]
        if not nan_cols:
            return df

        # Reference distribution: no-fault (baseline) class instances.
        # If y is None or baseline class is absent, fall back to overall median.
        if y is not None:
            # label index 0 is typically "baseline" after sorted() in loader.py
            # We identify baseline by checking which label has the smallest index
            # and fewest samples (as seen in data audit: 105 encoder, 70 decoder).
            # Robust fallback: use all correct-class samples (is_faulty==False).
            # Here we use the class with the minimum integer label (index 0) as proxy.
            ref_mask = (y == 0)
            if ref_mask.sum() < 5:
                ref_mask = np.ones(len(y), dtype=bool)
        else:
            ref_mask = np.ones(len(df), dtype=bool)

        for col in nan_cols:
            ref_vals = df.loc[ref_mask, col].dropna()
            med = float(ref_vals.median()) if len(ref_vals) > 0 else 0.0
            self._impute_values[col] = med

        return self._apply_impute(df)

    def _step4_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._apply_impute(df)

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, val in self._impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        # Fill any remaining NaN (columns not in impute_values, e.g. new agg cols)
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(0.0)
        return df

    # ── Step 5: CV-based low-relative-variance filter ─────────────────────────

    def _step5_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self._drop_step5 = []
        for col in df.columns:
            vals = df[col].dropna().values.astype(np.float64)
            if len(vals) == 0:
                self._drop_step5.append(col)
                continue
            mean = float(np.mean(vals))
            std  = float(np.std(vals))
            if abs(mean) < CV_TINY_MEAN:
                # Fallback: drop if absolute std is negligible
                if std < CV_FALLBACK_STD:
                    self._drop_step5.append(col)
            else:
                cv = std / abs(mean)
                if cv < CV_THRESHOLD:
                    self._drop_step5.append(col)
        return df.drop(columns=self._drop_step5)

    def _step5_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self._drop_step5 if c in df.columns]
        return df.drop(columns=present)

    # ── Step 6: Group alignment (assign final group names) ────────────────────

    def _step6_assign(self, feature_names: list[str]) -> tuple[list[str], dict]:
        """
        Assign each surviving feature to its FPG node group.
        training_dynamics features are split into logit_distribution,
        gradient_flow, and update_dynamics subgroups.
        Returns (feature_names unchanged, group_indices dict).
        """
        group_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(feature_names):
            grp = _assign_group(name)
            if grp == "training_dynamics":
                grp = _td_subgroup(name)
            group_indices.setdefault(grp, []).append(idx)
        return feature_names, group_indices

    # ── Log builder ───────────────────────────────────────────────────────────

    def _build_log(self):
        self.processing_log_ = {
            "arch": self.arch,
            "step1_dropped_nan": {
                "n": len(self._drop_step1),
                "cols": self._drop_step1[:20],
                "threshold": NAN_DROP_THRESHOLD,
            },
            "step2_log_transformed": {
                "n": len(self._log_cols),
                "cols": self._log_cols[:20],
            },
            "step3_layer_aggregated": {
                "n_families": len(self._layer_families),
                "families": list(self._layer_families.keys())[:10],
                "note": "encoder only" if self.arch == "encoder" else "no-op (decoder)",
            },
            "step4_imputed": {
                "n_cols": len(self._impute_values),
                "sample_values": dict(list(self._impute_values.items())[:5]),
            },
            "step5_dropped_cv": {
                "n": len(self._drop_step5),
                "cols": self._drop_step5[:20],
                "threshold": CV_THRESHOLD,
            },
            "step6_group_counts": {
                g: len(idxs) for g, idxs in sorted(self.group_indices_out_.items())
            },
            "features_in":  "unknown (call fit with known input size to log)",
            "features_out": len(self.feature_names_out_),
        }



# ── Convenience function for use in training scripts ─────────────────────────

def apply_processing_in_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    y_train: np.ndarray,
    arch: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict, dict]:
    """
    Fit processor on train fold, transform both train and test.
    Returns (X_train_clean, X_test_clean, feature_names_out, group_indices, log).

    IMPORTANT: X_train and X_test must be loaded with skip_impute=True from
    prepare_dataset_from_csv. The processor needs raw NaN values to correctly
    identify MNAR columns in Step 1. Pre-imputed data silently bypasses Step 1.

    This is the recommended entry point for use inside GroupKFold loops.
    The processor is fit ONLY on the training split — no leakage.
    """
    processor = FeatureProcessor(arch=arch)
    X_tr_clean, names, group_idx = processor.fit_transform(X_train, feature_names, y_train)
    X_te_clean, _,     _         = processor.transform(X_test, feature_names)
    return X_tr_clean, X_te_clean, names, group_idx, processor.processing_log_

