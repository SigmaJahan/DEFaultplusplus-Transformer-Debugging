"""Feature processing pipeline.

Six-step transformation that turns raw extracted training traces into the
fixed-length feature vector consumed by the diagnostic model.

  1. Structural drop    drop columns with NaN rate > 0.40 (treated as
                        missing-not-at-random rather than imputable)
  2. Scale normalise    log1p-transform columns whose variance or
                        max/median ratio is too large to scale
  3. Layer aggregation  collapse per-layer expansions to four statistics
                        (mean, std, min, max). Encoders only; decoder
                        traces already arrive layer-aggregated.
  4. Median imputation  fill remaining NaN with the within-fold median of
                        the no-fault reference distribution
  5. CV filter          drop columns whose coefficient of variation
                        std/|mean| is below 0.01
  6. Group assignment   route each surviving feature to its diagnostic
                        feature group (see ``feature_groups.py``)

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

# Match both short layer prefixes (foo_l3_bar) and long ones (foo_layer3_bar).
# The latter is what the offline raw collector emits (e.g.
# ``grad_norm_layer3_attention_*``); the former is the in-process extractor's
# convention.
LAYER_RE = re.compile(r"^(.*?)_l(?:ayer)?(\d+)_(.*)$")


# Step 6 delegates to feature_groups.assign_feature_to_group, which encodes
# the Table 7.7 mapping. The local copy that previously lived here drifted
# from the official mapping and has been removed.
from .feature_groups import assign_feature_to_group as _assign_group  # noqa: E402



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

    # ── Step 3: Layer aggregation ─────────────────────────────────────────────
    # Data-driven, not arch-gated: aggregation runs whenever the input
    # carries per-layer column families (``..._l{N}_...`` or
    # ``..._layer{N}_...``). The original encoder-only gate assumed
    # decoder traces arrived pre-aggregated by the in-process extractor;
    # offline raw CSVs from either arch may include per-layer columns,
    # so we let the data decide.

    def _step3_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        families: dict[str, list] = {}
        for col in df.columns:
            m = LAYER_RE.match(col.lower())
            if m:
                base = f"{m.group(1)}__{m.group(3)}"
                families.setdefault(base, []).append((int(m.group(2)), col))
        self._layer_families = {b: sorted(v) for b, v in families.items() if len(v) > 1}
        return self._apply_layer_agg(df)

    def _step3_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._layer_families:
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
        Assign each surviving feature to its DEFault++ feature group
        (Table 7.7). Group names are paper-aligned: structural groups receive
        FPG-derived neighbor edges, non-structural groups (representation_drift,
        training_dynamics, validation_perf) receive self-loops only.
        Returns (feature_names unchanged, group_indices dict).
        """
        group_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(feature_names):
            grp = _assign_group(name)
            if grp is None:
                continue
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

