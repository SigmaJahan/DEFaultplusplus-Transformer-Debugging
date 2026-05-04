"""Train the hierarchical diagnostic model and write a v1 checkpoint.

This is the offline training driver for the runtime diagnosis path.
It reads a paper-aligned benchmark CSV (as produced by
``defaultpp-benchmark`` plus the FrankenFormer mutation labels) or a
prebuilt ``(X, y_detect, y_cat, y_rc)`` numpy bundle, trains a
``HierarchicalDiagnosisModel``, and emits the
``defaultplusplus.diagnosis`` v1 checkpoint format.

For development without a real corpus, ``--synthetic`` synthesizes
labels from random data. The resulting checkpoint is meaningless as
a diagnoser but exercises every piece of the loading, scaler, and
prototype paths so we can wire the rest of the package against a
stable schema.

Usage:

    # Real run (after the benchmark has produced data/encoder.csv):
    python scripts/train_diagnoser.py \\
        --arch encoder \\
        --csv data/encoder_v1_killed_binary.csv \\
        --output src/defaultplusplus/pretrained/weights/encoder.pt

    # Synthetic smoke run for development:
    python scripts/train_diagnoser.py \\
        --arch encoder --synthetic \\
        --output /tmp/encoder_smoke.pt
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from defaultplusplus.diagnosis import save_checkpoint  # noqa: E402
from hierarchical_graph_category_rootcause.model import (  # noqa: E402
    HierarchicalDiagnosisModel,
)
from src.data.feature_processor import FeatureProcessor  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=("encoder", "decoder"))
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the v1 checkpoint (.pt).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path,
                     help="Paper-aligned benchmark CSV with feature columns "
                          "and ``is_faulty`` / ``fault_category`` / "
                          "``fault_subcategory`` label columns.")
    src.add_argument("--synthetic", action="store_true",
                     help="Synthesize random data (development smoke only).")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction held out for evaluation, stratified by "
                        "(is_faulty, fault_category). Set 0 to disable.")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping: stop if val AUROC has not "
                        "improved for this many evaluation rounds. "
                        "Set 0 to disable early stopping. Ignored when "
                        "val_split=0.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate val every N epochs. Default 0 = use "
                        "the existing 5-checkpoint cadence "
                        "(epochs // 5). Set to 1 for per-epoch eval.")
    # Experimental: opt-in alternate early-stopping criterion for runs
    # where AUROC and downstream-stage accuracy peak at different epochs
    # (decoder benchmark exhibits this). Default keeps the principled
    # "stop on detection AUROC" behavior so the trainer's contract is
    # unchanged for any caller that doesn't pass this flag.
    p.add_argument("--early-stop-metric", choices=("auroc", "auroc+cat"),
                   default="auroc",
                   help="Metric driving --patience. 'auroc' (default) "
                        "stops on val detection AUROC. 'auroc+cat' is "
                        "a temporary composite (0.5*AUROC + 0.5*cat_acc) "
                        "for runs where the categorization head trains "
                        "more slowly than the detection head.")
    # Synthetic data shape knobs.
    p.add_argument("--n-samples", type=int, default=256,
                   help="Synthetic mode: number of rows to generate.")
    p.add_argument("--n-features", type=int, default=64,
                   help="Synthetic mode: feature dimensionality.")
    return p


# ─────────────────────────────────────────────────────────────────────────
# Class-balance weighting
# ─────────────────────────────────────────────────────────────────────────
def _inverse_freq_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for cross-entropy.

    Returns a (num_classes,) float tensor where rare classes have a
    higher weight. Computed as ``N / (num_classes * count_c)`` so a
    perfectly balanced dataset yields all-ones (no-op).

    Classes absent from ``y`` get weight 1.0 (the head still has those
    output neurons; we just have no signal to upweight or downweight
    them with).
    """
    counts = torch.bincount(y[y >= 0], minlength=num_classes).float()
    total = counts.sum().clamp(min=1.0)
    weights = total / (num_classes * counts.clamp(min=1.0))
    weights = torch.where(counts > 0, weights, torch.ones_like(weights))
    return weights


# ─────────────────────────────────────────────────────────────────────────
# Train/val split + evaluation
# ─────────────────────────────────────────────────────────────────────────
def _stratified_split(
    y_detect: np.ndarray,
    y_cat: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified hold-out split by (is_faulty, fault_category).

    Each (is_faulty, y_cat) bucket contributes ``round(val_frac * n)``
    rows to validation, with at least 1 row per bucket whenever the
    bucket has 2 or more rows. Returns (train_idx, val_idx).
    """
    rng = np.random.default_rng(seed)
    n = len(y_detect)
    val_mask = np.zeros(n, dtype=bool)

    # Build strata. Baselines (y_detect == 0) all share one stratum;
    # faulty rows split by y_cat.
    strata: dict[tuple, np.ndarray] = {}
    for i in range(n):
        key = (0, -1) if y_detect[i] == 0 else (1, int(y_cat[i]))
        strata.setdefault(key, []).append(i)

    for key, idxs in strata.items():
        idxs_arr = np.array(idxs)
        rng.shuffle(idxs_arr)
        if len(idxs_arr) <= 1:
            continue  # singleton: keep in training
        n_val = max(1, int(round(val_frac * len(idxs_arr))))
        n_val = min(n_val, len(idxs_arr) - 1)  # always leave at least 1 train
        val_mask[idxs_arr[:n_val]] = True

    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx


def _detection_auroc(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Binary AUROC via the rank-based formula. No sklearn dep.

    Returns NaN if either class is missing.
    """
    pos = y_true == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    n_pos = pos.sum()
    n_neg = neg.sum()
    sum_ranks_pos = ranks[pos].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def _evaluate(
    model: "HierarchicalDiagnosisModel",
    X_t: torch.Tensor,
    y_det_t: torch.Tensor,
    y_cat_t: torch.Tensor,
    y_rc_t: torch.Tensor,
    category_names: list[str],
) -> dict[str, float]:
    """Compute val metrics: detection AUROC + accuracy, category accuracy,
    per-category root-cause accuracy."""
    was_training = model.training
    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        z, _ = model.encode(X_t)
        det_logits = model.detect(z)
        det_probs = F.softmax(det_logits, dim=-1)[:, 1].cpu().numpy()
        det_pred = det_logits.argmax(dim=-1).cpu().numpy()
        y_det_np = y_det_t.cpu().numpy()
        metrics["detection_auroc"] = _detection_auroc(det_probs, y_det_np)
        metrics["detection_acc"] = float((det_pred == y_det_np).mean())

        # Category accuracy (faulty rows only)
        faulty_mask = y_det_t == 1
        if faulty_mask.any():
            cat_logits = model.categorize(z[faulty_mask])
            cat_pred = cat_logits.argmax(dim=-1).cpu().numpy()
            cat_true = y_cat_t[faulty_mask].cpu().numpy()
            valid = cat_true >= 0
            metrics["category_acc"] = (
                float((cat_pred[valid] == cat_true[valid]).mean())
                if valid.any() else float("nan")
            )

            # Per-category root-cause accuracy
            rc_correct = 0
            rc_total = 0
            for cat_idx, cat_name in enumerate(category_names):
                cat_mask = faulty_mask & (y_cat_t == cat_idx) & (y_rc_t >= 0)
                if not cat_mask.any():
                    continue
                rc_logits = model.diagnose(z[cat_mask], cat_name)
                if rc_logits is None:
                    continue
                rc_pred = rc_logits.argmax(dim=-1).cpu().numpy()
                rc_true = y_rc_t[cat_mask].cpu().numpy()
                rc_correct += int((rc_pred == rc_true).sum())
                rc_total += int(len(rc_true))
            metrics["rootcause_acc"] = (
                float(rc_correct / rc_total) if rc_total > 0 else float("nan")
            )
        else:
            metrics["category_acc"] = float("nan")
            metrics["rootcause_acc"] = float("nan")
    if was_training:
        model.train()
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Data loading paths
# ─────────────────────────────────────────────────────────────────────────
def _load_real_csv(csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Metadata columns from the paper-aligned CSV pipeline. Anything not
    # in this set is treated as a feature.
    label_cols = {"Identifier", "arch", "model_name", "dataset_name", "seed",
                  "is_faulty", "fault_category", "fault_subcategory",
                  "layer_idx", "severity_params", "label", "killed",
                  "instance_id", "architecture", "model_task", "model",
                  "dataset", "fault_id", "status"}
    feature_names = [c for c in df.columns if c not in label_cols]
    if not feature_names:
        raise ValueError(f"{csv_path} has no feature columns")

    X = df[feature_names].to_numpy(dtype=np.float32)

    if "is_faulty" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain ``is_faulty`` (0/1) label column"
        )
    y_detect = df["is_faulty"].fillna(0).astype(np.int64).to_numpy()

    is_faulty = y_detect == 1
    cat_col = df.get("fault_category")
    rc_col = df.get("fault_subcategory")

    category_names = sorted(cat_col[is_faulty].dropna().unique().tolist()) \
        if cat_col is not None else []
    rootcause_names: dict[str, list[str]] = {}
    for cat in category_names:
        mask = is_faulty & (cat_col == cat)
        subs = sorted(rc_col[mask].dropna().unique().tolist()) if rc_col is not None else []
        rootcause_names[cat] = subs

    cat2idx = {c: i for i, c in enumerate(category_names)}
    y_cat = np.full(len(df), -1, dtype=np.int64)
    if cat_col is not None:
        for i, faulty in enumerate(is_faulty):
            if faulty and cat_col.iloc[i] in cat2idx:
                y_cat[i] = cat2idx[cat_col.iloc[i]]

    # Local-to-category root-cause indices (per category, 0..n_rc-1)
    y_rc_local = np.full(len(df), -1, dtype=np.int64)
    if rc_col is not None:
        for i, faulty in enumerate(is_faulty):
            if not faulty:
                continue
            cat = cat_col.iloc[i] if cat_col is not None else None
            sc = rc_col.iloc[i]
            if cat in rootcause_names and sc in rootcause_names[cat]:
                y_rc_local[i] = rootcause_names[cat].index(sc)

    return {
        "X": X,
        "feature_names": feature_names,
        "y_detect": y_detect,
        "y_cat": y_cat,
        "y_rc_local": y_rc_local,
        "category_names": category_names,
        "rootcause_names": rootcause_names,
    }


def _synthesize(*, n_samples: int, n_features: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{i:04d}" for i in range(n_features)]

    # Random feature matrix.
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    # Half clean, half faulty.
    y_detect = (rng.random(n_samples) < 0.5).astype(np.int64)

    # Two synthetic categories among the faulty rows; each has two
    # root causes. Enough to exercise stage-3 prototype matching.
    category_names = ["qkv", "masking"]
    rootcause_names = {
        "qkv": ["zero_query", "head_interaction"],
        "masking": ["mask_application", "mask_generation"],
    }
    y_cat = np.full(n_samples, -1, dtype=np.int64)
    y_rc_local = np.full(n_samples, -1, dtype=np.int64)
    faulty_idx = np.where(y_detect == 1)[0]
    half = len(faulty_idx) // 2
    y_cat[faulty_idx[:half]] = 0      # qkv
    y_cat[faulty_idx[half:]] = 1      # masking
    # Random root cause within category.
    y_rc_local[faulty_idx] = rng.integers(0, 2, size=len(faulty_idx))

    return {
        "X": X,
        "feature_names": feature_names,
        "y_detect": y_detect,
        "y_cat": y_cat,
        "y_rc_local": y_rc_local,
        "category_names": category_names,
        "rootcause_names": rootcause_names,
    }


# ─────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────
def train(
    *,
    arch: str,
    output: Path,
    data: dict[str, Any],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    embedding_dim: int,
    dropout: float,
    seed: int,
    val_split: float = 0.2,
    patience: int = 10,
    eval_every: int = 0,
    early_stop_metric: str = "auroc",
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw_feature_names = list(data["feature_names"])
    X_raw = data["X"].astype(np.float32)
    y_det_full = data["y_detect"]
    y_cat_full = data["y_cat"]
    y_rc_full = data["y_rc_local"]
    category_names = data["category_names"]
    rootcause_names = data["rootcause_names"]
    category_sizes = {cat: max(1, len(rcs)) for cat, rcs in rootcause_names.items()}

    # Stratified hold-out split BEFORE FeatureProcessor.fit so the
    # processor's median-imputation reference and layer-family inventory
    # are computed on training data only.
    if val_split > 0:
        train_idx, val_idx = _stratified_split(
            y_det_full, y_cat_full, val_frac=val_split, seed=seed,
        )
        print(f"[train] split: {len(train_idx)} train / {len(val_idx)} val "
              f"(val_frac={val_split})")
    else:
        train_idx = np.arange(len(X_raw))
        val_idx = np.array([], dtype=np.int64)
        print(f"[train] no val split (val_split=0)")

    X_raw_train = X_raw[train_idx]
    X_raw_val = X_raw[val_idx] if len(val_idx) > 0 else None
    y_det_train = y_det_full[train_idx]
    y_cat_train = y_cat_full[train_idx]
    y_rc_train = y_rc_full[train_idx]

    # FeatureProcessor implements the 6-step paper-aligned cleanup that
    # the trainer historically skipped: NaN-rate drop, log1p for huge
    # variance, layer aggregation, reference-class median imputation,
    # low-CV drop, and group routing. Fitting it on training data only
    # avoids leaking val statistics back into the model.
    processor = FeatureProcessor(arch=arch)
    X_train, feature_names, group_indices = processor.fit_transform(
        X_raw_train, raw_feature_names, y=y_det_train,
    )
    n_features = X_train.shape[1]
    print(f"[train] FeatureProcessor: {len(raw_feature_names)} raw -> "
          f"{n_features} processed features")

    if X_raw_val is not None and len(X_raw_val) > 0:
        X_val, _, _ = processor.transform(X_raw_val, raw_feature_names)
    else:
        X_val = None

    # StandardScaler-style normalization. Fit on training data only;
    # the checkpoint stores ``mean`` / ``scale`` so inference can
    # reproduce.
    scaler_mean = X_train.mean(axis=0)
    scaler_scale = X_train.std(axis=0)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    X_scaled = (X_train - scaler_mean) / scaler_scale
    X_val_scaled = (X_val - scaler_mean) / scaler_scale if X_val is not None else None

    # Group structure: in flat mode the encoder splits hidden dim into
    # n_groups equal chunks. We use a small fixed group count for
    # synthetic / unstructured runs; real benchmark CSVs would use the
    # FPG-derived group names.
    group_names = ["attention", "qkv_alignment", "ffn_output", "residual_stream",
                   "output", "training_dynamics", "validation_perf",
                   "representation_drift"]

    model_kwargs: dict[str, Any] = {
        "input_dim": n_features,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
        "mode": "flat",
        "n_categories": max(1, len(category_names)),
        "category_sizes": category_sizes,
        "group_names": group_names,
    }
    model = HierarchicalDiagnosisModel(**model_kwargs)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(X_scaled).float()
    y_det_t = torch.from_numpy(y_det_train).long()
    y_cat_t = torch.from_numpy(y_cat_train).long()
    y_rc_t = torch.from_numpy(y_rc_train).long()
    n = X_t.shape[0]

    if X_val_scaled is not None:
        X_val_t = torch.from_numpy(X_val_scaled).float()
        y_det_val_t = torch.from_numpy(y_det_full[val_idx]).long()
        y_cat_val_t = torch.from_numpy(y_cat_full[val_idx]).long()
        y_rc_val_t = torch.from_numpy(y_rc_full[val_idx]).long()
    else:
        X_val_t = y_det_val_t = y_cat_val_t = y_rc_val_t = None

    # Inverse-frequency class weights for cross-entropy. Replaces the
    # earlier baseline-oversampling band-aid: instead of inflating row
    # counts we upweight the loss contribution of rare classes.
    detect_weight = _inverse_freq_weights(y_det_t, num_classes=2)
    cat_weight = _inverse_freq_weights(
        y_cat_t[y_det_t == 1], num_classes=max(1, len(category_names)),
    )
    rc_weights: dict[str, torch.Tensor] = {}
    for cat_idx, cat_name in enumerate(category_names):
        mask = (y_det_t == 1) & (y_cat_t == cat_idx) & (y_rc_t >= 0)
        if mask.any():
            rc_weights[cat_name] = _inverse_freq_weights(
                y_rc_t[mask], num_classes=category_sizes[cat_name],
            )

    print(f"[train] arch={arch} samples={n} features={n_features} "
          f"categories={len(category_names)} epochs={epochs}")
    print(f"[train] detection class weights: "
          f"baseline={detect_weight[0].item():.3f}, "
          f"faulty={detect_weight[1].item():.3f}")

    # Early-stopping bookkeeping. Tracks the best val score seen so far
    # (under whichever criterion the caller chose) and the model state
    # at that point; on patience exhaustion we restore the best state
    # and stop training.
    eval_cadence = eval_every if eval_every > 0 else max(1, epochs // 5)
    use_early_stop = X_val_t is not None and patience > 0
    best_score = -float("inf")
    best_auroc = -float("inf")
    best_epoch = -1
    best_state = None
    rounds_without_improve = 0
    early_stopped = False

    def _score(metrics: dict[str, float]) -> float:
        """Composite or pure-AUROC early-stop score; NaN-safe."""
        auroc = metrics.get("detection_auroc", float("nan"))
        if early_stop_metric == "auroc+cat":
            cat = metrics.get("category_acc", float("nan"))
            if np.isnan(cat):
                return auroc
            return 0.5 * auroc + 0.5 * cat
        return auroc

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_t[idx]
            yd = y_det_t[idx]
            yc = y_cat_t[idx]
            yr = y_rc_t[idx]

            z, h_groups = model.encode(xb)
            det_logits = model.detect(z)
            loss = F.cross_entropy(det_logits, yd, weight=detect_weight)

            faulty_mask = yd == 1
            if faulty_mask.any():
                cat_logits = model.categorize(z[faulty_mask])
                cat_targets = yc[faulty_mask]
                if (cat_targets >= 0).any():
                    valid = cat_targets >= 0
                    if valid.any():
                        loss = loss + F.cross_entropy(
                            cat_logits[valid], cat_targets[valid],
                            weight=cat_weight,
                        )

                # Per-category root-cause CE.
                for cat_idx, cat_name in enumerate(category_names):
                    cat_mask = faulty_mask & (yc == cat_idx) & (yr >= 0)
                    if not cat_mask.any():
                        continue
                    rc_logits = model.diagnose(z[cat_mask], cat_name)
                    if rc_logits is None:
                        continue
                    loss = loss + F.cross_entropy(
                        rc_logits, yr[cat_mask],
                        weight=rc_weights.get(cat_name),
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        is_eval_epoch = (
            (epoch + 1) % eval_cadence == 0 or epoch == epochs - 1
        )
        if is_eval_epoch:
            msg = (f"[train]   epoch {epoch + 1}/{epochs}  "
                   f"loss={total_loss / max(1, n_batches):.4f}")
            if X_val_t is not None:
                m = _evaluate(model, X_val_t, y_det_val_t, y_cat_val_t,
                              y_rc_val_t, category_names)
                auroc = m["detection_auroc"]
                msg += (f"  val_auroc={auroc:.4f}"
                        f" val_det_acc={m['detection_acc']:.4f}"
                        f" val_cat_acc={m['category_acc']:.4f}"
                        f" val_rc_acc={m['rootcause_acc']:.4f}")
                if use_early_stop:
                    score = _score(m)
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_auroc = auroc
                        best_epoch = epoch + 1
                        best_state = {
                            k: v.detach().clone()
                            for k, v in model.state_dict().items()
                        }
                        rounds_without_improve = 0
                        msg += " *"
                    else:
                        rounds_without_improve += 1
                        msg += f"  ({rounds_without_improve}/{patience})"
            print(msg)

            if (use_early_stop
                    and rounds_without_improve >= patience):
                print(f"[train]   early stop: no {early_stop_metric} "
                      f"improvement for {patience} eval rounds "
                      f"(best score={best_score:.4f}, "
                      f"auroc={best_auroc:.4f} at epoch {best_epoch})")
                early_stopped = True
                break

    # Restore best-by-val state before final eval / prototype computation.
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[train] restored best checkpoint from epoch {best_epoch} "
              f"(val_auroc={best_auroc:.4f})")

    # Final eval summary on the held-out split.
    if X_val_t is not None:
        final = _evaluate(model, X_val_t, y_det_val_t, y_cat_val_t,
                          y_rc_val_t, category_names)
        print()
        print(f"[eval] final val metrics ({len(val_idx)} rows):")
        print(f"[eval]   detection AUROC : {final['detection_auroc']:.4f}")
        print(f"[eval]   detection acc   : {final['detection_acc']:.4f}")
        print(f"[eval]   category acc    : {final['category_acc']:.4f}")
        print(f"[eval]   root-cause acc  : {final['rootcause_acc']:.4f}")
        print()

    # Compute prototypes from training data so stage 3 explainability
    # works at inference time.
    model.eval()
    with torch.no_grad():
        _, h_groups_full = model.encode(X_t)
        for cat_idx, cat_name in enumerate(category_names):
            cat_mask = (y_det_t == 1) & (y_cat_t == cat_idx) & (y_rc_t >= 0)
            if cat_mask.any():
                model.compute_prototypes(
                    h_groups_full[cat_mask], y_rc_t[cat_mask], cat_name,
                )

    prototypes = {k: v.cpu() for k, v in model._prototypes.items()}

    output = Path(output)
    save_checkpoint(
        path=output,
        arch=arch,
        feature_names=feature_names,
        category_names=category_names,
        category_sizes=category_sizes,
        rootcause_names=rootcause_names,
        group_names=group_names,
        model_state_dict=model.state_dict(),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        prototypes=prototypes,
        model_kwargs=model_kwargs,
        extra={
            "raw_feature_names": raw_feature_names,
            "feature_processor": processor,
            "group_indices": group_indices,
        },
    )
    print(f"[train] wrote checkpoint to {output}")
    return output


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.synthetic:
        data = _synthesize(
            n_samples=args.n_samples,
            n_features=args.n_features,
            seed=args.seed,
        )
    else:
        data = _load_real_csv(args.csv)

    train(
        arch=args.arch,
        output=args.output,
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        seed=args.seed,
        val_split=args.val_split,
        patience=args.patience,
        eval_every=args.eval_every,
        early_stop_metric=args.early_stop_metric,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
