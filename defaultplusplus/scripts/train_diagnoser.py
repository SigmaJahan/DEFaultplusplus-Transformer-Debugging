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
    # Synthetic data shape knobs.
    p.add_argument("--n-samples", type=int, default=256,
                   help="Synthetic mode: number of rows to generate.")
    p.add_argument("--n-features", type=int, default=64,
                   help="Synthetic mode: feature dimensionality.")
    return p


# ─────────────────────────────────────────────────────────────────────────
# Data loading paths
# ─────────────────────────────────────────────────────────────────────────
def _load_real_csv(csv_path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    label_cols = {"Identifier", "arch", "model_name", "dataset_name", "seed",
                  "is_faulty", "fault_category", "fault_subcategory",
                  "layer_idx", "severity_params", "label", "killed"}
    feature_names = [c for c in df.columns if c not in label_cols]
    if not feature_names:
        raise ValueError(f"{csv_path} has no feature columns")

    X = df[feature_names].fillna(0.0).to_numpy(dtype=np.float32)

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
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)

    feature_names = data["feature_names"]
    X = data["X"].astype(np.float32)
    n_features = X.shape[1]
    category_names = data["category_names"]
    rootcause_names = data["rootcause_names"]
    category_sizes = {cat: max(1, len(rcs)) for cat, rcs in rootcause_names.items()}

    # StandardScaler-style normalization. Compute on training data; the
    # checkpoint stores ``mean`` / ``scale`` so inference can reproduce.
    scaler_mean = X.mean(axis=0)
    scaler_scale = X.std(axis=0)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    X_scaled = (X - scaler_mean) / scaler_scale

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
    y_det_t = torch.from_numpy(data["y_detect"]).long()
    y_cat_t = torch.from_numpy(data["y_cat"]).long()
    y_rc_t = torch.from_numpy(data["y_rc_local"]).long()
    n = X_t.shape[0]

    print(f"[train] arch={arch} samples={n} features={n_features} "
          f"categories={len(category_names)} epochs={epochs}")

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
            loss = F.cross_entropy(det_logits, yd)

            faulty_mask = yd == 1
            if faulty_mask.any():
                cat_logits = model.categorize(z[faulty_mask])
                cat_targets = yc[faulty_mask]
                if (cat_targets >= 0).any():
                    valid = cat_targets >= 0
                    if valid.any():
                        loss = loss + F.cross_entropy(
                            cat_logits[valid], cat_targets[valid],
                        )

                # Per-category root-cause CE.
                for cat_idx, cat_name in enumerate(category_names):
                    cat_mask = faulty_mask & (yc == cat_idx) & (yr >= 0)
                    if not cat_mask.any():
                        continue
                    rc_logits = model.diagnose(z[cat_mask], cat_name)
                    if rc_logits is None:
                        continue
                    loss = loss + F.cross_entropy(rc_logits, yr[cat_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"[train]   epoch {epoch + 1}/{epochs}  "
                  f"loss={total_loss / max(1, n_batches):.4f}")

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
