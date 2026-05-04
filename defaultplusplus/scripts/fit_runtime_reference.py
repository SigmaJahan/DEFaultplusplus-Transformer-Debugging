"""Fit a RuntimeReference (clean baseline summary) from a merged CSV.

Usage::

    python scripts/fit_runtime_reference.py \\
        --arch encoder \\
        --csv data/paper-aligned-csv/encoder_merged.csv \\
        --output src/defaultplusplus/pretrained/weights/encoder_reference.npz

The reference captures per-key (median, MAD, std, count) over the
``is_faulty == 0`` rows. RuntimeNormalizer.encode() uses these stats to
fill missing keys with the median (mode='raw') or to produce z-scores
(mode='anomaly').
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from defaultplusplus.processing import fit_reference  # noqa: E402


META_COLS = {
    "instance_id", "architecture", "model_task", "model", "dataset",
    "seed", "is_faulty", "fault_category", "fault_subcategory",
    "fault_id", "layer_idx", "severity_params", "status",
    "Identifier", "arch", "model_name", "dataset_name", "label", "killed",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=("encoder", "decoder"))
    p.add_argument("--csv", required=True, type=Path,
                   help="Merged trainer CSV (e.g. data/paper-aligned-csv/encoder_merged.csv)")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the .npz reference")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"missing {args.csv}", file=sys.stderr)
        return 1

    print(f"[fit] reading {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    if "is_faulty" not in df.columns:
        print(f"{args.csv} missing is_faulty column", file=sys.stderr)
        return 1

    feat_cols = [c for c in df.columns if c not in META_COLS]
    baseline = df[df["is_faulty"] == 0][feat_cols]
    print(f"[fit] {len(baseline)} baseline rows, {len(feat_cols)} feature cols")
    if len(baseline) == 0:
        print("no baseline rows; cannot fit reference", file=sys.stderr)
        return 1

    X = baseline.to_numpy(dtype=np.float32)
    ref = fit_reference(X, feat_cols, arch=args.arch)
    ref.save(args.output)
    print(f"[fit] wrote {args.output} "
          f"(median={ref.median.shape}, mad={ref.mad.shape}, "
          f"n_baseline={ref.n_baseline})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
