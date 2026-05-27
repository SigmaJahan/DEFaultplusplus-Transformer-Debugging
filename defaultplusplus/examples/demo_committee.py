"""DEFault++ committee-demo script (benchmark replay, ~5 seconds).

Loads two rows from the DEFault-bench v1 encoder benchmark - one
clean run, one with a real ``zero_query`` fault - and runs the
bundled encoder diagnoser on both. Both rows are pinned by CSV
index so the verdict is bit-exact reproducible across runs and
machines.

Why this shape: a live fine-tune is only ~30 seconds of training
data, which sits far outside the diagnoser's training distribution
(full fine-tunes, 1442 baseline rows). Detection is rock-solid in
both cases, but category/root-cause stability needs the full
trajectory. Using a real benchmark sample shows the full hierarchy
firing correctly: detection + category + root cause, all matching
ground truth.

Run (from the package root):

    pip install -e ".[viz]"
    python examples/demo_committee.py

Wall-clock: ~5 seconds. No model download, no training.

Outputs (written next to the script):

    demo_diagnosis_faulty.html
    demo_diagnosis_clean.html

Open both in a browser side by side for the committee walkthrough.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from defaultplusplus.diagnosis import (
    PretrainedWeightsMissingError,
    load_pretrained,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_CSV = REPO_ROOT / "dist" / "defaultpp-bench-v1" / "encoder_merged.csv"
OUT_DIR = Path(__file__).resolve().parent

# Pinned indices into BENCH_CSV. Both rows score 1.000 / 1.000 / 1.000
# (faulty) and 0.0002 (clean) on the bundled encoder.pt, verified at
# script-authoring time. Do not change without re-verifying.
FAULTY_INDEX = 771   # qkv / zero_query, fault_id E2.1
CLEAN_INDEX = 4777   # baseline, det prob ~0.0002

LABEL_COLS = ("is_faulty", "fault_category", "fault_subcategory", "fault_id")


def _banner(text: str) -> None:
    bar = "=" * max(60, len(text) + 4)
    print(f"\n{bar}\n  {text}\n{bar}")


def _row_to_features(row, schema):
    return {
        k: float(row[k]) if k in row and pd.notna(row[k]) else 0.0
        for k in schema
    }


def _print_ground_truth(row) -> None:
    is_faulty = int(row.get("is_faulty", 0))
    if is_faulty:
        print(f"  ground truth    : faulty "
              f"(category={row.get('fault_category')}, "
              f"root_cause={row.get('fault_subcategory')}, "
              f"fault_id={row.get('fault_id')})")
    else:
        print("  ground truth    : clean (is_faulty=0)")


def _print_diagnosis(diag) -> None:
    print(f"  is_faulty       : {diag.is_faulty}")
    print(f"  detection_prob  : {diag.detection_prob:.3f}")
    if diag.category:
        print(f"  category        : {diag.category}  "
              f"(p={diag.category_prob:.3f})")
        print(f"  root_cause      : {diag.root_cause}  "
              f"(p={diag.root_cause_prob:.3f})")
    if diag.group_importance:
        top = sorted(diag.group_importance.items(),
                     key=lambda kv: -kv[1])[:3]
        names = ", ".join(f"{n} ({v:+.2f})" for n, v in top)
        print(f"  top groups      : {names}")


def diagnose_row(label: str, row, predictor) -> None:
    _banner(f"Run: {label}")
    _print_ground_truth(row)

    feature_dict = _row_to_features(row, predictor.feature_names)
    diagnosis = predictor.predict(feature_dict)
    print()
    print("  -- DEFault++ prediction --")
    _print_diagnosis(diagnosis)

    try:
        from defaultplusplus.viz import save_diagnosis_report
    except ImportError:
        print("\n[demo] viz extras not installed; install with "
              "``pip install defaultplusplus[viz]`` for the HTML report")
        return
    report_path = OUT_DIR / f"demo_diagnosis_{label}.html"
    save_diagnosis_report(diagnosis, feature_dict, report_path)
    print(f"\n  HTML report     : {report_path.name}")


def main() -> None:
    _banner("DEFault++ committee demo")
    print(f"benchmark   : {BENCH_CSV.relative_to(REPO_ROOT)}")
    print(f"outputs     : {OUT_DIR}")

    if not BENCH_CSV.exists():
        raise SystemExit(
            f"Benchmark CSV not found at {BENCH_CSV}. Either re-stage "
            f"the v1 bundle with data/stage_release_bundle.py, or "
            f"download it from Zenodo (DOI 10.5281/zenodo.20018623)."
        )

    try:
        predictor = load_pretrained("encoder")
    except PretrainedWeightsMissingError as exc:
        raise SystemExit(f"Pretrained weights missing: {exc}")

    print(f"predictor   : encoder.pt "
          f"({len(predictor.feature_names)} features, "
          f"{len(predictor.category_names)} categories)")

    keep_cols = list(predictor.feature_names) + list(LABEL_COLS)
    df = pd.read_csv(BENCH_CSV, low_memory=False, usecols=lambda c: c in keep_cols)

    faulty_row = df.iloc[FAULTY_INDEX]
    clean_row = df.iloc[CLEAN_INDEX]

    diagnose_row("faulty", faulty_row, predictor)
    diagnose_row("clean", clean_row, predictor)

    _banner("Done")
    print("Open these two HTML files side by side in a browser:")
    print(f"  {OUT_DIR / 'demo_diagnosis_faulty.html'}")
    print(f"  {OUT_DIR / 'demo_diagnosis_clean.html'}")


if __name__ == "__main__":
    main()
