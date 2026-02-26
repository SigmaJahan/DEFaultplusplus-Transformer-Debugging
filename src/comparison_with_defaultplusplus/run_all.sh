#!/usr/bin/env bash
# RQ6: Baseline comparison experiments
# Reproduces Tables 1-3 from the paper (detection, categorization, graph diagnosis).
#
# Prerequisites:
#   - Python 3.10+ with packages: numpy, pandas, scikit-learn, xgboost, matplotlib
#   - Pickle data in ../detection_categorization_xai/data/
#   - NDG results in ../results/stage_3_diagnosis/
#
# Usage:
#   cd src/comparison_with_defaultplusplus/
#   bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== RQ6: Baseline Comparison Experiments ==="
echo "Output: results/"
echo ""

python run_rq6_baselines.py

echo ""
echo "=== Done ==="
echo "Results:  results/rq6_summary.json"
echo "Tables:   results/rq6_latex_tables.tex"
echo "Plots:    results/plots/"
