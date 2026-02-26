#!/usr/bin/env bash
# Baseline comparison experiments
# Reproduces Tables 1-3 from the paper (detection, categorization, graph diagnosis).
#
# Prerequisites:
#   - Python 3.10+ with packages: numpy, pandas, scikit-learn, xgboost, matplotlib
#   - Pickle data in ../detection_categorization_xai/data/
#   - NDG results in ../results/diagnosis/
#
# Usage:
#   cd src/comparison_with_defaultplusplus/
#   bash run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Baseline Comparison Experiments ==="
echo "Output: results/"
echo ""

python run_baseline_comparison.py

echo ""
echo "=== Done ==="
echo "Results:  results/summary.json"
echo "Tables:   results/baseline_tables.tex"
echo "Plots:    results/plots/"
