#!/bin/bash
# Run classifier + XAI pipeline for encoder and decoder configurations.
# Usage: bash run_all.sh
# Results are saved to results/ directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results

echo "============================================"
echo "STEP 1: Classification (4 models x 4 datasets)"
echo "============================================"

# Detection (binary: correct vs buggy)
echo ""
echo "--- Encoder Detection ---"
python run_classifiers.py --data data/enc_v1_detection.pkl --out results/enc_detection.json

echo ""
echo "--- Decoder Detection ---"
python run_classifiers.py --data data/dec_v1_detection.pkl --out results/dec_detection.json

# Categorization (multiclass: fault families)
echo ""
echo "--- Encoder Categorization ---"
python run_classifiers.py --data data/enc_v1_categorization.pkl --out results/enc_categorization.json

echo ""
echo "--- Decoder Categorization ---"
python run_classifiers.py --data data/dec_v1_categorization.pkl --out results/dec_categorization.json

echo ""
echo "============================================"
echo "STEP 2: XAI (SHAP + Counterfactuals + Rules)"
echo "============================================"

# XAI uses XGBoost best params from classifier results
echo ""
echo "--- Encoder Detection XAI ---"
python run_xai.py --data data/enc_v1_detection.pkl --results results/enc_detection.json --out results/xai_enc_detection.json

echo ""
echo "--- Decoder Detection XAI ---"
python run_xai.py --data data/dec_v1_detection.pkl --results results/dec_detection.json --out results/xai_dec_detection.json

echo ""
echo "--- Encoder Categorization XAI ---"
python run_xai.py --data data/enc_v1_categorization.pkl --results results/enc_categorization.json --out results/xai_enc_categorization.json

echo ""
echo "--- Decoder Categorization XAI ---"
python run_xai.py --data data/dec_v1_categorization.pkl --results results/dec_categorization.json --out results/xai_dec_categorization.json

echo ""
echo "============================================"
echo "ALL DONE. Results in results/"
echo "============================================"
ls -lh results/
