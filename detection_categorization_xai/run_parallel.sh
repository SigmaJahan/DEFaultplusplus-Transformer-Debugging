#!/bin/bash
# Parallel pipeline: 4 classifier jobs simultaneously, then 4 XAI jobs simultaneously.
# Usage: bash run_parallel.sh
# Logs per job in logs/. Results in results/.
# Wall time: ~2-3 hours on a 32+ core CPU node (vs ~7-10 hours sequential).
# No set -e: if one job fails, others still continue

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results logs

echo "============================================"
echo "STEP 1: Classification (4 configs in parallel)"
echo "============================================"
echo "Started: $(date)"

python run_classifiers.py --data data/enc_v1_detection.pkl       --out results/enc_detection.json       > logs/cls_enc_det.log 2>&1 &
python run_classifiers.py --data data/dec_v1_detection.pkl       --out results/dec_detection.json       > logs/cls_dec_det.log 2>&1 &
python run_classifiers.py --data data/enc_v1_categorization.pkl  --out results/enc_categorization.json  > logs/cls_enc_cat.log 2>&1 &
python run_classifiers.py --data data/dec_v1_categorization.pkl  --out results/dec_categorization.json  > logs/cls_dec_cat.log 2>&1 &
echo "  4 classifier jobs launched. Waiting..."
wait
echo "  All classifiers done: $(date)"

# Check for errors
FAILED=0
for f in logs/cls_*.log; do
    if grep -q "ERROR\|Traceback" "$f"; then
        echo "  WARN: errors in $f"
        FAILED=1
    fi
done
if [ $FAILED -eq 1 ]; then
    echo "  Some classifiers had errors. Check logs/cls_*.log"
fi

echo ""
echo "============================================"
echo "STEP 2: XAI (6 configs in parallel)"
echo "============================================"
echo "Started: $(date)"

python run_xai.py --data data/enc_v1_detection.pkl       --results results/enc_detection.json       --out results/xai_enc_detection.json       > logs/xai_enc_det.log 2>&1 &
python run_xai.py --data data/dec_v1_detection.pkl       --results results/dec_detection.json       --out results/xai_dec_detection.json       > logs/xai_dec_det.log 2>&1 &
python run_xai.py --data data/enc_v1_categorization.pkl  --results results/enc_categorization.json  --out results/xai_enc_categorization.json  > logs/xai_enc_cat.log 2>&1 &
python run_xai.py --data data/dec_v1_categorization.pkl  --results results/dec_categorization.json  --out results/xai_dec_categorization.json  > logs/xai_dec_cat.log 2>&1 &
echo "  4 XAI jobs launched. Waiting..."
wait
echo "  All XAI done: $(date)"

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo "Results:"
ls -lh results/
echo ""
echo "Logs:"
ls -lh logs/
