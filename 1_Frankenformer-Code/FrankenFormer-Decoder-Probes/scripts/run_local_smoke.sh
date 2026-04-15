#!/bin/bash
# Local GPU smoke test: distilgpt2 + wikitext-2, 1 baseline, 1 epoch, batch=2
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_ROOT/venv"
RESULTS_DIR="$PROJECT_ROOT/results/local_smoke"
HF_CACHE="$PROJECT_ROOT/hf-cache"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV. Run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

PYTHON_BIN="$VENV/bin/python"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_MODULES_CACHE="$HF_CACHE/modules"
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Local GPU Smoke Test"
echo "========================================"
echo "Model:   distilgpt2"
echo "Dataset: wikitext-2"
echo "Epochs:  1"
echo "Batch:   2"
echo "Results: $RESULTS_DIR"
echo "HF cache: $HF_CACHE"
echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Step 0: Validate pipeline components
echo "--- Step 0: Pipeline validation ---"
$PYTHON_BIN scripts/validate_pipeline.py --cuda
echo ""

# Step 1: Unit probe smoke test
echo "--- Step 1: Probe unit test ---"
$PYTHON_BIN scripts/smoke_test_probes.py --cuda
echo ""

# Step 2: Full pipeline (1 baseline, 1 epoch, batch=2, max_length=64)
echo "--- Step 2: Pipeline run ---"
$PYTHON_BIN scripts/run_pipeline.py \
    --matrix-config "$PROJECT_ROOT/config/local_smoke_matrix.yaml" \
    --config-file "$PROJECT_ROOT/config/smoke_test_pipeline.json" \
    --results-dir "$RESULTS_DIR" \
    --model-keys "distilgpt2" \
    --task-keys "wikitext-2" \
    --max-configs 1 \
    --min-gpu-mem-gb 0
echo ""

# Step 3: Verify outputs
echo "--- Step 3: Verify outputs ---"
COMBO_DIR="$RESULTS_DIR/distilgpt2/wikitext-2"
ERRORS=0

if [ -f "$COMBO_DIR/metrics.h5" ]; then
    echo "  HDF5: $(du -h "$COMBO_DIR/metrics.h5" | cut -f1)"
else
    echo "  ERROR: metrics.h5 not found"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$COMBO_DIR/dataset.db" ]; then
    echo "  SQLite: $(du -h "$COMBO_DIR/dataset.db" | cut -f1)"
else
    echo "  ERROR: dataset.db not found"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED ($ERRORS errors)"
fi

exit $ERRORS
