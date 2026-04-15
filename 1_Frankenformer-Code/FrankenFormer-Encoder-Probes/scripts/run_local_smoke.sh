#!/bin/bash
# Local GPU smoke test: distilbert-base-uncased + sst2, 1 baseline, 1 epoch, batch=2
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
echo "Local GPU Smoke Test (Encoder)"
echo "========================================"
echo "Model:   distilbert-base-uncased"
echo "Dataset: sst2"
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

# Step 2: Full pipeline (1 baseline, 1 epoch, batch=2)
echo "--- Step 2: Pipeline run ---"
$PYTHON_BIN scripts/run_pipeline.py \
    --matrix-config "$PROJECT_ROOT/config/local_smoke_matrix.yaml" \
    --fault-config "$PROJECT_ROOT/config/smoke_test_pipeline.json" \
    --results-dir "$RESULTS_DIR" \
    --cuda \
    --config-index 0
echo ""

# Step 3: Verify outputs
echo "--- Step 3: Verify outputs ---"
ERRORS=0

H5_COUNT=$(find "$RESULTS_DIR" -name "*.h5" 2>/dev/null | wc -l)
DB_COUNT=$(find "$RESULTS_DIR" -name "*.db" 2>/dev/null | wc -l)

if [ "$H5_COUNT" -gt 0 ]; then
    echo "  HDF5 files: $H5_COUNT"
else
    echo "  ERROR: No metrics.h5 found"
    ERRORS=$((ERRORS + 1))
fi

if [ "$DB_COUNT" -gt 0 ]; then
    echo "  SQLite files: $DB_COUNT"
else
    echo "  ERROR: No dataset.db found"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "SMOKE TEST PASSED"
else
    echo "SMOKE TEST FAILED ($ERRORS errors)"
fi

exit $ERRORS
