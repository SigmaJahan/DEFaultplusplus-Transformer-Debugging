#!/bin/bash
# Smoke test: 1 baseline config, distilbert-base-uncased, sst2, 1 epoch, batch_size=2
#
# Verifies end-to-end pipeline output:
#   - HDF5 metrics file with probe features
#   - SQLite database with configuration results
#   - Structural probes (ffn_delta, ln_std, residual_cos, etc.) are non-zero
#
#SBATCH --job-name=enc-probe-smoke
#SBATCH --account=def-mrdal22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-user=sigma.jahan@dal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=CC-logs/probes/enc-probe-smoke-%j.out
#SBATCH --error=CC-logs/probes/enc-probe-smoke-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/project/def-mrdal22/sjahan/FrankenFormer-Encoder-Probes}"
VENV_TARBALL="/project/def-mrdal22/sjahan/venv_encoder_packed.tar.gz"
VENV_FALLBACK="/project/def-mrdal22/sjahan/venv-encoder"

module load StdEnv/2023 gcc/12.3 arrow/14.0.1 python/3.10 cuda/12.2 rust/1.70.0

if [ -f "$VENV_TARBALL" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Extracting venv tarball to $SLURM_TMPDIR ..."
    tar -xzf "$VENV_TARBALL" -C "$SLURM_TMPDIR"
    PYTHON_BIN="$SLURM_TMPDIR/venv/bin/python"
    echo "Using fast local venv: $PYTHON_BIN"
elif [ -x "$VENV_FALLBACK/bin/python" ]; then
    echo "WARNING: Tarball not found, falling back to $VENV_FALLBACK"
    PYTHON_BIN="$VENV_FALLBACK/bin/python"
else
    echo "ERROR: No venv found. Run: bash scripts/setup.sh"
    exit 1
fi

source "$PROJECT_ROOT/scripts/env_config.sh"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

RESULTS_DIR="$PROJECT_ROOT/results/smoke_test"
mkdir -p "$PROJECT_ROOT/CC-logs/probes"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_ROOT" || exit 1

echo "================================================================================"
echo "Encoder Probe Smoke Test"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Model: distilbert-base-uncased"
echo "Dataset: sst2"
echo "================================================================================"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# Step 1: Unit-level probe smoke test
echo "--- Step 1: Probe unit test ---"
$PYTHON_BIN scripts/smoke_test_probes.py --cuda
echo "Step 1 PASSED"
echo ""

# Step 2: Full pipeline smoke test
echo "--- Step 2: Pipeline run (1 baseline, 1 epoch) ---"
$PYTHON_BIN scripts/run_pipeline.py \
    --matrix-config "$PROJECT_ROOT/config/local_smoke_matrix.yaml" \
    --fault-config "$PROJECT_ROOT/config/smoke_test_pipeline.json" \
    --results-dir "$RESULTS_DIR" \
    --cuda \
    --config-index 0
echo "Step 2 PASSED"
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
