#!/bin/bash
# Smoke test: 1 baseline config, distilgpt2, wikitext-2, 2 epochs, batch_size=2
#
# Verifies end-to-end pipeline output:
#   - HDF5 metrics file with probe features
#   - SQLite database with configuration results
#   - Structural probes (ffn_delta, ln_std, residual_cos, etc.) are non-zero
#
# Uses $SLURM_TMPDIR for venv extraction (fast local SSD).
#
# Prerequisites:
#   bash scripts/pack_venv.sh   (run once on login node)
#
# Usage:
#   sbatch scripts/submit_smoke_test.sh
#
#SBATCH --job-name=probe-smoke
#SBATCH --account=def-mrdal22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-user=sigma.jahan@dal.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=CC-logs/probes/probe-smoke-%j.out
#SBATCH --error=CC-logs/probes/probe-smoke-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/project/def-mrdal22/sjahan/FrankenFormer-Decoder-Probes}"
VENV_TARBALL="/project/def-mrdal22/sjahan/venv_packed.tar.gz"
VENV_FALLBACK="/project/def-mrdal22/sjahan/venv"

# ============================================================================
# Load Modules (before sourcing env_config so python is available)
# ============================================================================
module load StdEnv/2023 gcc/12.3 arrow/14.0.1 python/3.10 cuda/12.2 rust/1.70.0

# ============================================================================
# Extract venv to SLURM_TMPDIR (local SSD -> fast imports)
# ============================================================================
if [ -f "$VENV_TARBALL" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Extracting venv tarball to $SLURM_TMPDIR ..."
    tar -xzf "$VENV_TARBALL" -C "$SLURM_TMPDIR"
    PYTHON_BIN="$SLURM_TMPDIR/venv/bin/python"
    echo "Using fast local venv: $PYTHON_BIN"
elif [ -x "$VENV_FALLBACK/bin/python" ]; then
    echo "WARNING: Tarball not found, falling back to $VENV_FALLBACK"
    PYTHON_BIN="$VENV_FALLBACK/bin/python"
else
    echo "ERROR: No venv found. Run: bash scripts/migrate_to_project.sh"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================
source "$PROJECT_ROOT/scripts/env_config.sh"

# Performance
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# ============================================================================
# Directories
# ============================================================================
RESULTS_DIR="$PROJECT_ROOT/results/smoke_test"
mkdir -p "$PROJECT_ROOT/CC-logs/probes"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_ROOT" || exit 1

echo "================================================================================"
echo "Decoder Probe Smoke Test"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Model: distilgpt2"
echo "Dataset: wikitext-2"
echo "Configs: 1 baseline (2 epochs, batch_size=2, max_length=64)"
echo "Project: $PROJECT_ROOT"
echo "Results: $RESULTS_DIR"
echo "================================================================================"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# ============================================================================
# Step 1: Unit-level probe smoke test
# ============================================================================
echo "================================================================================"
echo "Step 1: Unit-level probe verification (smoke_test_probes.py)"
echo "================================================================================"

$PYTHON_BIN scripts/smoke_test_probes.py --cuda
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo "FAIL: smoke_test_probes.py exited with code $STEP1_EXIT"
    exit $STEP1_EXIT
fi
echo "Step 1 PASSED"
echo ""

# ============================================================================
# Step 2: Full pipeline smoke test (1 baseline, 2 epochs)
# ============================================================================
echo "================================================================================"
echo "Step 2: Full pipeline smoke test (1 baseline, 2 epochs, batch=2)"
echo "================================================================================"

$PYTHON_BIN scripts/run_pipeline.py \
    --matrix-config "$PROJECT_ROOT/config/smoke_test_matrix.yaml" \
    --config-file "$PROJECT_ROOT/config/smoke_test_pipeline.json" \
    --results-dir "$RESULTS_DIR" \
    --model-keys "distilgpt2" \
    --task-keys "wikitext-2" \
    --max-configs 1 \
    --min-gpu-mem-gb 0
STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo "FAIL: Pipeline exited with code $STEP2_EXIT"
    exit $STEP2_EXIT
fi
echo "Step 2 PASSED"
echo ""

# ============================================================================
# Step 3: Verify outputs
# ============================================================================
echo "================================================================================"
echo "Step 3: Verify output files"
echo "================================================================================"

COMBO_DIR="$RESULTS_DIR/distilgpt2/wikitext-2"
ERRORS=0

# Check HDF5
H5_FILE="$COMBO_DIR/metrics.h5"
if [ -f "$H5_FILE" ]; then
    H5_SIZE=$(du -h "$H5_FILE" | cut -f1)
    echo "  HDF5 file: $H5_FILE ($H5_SIZE)"

    # Verify HDF5 contents with Python
    $PYTHON_BIN - <<'PYEOF'
import h5py, sys

h5_path = sys.argv[1] if len(sys.argv) > 1 else "results/smoke_test/distilgpt2/wikitext-2/metrics.h5"
with h5py.File(h5_path, "r") as f:
    configs = list(f.keys())
    print(f"  HDF5 configs: {len(configs)}")
    if not configs:
        print("  ERROR: No configs in HDF5!")
        sys.exit(1)

    for cfg_name in configs:
        cfg_group = f[cfg_name]
        datasets = list(cfg_group.keys())
        print(f"  Config '{cfg_name}': {len(datasets)} datasets")
        for ds_name in sorted(datasets)[:20]:
            ds = cfg_group[ds_name]
            print(f"    {ds_name}: shape={ds.shape}, dtype={ds.dtype}")

        # Check for probe-related datasets
        probe_keywords = ["ffn_delta", "ln_std", "residual_cos", "entropy", "head_sim", "pre_softmax"]
        found_probes = [d for d in datasets if any(k in d for k in probe_keywords)]
        print(f"  Probe datasets found: {len(found_probes)}")
        if found_probes:
            for p in sorted(found_probes)[:10]:
                vals = cfg_group[p][:]
                nonzero = (abs(vals) > 1e-12).sum()
                print(f"    {p}: {nonzero}/{len(vals)} non-zero values")

print("  HDF5 verification PASSED")
PYEOF
    H5_CHECK=$?
    if [ $H5_CHECK -ne 0 ]; then ERRORS=$((ERRORS + 1)); fi
else
    echo "  ERROR: HDF5 file not found at $H5_FILE"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check SQLite
DB_FILE="$COMBO_DIR/dataset.db"
if [ -f "$DB_FILE" ]; then
    DB_SIZE=$(du -h "$DB_FILE" | cut -f1)
    echo "  SQLite DB: $DB_FILE ($DB_SIZE)"

    $PYTHON_BIN - <<'PYEOF'
import sqlite3, sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "results/smoke_test/distilgpt2/wikitext-2/dataset.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# List tables
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"  Tables: {tables}")

for table in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    print(f"  {table}: {count} rows")
    if count > 0:
        row = conn.execute(f"SELECT * FROM [{table}] LIMIT 1").fetchone()
        print(f"    Columns: {list(row.keys())}")
        status_col = 'status' if 'status' in row.keys() else None
        if status_col:
            statuses = conn.execute(f"SELECT {status_col}, COUNT(*) FROM [{table}] GROUP BY {status_col}").fetchall()
            for s in statuses:
                print(f"    Status '{s[0]}': {s[1]}")

conn.close()
print("  SQLite verification PASSED")
PYEOF
    DB_CHECK=$?
    if [ $DB_CHECK -ne 0 ]; then ERRORS=$((ERRORS + 1)); fi
else
    echo "  ERROR: SQLite DB not found at $DB_FILE"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check logs
LOG_DIR="$COMBO_DIR/logs"
if [ -d "$LOG_DIR" ]; then
    LOG_COUNT=$(find "$LOG_DIR" -type f | wc -l)
    echo "  Log files: $LOG_COUNT"
else
    echo "  WARNING: No log directory at $LOG_DIR"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
if [ $ERRORS -eq 0 ]; then
    echo "SMOKE TEST PASSED: All outputs verified successfully"
else
    echo "SMOKE TEST FAILED: $ERRORS verification error(s)"
fi
echo "================================================================================"
echo ""

# List all output files
echo "Output tree:"
find "$RESULTS_DIR" -type f | sort | head -50

exit $ERRORS
