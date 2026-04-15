#!/bin/bash
# Decoder structural-probe re-run: 336 stratified samples
# 6 (model, dataset) pairs x 56 configs each = 336 experiments
#
# Chunked into 18 array tasks (3 chunks per pair, 20 configs each).
# Timing from prior runs:
#   distilgpt2: ~36 min/config  -> 20 configs = 12h
#   gpt2/neo:   ~57.5 min/config -> 20 configs = 19.2h
# 24h walltime fits all chunks with margin.
#
# Usage (from project root):
#   sbatch scripts/submit_probes_336.sh
#
#SBATCH --job-name=dec-probes
#SBATCH --account=def-mrdal22
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-user=sigma.jahan@dal.ca
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --output=CC-logs/probes/%x-%A_%a.out
#SBATCH --error=CC-logs/probes/%x-%A_%a.err

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# 6 pairs: (model, dataset)
MODELS=("distilgpt2" "distilgpt2" "gpt2" "gpt2" "gpt-neo-125m" "gpt-neo-125m")
DATASETS=("wikitext-2" "lambada" "wikitext-2" "lambada" "wikitext-2" "lambada")

CONFIGS_PER_PAIR=56
CONFIGS_PER_CHUNK=20
CHUNKS_PER_PAIR=3    # ceil(56/20) = 3

# Map array task -> (pair, chunk)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PAIR_IDX=$((TASK_ID / CHUNKS_PER_PAIR))
CHUNK_IDX=$((TASK_ID % CHUNKS_PER_PAIR))

MODEL="${MODELS[$PAIR_IDX]}"
DATASET="${DATASETS[$PAIR_IDX]}"

START_INDEX=$((CHUNK_IDX * CONFIGS_PER_CHUNK))
REMAINING=$((CONFIGS_PER_PAIR - START_INDEX))
if [ "$REMAINING" -le 0 ]; then
    echo "No configs for chunk $CHUNK_IDX (start=$START_INDEX >= total=$CONFIGS_PER_PAIR). Exiting."
    exit 0
fi
MAX_CONFIGS=$((REMAINING < CONFIGS_PER_CHUNK ? REMAINING : CONFIGS_PER_CHUNK))

MIN_GPU_GB="${MIN_GPU_GB:-35}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
WAIT_MAX_MINUTES="${WAIT_MAX_MINUTES:-20}"

# ============================================================================
# Environment Setup
# ============================================================================
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/project/def-mrdal22/sjahan/FrankenFormer-Decoder-Probes}"

# ============================================================================
# Load Modules (before sourcing env_config so python is available)
# ============================================================================
module load StdEnv/2023 gcc/12.3 arrow/14.0.1 python/3.10 cuda/12.2 rust/1.70.0

# ============================================================================
# Extract venv to SLURM_TMPDIR (local SSD -> fast imports)
# ============================================================================
VENV_TARBALL="/project/def-mrdal22/sjahan/venv_packed.tar.gz"
if [ -f "$VENV_TARBALL" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "Extracting venv tarball to $SLURM_TMPDIR ..."
    tar -xzf "$VENV_TARBALL" -C "$SLURM_TMPDIR"
    PYTHON_BIN="$SLURM_TMPDIR/venv/bin/python"
    echo "Using fast local venv: $PYTHON_BIN"
else
    PYTHON_BIN="/project/def-mrdal22/sjahan/venv/bin/python"
    echo "WARNING: Tarball not found, using /project venv (slower): $PYTHON_BIN"
fi

source "$PROJECT_ROOT/scripts/env_config.sh"

validate_environment || exit 1

MATRIX_CONFIG="$PROJECT_ROOT/config/matrix_336.yaml"
PIPELINE_CONFIG="$PROJECT_ROOT/config/pipeline_configs_probes.json"
RESULTS_DIR="$PROJECT_ROOT/probe-results"

mkdir -p "$PROJECT_ROOT/CC-logs/probes"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# CUDA Configuration
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,roundup_power2_divisions:16"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "================================================================================"
echo "Decoder Structural Probe Re-run"
echo "================================================================================"
echo "Array Task: $TASK_ID  (pair=$PAIR_IDX, chunk=$CHUNK_IDX)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Config range: start=$START_INDEX, max=$MAX_CONFIGS (of $CONFIGS_PER_PAIR total)"
echo "================================================================================"

# ============================================================================
# Verify GPU
# ============================================================================
cd "$PROJECT_ROOT" || exit 1

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

max_wait_sec=$((WAIT_MAX_MINUTES * 60))
waited_sec=0
while true; do
    GPU_STATUS=$($PYTHON_BIN - <<PY
import torch
min_gb = float("$MIN_GPU_GB")
if not torch.cuda.is_available():
    print("NO_CUDA 0 0")
else:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    if total_gb + 1e-6 < min_gb:
        print(f"TOTAL_TOO_SMALL {free_gb:.2f} {total_gb:.2f}")
    elif free_gb + 1e-6 < min_gb:
        print(f"BUSY {free_gb:.2f} {total_gb:.2f}")
    else:
        print(f"OK {free_gb:.2f} {total_gb:.2f}")
PY
    )
    read -r GPU_STATE GPU_FREE GPU_TOTAL <<<"$GPU_STATUS"
    echo "  GPU Memory: free ${GPU_FREE} GB / total ${GPU_TOTAL} GB"

    case "$GPU_STATE" in
        OK) break ;;
        TOTAL_TOO_SMALL)
            echo "ERROR: GPU total memory ${GPU_TOTAL}GB below required ${MIN_GPU_GB}GB"
            exit 2 ;;
        NO_CUDA)
            echo "ERROR: CUDA not available"
            exit 2 ;;
        BUSY)
            if [ "$max_wait_sec" -gt 0 ] && [ "$waited_sec" -ge "$max_wait_sec" ]; then
                echo "ERROR: GPU busy after ${WAIT_MAX_MINUTES} minutes. Requeueing."
                scontrol requeue "$SLURM_JOB_ID"
                exit 0
            fi
            echo "GPU busy. Waiting ${WAIT_INTERVAL_SEC}s..."
            sleep "$WAIT_INTERVAL_SEC"
            waited_sec=$((waited_sec + WAIT_INTERVAL_SEC))
            ;;
        *) echo "ERROR: $GPU_STATUS"; exit 2 ;;
    esac
done

# ============================================================================
# Run Pipeline (chunked)
# ============================================================================
echo ""
echo "Starting pipeline for $MODEL / $DATASET (chunk $CHUNK_IDX: configs $START_INDEX..$((START_INDEX + MAX_CONFIGS - 1))) ..."
echo "================================================================================"

$PYTHON_BIN scripts/run_pipeline.py \
    --matrix-config "$MATRIX_CONFIG" \
    --config-file "$PIPELINE_CONFIG" \
    --results-dir "$RESULTS_DIR/${MODEL}_${DATASET}" \
    --model-keys "$MODEL" \
    --task-keys "$DATASET" \
    --append-seed-to-config-id \
    --start-index "$START_INDEX" \
    --max-configs "$MAX_CONFIGS" \
    --min-gpu-mem-gb "$MIN_GPU_GB"

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "Task $TASK_ID (pair=$PAIR_IDX chunk=$CHUNK_IDX) Complete: $MODEL / $DATASET"
echo "================================================================================"
echo "Exit code: $EXIT_CODE"
echo "Configs: $START_INDEX .. $((START_INDEX + MAX_CONFIGS - 1))"

if [ -d "$RESULTS_DIR/${MODEL}_${DATASET}" ]; then
    H5_COUNT=$(find "$RESULTS_DIR/${MODEL}_${DATASET}" -name "*.h5" 2>/dev/null | wc -l)
    echo "H5 files created: $H5_COUNT"
fi

exit $EXIT_CODE
