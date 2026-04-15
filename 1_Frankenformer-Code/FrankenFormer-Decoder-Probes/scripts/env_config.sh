#!/bin/bash
# FrankenFormer-Decoder Environment Configuration
# Edit the variables below to match your cluster setup
#
# IMPORTANT: This file should ONLY contain environment variable exports.
# Do NOT add module load commands here - they belong in submit_*.sh and setup.sh
#
# NOTE: #SBATCH directives in submit_*.sh scripts are processed before
# this file is sourced, so they use relative paths. Ensure #SBATCH
# --output and --error paths match the LOGS_DIR structure below.

# ============================================================================
# Core Paths - CUSTOMIZE THESE
# ============================================================================

# Project directory on /project (faster metadata ops than /scratch)
export PROJECT_ROOT="/project/def-mrdal22/sjahan/FrankenFormer-Decoder-Probes"

# Results and logs persist on /project (no 60-day purge)
export RESULTS_DIR="$PROJECT_ROOT/probe-results"
export LOGS_DIR="$PROJECT_ROOT/CC-logs"
export CONFIG_DIR="$PROJECT_ROOT/config"

# Venv on /project (used as fallback and for tarball packing)
# In SLURM jobs, the tarball is extracted to $SLURM_TMPDIR for fast imports.
export VENV_PATH="/project/def-mrdal22/sjahan/venv"
export VENV_TARBALL="/project/def-mrdal22/sjahan/venv_packed.tar.gz"

# PYTHON_BIN: use $SLURM_TMPDIR venv if available (fast local SSD), else fallback
if [ -n "${SLURM_TMPDIR:-}" ] && [ -x "${SLURM_TMPDIR}/venv/bin/python" ]; then
    export PYTHON_BIN="${SLURM_TMPDIR}/venv/bin/python"
else
    export PYTHON_BIN="$VENV_PATH/bin/python"
fi

# ============================================================================
# HuggingFace Cache - on /project (persistent, no purge)
# ============================================================================

export HF_HOME="/project/def-mrdal22/sjahan/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_MODULES_CACHE="$HF_HOME/modules"

# Force offline mode to prevent jobs from hanging on compute nodes without internet
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================================
# Cluster-Specific SLURM Settings - CUSTOMIZE THESE
# ============================================================================

# CHANGE THIS: Your SLURM account name
export SLURM_ACCOUNT="def-mrdal22"

# CHANGE THIS: Your SLURM partition name (if different)
# export SLURM_PARTITION="gpubase_bygpu_b4"

# CHANGE THIS: Your email for job notifications
export SLURM_EMAIL="sigma.jahan@dal.ca"

# CHANGE THIS: Nodes to exclude from scheduling (comma-separated, or empty)
# export EXCLUDED_NODES="g30,g31,g32,g33,g34,g35,g36"

# ============================================================================
# Validation Function - Do not modify
# ============================================================================
validate_environment() {
    local errors=0

    if [ ! -d "$PROJECT_ROOT" ]; then
        echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT"
        errors=$((errors + 1))
    fi

    if [ ! -d "$CONFIG_DIR" ]; then
        echo "ERROR: CONFIG_DIR does not exist: $CONFIG_DIR"
        errors=$((errors + 1))
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        echo "ERROR: Python not found or not executable: $PYTHON_BIN"
        errors=$((errors + 1))
    fi

    if [ ! -d "$HF_HOME" ]; then
        echo "ERROR: HF_HOME does not exist: $HF_HOME"
        echo "Hint: Create the cache dir before submitting jobs."
        errors=$((errors + 1))
    fi

    mkdir -p "$LOGS_DIR/slurm" || {
        echo "ERROR: Failed to create logs directory: $LOGS_DIR/slurm"
        errors=$((errors + 1))
    }

    return $errors
}
