#!/bin/bash
# FrankenFormer-Encoder Environment Configuration
# Edit the variables below to match your cluster setup
#
# IMPORTANT: This file should ONLY contain environment variable exports.
# Do NOT add module load commands here - they belong in submit_*.sh and setup.sh

# ============================================================================
# Core Paths - CUSTOMIZE THESE
# ============================================================================

export PROJECT_ROOT="/project/def-mrdal22/sjahan/FrankenFormer-Encoder-Probes"

export RESULTS_DIR="$PROJECT_ROOT/probe-results"
export LOGS_DIR="$PROJECT_ROOT/CC-logs"
export CONFIG_DIR="$PROJECT_ROOT/config"

export VENV_PATH="/project/def-mrdal22/sjahan/venv-encoder"
export VENV_TARBALL="/project/def-mrdal22/sjahan/venv_encoder_packed.tar.gz"

if [ -n "${SLURM_TMPDIR:-}" ] && [ -x "${SLURM_TMPDIR}/venv/bin/python" ]; then
    export PYTHON_BIN="${SLURM_TMPDIR}/venv/bin/python"
else
    export PYTHON_BIN="$VENV_PATH/bin/python"
fi

# ============================================================================
# HuggingFace Cache
# ============================================================================

export HF_HOME="/project/def-mrdal22/sjahan/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_MODULES_CACHE="$HF_HOME/modules"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================================
# Cluster-Specific SLURM Settings - CUSTOMIZE THESE
# ============================================================================

export SLURM_ACCOUNT="def-mrdal22"
export SLURM_EMAIL="sigma.jahan@dal.ca"

# ============================================================================
# Validation Function
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
