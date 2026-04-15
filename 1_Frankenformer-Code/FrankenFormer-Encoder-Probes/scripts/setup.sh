#!/bin/bash
# FrankenFormer-Encoder fully automated environment setup
# Usage: bash scripts/setup.sh
#
# Steps:
# 1. Loads Compute Canada modules
# 2. Creates Python virtual environment & installs PyTorch and requirements
# 3. Caches all Hugging Face models and datasets offline
# 4. Performs a pipeline dry-run check
# 5. Performs a CPU-based structural probe smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$SCRIPT_DIR/env_config.sh" ]; then
    echo "ERROR: env_config.sh not found in project directory"
    exit 1
fi

source "$SCRIPT_DIR/env_config.sh"

echo "========================================="
echo "FrankenFormer-Encoder Automated Setup"
echo "Project:  $PROJECT_ROOT"
echo "Venv:     $VENV_PATH"
echo "Results:  $RESULTS_DIR"
echo "Logs:     $LOGS_DIR"
echo "HF Cache: $HF_HOME"
echo "========================================="

# Load Compute Canada modules
echo "Loading CC modules..."
module --force purge
module load StdEnv/2023 gcc/12.3 arrow/14.0.1 python/3.10 cuda/12.2 rust/1.70.0

# Create or update venv
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..."
    python -m venv "$VENV_PATH"
else
    echo "Virtual environment already exists at $VENV_PATH"
fi

echo "Verifying venv python..."
"$PYTHON_BIN" -c 'import sys; print("sys.executable:", sys.executable)'

echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

echo "Installing PyTorch..."
"$PYTHON_BIN" -m pip install \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "Installing requirements..."
grep -v 'datasets' "$PROJECT_ROOT/requirements.txt" > "$PROJECT_ROOT/reqs_temp.txt" || true
"$PYTHON_BIN" -m pip install --no-index -r "$PROJECT_ROOT/reqs_temp.txt"
"$PYTHON_BIN" -m pip install --no-index --no-deps datasets==2.18.0
rm -f "$PROJECT_ROOT/reqs_temp.txt"

echo "========================================="
echo "Packing venv tarball for SLURM_TMPDIR ..."
echo "========================================="
bash "$SCRIPT_DIR/pack_venv.sh"

echo "========================================="
echo "Caching models and datasets offline..."
echo "========================================="
"$PYTHON_BIN" "$SCRIPT_DIR/cache-models-datasets.py" --matrix-config "$PROJECT_ROOT/config/matrix_encoder.yaml"

echo "========================================="
echo "Running Dry-Run Pipeline Check..."
echo "========================================="
"$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" \
    --matrix-config "$PROJECT_ROOT/config/matrix_encoder.yaml" \
    --fault-config "$PROJECT_ROOT/config/pipeline_configs_probes.json" \
    --results-dir "$RESULTS_DIR/dry_run" \
    --list-configs
echo "Dry run successful!"

echo "========================================="
echo "Running CPU Smoke Test (Structural Probes)..."
echo "========================================="
"$PYTHON_BIN" "$SCRIPT_DIR/smoke_test_probes.py"
echo "CPU Smoke test successful!"

echo "========================================="
echo "Setup is fully complete! All checks passed."
echo "You can now run your main array job:"
echo "  sbatch scripts/submit_encoder_probes.sh"
echo "========================================="
