#!/bin/bash
# Pack the virtual environment into a tarball for fast $SLURM_TMPDIR extraction.
#
# Run once from the login node after pip install is done:
#   bash scripts/pack_venv.sh
#
# This creates /scratch/sjahan/venv_packed.tar.gz (~1-2 GB).
# SLURM jobs extract it to $SLURM_TMPDIR (local SSD) for instant imports.

set -euo pipefail

VENV_SRC="/project/def-mrdal22/sjahan/venv"
TARBALL="/project/def-mrdal22/sjahan/venv_packed.tar.gz"

if [ ! -d "$VENV_SRC" ]; then
    echo "ERROR: venv not found at $VENV_SRC"
    exit 1
fi

echo "Packing venv from $VENV_SRC ..."
echo "This may take a few minutes on Lustre."

tar -czf "$TARBALL" -C "$(dirname "$VENV_SRC")" "$(basename "$VENV_SRC")"

SIZE=$(du -h "$TARBALL" | cut -f1)
echo "Done: $TARBALL ($SIZE)"
echo ""
echo "SLURM jobs can now use:"
echo "  tar -xzf $TARBALL -C \$SLURM_TMPDIR"
echo "  export PYTHON_BIN=\$SLURM_TMPDIR/.venv/bin/python"
