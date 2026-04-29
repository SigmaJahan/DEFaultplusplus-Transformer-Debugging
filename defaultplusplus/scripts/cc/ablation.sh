#!/usr/bin/env bash
#SBATCH --job-name=defaultpp-ablation
#SBATCH --account=def-yourgroup
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Run the four ablation variants (basic, graph, sep, graph_sep) for
# both architectures. Plots and JSON summaries land under $RESULTS_ROOT.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/cc/env.sh"

cd "${CODE_ROOT}"
python -m hierarchical_graph_category_rootcause.evaluate \
    --arch both \
    --output "${RESULTS_ROOT}/hierarchical_graph_category_rootcause"

echo "[ablation] done; figures under ${RESULTS_ROOT}/.../figures/"
