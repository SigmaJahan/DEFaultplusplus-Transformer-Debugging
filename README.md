# DEFault++ Transformer Debugging Workspace

This README is workspace navigation only. The numbered top-level directories are intentional: the active DEFault++ implementation lives in `7_DEFaultpp-code`, while the mutation data, manuscript, baselines, benchmarks, and user-study materials live alongside it.

Research and reproduction usage live in [`7_DEFaultpp-code/README.md`](7_DEFaultpp-code/README.md). The runtime-v1 contract lives in [`docs/runtime_v1_contract.md`](docs/runtime_v1_contract.md). Runtime product architecture lives in [`7_DEFaultpp-code/defaultplusplus_runtime_roadmap.md`](7_DEFaultpp-code/defaultplusplus_runtime_roadmap.md).

## Workspace Layout

- `1_Frankenformer-Code/`: upstream FrankenFormer encoder/decoder probe code
- `2_Frakenformer-DEFaultpp-Manuscript/`: manuscript sources and figures
- `3_Mutation-Data-from-Frakenformer/`: tracked mutation datasets used by DEFault++
- `4_Baseline-comparison_with_defaultpp/`: baseline comparison scripts
- `5_Benchmarks-realworld_defaultpp/`: real-world benchmark cases and metadata
- `6_User-study-defaultpp/`: user-study assets
- `7_DEFaultpp-code/`: active DEFault++ package, experiments, and tests
- `results/`: generated outputs only; ignored by Git

## Document Roles

- `2_Frakenformer-DEFaultpp-Manuscript/Chapter_7_8.pdf`: scientific source of truth for the taxonomy, grouped diagnosis design, and benchmark construction.
- `docs/runtime_v1_contract.md`: normative runtime-v1 contract and document-precedence reference.
- `7_DEFaultpp-code/defaultplusplus_runtime_roadmap.md`: runtime/product architecture source of truth.
- `7_DEFaultpp-code/Plan.md`: subordinate implementation backlog and test bank.
- `7_DEFaultpp-code/README.md`: canonical research/reproduction README for the current artifact.
- `7_DEFaultpp-code/features.md`: runtime feature reference derived from the runtime-v1 contract.

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The environment is created at the repository root and installs the editable package from `7_DEFaultpp-code`.

## Common Commands

```bash
make data-check
make train
make ablation
make baselines
pytest 7_DEFaultpp-code/tests/test_phase0_gate.py
```

## Notes

The canonical data location is `3_Mutation-Data-from-Frakenformer/`. Do not copy those CSVs into `7_DEFaultpp-code/data`; the experiment code resolves the shared dataset directory directly.
