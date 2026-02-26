# DEFault++ Fault Debugging Study Artifact

![Artifact](https://img.shields.io/badge/artifact-study-blue)
![Access](https://img.shields.io/badge/access-private-critical)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)

Codebase and curated datasets for the DEFault++ fault debugging study.

## Scope

This repository contains the end-to-end study pipeline across detection, categorization/XAI, diagnosis/root-cause, and RQ6 comparison.

Included:
- `detection_categorization_xai/`
- `diagnosis_root_cause/`
- `comparison_with_defaultplusplus/`
- `defaultplusplus_results/` (frozen historical outputs)
- `results/` (canonicalized output layout)
- `scripts/` (top-level execution entry points)

## Quick Start

Run from repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r detection_categorization_xai/requirements.txt
pip install -r diagnosis_root_cause/requirements.txt
```

Core stage execution:

```bash
bash scripts/run_stage1_preprocess.sh
bash scripts/run_stage1_classifiers.sh
bash scripts/run_stage2_xai.sh
bash scripts/run_stage2_rq3_ablation.sh
bash scripts/run_stage3_diagnosis_builder.sh
bash scripts/run_stage3_signature_matching_rq5.sh
bash scripts/run_stage3_ndg_cli.sh
bash scripts/run_rq6_baseline_comparison.sh
```

## Repository Layout

- `detection_categorization_xai/`: Stage-1 detection and Stage-2 categorization/XAI
- `diagnosis_root_cause/`: Stage-3 diagnosis, signature matching, NDG
- `comparison_with_defaultplusplus/`: RQ6 baseline comparison workflows
- `defaultplusplus_results/`: preserved original outputs
- `results/`: canonicalized stage outputs for stable script paths
- `manifests/`: file-level integrity manifests

## Data Policy

This repository retains code and final/processed data required for replication while avoiding unnecessary transient artifacts.

## Integrity

- `manifests/pre_patch_sha256.txt`
- `manifests/post_patch_sha256.txt`
- `manifests/changes_since_copy.txt`
