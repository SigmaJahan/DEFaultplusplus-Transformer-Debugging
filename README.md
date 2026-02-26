# DEFault++ Fault Debugging Study Artifact

Codebase and curated datasets for the DEFault++ fault debugging study.

## Artifact Structure

- `src/`: executable code for study stages
- `results/`: stage outputs and legacy frozen outputs
- `scripts/`: root-level execution wrappers

## Source Layout

- `src/detection_categorization_xai/`: Stage 1 detection and Stage 2 categorization/XAI
- `src/diagnosis_root_cause/`: Stage 3 diagnosis and NDG workflows
- `src/comparison_with_defaultplusplus/`: RQ6 baseline comparisons

## Results Layout

- `results/stage_1_detection/`
- `results/stage_2_categorization/`
- `results/stage_2_1_categorization_xai/`
- `results/stage_3_diagnosis/`
- `results/rq3_ablation/`
- `results/rq6/`
- `results/legacy/defaultplusplus_results/` (frozen historical outputs)

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/detection_categorization_xai/requirements.txt
pip install -r src/diagnosis_root_cause/requirements.txt
```

## Execution

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

## Data Policy

Includes code, curated datasets, and final study outputs required for reproducibility. Excludes unnecessary transient artifacts.
