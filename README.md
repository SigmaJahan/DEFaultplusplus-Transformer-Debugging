# DEFault++ Fault Debugging Study Artifact

Codebase and curated datasets for the DEFault++ fault debugging study.

## Artifact Structure

- `src/`: executable study code
- `results/`: reproducible outputs
- `scripts/`: root-level execution wrappers

## Source Layout

- `src/detection_categorization_xai/`: detection, categorization, and XAI workflows
- `src/diagnosis_root_cause/`: diagnosis and NDG workflows
- `src/comparison_with_defaultplusplus/`: baseline comparisons

## Results Layout

- `results/detection/`
- `results/categorization/`
- `results/explanations/`
- `results/diagnosis/`
- `results/feature_ablation/`
- `results/baseline_comparison/`

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/detection_categorization_xai/requirements.txt
pip install -r src/diagnosis_root_cause/requirements.txt
```

## Execution

```bash
bash scripts/run_preprocess.sh
bash scripts/run_classifiers.sh
bash scripts/run_explanations.sh
bash scripts/run_feature_ablation.sh
bash scripts/run_diagnosis_builder.sh
bash scripts/run_signature_matching.sh
bash scripts/run_diagnosis_graph.sh
bash scripts/run_baseline_comparison.sh
```

## Data Policy

Includes code, curated datasets, and final study outputs required for reproducibility. Excludes unnecessary transient artifacts.
