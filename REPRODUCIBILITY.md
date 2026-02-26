# Reproducibility Protocol

## Environment

- Python 3.10+
- Linux/macOS shell environment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 1_Detection_Categorization_XAI/requirements.txt
pip install -r 2_Diagnosis_Root_Cause/requirements.txt
```

## Execution Order

From repository root:

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

## Expected Output Locations

- Stage outputs: `results/stage_1_detection/`, `results/stage_2_categorization/`, `results/stage_2.1_categorization_xai/`, `results/stage_3_diagnosis/`
- Frozen legacy outputs: `default++_results/`
- RQ6 artifacts: `3_Comparison_with_defaultplusplus/results/`

## Verification

- Confirm stage artifacts are generated in expected folders.
- Validate key JSON/CSV/PDF outputs are produced and non-empty.
- Use `manifests/` to audit file integrity when needed.
