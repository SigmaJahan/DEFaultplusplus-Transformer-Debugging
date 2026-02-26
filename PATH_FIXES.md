# Path and Runtime Fixes Applied in Split Copy

- `3_Comparison_with_defaultplusplus/run_rq6_baselines.py`
  - Updated data path from legacy `2_A_Detection_Categorization_XAI` to `1_Detection_Categorization_XAI`.
- `3_Comparison_with_defaultplusplus/run_all.sh`
  - Updated legacy directory references in comments.
- `2_Diagnosis_Root_Cause/ndg_diagnosis_builder/run_stage3_diagnosis_ndg_v1.py`
  - Updated legacy dataset roots to local package paths.
  - Updated default `--feature_core_map` path to package-local file.
- `2_Diagnosis_Root_Cause/run_ndg_stage3.sh`
  - Made `.venv` activation optional.
- `1_Detection_Categorization_XAI/run_all.sh` and `run_parallel.sh`
  - Removed cross-dataset calls requiring absent cross PKLs.
- `2_Diagnosis_Root_Cause/data/`
  - Replaced broken symlinks with local copied files.
