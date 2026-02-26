# Path and Runtime Fixes Applied in Split Copy

- `comparison_with_defaultplusplus/run_rq6_baselines.py`
  - Updated data path from legacy `2_A_Detection_Categorization_XAI` to `detection_categorization_xai`.
- `comparison_with_defaultplusplus/run_all.sh`
  - Updated legacy directory references in comments.
- `diagnosis_root_cause/ndg_diagnosis_builder/run_stage3_diagnosis_ndg_v1.py`
  - Updated legacy dataset roots to local package paths.
  - Updated default `--feature_core_map` path to package-local file.
- `diagnosis_root_cause/run_ndg_stage3.sh`
  - Made `.venv` activation optional.
- `detection_categorization_xai/run_all.sh` and `run_parallel.sh`
  - Removed cross-dataset calls requiring absent cross PKLs.
- `diagnosis_root_cause/data/`
  - Replaced broken symlinks with local copied files.
