# DEFault++ Replication Package

This package is a copy-only split of the DEFault++ study artifacts from the combined thesis workspace.

## Included Components

- `1_Detection_Categorization_XAI/` (Stage 1 + Stage 2 + XAI code/data)
- `2_Diagnosis_Root_Cause/` (Stage 3 diagnosis + NDG)
- `3_Comparison_with_defaultplusplus/` (RQ6 baseline comparisons)
- `default++_results/` (frozen original result outputs and final plots)
- `results/` (canonicalized stage result layout for script compatibility)

## Applied Hardening in This Copy

- Replaced broken `2_Diagnosis_Root_Cause/data` symlinks with local file copies.
- Canonicalized stage outputs into:
  - `results/stage_1_detection/`
  - `results/stage_2_categorization/`
  - `results/stage_2.1_categorization_xai/`
  - `results/stage_3_diagnosis/`
- Patched stale path constants in Stage-3 and RQ6 scripts.
- Updated run scripts to avoid missing cross-arch PKL assumptions.

## Canonical Naming and Navigation

To improve readability and reduce command ambiguity, use canonical wrapper scripts in:
- `scripts/`

Canonical index and legacy-name mapping:
- `CANONICAL_SCRIPT_INDEX.md`

These wrappers do not replace legacy files; they only provide clearer names.

## Integrity

- Pre-edit file manifest for this package copy:
  - `manifests/pre_patch_sha256.txt`
- Post-edit file manifest:
  - `manifests/post_patch_sha256.txt`
- Thesis alignment map:
  - `THESIS_ALIGNMENT.md`
