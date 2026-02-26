# Diagnosis JSON Builder (mutation-grounded) — NDG-aligned mapping

This script is a patched version of `run_stage3_diagnosis_old.py` that generates:

- `results/enc_diagnosis.json`
- `results/dec_diagnosis.json`

It differs from the old version in one critical way:

It computes `subsystem_impact` and propagation summaries using the **authoritative subsystem mapping**
from `feature_core_map.md` (instead of a keyword heuristic).

These diagnosis JSONs are the inputs consumed by **NDG Stage-3 V3** to add:
- `SIGNATURE` edges (FaultFamily → CoreFeature; direction + effect size)
- `IMPACTS` edges (FaultFamily → Subsystem; impact weights)

## Inputs

Required (same as your existing Stage-3 diagnosis script expects):
- Stage-1/Stage-2 prediction + report artifacts already produced by your pipeline (as configured inside the script)
- The run/report directories referenced in the script constants (DATA_DIR / RESULTS_DIR)

New required for NDG alignment:
- `feature_core_map.md`

## Output structure

Each `{arch}_diagnosis.json` includes:
- `differential_signatures[fault_family].top20_differential_features`
- `differential_signatures[fault_family].subsystem_impact`
- `differential_signatures[fault_family].propagation_profile`
- other summary fields already present in your old script (cascade metrics, etc.)

## Usage

From the directory containing your JSON artifacts:

```bash
python run_stage3_diagnosis_ndg_v1.py --arch both --feature_core_map feature_core_map.md
```

You can also run one side:

```bash
python run_stage3_diagnosis_ndg_v1.py --arch enc --feature_core_map feature_core_map.md
python run_stage3_diagnosis_ndg_v1.py --arch dec --feature_core_map feature_core_map.md
```

The script writes results to the existing `RESULTS_DIR` used inside the script (typically `results/`).

## Integration with NDG Stage-3 V3

After generating the diagnosis JSONs, run the NDG builder (V3) with:

```bash
python -m ndg_stage3.cli \
  --enc_detection enc_detection.json \
  --enc_categorization enc_categorization.json \
  --xai_enc xai_enc_categorization.json \
  --dec_detection dec_detection.json \
  --dec_categorization dec_categorization.json \
  --xai_dec xai_dec_categorization.json \
  --feature_core_map feature_core_map.md \
  --enc_diagnosis results/enc_diagnosis.json \
  --dec_diagnosis results/dec_diagnosis.json \
  --out_dir ndg_out \
  --plots
```

## Notes on mapping

The patched script maps a feature *variant* (e.g., `abs_ffn_delta_l1_mean_final`) to a core feature by:
- stripping training-phase suffixes (`_early_mean`, `_mid_slope`, `_final`, etc.)
- handling per-layer tokens (`_l0`..`_l11`) when present
- falling back to a conservative token-based heuristic if no match is found in `feature_core_map.md`

This ensures subsystem impact profiles remain consistent across:
- your dataset schema,
- your thesis definitions,
- and the NDG abstraction layer.
