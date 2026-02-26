# FrankenFormer Detection & Categorization Pipeline

Self-contained pipeline for fault detection and categorization of transformer mutations, with XAI explanations.

## Directory Structure

```
1_Detection_Categorization_XAI/
  data/                          # Preprocessed pkl + source CSV files
    enc_v1_detection.pkl         # Encoder detection (9560 x 546)
    enc_v1_categorization.pkl    # Encoder categorization (9455 x 546, 11 classes)
    dec_v1_detection.pkl         # Decoder detection (9310 x 218)
    dec_v1_categorization.pkl    # Decoder categorization (9240 x 218, 12 classes)
    encoder_v1_killed_binary.csv # Source CSV (for reprocessing if needed)
    decoder_v1_killed_binary.csv # Source CSV (for reprocessing if needed)
  preprocess.py                  # Preprocessing library (also --batch to regenerate all pkl)
  run_classifiers.py             # 4-model classifier pipeline
  run_xai.py                     # XAI pipeline (SHAP + DiCE + rules)
  run_all.sh                     # Sequential execution (encoder + decoder)
  run_parallel.sh                # Parallel execution with logging
  requirements.txt               # Python dependencies
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the entire pipeline (classifiers + XAI for encoder/decoder configurations):

```bash
bash run_all.sh
```

This runs:
1. **Classification** (4 models x 4 datasets) -- outputs to `results/*.json`
2. **XAI** (SHAP + counterfactuals + rules for 4 datasets) -- outputs to `results/xai_*.json`

## Running Individually

### Classifiers only

```bash
python run_classifiers.py --data data/enc_v1_detection.pkl --out results/enc_detection.json
python run_classifiers.py --data data/dec_v1_categorization.pkl --out results/dec_categorization.json
```

### XAI only (after classifiers)

```bash
# Pass --results to use best XGBoost params from classifier run
python run_xai.py --data data/enc_v1_categorization.pkl \
                  --results results/enc_categorization.json \
                  --out results/xai_enc_categorization.json
```

### Regenerate pkl files from CSVs (if needed)

```bash
python preprocess.py --batch
```

## Pipeline Details

### Classifiers (run_classifiers.py)

4 models as specified in doing.MD:

| Model | Key Settings |
|-------|-------------|
| ElasticNet LR | saga solver, balanced weights, max_iter=5000 |
| Calibrated RBF SVM | sigmoid calibration, balanced weights |
| XGBoost | hist, subsample=0.8, colsample=0.8, early stopping |
| EasyEnsemble | AdaBoost(DT stump, 200, lr=0.5), n_estimators in {10,20} |

- **CV**: 5-fold GroupKFold by (model_name, dataset_name, seed) -- prevents leakage
- **Normalization**: Per-fold group z-score (fit on training indices only)
- **NaN handling** (cross-arch union mode): XGBoost gets NaN preserved; LR/SVM/EasyEnsemble get NaN filled with 0
- **Grid search**: On fold-0 training data with 3-fold inner CV
- **Detection metrics**: AUROC, AUPRC, Recall@Precision=0.95, confusion matrix
- **Categorization metrics**: Macro-F1, balanced accuracy, Top-3/5 accuracy, per-class F1/precision/recall

### XAI (run_xai.py)

Three explanation types, all on held-out test folds:

1. **SHAP TreeExplainer**: Per-column SHAP values aggregated to core features (time-window and per-layer columns merged). Reports top-30 core features + stability (Jaccard) + layer-depth pattern.
2. **DiCE Counterfactuals**: Minimal feature changes to flip prediction. Targets misclassified + low-margin instances. Structural features (arch, layer_idx, severity) are immutable.
3. **Surrogate Rules**: KBinsDiscretizer (5 quantile bins, fit on train fold) + DecisionTree (depth=4). Reports fidelity to XGBoost, per-leaf precision/coverage, stable rules across folds.

### Data Variants

| Dataset | Rows | Features | NaN | Task |
|---------|------|----------|-----|------|
| enc_v1_detection | 9,560 | 546 (543 abs + 3 struct) | 0% | binary |
| enc_v1_categorization | 9,455 | 546 | 0% | 11-class |
| dec_v1_detection | 9,310 | 218 (215 abs + 3 struct) | 0% | binary |
| dec_v1_categorization | 9,240 | 218 | 0% | 12-class |

**Labels**: Detection uses `correct` (baseline) vs `buggy` (all injected faults, regardless of kill status). Categorization uses the 11-12 fault families (e.g., ffn, qkv, layernorm, embedding, etc.).

**Structural features**: `arch_enc` (binary), `layer_idx_num` (integer), `severity_scalar` (max abs value from severity_params JSON).

## Runtime Notes

- Typical runtime: ~30-60 min per classifier config, ~15-30 min per XAI config
- Total: ~3-6 hours for the full `run_all.sh`
- Memory: peak ~6 GB on encoder runs
