# Experiment: Hierarchical Fault Diagnosis

## Architecture

```
Input features (410-540 dims after processing)
    |
    v
Shared Backbone (group-aware encoding)
    |-- Flat mode:  single MLP, output split into n_groups chunks
    |-- Graph mode: per-group MLPs + FPG message passing
    |
    +--> z (64-dim projected embedding)     --> Stage 1: Detection head (binary)
    |                                       --> Stage 2: Category head (11-12 classes)
    |
    +--> h_groups (n_groups x 32-dim)       --> Stage 3: Per-category root-cause heads
                                            --> Prototypical diagnosis + FPG explanation
```

**Stage 1 and 2** operate on the projected embedding `z`.
**Stage 3** operates on group-level embeddings `h_groups` for prototypical matching and per-component explanation.
The prototype matcher is the primary Stage 3 evaluation path; the per-category CE head is auxiliary during training and can be checked against the prototype head for consistency.

## Losses

```
L_total = L_detect + alpha * L_category + lambda * L_rootcause + beta * L_contrastive
```

| Loss | Scope | Purpose |
|------|-------|---------|
| L_detect | All samples | Binary CE (class-weighted) for clean vs faulty |
| L_category | Faulty only | CE over 11-12 fault families |
| L_rootcause | Per category | CE over 2-7 root causes within each family |
| L_contrastive | Per category | Intra-family contrastive: separates confusable root causes |

## Training Protocol

1. **Outer loop**: 5-fold GroupKFold (grouped by model+dataset+seed)
2. **Inner split**: 80/20 stratified train/val within each fold
3. **Oversampling**: clean class upsampled to 50% of faulty count (training only)
4. **Early stopping**: on a hierarchy-aware validation metric that includes Stage 3 prototype F1
5. **After training**: compute root-cause prototypes from training embeddings

## Evaluation Protocol

**Stage 1**: AUROC + macro-F1 on full test set.

**Stage 2**: Macro-F1 on faulty test samples only.

**Stage 3**: Use the *predicted* category from Stage 2. If category prediction is wrong, the sample is counted as an error for root-cause diagnosis. This is the primary Stage 3 metric and the only one that should appear in the main evaluation.

**Explainability**: For each diagnosed sample, decompose the prototype distance into per-FPG-group contributions. The explanation is aligned with the prototype-based Stage 3 decision path.

## Files

| File | What it does |
|------|-------------|
| `model.py` | `HierarchicalDiagnosisModel`: shared backbone + 3 stage heads + prototypical explainability |
| `losses.py` | Detection (weighted CE), category (CE), root-cause (CE), intra-family contrastive |
| `train.py` | `load_data` (with killed-column merge), `train_one_fold`, `evaluate_one_fold`, `run_experiment` |
| `evaluate.py` | Runs all 4 ablation variants, generates plots, saves JSON results |
| `plotting.py` | Publication-quality figures: training dynamics, confusion, ROC, explanation heatmaps |

## Outputs

```
results/hierarchical_graph_category_rootcause/
  {arch}_{variant}.json          # Per-variant metrics
  full_ablation_table.json       # Comparison table
  figures/{arch}/
    train_val_loss.pdf           # Overfitting check: train vs val loss
    confusion_category.pdf       # Category confusion matrix (full method)
    roc_detection.pdf            # Stage 1 ROC curve
    explain_*.pdf                # Per-category FPG explanation barplots
    explain_heatmap_*.pdf        # Per-root-cause diagnosis signatures (key figure)
```
