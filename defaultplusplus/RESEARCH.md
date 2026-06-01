# DEFault++ research runbook

This document is the **researcher's notebook** for DEFault++. It
covers the end-to-end pipeline, every command, the data flow, where
each piece of code lives, how to extend the operator catalog or the
diagnostic model, the Compute Canada workflow, and debugging tips.

It is *not* user-facing. The user-facing PyPI documentation lives in
[`README.md`](README.md) (package) and [`../README.md`](../README.md)
(repository root).

The scientific reference is the DEFault++ manuscript (in preparation).
The method is walked through in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md), and the frozen
runtime output schema is [`../docs/SPEC.md`](../docs/SPEC.md).

---

## 1. Big picture

DEFault++ has four moving parts. All four are in this repository.

```
       ┌─────────────────────┐                   ┌──────────────────────┐
       │  DEForm injector    │                   │  Diagnostic model    │
       │  (defaultplusplus.  │                   │  (hierarchical_      │
       │   deform)           │                   │   graph_category_    │
       │                     │                   │   rootcause)         │
       └──────────┬──────────┘                   └──────────▲───────────┘
                  │ injects fault                           │ predicts
                  ▼                                         │
       ┌─────────────────────┐  paired clean/faulty   ┌─────┴────────────┐
       │  Fine-tuning runs   │  fine-tuning + matched │  Feature vector  │
       │  (HF Trainer or     │     seeds, sign-flip   │  (data/*.csv)    │
       │   manual loop)      │  ─────────────────────►│                  │
       └──────────┬──────────┘                        └──────────────────┘
                  │ MetricCollector
                  ▼
       ┌─────────────────────┐
       │  FeatureExtractor   │  ← public API surface (api.py / hf_callback.py)
       │  (defaultplusplus.  │
       │   api)              │
       └─────────────────────┘
```

Concretely:

| Stage                         | Code path                                                                | Output                       |
|-------------------------------|--------------------------------------------------------------------------|------------------------------|
| Mutation injection            | `defaultplusplus.deform.{operators,injection,validation}`                | `Mutant` records             |
| Paired training + extraction  | `defaultplusplus.benchmark.runner` + `defaultplusplus.api.FeatureExtractor` | per-config feature vectors |
| Benchmark assembly            | `defaultplusplus.benchmark.dataset_writer`                               | `data/*.csv`                 |
| Diagnostic model training     | `hierarchical_graph_category_rootcause.train`                            | `results/.../*.json`         |
| Ablation + plots              | `hierarchical_graph_category_rootcause.evaluate`                         | comparison table + figures   |
| Post-hoc importance           | `hierarchical_graph_category_rootcause.posthoc_analysis`                 | importance JSON + plots      |
| Baseline comparison           | external (paper-artifact bundle, not in this repo)                       | per-baseline JSONs           |

---

## 2. Repository tour

```
DEFaultplusplus-Transformer-Debugging/
  defaultplusplus/                          installable package + research drivers
    src/defaultplusplus/                    PyPI-shipped code only
      __init__.py                           public API (FeatureExtractor, ...)
      _version.py                           single source of truth for __version__
      api.py                                FeatureExtractor (manual loop)
      hf_callback.py                        DEFaultPlusCallback (HF Trainer)
      config.py                             ExtractionConfig
      extraction/
        inspector.py                        auto-discovers HF model structure
        collector.py                        orchestrates per-step metric modules
        aggregator.py                       Welford running statistics
        sublayer_capture.py                 forward hooks for FFN/LN/Q/K/V taps
        feature_construction.py             layer/step/epoch/phase aggregator
        metrics/{attention,gradient,...}.py per-component metric modules
      deform/
        operators.py                        52 mutation operators (the catalog)
        operator_impls/                     per-operator injector implementations
        injection.py                        StaticFault, DynamicFault context managers
        validation.py                       sign-flip permutation test, verifier
        fault_config.py                     FaultConfiguration, Mutant types
      benchmark/
        cli.py                              defaultpp-benchmark entry point
        config_grid.py                      enumerate (model, task, op, layer, sev)
        runner.py                           paired runs + crash isolation
        task_metrics.py                     per-task kill-test metric registry
        dataset_writer.py                   shard CSV writer
      diagnosis/                            Predictor + load_pretrained()
      processing/                           FeatureProcessor + RuntimeNormalizer
      pretrained/weights/                   shipped encoder/decoder checkpoints
      data/                                 benchmark download (defaultpp-bench-download)
      viz/                                  matplotlib plots + HTML report
      ui/                                   reserved for future CLI helpers

    hierarchical_graph_category_rootcause/  research-side training (NOT in wheel)
      train.py                              nested grouped CV training driver
      evaluate.py                           4-variant ablation driver
      posthoc_analysis.py                   permutation feature/group importance
      model.py                              HierarchicalDiagnosisModel
      losses.py                             L_DEFault++ formula
      plotting.py                           matplotlib figures

    src/data/, src/models/                  research-side helpers (NOT in wheel)
      feature_groups.py                     12/13 feature group taxonomy
      feature_processor.py                  six-step preprocessing pipeline
      fundamental_fpg.py                    FPG construction + group adjacency
      loader.py                             CSV / pickle loaders
      group_encoder.py                      GroupEncoder + GraphAggregator

    configs/base.yaml                       hyperparameter config
    examples/extract_*.py                   runnable demos
    scripts/
      setup.sh                              local venv setup
      dry_run_local.sh                      end-to-end smoke test
      build_pypi.sh                         clean → build → twine check → upload
      cc/                                   Compute Canada SLURM scripts
        env.sh setup_env.sh bench_array.sh merge_shards.sh train.sh ablation.sh
    tests/                                  pytest suite
      conftest.py                           sys.path shim for src/ + src/defaultplusplus
      test_phase0_gate.py                   structural gate
      test_phase1_gate.py                   feature-group gate
      test_dry_run.py                       fast smoke (~1s)
      test_feature_extractor.py             real DistilBERT + GPT-2 + HF Trainer
    pyproject.toml                          PEP 621 metadata + build config
    LICENSE                                 Apache-2.0
    CHANGELOG.md                            version history
    Makefile                                research-side commands

    README.md                               package-side user reference
    RESEARCH.md                             this file

  docs/
    ARCHITECTURE.md                         figure-by-figure method walk-through
    SPEC.md                                 output schema + architecture + scope
    figures/                                diagrams used across the docs

  data/                                     DEFault-bench CSVs (fetched, not committed)
    encoder_dataset.csv                     encoder feature traces + mutation killed labels
    decoder_dataset.csv                     decoder feature traces + mutation killed labels
  results/                                  generated outputs (gitignored)
```

The benchmark CSVs under `data/` are downloaded from Zenodo with
`defaultpp-bench-download` (or `download_bench(version="v1")`) rather
than committed. The real-world GitHub-issue evaluation lives in
`../realworld_evaluation/` (11 reproduced faults, one case file each).
The baseline scripts and the developer-study assets live in the
paper-artifact bundle outside this code repository.

---

## 3. Setup

### Local

```bash
cd defaultplusplus
bash scripts/setup.sh                          # creates ../.venv, pip install -e .[dev]
source ../.venv/bin/activate
make data-check                                # confirms data/*.csv are present
```

If you need the HF Trainer test path or the example notebooks:

```bash
pip install -e ".[dev,hf]"
```

### Compute Canada

```bash
# One-time on a login node:
bash defaultplusplus/scripts/cc/setup_env.sh
```

This creates `$SCRATCH/venvs/defaultplusplus`, pre-fetches all
HuggingFace tokenizers, and points caches to scratch. Every SLURM job
script then sources `defaultplusplus/scripts/cc/env.sh` to load the
module stack and activate the venv. See
[`scripts/cc/README.md`](scripts/cc/README.md) for the full pipeline.

---

## 4. Commands

All commands assume you are inside `defaultplusplus/`.

### Daily research workflow

```bash
make data-check                  # verify ../data/*.csv exist
make train                       # nested grouped CV, both archs (~30 min CPU smoke)
make ablation                    # 4 variants × 2 archs × 5 folds
make baselines                   # DEFault, DeepFD, AutoTrainer, DeepDiagnosis
make all                         # data-check + train + ablation + baselines
make clean                       # rm -rf ../results/, drops __pycache__/
```

### Module-level invocations (more flexible than `make`)

```bash
python -m hierarchical_graph_category_rootcause.train \
       --arch encoder --epochs 1                           # 1-epoch smoke
python -m hierarchical_graph_category_rootcause.train \
       --arch both --no-graph                              # ablate FPG
python -m hierarchical_graph_category_rootcause.train \
       --arch both --no-sep                                # ablate separation loss
python -m hierarchical_graph_category_rootcause.train \
       --arch both --no-graph --no-sep                     # basic variant

python -m hierarchical_graph_category_rootcause.evaluate \
       --arch both --no-plots                              # all 4 variants, no figures
python -m hierarchical_graph_category_rootcause.evaluate \
       --arch encoder --output ../results/runs/2026-04-29  # custom output dir

python -m hierarchical_graph_category_rootcause.posthoc_analysis \
       --arch both                                         # permutation importance

# Baseline comparison (DEFault, DeepFD, AutoTrainer, DeepDiagnosis) runs
# from the paper-artifact bundle, which is not part of this code repo.
```

### Tests

```bash
pytest tests/                              # full suite
pytest tests/test_dry_run.py -v            # fast smoke tests (~1s)
pytest tests/test_feature_extractor.py -v  # real-model API tests (~15s)
bash scripts/dry_run_local.sh --quick      # imports + smoke + FPG sanity
bash scripts/dry_run_local.sh --train      # also runs 1-epoch training
```

### Build / publish (when releasing a new version)

```bash
# 1. Bump src/defaultplusplus/_version.py and add a CHANGELOG entry.
# 2. Build:
bash scripts/build_pypi.sh                 # clean → build → twine check
# 3. Smoke test on TestPyPI:
bash scripts/build_pypi.sh --testpypi
# 4. Publish:
bash scripts/build_pypi.sh --pypi
```

### Compute Canada

```bash
sbatch scripts/cc/bench_array.sh           # build DEFault-bench (GPU array)
sbatch scripts/cc/merge_shards.sh          # concat per-task shards
sbatch scripts/cc/train.sh                 # train diagnostic model
sbatch scripts/cc/ablation.sh              # 4-variant ablation
```

---

## 5. The diagnostic model

### Architecture

A shared encoder feeds three classification levels:

```
input_features
   │
   ▼
GroupEncoder           per-group MLPs (one per of 12/13 feature groups)
   │
   ▼
GraphAggregator        H = ReLU(A_hat · H · W_msg)  ×3 rounds
   │
   ▼
projection             64-dim shared embedding z, plus group-stacked H
   │
   ├─► detection head            binary logits  (faulty vs clean)
   ├─► category head             C-class logits (11 enc / 12 dec)
   └─► per-category root-cause heads  +  prototype matcher in H-space
```

Code: [`hierarchical_graph_category_rootcause/model.py`](hierarchical_graph_category_rootcause/model.py).

### Loss

```
L = L_detect + α·L_cat + λ_rc·L_rc + L_sep
L_sep = β·L_ctr + γ·L_pm
```

| Term | Code | Default |
|---|---|---|
| `L_detect` | `detection_loss` (BCE on detection head) | weight = 1.0 |
| `L_cat` | `category_loss` (CE on category head, faulty only) | α = 1.0 |
| `L_rc` | `rootcause_loss` (per-category CE) | λ_rc = 1.0 |
| `L_ctr` | `contrastive_separation_loss` (SupCon over `vec(H)`) | β = 0.5 |
| `L_pm` | `prototype_matching_loss` (CE over -d/τ) | γ = 0.3 |

Code: [`hierarchical_graph_category_rootcause/losses.py`](hierarchical_graph_category_rootcause/losses.py).
Hyperparameters: [`configs/base.yaml`](configs/base.yaml).

### Feature groups (Table 7.7)

12 encoder + 13 decoder groups defined in
[`src/data/feature_groups.py`](src/data/feature_groups.py):

```
structural (FPG nodes):
  attention, score, ffn_output, layernorm, residual_stream,
  qkv_alignment, embedding, positional, output, cache (decoder only)
non-structural (self-loop only):
  representation_drift, training_dynamics, validation_perf
```

### FPG message passing (Section 7.4.4.2)

Component-level FPG → group-level adjacency Â (row-normalized,
self-loops). Three rounds of `H = ReLU(Â · H · W_msg)`.
Code: [`src/models/group_encoder.py`](src/models/group_encoder.py),
[`src/data/fundamental_fpg.py`](src/data/fundamental_fpg.py).

### Explanation (Equations 7.29–7.30)

Per-group importance is computed from the predicted vs. nearest-
alternative prototype margin:

```
Δ_g = d_g(π_alt) − d_g(π_pred)
w_g = max(Δ_g, 0) / Σ max(Δ_g', 0)
```

Code: `HierarchicalDiagnosisModel.explain_diagnosis` in `model.py`.

---

## 6. The DEForm engine

### Operator catalog (Tables 7.1 + 7.2)

52 operators across 12 components, with three-letter IDs. They cover 45
root causes (40 for encoders, 45 for decoders; KV Cache is decoder-only).
Code: [`src/defaultplusplus/deform/operators.py`](src/defaultplusplus/deform/operators.py).

Operators have four search-type categories:

| Type | Meaning | Example |
|---|---|---|
| `B`  | binary on/off | `MZM` (zero attention mask) |
| `EU` | numeric grid | `ETZ` (zero N% of token embeddings) |
| `EL` | categorical | `FCA` (replace activation: ReLU/GELU/Tanh/Sigmoid) |

### Injection mechanisms

| Type | Class | What it does |
|---|---|---|
| Static  | `StaticFault`  | back up parameters → mutate in place → restore on context exit |
| Dynamic | `DynamicFault` | wrap `module.forward` with a closure → restore on context exit |

Both subclass `FaultInjector`, which is an `AbstractContextManager`.
Use them inside a `with` block:

```python
from defaultplusplus.deform.injection import StaticFault

class _ZeroQ(StaticFault):
    def parameters_to_mutate(self, model):
        return [model.bert.encoder.layer[0].attention.self.query.weight]
    def mutate_parameters(self, params):
        for p in params:
            p.zero_()

with _ZeroQ(model):
    # forward passes here see the zero query projection
    outputs = model(**batch)
# parameters restored automatically here
```

### Mutation killing (Section 7.3.4)

Five matched seeds, exact one-sided sign-flip permutation test.
Floor p-value 1/2⁵ ≈ 0.031. Code:
[`src/defaultplusplus/deform/validation.py`](src/defaultplusplus/deform/validation.py).

```python
from defaultplusplus.deform.validation import is_killed

clean = [0.90, 0.91, 0.89, 0.92, 0.90]
faulty = [0.70, 0.72, 0.68, 0.74, 0.71]
killed, p = is_killed(clean, faulty, higher_is_better=True, alpha=0.05)
# killed=True, p=0.03125 (= 1/32)
```

### Correct class (Section 7.3.5)

The faulty class comes from killed mutants. The correct class comes from
clean base models with label-preserving perturbations. For each base model
that produces `k` killed mutants, generate `k` clean variants, test each
against the base model with the same kill test, and keep the ones that stay
indistinguishable. Code:
[`src/defaultplusplus/deform/clean_variants.py`](src/defaultplusplus/deform/clean_variants.py).

```python
from defaultplusplus.deform import generate_clean_variants, run_one_clean_variant

variants = generate_clean_variants("bert-base-uncased", "sst2", k, base_seed=42)
for variant in variants:
    sample = run_one_clean_variant(
        variant, fine_tune, feature_builder,
        higher_is_better=True, seeds=SEEDS, base_hyperparams=base_hp)
    # sample.retained is True only when the variant is NOT killed; a
    # retained sample is written with detection_label = 0.
```

The CLI generates the correct class with `defaultpp-benchmark
--clean-variants N` (N variants per model-task pair).

---

## 7. The benchmark pipeline

```
┌──────────────────┐  enumerate_configurations  ┌──────────────────┐
│  BenchmarkSpec   │ ─────────────────────────► │ FaultConfig grid │
└──────────────────┘                            └──────────┬───────┘
                                                           │ for each
                                                           ▼
                                                ┌──────────────────┐
                                                │ run_one_config   │
                                                │  (5 paired seeds)│
                                                └──────────┬───────┘
                                                           │ Mutant
                                                           ▼
                                                ┌──────────────────┐
                                                │ DatasetWriter    │
                                                │  (CSV shard)     │
                                                └──────────┬───────┘
                                                           │ merge
                                                           ▼
                                                ┌──────────────────┐
                                                │ data/*.csv       │
                                                └──────────────────┘
```

Code: `src/defaultplusplus/benchmark/`. The runner takes pluggable
`FineTuneFn` and `FeatureBuilderFn` callables so it can be unit-
tested without HF or GPUs (see `tests/test_dry_run.py`).

---

## 8. How to extend

### Add a new mutation operator

1. **Define the operator** in [`src/defaultplusplus/deform/operators.py`](src/defaultplusplus/deform/operators.py):

   ```python
   Operator("XYZ", OperatorComponent.SCORE, "my_root_cause",
            "Description shown in tables.",
            OperatorSearchType.NUMERIC_GRID,
            param_name="factor", param_grid=(0.5, 2.0, 5.0),
            scope="both")
   ```

2. **Wire the injection** in
   [`src/defaultplusplus/deform/operator_impls/registry.py`](src/defaultplusplus/deform/operator_impls/registry.py):
   add an entry to `_STATIC_SPECS` (parameter mutation), `_DYNAMIC_PATTERNS`
   (forward wrap), or `_ATTRIBUTE_OPS` (model-attribute toggle), depending
   on the operator's effect. `get_injector("XYZ")` will then return the
   right injector class without further wiring; the benchmark runner
   resolves operators by ID through the same path.

3. **Add coverage in two places**:
   - `tests/test_operator_coverage.py` — append `"XYZ"` to
     `EXPECTED_OPERATOR_IDS` so the locked operator-id list grows in step.
   - `tests/test_dry_run.py::test_all_operator_injectors_construct_verify_and_restore`
     iterates the full catalog, so the operator must construct on the
     tiny model and pass the structural verifier with no extra work.

### Add a new metric

1. Add the metric to one of the existing modules in
   [`src/defaultplusplus/extraction/metrics/`](src/defaultplusplus/extraction/metrics/)
   (or create a new one inheriting `MetricModule`).
2. The metric's `collect()` returns `dict[str, float]`. Keys must
   follow the naming convention in the schema.
3. Update `../docs/SPEC.md` to add the metric and tag it as
   `exact` / `approximate` / `reconstructed`.
4. Add a regex token in [`src/data/feature_groups.py`](src/data/feature_groups.py)
   so the column routes to the right group.

### Add a new benchmark task

The kill-test scoring is registry-driven so adding a task is a
single-file change:

1. **Register the spec** in
   [`src/defaultplusplus/benchmark/task_metrics.py`](src/defaultplusplus/benchmark/task_metrics.py):

   ```python
   "mytask": TaskMetricSpec(
       name="mytask", arch="encoder", higher_is_better=True,
       raw_metrics=("accuracy", "f1"),  # what compute_metrics must emit
       aggregator=lambda m: 0.5 * (m["eval_accuracy"] + m["eval_f1"]),
   ),
   ```

2. **Wire the dataset loader** in
   [`src/defaultplusplus/benchmark/cli.py`](src/defaultplusplus/benchmark/cli.py):
   if the task uses GLUE, add the text column tuple to
   `_glue_text_columns` and the `num_labels` to `_GLUE_NUM_LABELS`. For
   non-GLUE encoder tasks or new decoder tasks, extend
   `_load_encoder_dataset` / `_load_decoder_dataset`.

3. **Add tests** in `tests/test_task_metrics.py` covering the
   aggregator's arithmetic on a sample `eval_*` dict and the
   `compute_metrics` callable on synthetic logits.

The runner reads `higher_is_better` per configuration from the spec,
so encoder + decoder tasks can mix in a single CLI invocation.

### Add a new feature group

1. Add the group name to `STRUCTURAL_GROUPS` or `NON_STRUCTURAL_GROUPS`
   in [`src/data/feature_groups.py`](src/data/feature_groups.py).
2. Add a token rule to `_TOKEN_RULES` so columns route into it.
3. If structural, add component → group mappings in
   [`src/data/fundamental_fpg.py:_COMP_TO_GROUP`](src/data/fundamental_fpg.py).
4. Re-run the FPG sanity portion of the dry-run script:
   `bash scripts/dry_run_local.sh --quick`.

### Add a new ablation variant

1. Extend `ABLATION_VARIANTS` in
   [`hierarchical_graph_category_rootcause/evaluate.py`](hierarchical_graph_category_rootcause/evaluate.py).
2. Add a CLI flag in `train.py:main` and thread it through
   `run_experiment(use_graph=, use_sep=)`.

### Bump the package

1. `defaultplusplus/src/defaultplusplus/_version.py`: bump version.
2. `defaultplusplus/CHANGELOG.md`: add a dated entry.
3. `bash scripts/build_pypi.sh` → check artifacts.
4. `bash scripts/build_pypi.sh --testpypi` → install in throwaway venv.
5. `bash scripts/build_pypi.sh --pypi`.

---

## 9. Data flow

### Training-time feature collection

```
HF model + batch
   │
   ▼
ModelInspector.discover_structure()        # auto-detect QKV/FFN/LN paths
   │
   ▼
MetricCollector.collect_step(...)
   ├─ TrainingMetrics       (loss, lr, step time, memory)
   ├─ GradientMetrics       (per-component grad norms, update ratios)
   ├─ AttentionMetrics      (entropy, mass_pad, head_similarity, ...)
   ├─ StructuralMetrics     (residual cosine, ffn norms, LN stats, CKA)
   ├─ LogitMetrics          (NLL, ECE, margin, accuracy, F1)
   ├─ PositionalMetrics     (positional sensitivity)
   └─ CacheMetrics          (decoder only: cache_hidden_sim)
   │
   ▼
EpochAggregator.update(per-step dict)      # Welford running stats
   │
   ▼ (every epoch_end)
EpochAggregator.finalize_epoch()           # mean/var/burst per metric
   │
   ▼ (at finalize)
get_final_features()                       # schema keys, flat
   +
build_feature_vector(TrainingTrace)        # layer/step/epoch/phase aggregates
   │
   ▼
flat dict[str, float]                      # the feature vector
```

### Diagnostic-model training

```
data/*.csv (DEFault-bench instances)
   │
   ▼
prepare_dataset_from_csv (loader.py)       # X, y_detect, y_category, y_rootcause
   │
   ▼ (per outer CV fold; preprocessing fit on the fold's training data)
apply_processing_in_fold (feature_processor.py)
   ├─ Step 1: drop NaN > 40%
   ├─ Step 2: log1p high-variance cols
   ├─ Step 3: per-layer aggregation (encoder)
   ├─ Step 4: median imputation in fold
   ├─ Step 5: drop CV < 0.01
   └─ Step 6: route columns to feature groups (Table 7.7)
   │
   ▼
HierarchicalDiagnosisModel(group_dims, adjacency, ...)
   │
   ▼
hierarchical_loss (5 components)
   │
   ▼
train_one_fold (early stopping on val_metric blend)
   │
   ▼
evaluate_one_fold (all 3 levels, oracle + predicted-category routes)
   │
   ▼
results/.../{arch}_{variant}.json
```

---

## 10. Compute Canada workflow

`$PROJECT` and `$SCRATCH` are CC environment variables. Code lives in
`$PROJECT`; HF caches and transient artifacts live in `$SCRATCH`.

```bash
# One-time on the login node
bash defaultplusplus/scripts/cc/setup_env.sh

# Stage 1: build DEFault-bench (GPU array; one task per configuration)
sbatch defaultplusplus/scripts/cc/bench_array.sh

# Concatenate per-task shards
sbatch defaultplusplus/scripts/cc/merge_shards.sh

# Stage 2: train the diagnostic model
sbatch defaultplusplus/scripts/cc/train.sh

# Stage 3: 4-variant ablation study
sbatch defaultplusplus/scripts/cc/ablation.sh
```

Edit `--account=def-yourgroup` in each `*.sh` to match your CC
allocation. Edit the `--array=0-N%64` range in `bench_array.sh` to
match the number of configurations you generated (`N` is the config
count minus one).

---

## 11. Debugging tips

| Symptom | First check |
|---|---|
| `from defaultplusplus import FeatureExtractor` raises `ModuleNotFoundError` | run `pip install -e .` from inside `defaultplusplus/`, then re-source the venv |
| `make train` complains about missing CSVs | `make data-check` |
| `inspector.py` fails with "Could not detect attention pattern" | the model is from an unsupported family (e.g. T5). Pass `arch=` only for `bert`/`gpt` family aliases. Pull request welcome to extend `inspector.py`. |
| `cache.py` raises `'DynamicCache' object is not subscriptable` | already fixed in 0.2.0; if you hit it again on a newer transformers release, extend `_extract_kv_pairs` |
| `feature_construction.py` `IndexError: index 1 out of bounds` | already fixed for n=1 epochs; if it returns, check `_band_indices` for the new shape |
| HF Trainer test skips with "accelerate is required" | `pip install -e ".[hf]"` |
| `build_pypi.sh` reports "version is not dynamic" | `pyproject.toml`'s `dynamic = ["version"]` line was removed; restore it |
| sign-flip test always returns p=1.0 | argument order: `is_killed(clean, faulty, higher_is_better)`; perplexity needs `False` |

### Useful inspection commands

```bash
# Inspect the wheel that PyPI would receive
python -c "import zipfile, sys; z=zipfile.ZipFile(sys.argv[1]); print('\n'.join(z.namelist()))" \
       dist/defaultplusplus-*-py3-none-any.whl

# Inspect the FPG group-level adjacency
python -c "from src.data.fundamental_fpg import fundamental_to_feature_group_adjacency; \
           names, adj, _ = fundamental_to_feature_group_adjacency('decoder'); \
           print(names); print(adj)"

# Print the operator catalog
python -c "from defaultplusplus.deform import OPERATORS; \
           [print(f'{op.op_id}  {op.component.value:11s}  {op.action}') \
            for op in OPERATORS.values()]"
```

---

## 12. Pointers to the manuscript

The DEFault++ manuscript (in preparation) is the scientific reference.
Section numbers below follow the thesis-chapter numbering.

| Topic | Location in the manuscript |
|---|---|
| Introduction + motivating example | Sections 7.1–7.2 |
| Fault taxonomy + operator catalog | Section 7.3, Tables 7.1, 7.2, Figure 7.3 |
| DEForm injection mechanism | Section 7.3.2, Figure 7.4 |
| Subject models / tasks | Section 7.3.3, Table 7.3 |
| Mutation validation | Section 7.3.4, Algorithm 1 |
| Benchmark statistics | Section 7.3.5, Tables 7.4, 7.5 |
| FPG construction | Section 7.4.2, Figure 7.5, Table 7.6 |
| Feature representation + Table 7.7 / 7.8 | Section 7.4.3 |
| Feature-vector construction (Equation 7.19) | Section 7.4.3.1, Figure 7.7 |
| Diagnostic model + algorithms | Section 7.4.4, Figure 7.8 |
| Equation 7.20 (group encoder) | Section 7.4.4.1 |
| Equation 7.22 (message passing) | Section 7.4.4.2 |
| Loss formulas (7.25–7.28) | Section 7.4.5, Figure 7.9 |
| Equations 7.29 – 7.30 (explanation) | Section 7.4.6 |
| Evaluation results | Section 7.5 |
| Real-world bug evaluation | Section 7.6, Table 7.20 |
| Developer study | Section 7.7 |

---

## 13. What's next

The scope and future extensions live in
[`../docs/SPEC.md`](../docs/SPEC.md) §3. The runtime
extractor, the diagnostic model (`diagnosis/`), the single-run normalizer
(`processing/`), the shipped checkpoints (`pretrained/`), and the
benchmark download path (`data/`) are all complete. The remaining items
are the out-of-v1-scope architectures (encoder-decoder, sparse-attention,
Mixture-of-Experts) and distributed-training signals. The `ui/`
subpackage is reserved for future CLI helpers.

---

## 14. License + citation

Apache-2.0 (`LICENSE`).

DEFault++ is unpublished research. If you use this code, please cite
the repository directly:

```bibtex
@software{defaultplusplus,
  title   = {{DEFault++}: Hierarchical Fault Detection and Diagnosis for
             Transformer Architectures},
  author  = {Jahan, Sigma and Rajput, Saurabhsingh and Sharma, Tushar and
             Rahman, Mohammad Masudur},
  year    = {2026},
  url      = {https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging},
  version = {0.4.1},
  doi     = {10.5281/zenodo.20019817},
  note    = {Software repository; manuscript in preparation.}
}
```
