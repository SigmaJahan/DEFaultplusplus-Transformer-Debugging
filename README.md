# DEFault++

[![CI](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SigmaJahan/DEFaultplusplus-Transformer-Debugging/actions/workflows/ci.yml)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20019817.svg)](https://doi.org/10.5281/zenodo.20019817)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20481557.svg)](https://doi.org/10.5281/zenodo.20481557)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](defaultplusplus/LICENSE)

**Hierarchical fault detection and diagnosis for transformer architectures.**

Faults in a transformer's attention mechanism, projections, masking, or
other internal parts can change behavior silently. The loss curve still
goes down, no NaN appears, and the run finishes without an error, yet the
model has a bug. DEFault++ watches a fine-tuning run at the level of
individual transformer components and answers three questions in order.

1. **Detection.** Is this run faulty at all?
2. **Categorization.** Which transformer subsystem is responsible (QKV,
   masking, LayerNorm, KV cache, and so on)?
3. **Root cause.** Which specific bug pattern fits the evidence, and which
   feature groups support that diagnosis?

<p align="center">
  <img src="docs/figures/fig_technique_overview.png" width="90%"
       alt="DEFault++ takes runtime information from a transformer program, processes it through feature-group encoding and Fault Propagation Graph message passing, then runs three classifiers: binary detection, multi-class categorization, and prototypical root-cause diagnosis.">
</p>

The project ships two things. A clean, installable Python package
(`defaultplusplus`) that extracts runtime features and runs the trained
diagnostic model, and the research code that builds the benchmark and
trains that model.

---

## Contents

- [How it works](#how-it-works)
- [Install](#install)
- [Quick start](#quick-start)
- [The fault taxonomy](#the-fault-taxonomy)
- [How DEFault-bench is built](#how-default-bench-is-built)
- [The Fault Propagation Graph](#the-fault-propagation-graph)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## How it works

DEFault++ turns a fine-tuning run into a fixed-length feature vector, then
diagnoses that vector in three levels. Two ideas make the diagnosis work
on transformer faults that generic deep-learning debuggers miss.

**Component-level features.** Generic features such as loss curves and
gradient norms do not separate transformer fault categories. DEFault++
measures attention entropy, padding attention mass, QKV alignment,
residual cosine similarity, KV-cache divergence, and other component-level
quantities during training (see the feature-construction process below).

**The Fault Propagation Graph (FPG).** A fault at one component shifts
measurements at the components it feeds. The FPG is a structural prior,
read off the transformer's forward and backward equations, that tells the
model which components can affect each other. Message passing over the FPG
mixes evidence across related feature groups before classification.

<p align="center">
  <img src="docs/figures/fig_feature_construction.png" width="85%"
       alt="Feature construction. Layer-internal, gradient, behavioral, and validation metrics are collected during training, then aggregated by layer, step, epoch, and training phase into one fixed-length feature vector.">
</p>

The diagnostic model trains all three levels jointly with a shared encoder
and four losses: detection, categorization, root cause, and a separation
loss that pulls same-root-cause samples together and pushes different ones
apart.

<p align="center">
  <img src="docs/figures/fig_training_view.png" width="80%"
       alt="Training view. The shared encoder feeds four losses: detection, categorization, root cause, and separation, summed into one total loss.">
</p>

---

## Install

```bash
pip install defaultplusplus           # core runtime feature extractor
pip install defaultplusplus[hf]       # + HuggingFace Trainer callback
pip install defaultplusplus[viz]      # + matplotlib / seaborn / rich report
pip install defaultplusplus[all]      # everything
```

Editable install from a local checkout:

```bash
cd defaultplusplus
pip install -e ".[dev,hf]"
```

---

## Quick start

Extract features during a fine-tuning run, then diagnose them.

```python
from defaultplusplus import FeatureExtractor
from defaultplusplus.diagnosis import load_pretrained

with FeatureExtractor(model, arch="encoder") as fx:
    for epoch in range(num_epochs):
        for batch in loader:
            outputs = model(**batch, output_attentions=True,
                            output_hidden_states=True)
            outputs.loss.backward()
            optimizer.step(); optimizer.zero_grad()
            fx.step(loss=outputs.loss, outputs=outputs,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"], optimizer=optimizer)
        fx.epoch_end(epoch)
        fx.record_validation(epoch, eval_loop(model))
    features = fx.finalize()

predictor = load_pretrained("encoder")        # ships inside the wheel
diagnosis = predictor.predict(features)
print(diagnosis.to_dict())
# {'is_faulty': True, 'detection_prob': 0.92,
#  'category': 'qkv', 'category_prob': 0.81,
#  'root_cause': 'parameter_initialization', 'root_cause_prob': 0.74,
#  'group_importance': {'qkv_alignment': 0.45, 'attention': 0.17, ...}}
```

For the HuggingFace `Trainer`, use the drop-in callback instead:

```python
from defaultplusplus.hf_callback import DEFaultPlusCallback

trainer = Trainer(
    model=model, args=args,
    callbacks=[DEFaultPlusCallback(out_path="features.json", arch="encoder")],
)
trainer.train()
```

Runnable examples live in
[`defaultplusplus/examples/`](defaultplusplus/examples/). The full API,
visualization helpers, and the benchmark CLI are documented in the
[package README](defaultplusplus/README.md).

---

## The fault taxonomy

DEFault++ covers **12 fault categories** and **45 root causes**. Seven
categories are attention-internal and come from an attention-fault study
of 555 real faults. Five are architecture-level (Embedding, FFN,
LayerNorm, Residual, Output) and come from prior deep-learning fault
studies. KV Cache is decoder-only, so encoders use 11 categories and 40
root causes while decoders use all 12 and 45.

<p align="center">
  <img src="docs/figures/fig_fault_taxonomy.png" width="95%"
       alt="Taxonomy tree of transformer faults. Five architecture-level categories (Input/Embedding, LayerNorm, FFN, Residual, Output) and seven attention-specific categories (Masking, QKV, Kernel, Positional, Score, KV Cache, Variant), each expanded into its root causes.">
</p>

Each category maps onto a specific place in the transformer block, which
is where its operators inject faults.

<p align="center">
  <img src="docs/figures/fig_fault_categories.png" width="85%"
       alt="Fault categories placed on the transformer block. The left panel maps Embedding, Attention Variant, Normalization, FFN, Residual, and Output faults onto the block. The right panel expands the attention internals and places Masking, Score, QKV, KV-Cache, Positional, and Kernel faults on the attention computation.">
</p>

The faults are injected by **DEForm**, a transformer-specific mutation
engine with **52 operators** over the taxonomy. Every operator maps to one
root cause, and several root causes are covered by more than one operator
(for example, the QKV parameter-initialization root cause is covered by
three operators that zero the Q, K, and V projections separately). The
full operator catalog is in
[`deform/operators.py`](defaultplusplus/src/defaultplusplus/deform/operators.py).

---

## How DEFault-bench is built

DEFault-bench is a benchmark of labeled fine-tuning runs across seven
transformer models and nine downstream tasks. DEForm injects a fault into
a clean model, then a clean run and a faulty run are trained under matched
seeds. A one-sided sign-flip permutation test over five seeds decides
whether the fault changed task performance. A fault that passes is kept as
a labeled faulty instance. Clean, label-preserving variants form the
correct class.

<p align="center">
  <img src="docs/figures/fig_benchmark_construction.png" width="80%"
       alt="Benchmark construction workflow. A clean model and a fault-injected model are trained on a downstream task, features are collected during training, and a mutant-validation step compares clean versus injected performance. Validated mutants are labeled faulty, and clean runs are labeled correct, together forming DEFault-bench.">
</p>

The benchmark CSVs (about 360 MB) are hosted on Zenodo and fetched on
demand:

```bash
defaultpp-bench-download        # downloads to ~/.cache/defaultplusplus/bench/v1
```

---

## The Fault Propagation Graph

The FPG is the structural prior at the center of DEFault++. Nodes are
transformer components. Edges are forward data-flow dependencies read off
the architecture, so an edge means a perturbation at the source has a path
to the target. The model passes messages over the FPG so each feature
group's embedding reflects evidence from its structural neighbors before
classification.

<p align="center">
  <img src="docs/figures/fig_fpg.png" width="60%"
       alt="The Fault Propagation Graph. Nodes are transformer components (embedding, positional, QKV projection, score, masking, attention weights and output, residual junctions, LayerNorm, FFN, KV cache, output head). Edges are labeled by mechanism: forward, simultaneous, residual, cross-layer, and cache-time. Softmax, layer norm, and activation are marked as nonlinear operations that limit fault propagation.">
</p>

Seven propagation mechanisms are derived from the forward and backward
equations. The forward and structural mechanisms (M1, M2, M3, M4, M7)
become edges in the message-passing graph. Backward gradient coupling (M5)
enters through gradient features, and architecture-wide intervention (M6)
enters through the fault labels. A full walk-through of every figure,
including the group-level adjacency matrix, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```text
DEFaultplusplus-Transformer-Debugging/
  README.md                 this file (project landing page)
  CITATION.cff              how to cite the code and dataset
  docs/
    ARCHITECTURE.md         figure-by-figure walk-through of the method
    SPEC.md                 frozen feature-vector output schema
    figures/                the diagrams used across the docs
  defaultplusplus/          the installable package and research drivers
    README.md               package reference: full API, CLI, build/publish
    RESEARCH.md             research-side runbook (benchmark + training)
    CHANGELOG.md            version history
    LICENSE                 Apache-2.0
    pyproject.toml          PEP 621 metadata + build config
    src/defaultplusplus/    the importable package
      api.py                FeatureExtractor (manual loop)
      hf_callback.py        DEFaultPlusCallback (HF Trainer)
      extraction/           metric collection + aggregation
      deform/               mutation engine (52 operators)
      benchmark/            benchmark construction + kill test
      diagnosis/            Predictor + load_pretrained()
      processing/           feature processor + runtime normalizer
      pretrained/           shipped diagnostic-model checkpoints
      viz/                  matplotlib plots + HTML report
    hierarchical_graph_category_rootcause/
                            diagnostic-model training driver (nested CV)
    examples/               runnable demos
    scripts/                local + cluster reproduction scripts
    tests/                  pytest suite
  realworld_evaluation/     real-world GitHub-issue fault reproductions
    cases/                  one reproduction script per issue
    metadata/               per-issue source, root cause, and contract
    contract_checks.py      mechanism / symptom / buggy-vs-fixed checks
    run_benchmarks.py       runs every case and reports the contracts
```

---

## Documentation

| Document | What it covers |
| --- | --- |
| [`README.md`](README.md) | this landing page: overview, install, quick start |
| [`default++_manuscript.pdf`](default++_manuscript.pdf) | the full manuscript (in preparation) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the method explained figure by figure |
| [`docs/SPEC.md`](docs/SPEC.md) | the frozen feature-vector output schema |
| [`defaultplusplus/README.md`](defaultplusplus/README.md) | package reference: full API, visualization, benchmark CLI, build/publish |
| [`defaultplusplus/RESEARCH.md`](defaultplusplus/RESEARCH.md) | research runbook: rebuild the benchmark and retrain the model |

---

## Citation

If you use this code or the benchmark, please cite the repository and the
dataset. See [`CITATION.cff`](CITATION.cff) for both DOIs.

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

---

## License

Apache-2.0. See [`defaultplusplus/LICENSE`](defaultplusplus/LICENSE).
