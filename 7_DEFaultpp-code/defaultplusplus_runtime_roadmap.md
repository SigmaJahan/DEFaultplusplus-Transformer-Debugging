# DEFault++ Runtime Library: Product, Research, and Engineering Roadmap

## 1. Executive decision

The best long-term architecture for DEFault++ is **not** a model-specific hook library and **not** a generic graph-tracing tool. It is a **single-run, transformer-specific telemetry and diagnosis system** built on four decisions:

1. **Canonical semantic feature schema**. The invariant part of DEFault++ is the meaning of the measurements: attention entropy, padding/future leakage, QKV alignment, pre-softmax score statistics, update ratios, FFN output behavior, LayerNorm statistics, residual-stream integrity, output uncertainty, cache consistency, runtime/memory, and validation quality. The extraction path may differ by model family, but the feature definitions must not. Chapter 8 already structures diagnosis around grouped transformer-specific measurements rather than generic DNN features. fileciteturn3file7turn4file11

2. **Family-level adapters, not per-model hooks**. Compatibility should be defined over structural families such as BERT-style encoders, split-QKV decoders, fused-QKV decoders, RMSNorm decoders, and later GQA/MoE variants. This matches the actual source of incompatibility: implementation structure, not model name.

3. **Single-run anomaly encoding, not paired clean-vs-faulty deltas at deployment**. This is the most important architectural correction. FrankenFormer and the current Chapter 7 dataset construct training examples as direction-aligned deltas between matched clean and faulty runs. That is valid for supervised benchmark generation, but it is not the correct product contract for a runtime package that observes only one user run. The production path must therefore transform one live run into anomaly features against a learned clean reference, then train the production diagnosis model on the same single-run feature contract. fileciteturn4file14turn4file18

4. **Hugging Face outputs + targeted hooks + sampled profiler + streaming aggregation**. This is the most practical and correct extraction substrate. Hugging Face already exposes `ModelOutput` objects, optional `hidden_states`, and optional `attentions`; its newer output-tracing mechanism can also capture submodule outputs into `ModelOutput`. Trainer callbacks are read-only and cannot change the forward path, so Trainer integration must use either preinstalled hooks or a thin `Trainer` subclass. PyTorch profiler should be used only in sampled windows for time/memory/operator features. citeturn600318view1turn600318view0turn600318view2turn600318view4

This design preserves the scientific core of DEFault++ while making the package usable in realistic training loops.

## 2. What already exists, what is partially done, and what is still missing

### 2.1 Already done and reusable

The current repository already contains the **core offline diagnosis stack**: the feature processor, feature-group mapping, FPG construction, grouped encoder, and hierarchical diagnosis model used in the Chapter 8 experiments. The README and replication package clearly position the current repo as an offline diagnosis pipeline over pre-existing trace files, with three levels of diagnosis and FPG-based explanations. fileciteturn1file14turn1file15

The research foundations are also already done:

- The transformer fault taxonomy and fault label spaces are defined in Chapters 5, 7, and 8, covering 11 encoder categories and 12 decoder categories, with 40 encoder root causes and 45 decoder root causes. fileciteturn4file12turn4file15
- FrankenFormer already generates labeled transformer fault traces from matched clean/faulty runs across seven models and nine tasks. fileciteturn4file13turn4file17
- DEFault++ already demonstrates that transformer-specific grouped features and FPG-based message passing outperform generic DNN baselines on detection and categorization. fileciteturn4file10turn4file11

The repository also contains a **partial extraction scaffold** under `src/defaultplusplus/extraction`: `inspector.py`, `collector.py`, `aggregator.py`, `export.py`, and metric modules for training, gradients, attention, structural, logits, positional, and cache. This is an important head start.

### 2.2 Partially done but not yet product-ready

From direct inspection of the uploaded repository, the `defaultplusplus` package already has:

- a category-based `ModelInspector` that discovers encoder vs decoder families structurally rather than by model name;
- a low-level `MetricCollector`;
- an `ExtractionConfig` dataclass;
- multiple metric modules;
- epoch aggregation and export helpers.

That is the correct *direction*, but it is not yet a usable runtime product.

The package is currently missing its public orchestration layer. `src/defaultplusplus/__init__.py` only exposes `__version__`. There is no `core.py`, no `DEFaultPP` class, no `callback.py`, no `DiagnosisReport`, no report UI, no visualization module, no pretrained registry, and no diagnosis/inference wrapper under the package namespace. The `processing`, `diagnosis`, `pretrained`, and `ui` package directories are mostly stubs at the moment.

### 2.3 Architecturally incomplete or incorrect for the runtime goal

There are five critical gaps between the current research artifact and the desired runtime library.

**Gap 1: the training/inference data contract is mismatched.** The current Chapter 7/8 supervised dataset is built from paired clean-vs-faulty deltas averaged across seeds. A runtime package for arbitrary users will not have that clean matched baseline. This is the single most important issue to fix before treating the package as production-ready. fileciteturn4file14turn4file18

**Gap 2: the feature schema is not yet frozen and versioned.** The diagnosis model depends on stable feature groups and dimensions. Without a versioned schema, any improvement to the extractor will silently break inference reproducibility.

**Gap 3: several current metric implementations are approximations or incomplete.** For example, the current structural metrics infer FFN and residual behavior mostly from adjacent hidden states, which is not the same as instrumenting the actual sublayer boundaries. The current cache metric returns `cache_hidden_sim` but leaves `cache_nll_divergence` effectively unimplemented. The collector’s `feature_names` property is not yet a canonical schema contract.

**Gap 4: backend-aware attention collection is not formalized.** Recent Hugging Face models support multiple attention backends through `attn_implementation`, including `eager`, `sdpa`, and FlashAttention-family variants. These backends expose different observability trade-offs. The package needs an explicit capability policy. citeturn600318view3turn985073search14

**Gap 5: there is no test suite yet.** The plan defines detailed phase gates, but the uploaded repo currently contains no corresponding `tests/` implementation.

## 3. The product boundary: what DEFault++ is and is not

DEFault++ should be treated as three related but distinct systems.

### 3.1 The benchmark generator

FrankenFormer remains the **training-data generator**. It produces the labeled fault corpus needed to train diagnosis models. It is not part of the runtime path used by end users. fileciteturn4file13

### 3.2 The diagnosis model

The hierarchical diagnosis model, group encoders, FPG message passing, and prototype-based explanation remain the **reasoning core** of DEFault++. This is the part worth preserving from the current repo with minimal conceptual changes. Chapter 8’s grouped encoding and explanation design is already the right backbone. fileciteturn3file12turn2file1

### 3.3 The runtime library

The new package is the missing **online instrumentation and inference layer**. Its job is to observe one live training run, transform observations into the same semantic space the diagnosis model expects, and emit a structured diagnosis report.

The runtime library must **not** require paired clean/faulty runs, mutation injection, or post-hoc CSV engineering by the user.

## 4. The correct data contract for production

## 4.1 Why the current paired-delta contract cannot be the final runtime contract

The current supervised examples are explicitly defined as direction-aligned deltas between clean and faulty runs, then averaged across seeds. That is scientifically valid for mutation-based supervised learning, but it is not directly available during real-world use. fileciteturn4file14turn4file18

If the package simply feeds raw one-run features into a model trained on paired deltas, inference becomes distributionally mismatched. That would be a band-aid solution.

## 4.2 The best long-term solution

The production model must be retrained on **single-run anomaly features**.

That means the new training and inference contract should be:

1. Collect raw runtime features from one run.
2. Normalize each feature against a **clean reference distribution**.
3. Aggregate the anomaly trajectory into the fixed grouped representation.
4. Train and serve the diagnosis model on that same representation.

This preserves the practical one-run use case without abandoning FrankenFormer. FrankenFormer still provides the labels and the raw healthy/faulty traces needed to build the reference distributions.

## 4.3 How to build the clean reference model

The clean reference model should be learned from healthy training runs and expressed in a way that is broad enough to generalize but specific enough to stay informative.

The reference statistics should be conditioned on a **context signature** that contains only variables that materially affect the expected metric scale. The best initial context signature is:

- architecture family: encoder vs decoder;
- objective type: classification vs causal language modeling;
- training phase: early, mid, final;
- capability mode: exact / hybrid / lite;
- optional scale buckets only when mathematically necessary, such as sequence-length bucket for attention-specific quantities.

The package should **first normalize the metrics mathematically** so that fewer context buckets are required. For example:

- entropy should be normalized by the maximum entropy of the visible attention set;
- score magnitudes should be scaled by head dimension where applicable;
- norms should be converted to per-dimension or relative quantities when possible;
- update ratios are already scale-normalized and are therefore naturally portable.

After this normalization, robust statistics such as median and MAD are the right default for the clean reference library.

## 4.4 The transitional bridge

For reproduction and short-term continuity, keep the current paired-delta dataset path and current pretrained models in the repository, but mark them clearly as **research/reproduction mode**, not runtime product mode.

That bridge is useful for validation and regression testing. It should not define the product contract.

## 5. The target runtime architecture

The runtime system should be implemented as seven layers.

### 5.1 Layer A: execution substrate

Use Hugging Face `ModelOutput` and standard forward returns as the primary substrate. The package should request `return_dict=True`, and, when needed, `output_hidden_states=True` and `output_attentions=True`. Hugging Face’s `ModelOutput` contract and output-tracing mechanism are the most stable high-value interfaces currently available. citeturn600318view1turn600318view0

### 5.2 Layer B: semantic model adapter

The adapter translates a concrete implementation into semantic observables. It must answer questions like:

- where is the layer stack?
- what is the attention family?
- are Q/K/V split or fused?
- what normalization layers are used?
- can the backend expose explicit attention tensors exactly?
- what cache structure exists, if any?

This is where architecture dependence belongs. It must not leak upward into metric logic.

### 5.3 Layer C: metric modules

Metric modules compute the canonical observables. These should be organized exactly around the DEFault++ feature groups from Chapter 8:

- Attention
- Score
- FFN output
- LayerNorm
- Residual stream
- Representation drift
- QKV alignment
- Embedding
- Positional
- Training dynamics
- Output
- Cache
- Validation performance. fileciteturn3file7turn3file0

### 5.4 Layer D: streaming aggregation

Step-level metrics should be aggregated online with numerically stable streaming statistics. The final representation should preserve the thesis’ temporal insight: early, mid, and final training behavior matters. Chapter 8’s phase-based summarization is a good foundation, but the runtime package should use **proportional windows** instead of hardcoded epoch ranges. fileciteturn2file11

### 5.5 Layer E: anomaly encoding

After phase summaries are computed, transform them into clean-reference anomaly features. This is the contract the new production diagnosis model should consume.

### 5.6 Layer F: diagnosis and explanation

Reuse the existing grouped processor, FPG adjacency, group encoders, hierarchical heads, and prototype-distance explanation. The group structure and explanation decomposition are already aligned with the scientific design. fileciteturn2file1turn3file12

### 5.7 Layer G: report and sinks

The package should emit:

- structured report objects;
- Rich terminal warnings and summaries;
- JSON export;
- local TensorBoard logging by default;
- optional W&B logging.

TensorBoard is a strong default local sink because `SummaryWriter` updates files asynchronously. W&B is an optional observability/export sink, while the W&B Public API should be used only for post-hoc querying/export, not as the runtime engine. citeturn600318view7turn600318view5turn600318view6

## 6. Best use of existing tools

### 6.1 Hugging Face: primary runtime substrate

This is the single most important external dependency for the runtime collector.

Use Hugging Face for:

- `ModelOutput` returns (`loss`, `logits`, `hidden_states`, `attentions`);
- output tracing where supported;
- model/config metadata;
- attention backend control via `attn_implementation`. citeturn600318view1turn600318view0turn600318view3

Do **not** make `TrainerCallback` the primary integration path. Callbacks cannot modify the forward pass or the training loop, so any integration that needs `output_attentions`, `output_hidden_states`, or probe passes must use either preinstalled hooks or a thin `Trainer` subclass. citeturn600318view2

### 6.2 PyTorch profiler: sampled infrastructure features only

Use `torch.profiler` to collect operator time, step time, and memory features in sparse windows. The profiler is designed for exactly these metrics, and the official recipe explicitly covers time, memory, and scheduled profiling for long-running jobs. Shape and stack collection add overhead, so they should be disabled in default mode. citeturn600318view4turn985073search7

### 6.3 TensorBoard: default local sink

Make TensorBoard the default local logging sink. It is local, robust, async, and does not couple the package to a hosted platform. citeturn600318view7

### 6.4 Weights & Biases: optional sink and artifact manager

Use the W&B SDK optionally for experiment tracking and artifact upload. `Run.watch()` can monitor gradients/parameters, but it does not compute DEFault++’s transformer-specific metrics. The W&B Public API is explicitly for post-hoc querying/export and should remain optional. citeturn600318view6turn600318view5

### 6.5 TorchLens: developer bring-up tool, not production runtime dependency

TorchLens is excellent for onboarding unfamiliar architectures because it can extract activations from every intermediate operation and visualize the computational structure. It should be used to validate adapters and inspect unsupported families. It should not be a core runtime dependency. citeturn600318view8

### 6.6 FX / torchvision feature extraction: offline structure aid only

TorchVision’s FX-based feature extraction is elegant for returning internal nodes from traceable models, but FX symbolic tracing has known limitations, especially around dynamic control flow and certain in-place patterns. It is a developer tool for discovery and debugging, not the primary runtime extraction backbone. citeturn276351search0

### 6.7 torch.export and Netron: offline inspection only

`torch.export` and Netron are valuable for offline structure inspection, backend sanity checks, and adapter development. Netron supports `torch.export` and PyTorch models, which makes it useful for understanding new families. But `torch.export` is an export/compiler IR path, not the right abstraction for live telemetry. citeturn276351search1turn276351search2turn600318view9

### 6.8 VisualTorch and torchviz: documentation/debugging only

VisualTorch is for architecture visualization, and torchviz is for autograd graph visualization. Neither is a runtime feature collector. They may be useful for docs, debugging, and onboarding, but they should remain non-core. citeturn959217search9turn959217search3

## 7. Compatibility strategy: universal where possible, explicit where not

DEFault++ should not claim universal support for all Hugging Face models. It should claim **family-level support with explicit capability reporting**.

## 7.1 Family coverage for v1

The best initial supported families are:

- BERT-style encoders: BERT, RoBERTa, DistilBERT, ALBERT, Electra-like descendants;
- GPT-style decoders with split QKV;
- GPT-style decoders with fused QKV such as GPT-2/GPT-Neo style descendants.

This aligns with the original research coverage and the user requirements in the plan. fileciteturn1file10

## 7.2 Explicitly out of scope for v1

The first production release should **not** target the following families yet:

- encoder-decoder models such as T5 and BART;
- mixture-of-experts architectures;
- multimodal composite models;
- diffusion U-Nets;
- non-transformer sequence models.

This is not a limitation of the science; it is a necessary product scoping decision.

## 7.3 Capability modes

Every run should advertise a capability profile.

- **Exact mode**: all required tensors are directly observable.
- **Hybrid mode**: some metrics require occasional eager probe batches or targeted hooks.
- **Lite mode**: attention-map-dependent metrics are unavailable, but training/output/runtime metrics still function.

This is better than pretending all models are equally observable.

## 7.4 Attention backend policy

The package should default to the fastest backend for ordinary training, but use **eager probe mode** for metrics that require explicit attention tensors or easily reconstructible per-head scores. Hugging Face explicitly supports runtime backend switching via `attn_implementation` / `set_attn_implementation()`. citeturn600318view3turn985073search14

That gives the right long-term trade-off: low overhead in normal steps, exact observability on scheduled probe steps.

## 8. The canonical schema: what must be frozen before major implementation continues

Before further implementation, the package needs a **versioned feature specification**.

For every feature, the spec must record:

- feature name;
- feature group;
- mathematical definition;
- collection point in the computation;
- required observables;
- exactness level (`exact`, `reconstructed`, `approximate`, `not_available`);
- aggregation path (layer, epoch, phase);
- units and normalization;
- availability by architecture family.

The schema should be frozen as `runtime_v1` **before** adding non-essential new metrics.

## 8.1 What belongs in `runtime_v1`

The best `runtime_v1` core is the Chapter 8 feature space, because that is the scientifically validated set. The thesis already identifies the grouped feature design and their aggregation factors. fileciteturn3file7turn2file11

The mandatory core should include:

- attention entropy, pad mass, future mass, cross-example leakage, head similarity, head utilization, attention rank;
- pre-softmax score statistics;
- FFN output norm;
- LayerNorm scale and post-norm moments;
- residual cosine;
- representation drift (CKA or a faithful surrogate where CKA is too expensive);
- QKV alignment;
- component-level gradient norm, update ratio, update activity;
- embedding norm and token variance;
- positional sensitivity;
- loss trajectory, gradient noise scale, step time, peak memory;
- output confidence, output entropy, margin statistics, ECE;
- cache hidden similarity and cache divergence for decoders;
- task accuracy/perplexity and calibration.

## 8.2 What should not block `runtime_v1`

The extended metrics proposed in the plan such as dead-neuron fractions, representation rank, token isotropy, attention sink score, dead-head counts, loss spike ratios, and NaN/Inf counters are useful. But they should be introduced **after** the Chapter 8 core is reproduced faithfully and the schema is frozen.

This is the right engineering choice. Stability first, feature expansion second.

## 9. The engineering plan

## Phase 0 — Architectural reset and contract freeze

### Objective

Turn the current plan into a stable product contract before writing more code.

### Deliverables

- one versioned feature-spec document (`runtime_v1`);
- one adapter capability-spec document;
- one product support matrix;
- explicit distinction between `research/reproduction mode` and `runtime production mode`.

### Tests and checks

- every feature in the spec has a mathematical definition and group assignment;
- every feature is labeled as exact/reconstructed/approximate by family;
- every existing metric implementation is mapped to the spec or rejected.

### Exit criterion

No metric or API work proceeds until the feature contract and deployment data contract are frozen.

## Phase 1 — Harden the extraction core

### Objective

Upgrade the current `src/defaultplusplus/extraction` scaffold from “promising prototype” to “reliable collector.”

### Required implementation work

1. **Refactor `ModelInspector` into a real adapter layer.**
   The current structural discovery logic is a good start, but it needs explicit family classes and capability reports.

2. **Fix metric correctness at the semantic collection point.**
   - attention scores must support both split and fused QKV paths;
   - structural metrics must observe real sublayer boundaries when the boundary is semantically important;
   - cache metrics must either be fully implemented or removed from the contract until implemented.

3. **Replace fixed epoch windows with proportional windows.**
   The current hardcoded 10-epoch windows are not a valid production assumption.

4. **Always emit a deterministic feature dictionary.**
   Missing metrics must become explicit zeros or explicit `not_available` states that are handled at schema alignment time, not silent omissions.

5. **Add backend capability handling.**
   The collector must know whether the current run is exact, hybrid, or lite.

### Tests

- family-discovery tests on representative BERT-style and GPT-style models;
- fused-QKV tests on GPT-2/GPT-Neo style models;
- synthetic tensor tests for every metric formula;
- differential tests comparing collector outputs to manual computations on toy models;
- smoke tests for exact/hybrid/lite modes;
- determinism tests: repeated collection on identical input produces identical feature keys;
- unsupported-family tests: clear and correct failure reporting.

### Exit criterion

The collector can produce a stable, schema-conformant raw feature stream for supported families without silent feature dropouts.

## Phase 2 — Build the single-run anomaly encoding pipeline

### Objective

Create the deployment data contract that replaces paired clean-vs-faulty deltas.

### Required implementation work

1. **Generate raw per-run traces from FrankenFormer runs.**
   Treat each clean and faulty run as its own single-run sample.

2. **Build clean reference distributions.**
   Use only healthy runs for the reference library.

3. **Transform raw phase summaries into anomaly features.**
   Recommended defaults:
   - robust center: median;
   - robust scale: MAD with epsilon floor;
   - signed direction where semantics are known;
   - absolute deviation where any shift is harmful.

4. **Create a runtime reference package artifact.**
   This artifact must be versioned and shipped alongside pretrained diagnosis weights.

5. **Keep the old paired-delta path for reproducibility only.**
   Do not delete it yet.

### Tests

- reference-statistics fit/transform tests;
- train/test leakage tests ensuring clean references are fit only on training folds;
- parity tests showing the anomaly pipeline is stable across repeated resampling;
- ablation tests on context conditioning to ensure the reference model is neither too coarse nor too brittle.

### Exit criterion

Single-run anomaly features can be computed both offline and online using the same transformation code.

## Phase 3 — Retrain the diagnosis model on the production contract

### Objective

Replace the paired-delta inference dependency with a production-ready single-run model.

### Required implementation work

1. **Refactor current diagnosis code into package modules.**
   Move the feature processor, group mapping, FPG, grouped encoder, and hierarchical model under `src/defaultplusplus/processing` and `src/defaultplusplus/diagnosis`.

2. **Retrain separate encoder and decoder production models.**
   Keep the architecture split; Chapter 8 already trains encoder and decoder models separately because their group structures differ. fileciteturn3file12

3. **Evaluate with grouped cross-validation and leakage control.**
   Preserve grouped splits by model/dataset/seed as in Chapter 8. fileciteturn2file1turn3file16

4. **Preserve explanation semantics.**
   The prototype-distance decomposition and group importance computation must remain unchanged conceptually. fileciteturn2file1

### Tests

- training/inference parity tests;
- fold leakage tests;
- feature-group dimensionality tests;
- explanation decomposition checks ensuring group contributions sum to the prototype gap;
- regression tests against the real-world case-study faults where available. The stale fused-QKV/LoRA case is especially important because it directly validates the product rationale. fileciteturn4file7

### Exit criterion

The package can run end-to-end from one live run to a valid diagnosis using a model trained on the same single-run contract.

## Phase 4 — Build the public runtime API

### Objective

Expose a simple and correct API that reflects the real integration constraints.

### Best primary API

The primary API should be the manual training-loop / context-manager API, because it gives the package direct access to the forward pass, backward pass, optimizer step, and validation boundaries.

```python
from defaultplusplus import DEFaultPP

monitor = DEFaultPP(model, optimizer, mode="hybrid")

for epoch in range(num_epochs):
    with monitor.epoch(epoch_idx=epoch):
        for batch in dataloader:
            outputs = model(
                **batch,
                return_dict=True,
                output_hidden_states=monitor.need_hidden_states,
                output_attentions=monitor.need_attentions,
            )
            loss = outputs.loss
            loss.backward()
            with monitor.step(batch=batch, outputs=outputs, loss=loss):
                optimizer.step()
                optimizer.zero_grad()

report = monitor.diagnose()
```

### Best secondary integration

Provide a **thin `Trainer` subclass or wrapper**, not a callback-only solution, because callbacks cannot modify the training loop or the forward pass. citeturn600318view2

### Required implementation work

- `core.py`: orchestrator;
- `callback.py` replaced or complemented by `trainer.py` / subclass wrapper;
- run/session lifecycle management;
- feature export and logging;
- failure-safe cleanup for hooks and profiler contexts.

### Tests

- manual-loop integration tests;
- context-manager correctness tests;
- Trainer subclass integration tests;
- interrupted-run cleanup tests;
- CPU-only and CUDA smoke tests.

### Exit criterion

A user can pip-install the package and diagnose a supported Hugging Face training run without touching CSV files.

## Phase 5 — Build reporting, sinks, and usability

### Objective

Make the output usable for debugging and research.

### Required implementation work

- `DiagnosisReport` object with structured fields;
- Rich terminal rendering;
- JSON serialization;
- Matplotlib plotting;
- TensorBoard sink;
- optional W&B sink.

### Tests

- report serialization/deserialization;
- rendering tests;
- sink integration tests;
- no-network default behavior;
- malformed-feature error reporting.

### Exit criterion

The package outputs a diagnosis that a practitioner can read and act on.

## Phase 6 — Pretrained artifacts and release engineering

### Objective

Ship a stable first release.

### Required implementation work

- pretrained encoder and decoder production weights;
- clean reference artifacts;
- processor/scaler/group metadata;
- semantic-schema version stamps;
- compatibility checks during load;
- first-use artifact download and checksum validation.

### Tests

- artifact loading tests;
- checksum tests;
- schema mismatch tests;
- cross-platform install smoke tests;
- end-to-end example scripts.

### Exit criterion

`pip install defaultplusplus` works, artifacts auto-load correctly, and one example script reproduces a diagnosis report.

## 10. Testing strategy in detail

The package needs five layers of testing.

### 10.1 Mathematical unit tests

Every metric with a formal definition must have a toy-tensor test where the expected value can be computed by hand.

Examples:

- attention entropy on uniform and peaked attention;
- future mass on a causal mask vs violated causal mask;
- QKV alignment on aligned vs orthogonal projection outputs;
- residual cosine on identity vs perturbed residual blocks;
- ECE on perfectly calibrated vs deliberately miscalibrated logits.

### 10.2 Adapter and compatibility tests

- BERT/RoBERTa/DistilBERT/ALBERT representatives;
- GPT-2/GPT-Neo/fused-QKV representatives;
- probe-mode eager attention;
- unsupported families fail with informative capability reports, not opaque errors.

### 10.3 Differential runtime tests

These are the most important practical tests.

For supported reference models, compare package-produced raw metrics against manual or notebook-computed metrics for the same forward/backward step.

This is how extraction correctness should be established.

### 10.4 Diagnosis regression tests

- old research pipeline vs package inference on historical benchmark traces;
- single-run anomaly model accuracy/F1/AUROC regression guards;
- explanation consistency checks;
- real-world reproduced bug cases from Chapter 8. fileciteturn3file10turn4file7

### 10.5 Performance tests

The package needs explicit overhead budgets.

Recommended budgets:

- lite mode: less than 10% slowdown;
- hybrid mode: less than 25% average slowdown with sparse probe batches;
- profiler windows: less than 5% amortized overhead at default schedule.

These are design targets, not currently measured repository results.

## 11. What to do immediately, in order

The critical path should be:

1. Freeze `runtime_v1` feature schema and capability metadata.
2. Finish and correct the extraction layer until it can produce faithful raw semantic metrics.
3. Build the single-run anomaly transformation and reference-stat package.
4. Retrain encoder and decoder diagnosis models on that production contract.
5. Refactor diagnosis code into the package.
6. Implement the primary manual/context API.
7. Implement report/UI/sinks.
8. Ship pretrained artifacts and release.

Everything else is secondary.

## 12. What should explicitly wait until after v1

The following are valuable, but they should not block the first correct release:

- relation-typed FPG message passing, which Chapter 8 identifies as a natural extension; fileciteturn2file4turn2file6
- selective Hessian or curvature-aware features at scheduled checkpoints; fileciteturn1file0turn1file4
- encoder-decoder architectures;
- MoE-specific routing diagnostics;
- multimodal model support;
- automated repair suggestion generation.

These belong in v1.1+ or the next paper, not in the critical path of a correct runtime package.

## 13. Final recommendation

The most practical, correct, and durable plan is:

- keep FrankenFormer as the labeled-data engine;
- keep the Chapter 8 grouped/FPG diagnosis design;
- harden the existing extractor scaffold into a family-adapter-based semantic collector;
- stop treating paired clean-vs-faulty deltas as the deployment contract;
- retrain the production diagnosis model on single-run anomaly features derived from clean-reference normalization;
- make the manual/context API the primary integration path;
- treat Trainer integration, W&B integration, Netron, TorchLens, VisualTorch, FX, and export tooling as supporting layers, not the core architecture.

That is the cleanest path that respects the science, the codebase you already have, the real-world deployment goal, and the engineering constraints of modern transformer stacks.
