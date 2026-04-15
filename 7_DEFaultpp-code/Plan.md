> Document role note:
> `defaultplusplus_runtime_roadmap.md` controls sequencing, and `../docs/runtime_v1_contract.md` controls runtime schema/capability truth.
> This file is a subordinate backlog and test bank, not the top-level product authority.
> Post-v1 expansions in this file are not on the critical path unless promoted by the roadmap or the runtime contract.

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Build DEFault++ as an Importable Fault Diagnosis Library

 Context

 The DEFault++ repo has a working diagnosis pipeline that operates on pre-existing CSV files. The feature
 extraction code lives in a separate repo (last_project_phd/0_Fault-Injection), tightly coupled to
 BERT/DistilBERT.

 Goal: Build a complete, pip-installable library where users can from defaultplusplus import DEFaultPP,
 instrument any HuggingFace transformer training, collect features in real-time, and get hierarchical fault
 diagnosis with explanations.

 Requirements (from user):
 - Support commong catgeories of HuggingFace encoder & decoder model types: encoders such as BERT, RoBERTa, DistilBERT, ALBERT and decoders such as GPT-2, GPT-Neo, DistilGPT2 and similar types.
 - Rich terminal output + logging for real-time warnings
 - Ship pretrained weights AND support custom training
 - Full importable Python API


 Here are important sources to look to get some more insights relevant to extract features from the model training: 
 1. torchvision.models.feature_extraction - https://docs.pytorch.org/vision/main/feature_extraction.html & https://docs.pytorch.org/docs/stable/fx.html
 2. weights & biases API - https://docs.wandb.ai/models/integrations/pytorch & https://docs.wandb.ai/models/ref/python/public-api/api & their github: https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py 
 3. pytorch profiler - https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
 4. torchlens - https://github.com/johnmarktaylor91/torchlens 
 5. visualtorch - https://visualtorch.readthedocs.io/en/latest/ & https://github.com/willyfh/visualtorch
 6. PyTorch’s Dynamic Computational Graph - https://medium.com/@StackGpu/understanding-pytorchs-dynamic-computational-graphs-92c42f41e334 
 7. torchviz - https://pypi.org/project/torchviz/
 8. Netron - https://pypi.org/project/netron/ & https://github.com/lutzroeder/netron 
 9. tensorboard - https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html 
 10. onx - https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html
 ---
 Target API

 # Option 1: Manual training loop
 from defaultplusplus import DEFaultPP

 monitor = DEFaultPP(model, optimizer)

 for epoch in range(num_epochs):
     for batch in dataloader:
         outputs = model(**batch)
         loss = outputs.loss
         loss.backward()
         monitor.step(loss=loss, outputs=outputs, labels=batch['labels'])
         optimizer.step()
         optimizer.zero_grad()
     monitor.end_epoch(val_metrics=val_results)

 report = monitor.diagnose()
 report.show()      # Rich terminal
 report.save()      # JSON export
 report.plot()      # Matplotlib figures

 # Option 2: Context manager (cleaner PyTorch)
 from defaultplusplus import DEFaultPP

 monitor = DEFaultPP(model, optimizer)

 for epoch in range(num_epochs):
     with monitor.epoch():                    # auto end_epoch + val_metrics
         for batch in dataloader:
             outputs = model(**batch)
             loss = outputs.loss
             loss.backward()
             with monitor.step(loss=loss, outputs=outputs, labels=batch['labels']):
                 optimizer.step()             # step() context wraps optimizer.step
                 optimizer.zero_grad()

 report = monitor.diagnose()

 # Option 3: HuggingFace Trainer callback (ONLY for HF Trainer users, not raw PyTorch)
 # NOTE: PyTorch itself has NO callback system. This only works with the HuggingFace
 # `Trainer` class which provides TrainerCallback hooks. Most users doing custom
 # PyTorch training should use Option 1 or Option 2 above.
 from defaultplusplus import DEFaultPPCallback
 from transformers import Trainer
 callback = DEFaultPPCallback()
 trainer = Trainer(model=model, callbacks=[callback])
 trainer.train()
 report = callback.diagnose()

 ---
 Package Structure

 src/defaultplusplus/
 ├── __init__.py                         # Public API: DEFaultPP, DEFaultPPCallback
 ├── core.py                             # DEFaultPP main class (orchestrator)
 ├── callback.py                         # HuggingFace TrainerCallback integration
 ├── config.py                           # Configuration dataclasses
 │
 ├── extraction/                         # Feature extraction from live training
 │   ├── __init__.py
 │   ├── inspector.py                    # Auto-detect HF model architecture + register hooks
 │   ├── collector.py                    # MetricCollector: orchestrates all metric modules
 │   ├── aggregator.py                   # Epoch-level Welford aggregation + windowed features
 │   ├── metrics/
 │   │   ├── __init__.py
 │   │   ├── base.py                     # Abstract MetricModule interface
 │   │   ├── training.py                 # Loss, accuracy, LR, step_time, memory
 │   │   ├── gradient.py                 # Per-layer gradient norms, update ratios, vanish/explode
 │   │   ├── attention.py                # Entropy, sparsity, padding mass, head similarity, positional
 │   │   ├── structural.py              # FFN delta, residual cosine, LN stats, embedding norms
 │   │   ├── logit.py                    # Entropy, confidence, ECE, margin stats
 │   │   ├── positional.py              # Early/late window accuracy and margins
 │   │   └── cache.py                    # Decoder-only: cache_hidden_sim
 │   └── export.py                       # Feature vector → CSV/DataFrame export
 │
 ├── processing/                         # Feature processing (reuse existing code)
 │   ├── __init__.py
 │   ├── pipeline.py                     # 6-step feature processor (from src/data/feature_processor.py)
 │   └── groups.py                       # Feature-to-FPG-group mapping (from src/data/feature_groups.py)
 │
 ├── diagnosis/                          # Hierarchical inference
 │   ├── __init__.py
 │   ├── fpg.py                          # Fault Propagation Graph (from src/data/fundamental_fpg.py)
 │   ├── model.py                        # HierarchicalDiagnosisModel (from hierarchical_.../model.py)
 │   ├── encoder.py                      # GroupEncoder + GraphAggregator (from src/models/group_encoder.py)
 │   ├── inference.py                    # End-to-end inference pipeline (NEW)
 │   └── explanation.py                  # Group decomposition + formatting (NEW)
 │
 ├── pretrained/                         # Pretrained model weights
 │   ├── __init__.py
 │   ├── registry.py                     # Model registry: load weights by arch type
 │   └── weights/                        # .pt files (gitignored, downloaded on first use)
 │       ├── encoder_model.pt
 │       ├── encoder_scaler.pkl
 │       ├── encoder_processor.pkl
 │       ├── decoder_model.pt
 │       ├── decoder_scaler.pkl
 │       └── decoder_processor.pkl
 │
 └── ui/                                 # User-facing output
     ├── __init__.py
     ├── console.py                      # Rich live display: progress, warnings, tables
     ├── report.py                       # DiagnosisReport class: show/save/export
     └── visualization.py                # Matplotlib plots: FPG, explanations, training curves

╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
EXPANDED IMPLEMENTATION PLAN — GRANULAR SUB-TASKS
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

================================================================================
PHASE 0: PROJECT SCAFFOLDING
================================================================================
Goal: Create the directory tree, __init__.py stubs, and update pyproject.toml
      so the package is importable from day one.
Dependencies: None (this is the foundation)
Estimated size: ~50 lines of boilerplate

 0.1  Create directory structure
      - mkdir -p src/defaultplusplus/{extraction/metrics,processing,diagnosis,pretrained/weights,ui}
      - This creates all 6 sub-packages in one shot

 0.2  Create all __init__.py stub files (8 files)
      Files to create (all initially empty or with TODO docstrings):
        src/defaultplusplus/__init__.py           → docstring + version = "0.2.0"
        src/defaultplusplus/extraction/__init__.py
        src/defaultplusplus/extraction/metrics/__init__.py
        src/defaultplusplus/processing/__init__.py
        src/defaultplusplus/diagnosis/__init__.py
        src/defaultplusplus/pretrained/__init__.py
        src/defaultplusplus/ui/__init__.py
      DO NOT add any imports yet — they'll be added as modules are completed

 0.3  Update pyproject.toml
      Sub-steps:
      a) Change package name from "graph-fault-diagnosis" to "defaultplusplus"
      b) Add new dependencies: rich>=13.0, joblib>=1.3, transformers>=4.30
      c) Update [tool.setuptools.packages.find] to include "src/defaultplusplus*"
      d) Keep existing src* and hierarchical* includes (don't break current pipeline)
      e) Add [project.optional-dependencies] for dev: pytest, pytest-cov

 0.4  Add .gitignore entry for pretrained weights
      - Add: src/defaultplusplus/pretrained/weights/*.pt
      - Add: src/defaultplusplus/pretrained/weights/*.pkl

 0.5  Verify: `python -c "import src.defaultplusplus"` works without error

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0 GATE TEST — Must pass before starting Phase 1                       │
│                                                                              │
│ tests/test_phase0_gate.py:                                                   │
│  T0.1  All 7 __init__.py files exist and are importable                      │
│  T0.2  `from src.defaultplusplus import __version__` returns "0.2.0"         │
│  T0.3  Directory structure matches package layout (glob for all dirs)        │
│  T0.4  pyproject.toml parses without error and includes defaultplusplus      │
│  T0.5  .gitignore contains pretrained weights exclusion                      │
│                                                                              │
│ Run: pytest tests/test_phase0_gate.py -v                                     │
│ All 5 must PASS. If any fail → fix before proceeding.                        │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 1: FEATURE EXTRACTION INFRASTRUCTURE
================================================================================
Goal: Build the entire real-time metric collection system that hooks into
      any HuggingFace model during training and produces feature vectors.
Source: Port from last_project_phd/.../base_metrics.py (1050 lines)
              + last_project_phd/.../metric_collector.py
              + last_project_phd/.../statistics.py
              + last_project_phd/.../running_metrics.py

────────────────────────────────────────────────────────────────────────────────
STEP 1: extraction/inspector.py — ModelInspector + MODEL_REGISTRY
────────────────────────────────────────────────────────────────────────────────
Source: NEW (no direct source — architecture discovery logic)
Size: ~250-300 lines
Why first: Every metric module depends on inspector to locate model internals

 1.1  Design: CATEGORY-BASED auto-discovery (NOT per-model registry)
      ════════════════════════════════════════════════════════════════════
      CORE PRINCIPLE: We do NOT maintain a list of model names. Instead we
      define TWO structural categories and AUTO-DETECT which one a model
      belongs to by probing its actual module tree at runtime.

      Why? Because HuggingFace has 100+ model types and new ones appear
      monthly. A per-model registry would need constant updates and would
      never truly be complete. Instead:

      CATEGORY 1: BERT-STYLE ENCODER
        Structural signature (what makes a model "BERT-style"):
        - config.is_encoder_decoder == False (or missing) AND no causal mask
        - Has a backbone module containing an 'encoder' or 'transformer' with
          a list of layers (ModuleList)
        - Each layer has: attention (with Q,K,V projections), FFN, LayerNorm
        - Bidirectional attention (no causal mask)
        Covers: bert, roberta, distilbert, albert, electra, deberta, xlm-roberta,
                camembert, flaubert, modernbert, ernie, funnel, and ANY future
                model that follows this structure

      CATEGORY 2: GPT-STYLE DECODER
        Structural signature (what makes a model "GPT-style"):
        - config.is_decoder == True OR uses causal attention mask
        - Has a 'transformer' or similar backbone with a list of layers
        - Each layer has: attention (causal), FFN/MLP, LayerNorm
        - Unidirectional / causal attention
        Covers: gpt2, gpt_neo, gpt_neox, gpt_j, opt, bloom, distilgpt2,
                llama, mistral, phi, qwen, and ANY future model that follows
                this structure

      The inspector does NOT need to know the model's name. It needs to:
        a) Detect which category (encoder vs decoder)
        b) Discover where the layers/attention/FFN/LN modules live
      Both are done by probing the model's actual nn.Module tree.

 1.2  Implement _detect_family() — category detection via structural probing
      Sub-steps:
      a) Check config attributes (fast path):
         - hasattr(config, 'is_decoder') and config.is_decoder → 'decoder'
         - hasattr(config, 'is_encoder_decoder') and not config.is_encoder_decoder
           → likely 'encoder' (but verify with structure)
      b) If config is ambiguous, probe model structure (reliable path):
         - Walk model.named_modules()
         - If any module has 'causal' in its name or is CausalSelfAttention → 'decoder'
         - If model has 'encoder.layer' or bidirectional attention → 'encoder'
      c) Final fallback: check config.architectures[0] string for known keywords
         - Contains 'ForCausalLM', 'GPT', 'LLama', 'OPT' → decoder
         - Contains 'ForMaskedLM', 'ForSequenceClassification', 'Bert' → encoder
      d) If STILL can't determine → raise ValueError with helpful message:
         "Could not detect architecture family. Supported: BERT-style encoders,
          GPT-style decoders. Your model: {type(model).__name__}"

 1.3  Implement _discover_backbone() — find the main transformer body
      Sub-steps:
      a) Try common backbone attribute names in order:
         model.bert, model.roberta, model.distilbert, model.albert,
         model.transformer, model.gpt_neox, model.model, model.encoder
      b) If none found: walk model.named_children() looking for the child that
         contains a ModuleList (that's the backbone with the layer stack)
      c) This is DISCOVERY, not lookup — works for any model that has a
         standard transformer body, even ones we've never seen before

 1.4  Implement _discover_layers() — find the repeating layer stack
      Sub-steps:
      a) Starting from backbone, search for a nn.ModuleList
      b) Try common paths: backbone.encoder.layer, backbone.layer,
         backbone.h, backbone.layers, backbone.transformer.layer
      c) If none of those: walk backbone.named_modules() and find the
         first ModuleList whose children each contain attention-like modules
      d) Return the ModuleList + its attribute path (for debugging)
      e) Set self.num_layers = len(discovered_layer_list)

 1.5  Implement _discover_attention(layer) — find attention within one layer
      Sub-steps:
      a) Try common attribute paths on a single layer module:
         layer.attention, layer.attn, layer.self_attn, layer.attention.self
      b) If none: search layer.named_modules() for module with 'attention' or
         'attn' in the name that has Q/K/V children
      c) Once found, discover Q/K/V projection names:
         Try: (query, key, value), (q_proj, k_proj, v_proj),
              (q_lin, k_lin, v_lin), (Wq, Wk, Wv)
         Pick whichever exists as attributes on the attention module
      d) Store discovered pattern for reuse across all layers

 1.6  Implement _discover_ffn(layer) — find FFN/MLP within one layer
      Sub-steps:
      a) Try: layer.intermediate, layer.mlp, layer.feed_forward, layer.ffn,
              layer.output.dense
      b) If none: search for module with 'mlp', 'ffn', 'intermediate' in name

 1.7  Implement _discover_layernorm(layer) — find LayerNorms within one layer
      Sub-steps:
      a) Collect ALL nn.LayerNorm instances within the layer
      b) Typically 2 per layer (post-attention LN, post-FFN LN)
      c) Return list of (name, module) tuples

 1.8  Implement _discover_embedding() — find word embedding layer
      Sub-steps:
      a) Try: backbone.embeddings.word_embeddings, backbone.embed_tokens,
              backbone.wte, backbone.embed_in
      b) Fallback: first nn.Embedding in backbone.named_modules()

 1.9  Implement find_classifier_head()
      - Try: model.classifier, model.lm_head, model.cls, model.score, model.qa_outputs
      - Return the output head module (or None if not found)

 1.10 Implement ModelInspector.__init__() — orchestrate discovery
      Sub-steps:
      a) Store model reference, extract model.config
      b) self.arch_family = self._detect_family()
      c) self.num_layers = config.num_hidden_layers or config.n_layer (config is reliable for counts)
      d) self.num_heads = config.num_attention_heads or config.n_head
      e) self.hidden_size = config.hidden_size or config.n_embd
      f) self.backbone = self._discover_backbone()
      g) self.layers = self._discover_layers()
      h) Probe layer[0] to discover attention/ffn/ln patterns:
         self._attn_pattern = self._discover_attention(self.layers[0])
         self._ffn_pattern = self._discover_ffn(self.layers[0])
         self._ln_modules = self._discover_layernorm(self.layers[0])
      i) self.embedding = self._discover_embedding()
      j) Log discovered structure for transparency

 1.11 Implement get_parameter_groups() — uses discovered patterns
      - Use discovered layers + patterns to build parameter group mapping
      - For each layer i:
          layer{i}_attention → attention module params (from _attn_pattern)
          layer{i}_qkv      → Q, K, V projection params (from _attn_pattern.qkv)
          layer{i}_ffn       → FFN/MLP params (from _ffn_pattern)
          layer{i}_layernorm → LayerNorm params (from _ln_modules)
      - Special groups: 'embedding', 'classifier'
      - Return Dict[str, List[str]] mapping group_name → list of param names
      - This replaces _layer_group_patterns() from base_metrics.py

 1.12 Implement register_attention_hooks(callback)
      - Set model.config.output_attentions = True
      - As fallback, register forward hooks on discovered attention modules
      - The callback receives (module, input, output) → stores attention weights
      - Return list of hook handles (for cleanup)

 1.13 Implement dynamic layer sampling helper
      - _get_sampled_layer_indices(strategy='early_mid_late') -> List[int]
      - Divides num_layers into 3 equal bands, picks one from each
      - E.g., 12 layers → [0, 5, 11]  (first, mid, last)
      - E.g., 6 layers → [0, 2, 5]
      - Used by attention, structural, gradient metrics to reduce overhead

 1.14 Smoke test — test CATEGORIES, not specific models
      We test with one representative per category + verify the discovery
      mechanism works, NOT that we have a registry entry for each model:
      a) BERT-style representative (bert-tiny):
         - arch_family='encoder', layers discovered, attention discovered
      b) GPT-style representative (gpt2-tiny):
         - arch_family='decoder', layers discovered, attention discovered
      c) UNSUPPORTED model (e.g., ViT, wav2vec):
         - Raises clear ValueError
      d) UNKNOWN-BUT-COMPATIBLE model test:
         - Create a dummy nn.Module that follows BERT structure
         - Inspector should still discover it even though it's not a real HF model
         - THIS proves the category approach works for future models

────────────────────────────────────────────────────────────────────────────────
STEP 2: extraction/metrics/base.py — MetricModule ABC
────────────────────────────────────────────────────────────────────────────────
Source: NEW (interface definition)
Size: ~40-50 lines
Why second: All 7 metric modules implement this interface

 2.1  Define MetricModule abstract base class
      @abstractmethod collect(self, *, loss, model, optimizer, outputs, labels,
                              attention_weights, hidden_states,
                              batch_idx, epoch, step_time) -> Dict[str, float]

      Properties (with defaults):
        requires_attention_weights -> bool = False
        requires_hidden_states     -> bool = False

      Constructor: takes ModelInspector instance, stores as self.inspector

 2.2  Define CANONICAL_FEATURES registry (Dict[str, str])
      - Maps feature_name → description
      - This is THE authoritative list that extraction MUST produce
      - Must align exactly with what FeatureProcessor expects
      - Build incrementally as each metric module is implemented
      - Final validation: set(CANONICAL_FEATURES.keys()) == set(pretrained model's expected features)

 2.3  Add helper utilities used across metric modules
      - _safe_mean(tensor) → float (handles empty/NaN)
      - _safe_std(tensor) → float
      - _safe_skew(values) → float (from scipy, with NaN fallback)
      - _safe_kurtosis(values) → float
      - _to_float(value) → float (detach, cpu, item)
      Port from: base_metrics.py lines ~980-1050 (_safe_skew, _safe_kurtosis, etc.)

────────────────────────────────────────────────────────────────────────────────
STEP 3: extraction/metrics/training.py — TrainingMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_training_metrics() (lines ~90-165)
Size: ~80-100 lines
Why third: Simplest module — loss, LR, memory. Good first test of the interface.

 3.1  Implement TrainingMetrics(MetricModule)
      Port compute_training_metrics() with these adaptations:
      a) Extract loss value: loss.item() if tensor, else float(loss)
      b) Extract learning rate: optimizer.param_groups[0]['lr']
      c) Step time: accept from kwargs (measured by caller)
      d) Memory: torch.cuda.memory_allocated() / (1024**2) if CUDA available
      e) Memory reserved: torch.cuda.memory_reserved() / (1024**2)

 3.2  Output keys (must match canonical names):
      - train_loss, loss (alias)
      - train_learning_rate
      - runtime_step_time, runtime_steps_per_sec
      - runtime_memory_alloc_mb, runtime_memory_reserved_mb

 3.3  Remove kernel fault state tracking from source
      - base_metrics.py tracks kernel_flash_enabled, kernel_fault_*_active
      - These are fault-injection-specific, NOT needed in the library
      - Skip entirely

────────────────────────────────────────────────────────────────────────────────
STEP 4: extraction/metrics/gradient.py — GradientMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_gradient_metrics() (lines ~165-280)
      + base_metrics.py::compute_update_ratio_metrics() (lines ~280-340)
Size: ~150-180 lines
Why fourth: Gradient metrics are essential for vanishing/exploding detection.

 4.1  Implement GradientMetrics(MetricModule)
      Key adaptation: Use inspector.get_parameter_groups() instead of
      base_metrics._layer_group_patterns() which hardcodes DistilBERT paths.

 4.2  Port compute_gradient_metrics() core logic
      Sub-steps:
      a) Iterate parameter groups from inspector
      b) For each group, compute: grad_norm = sqrt(sum(p.grad.norm()**2))
         Only for params where p.grad is not None
      c) Compute per-group: grad_norm_{group_name}
      d) Compute global: grad_norm_total, grad_abs_min, grad_abs_max, grad_zero_ratio
      e) Compute binary flags: gradient_vanish (total < threshold), gradient_explode (total > threshold)
      f) Thresholds from config: grad_vanish_threshold=1e-4, grad_explode_threshold=100.0

 4.3  Port compute_update_ratio_metrics() core logic
      Sub-steps:
      a) For each parameter group, compute: update_ratio = ||grad|| / (||param|| + eps)
      b) Output: update_ratio_{group_name}, update_ratio_total
      c) Also compute: update_active_{group_name} (1.0 if grad_norm > activity_threshold)

 4.4  Apply dynamic layer sampling
      - For per-layer metrics, only compute for sampled layers (early/mid/late)
      - Use inspector._get_sampled_layer_indices()
      - Still compute global aggregates (grad_norm_total) from ALL layers
      - Per-layer breakdown only for sampled indices → reduces from N layers to 3

 4.5  Output keys (canonical):
      Per-group: grad_norm_{group}, update_ratio_{group}, update_active_{group}
      Global: grad_norm_total, grad_abs_min, grad_abs_max, grad_zero_ratio
             gradient_vanish, gradient_explode, update_ratio_total

────────────────────────────────────────────────────────────────────────────────
STEP 5: extraction/metrics/attention.py — AttentionMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_attention_metrics() (lines ~440-720)
      + compute_head_similarity() (~720-780)
      + compute_positional_attention_profile() (~780-860)
Size: ~250-300 lines (MOST COMPLEX module)
Why fifth: Attention is the richest feature source, most model-specific.

 5.1  Implement AttentionMetrics(MetricModule)
      Set: requires_attention_weights = True
      Key adaptation: attention_weights come from model forward pass via
      output_attentions=True, shape differs by model type.

 5.2  Port entropy + sparsity computation
      Sub-steps:
      a) For each sampled layer's attention weights (batch, heads, seq, seq):
         - attention_entropy: -sum(a * log(a + eps)) averaged over heads
         - attention_sparsity: fraction of weights < 0.01
         - attention_max_mean: mean of max attention per query position
         - attention_weight_magnitude: mean absolute attention weight
      b) Aggregate across sampled layers: mean and std variants

 5.3  Port padding/special token mass computation
      Sub-steps:
      a) Identify pad positions from attention_mask (0 = pad)
      b) attention_mass_pad_mean: mean attention mass on pad tokens
      c) attention_mass_pad_max: max attention mass on pad tokens
      d) attention_mass_special_mean: mass on [CLS]/[SEP] or BOS/EOS
      e) attention_mass_leak: total mass on invalid positions
      f) attention_mass_leak_max: worst-case leak
      g) For decoders: pad detection via causal mask lower triangle

 5.4  Port head similarity computation
      Sub-steps:
      a) For each sampled layer, compute pairwise cosine similarity between heads
      b) head_similarity_mean, head_similarity_std, head_similarity_max
      c) Flatten heads' attention patterns to vectors, compute cosine between all pairs
      Port from: base_metrics.py::compute_head_similarity() (~60 lines)

 5.5  Port positional attention profile
      Sub-steps:
      a) For each head, compute mean received attention at each position
      b) Divide sequence into early/mid/late thirds
      c) positional_recv_early, positional_recv_mid, positional_recv_late
      d) positional_recv_mean, positional_recv_var, positional_recv_skew
      e) Ratios: positional_recv_mid_over_early, positional_recv_late_over_early
      Port from: base_metrics.py::compute_positional_attention_profile() (~80 lines)

 5.6  Port pre-softmax score statistics
      Sub-steps:
      a) Need to recompute QK^T scores — requires accessing Q, K projections
      b) Use inspector.find_attention_modules() to get attention module
      c) Use registry['qkv_names'] to find Q, K weight matrices
      d) Compute: scores = (Q @ K^T) / sqrt(d_k)
      e) pre_softmax_score_mean, pre_softmax_score_var, pre_softmax_score_skew, pre_softmax_score_kurt
      f) attention_score_var, attention_score_skew
      ADAPTATION: base_metrics._compute_pre_softmax_stats() hardcodes q_lin/k_lin (DistilBERT)
                  → use inspector registry to find correct attribute names

 5.7  Port cross-example leak detection
      - attention_cross_example_leak: detect if attention bleeds across batch examples
      - Only meaningful for padded batches
      - Compute mask of valid positions per example, check for cross-boundary attention
      Port from: base_metrics._compute_cross_example_mask() (~30 lines)

 5.8  Output keys (canonical, ~25 features):
      attention_entropy_mean, attention_entropy_std, attention_sparsity,
      attention_max_mean, attention_max_std, attention_weight_magnitude,
      attention_mass_pad_mean, attention_mass_pad_max,
      attention_mass_special_mean, attention_mass_special_std,
      attention_mass_leak, attention_mass_leak_max, attention_cross_example_leak,
      head_similarity_mean, head_similarity_std, head_similarity_max,
      positional_recv_mean, positional_recv_var, positional_recv_skew,
      positional_recv_early, positional_recv_mid, positional_recv_late,
      positional_recv_mid_over_early, positional_recv_late_over_early,
      pre_softmax_score_mean, pre_softmax_score_var,
      pre_softmax_score_skew, pre_softmax_score_kurt,
      attention_score_var, attention_score_skew

────────────────────────────────────────────────────────────────────────────────
STEP 6: extraction/metrics/structural.py — StructuralMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_structural_metrics() (lines ~860-980)
Size: ~150-180 lines
Why sixth: Depends on inspector for layer/FFN/LN module discovery.

 6.1  Implement StructuralMetrics(MetricModule)
      Set: requires_hidden_states = True
      Key adaptation: needs hidden_states from model output (output_hidden_states=True)

 6.2  Port per-layer FFN delta computation
      Sub-steps:
      a) For each sampled layer, get hidden states before and after FFN
         - hidden_states[i] = input to layer i
         - hidden_states[i+1] = output of layer i (or use hooks on FFN)
      b) ffn_delta_l{i}_mean = ||h_out - h_in|| averaged over tokens
      c) Aggregate: ffn_delta_mean across sampled layers
      ADAPTATION: Use inspector layers, not hardcoded paths

 6.3  Port residual cosine similarity
      Sub-steps:
      a) For each sampled layer:
         residual_cos_l{i}_mean = cosine_similarity(h_in, h_out) averaged over tokens
      b) Aggregate: residual_cos_mean

 6.4  Port FFN variance ratio + active dimension fraction
      Sub-steps:
      a) ffn_var_ratio_l{i} = var(h_out) / (var(h_in) + eps)
      b) ffn_active_dim_frac_l{i} = fraction of hidden dims with var > threshold
      c) ffn_out_skew_l{i} = skewness of output distribution
      d) Aggregate: ffn_var_ratio_mean, ffn_active_dim_frac_mean, ffn_out_skew_mean

 6.5  Port LayerNorm statistics
      Sub-steps:
      a) For each sampled layer, find LN modules via inspector registry['ln_paths']
      b) Hook or access LN input/output
      c) ln_std_l{i}_mean = mean of per-token std after LN
      d) ln_mean_abs_l{i}_mean = mean of per-token |mean| after LN
      e) Aggregate: ln_std_mean, ln_mean_abs_mean

 6.6  Port embedding norm computation
      Sub-steps:
      a) Use inspector.find_embedding_module() to get word_embeddings
      b) embedding_norm_mean = mean(||embedding_vector||) across vocab sample
      c) embedding_norm_std = std(||embedding_vector||)
      ADAPTATION: base_metrics hardcodes model.distilbert.embeddings.word_embeddings
                  → use inspector.find_embedding_module() instead

 6.7  Port first-layer drift metric
      - h1_delta_norm_mean = ||hidden_states[1] - hidden_states[0]|| mean
      - Simple: just uses first two hidden states

 6.8  Output keys (canonical, ~20+ features):
      Per-layer (sampled): ffn_delta_l{i}_mean, residual_cos_l{i}_mean,
          ffn_var_ratio_l{i}, ln_std_l{i}_mean, ln_mean_abs_l{i}_mean,
          ffn_active_dim_frac_l{i}, ffn_out_skew_l{i}
      Aggregated: ffn_delta_mean, residual_cos_mean, ffn_var_ratio_mean,
          ln_std_mean, ln_mean_abs_mean, ffn_active_dim_frac_mean, ffn_out_skew_mean
      Embedding: embedding_norm_mean, embedding_norm_std
      Drift: h1_delta_norm_mean

────────────────────────────────────────────────────────────────────────────────
STEP 7: extraction/metrics/logit.py — LogitMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_performance_metrics() (lines ~340-440)
Size: ~120-150 lines
Why seventh: Logit statistics are task-agnostic once you have the output tensor.

 7.1  Implement LogitMetrics(MetricModule)
      Key: Extract logits from model output (outputs.logits or outputs[0])

 7.2  Port classification performance metrics
      Sub-steps:
      a) Extract predictions: argmax(logits, dim=-1)
      b) accuracy = (preds == labels).float().mean()
      c) f1_score, precision, recall via sklearn (if available) or manual
      d) Handle both classification and regression tasks

 7.3  Port logit health metrics
      Sub-steps:
      a) logit_nan_ratio = fraction of NaN values in logits
      b) logit_inf_ratio = fraction of Inf values in logits
      c) logit_entropy = -sum(softmax * log_softmax) averaged
      d) logit_confidence_mean = max(softmax, dim=-1).mean()
      e) logit_kl_uniform = KL(softmax || uniform)
      f) nll = F.cross_entropy(logits, labels)

 7.4  Port ECE (Expected Calibration Error) computation
      Sub-steps:
      a) Bin predictions by confidence (15 bins)
      b) Per-bin: |accuracy - confidence| weighted by bin count
      c) ece = weighted average across bins
      Port from: base_metrics._compute_ece() (~30 lines)

 7.5  Port logit margin statistics
      Sub-steps:
      a) For each example: margin = top1_prob - top2_prob
      b) logit_margin_mean, logit_margin_var
      c) logit_margin_p25, logit_margin_p50, logit_margin_p75, logit_margin_min
      Port from: base_metrics._compute_logit_margin_stats() (~20 lines)

 7.6  Output keys (canonical, ~20 features):
      accuracy, f1_score, precision, recall,
      logit_nan_ratio, logit_inf_ratio, nll, ece,
      logit_entropy, logit_confidence_mean, logit_kl_uniform,
      logit_margin_mean, logit_margin_var,
      logit_margin_p25, logit_margin_p50, logit_margin_p75, logit_margin_min

────────────────────────────────────────────────────────────────────────────────
STEP 8: extraction/metrics/positional.py — PositionalMetrics
────────────────────────────────────────────────────────────────────────────────
Source: base_metrics.py::compute_positional_performance_metrics() (lines ~400-440)
Size: ~80-100 lines

 8.1  Implement PositionalMetrics(MetricModule)
      Key: Splits sequence into early/late windows and computes per-window task performance

 8.2  Port early/late masking logic
      Sub-steps:
      a) Determine sequence length from outputs or labels shape
      b) Split into thirds: early = [0, L/3), mid = [L/3, 2L/3), late = [2L/3, L)
      c) Create position masks for each window
      d) Mask labels + logits to each window

 8.3  Port per-window performance computation
      Sub-steps:
      a) For each window (early, late):
         - positional_accuracy_{window} = accuracy on masked subset
         - positional_margin_{window} = mean logit margin on subset
         - positional_loss_{window} = CE loss on subset
      b) Deltas: positional_accuracy_delta = late - early, positional_margin_delta = late - early

 8.4  Output keys (canonical, 8 features):
      positional_accuracy_early, positional_accuracy_late, positional_accuracy_delta,
      positional_margin_early, positional_margin_late, positional_margin_delta,
      positional_loss_early, positional_loss_late

────────────────────────────────────────────────────────────────────────────────
STEP 9: extraction/metrics/cache.py — CacheMetrics (Decoder-only)
────────────────────────────────────────────────────────────────────────────────
Source: NEW (no direct source — decoder-specific feature)
Size: ~60-80 lines

 9.1  Implement CacheMetrics(MetricModule)
      Only instantiated for decoder models (inspector.arch_family == "decoder")

 9.2  Implement KV-cache hidden similarity
      Sub-steps:
      a) Run model with use_cache=True to get past_key_values
      b) For sampled layers, compute cosine similarity between consecutive key vectors
      c) cache_hidden_sim = mean cosine similarity (high = repetitive generation)

 9.3  [REMOVED] cache_nll_divergence — see Phase 1.6 Bugfix 2
      Originally planned: compare loss with/without KV-cache.
      Removed: requires 2× forward passes (prohibitive for transformers).
      cache_hidden_sim is sufficient for cache health diagnosis.

 9.4  Output keys (canonical, 1 feature):
      cache_hidden_sim
      (Keep minimal — supplementary decoder diagnostic)

────────────────────────────────────────────────────────────────────────────────
STEP 10: extraction/collector.py — MetricCollector
────────────────────────────────────────────────────────────────────────────────
Source: metric_collector.py::MetricCollector (last_project_phd)
Size: ~120-150 lines
Why tenth: Orchestrates all metric modules built in Steps 3-9.

 10.1 Implement MetricCollector.__init__(inspector, config=None)
      Sub-steps:
      a) Instantiate all metric modules based on arch_family:
         - Always: TrainingMetrics, GradientMetrics, AttentionMetrics,
                   StructuralMetrics, LogitMetrics, PositionalMetrics
         - Decoder only: CacheMetrics
      b) Create EpochAggregator instance
      c) Store config: activation_interval, gradient_window, etc.
      d) Initialize step counter, batch history list

 10.2 Implement collect_step(**kwargs) -> Dict[str, float]
      Sub-steps:
      a) Accept a `batch` parameter (Dict[str, Tensor]) in addition to the
         individual kwargs (input_ids, attention_mask, labels, etc.).
         The batch dict is needed by curvature metrics (Phase 1.5) which
         must run model(**batch) at perturbed weights. If batch is not
         provided, reconstruct it from individual kwargs.
      b) Prepare model outputs:
         - If outputs has attention weights → pass to modules needing them
         - If outputs has hidden states → pass to modules needing them
         - If neither, set output_attentions=True / output_hidden_states=True on config
      c) Call each module.collect() and merge results
      d) Pass merged metrics to aggregator.update()
      e) Increment step counter
      f) Return merged metrics dict

      Optimization: only call attention/structural modules every `activation_interval`
      steps (these are expensive). Training + gradient collected every step.

 10.3 Implement finalize_epoch(epoch_idx) -> Dict[str, float]
      - Delegate to aggregator.finalize_epoch(epoch_idx)
      - Store result in epoch_history
      - Reset aggregator for next epoch
      - Return epoch-level features (mean/var per metric)

 10.4 Implement get_final_features(epoch_history) -> (np.ndarray, List[str])
      Sub-steps:
      a) Compute windowed features from epoch history
      b) Compute final-epoch features
      c) Compute best-of-training features (best_train_loss, best_val_accuracy, etc.)
      d) Concatenate into single feature vector
      e) Return (feature_vector, feature_names) — names MUST match canonical registry
      Port logic from: metric_collector.py::get_final_metrics()

 10.5 Implement feature_names property
      - Returns ordered list of all feature names the collector produces
      - Must be deterministic (always same order)
      - Used for alignment validation against pretrained model

────────────────────────────────────────────────────────────────────────────────
STEP 11: extraction/aggregator.py — Welford + Windowed Features
────────────────────────────────────────────────────────────────────────────────
Source: statistics.py (OnlineStatistic, EpochAggregator, compute_window_features)
      + running_metrics.py (RunningMetrics)
Size: ~150-180 lines

 11.1 Port OnlineStatistic dataclass (Welford's algorithm)
      Sub-steps:
      a) Fields: count, mean, m2
      b) update(value): skip NaN/inf, apply Welford formula
      c) variance property: m2 / (count - 1) if count > 1 else 0
      d) reset(): zero all fields
      Direct port from: statistics.py (~40 lines)

 11.2 Port EpochAggregator class
      Sub-steps:
      a) Maintains Dict[str, OnlineStatistic] for current epoch
      b) update(metrics): for each key, update the OnlineStatistic
      c) finalize_epoch(epoch_idx): return {key_mean, key_var} for all metrics
      d) Store history: Dict[str, List[Tuple[int, float]]] for windowing
      e) reset(): clear current epoch stats
      Direct port from: statistics.py (~60 lines)

 11.3 Port compute_window_features(metric_history, total_epochs)
      Sub-steps:
      a) Define windows: early=(1,3), mid=(4,7), late=(8,10)
      b) For each metric in history:
         - {metric}_early_mean, {metric}_early_slope
         - {metric}_mid_mean, {metric}_mid_slope
         - {metric}_late_mean, {metric}_late_slope (if applicable)
         - {metric}_final (last epoch value)
      c) Slope via np.polyfit(x, y, 1)[0]
      d) Handle edge cases: < 3 epochs, missing data
      Direct port from: statistics.py (~50 lines)

 11.4 Port RunningMetrics (rolling-window gradient stats)
      Sub-steps:
      a) Window implemented via collections.deque(maxlen=window_size)
      b) update(name, value): append to deque
      c) get_variance(name), get_mean(name), get_noise_scale(name)
      d) noise_scale = variance / (|mean| + eps) — gradient noise proxy
      Direct port from: running_metrics.py (~50 lines)

────────────────────────────────────────────────────────────────────────────────
STEP 12: extraction/export.py — Feature Vector Export
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~50-60 lines

 12.1 Implement export_to_dataframe(feature_vector, feature_names) -> pd.DataFrame
      - Single row DataFrame with feature names as columns
      - Validates length match

 12.2 Implement export_to_csv(feature_vector, feature_names, path)
      - Writes DataFrame to CSV
      - Useful for debugging / offline diagnosis

 12.3 Implement export_to_dict(feature_vector, feature_names) -> Dict
      - Simple dict mapping name → value
      - For programmatic access

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1 GATE TEST — Must pass before Phase 1 is considered complete         │
│                                                                              │
│ tests/test_phase1_gate.py:                                                   │
│                                                                              │
│ Category discovery tests (test the DISCOVERY mechanism, not model names):    │
│  T1.1  BERT-style category: ModelInspector(bert-tiny)                       │
│         → arch_family='encoder', discovers layers, attention, FFN, LN       │
│  T1.2  GPT-style category: ModelInspector(gpt2-tiny)                        │
│         → arch_family='decoder', discovers layers, attention, MLP, LN       │
│  T1.3  Discovery works on VARIANT within same category:                     │
│         ModelInspector(distilbert) also resolves as 'encoder'               │
│         despite different internal paths (transformer.layer vs encoder.layer)│
│  T1.4  find_attention_modules() returns correct count (== num_layers)       │
│  T1.5  find_embedding_module() returns nn.Embedding instance                │
│  T1.6  get_parameter_groups() has 'embedding', 'layer0_attention', etc.     │
│  T1.7  Unsupported model (e.g., ViT) raises clear ValueError               │
│  T1.7b UNKNOWN-BUT-COMPATIBLE: custom nn.Module with BERT-like structure    │
│         → inspector discovers it without any registry entry (proves auto)   │
│                                                                              │
│ Metric module tests (run with real model forward pass):                      │
│  T1.8  TrainingMetrics.collect() returns dict with 'train_loss'             │
│  T1.9  GradientMetrics.collect() returns 'grad_norm_total', no NaN          │
│  T1.10 AttentionMetrics.collect() returns 'attention_entropy_mean'           │
│  T1.11 StructuralMetrics.collect() returns 'ffn_delta_mean'                 │
│  T1.12 LogitMetrics.collect() returns 'accuracy', 'logit_entropy'           │
│  T1.13 PositionalMetrics.collect() returns 'positional_accuracy_early'      │
│  T1.14 CacheMetrics only instantiated for decoder, NOT for encoder          │
│  T1.15 No metric module returns NaN under normal (non-faulty) input         │
│                                                                              │
│ Collector integration tests:                                                 │
│  T1.16 MetricCollector.collect_step() returns >50 metrics, all floats       │
│  T1.17 MetricCollector.finalize_epoch() returns keys ending in _mean/_var   │
│  T1.18 MetricCollector.get_final_features() returns (ndarray, names_list)   │
│  T1.19 feature_names are deterministic (same across 3 calls)                │
│  T1.20 No duplicate feature names in feature_names list                     │
│                                                                              │
│ Aggregator tests:                                                            │
│  T1.21 OnlineStatistic matches numpy mean/var on 100 random values          │
│  T1.22 EpochAggregator produces {key}_mean, {key}_var for each input key   │
│  T1.23 compute_window_features produces early/mid/late/final keys           │
│                                                                              │
│ Run: pytest tests/test_phase1_gate.py -v --timeout=120                       │
│ All tests must PASS. If any fail → fix before proceeding.                    │
│ (Phase 2 can proceed in parallel since it's independent.)                    │
│                                                                              │
│ NOTE: Phase 1.6 renames some keys (mass_pad → attention_pad_mass_max,       │
│ removes 'loss' alias). When Phase 1.6 is implemented, these gate tests      │
│ MUST be updated to assert the new key names. See Phase 1.6 BUGFIX 5         │
│ migration task for the exact renaming table.                                 │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 1.5: ENHANCED DIAGNOSTIC METRICS (15 NEW METRICS)
================================================================================
Goal: Add 15 new metric groups across 4 existing modules + 1 new module,
      filling diagnostic blind spots and porting DEFault coverage gaps.
      Each targets a specific failure mode that no existing metric captures.
Dependencies: Phase 1 complete (all 7 metric modules exist)
Estimated size: ~400 lines of implementation + ~500 lines of tests

These metrics were identified by analyzing what transformer training pathologies
the current feature set CANNOT distinguish between. Each metric below includes:
  - The invariant mathematical/theoretical definition
  - Encoder vs decoder specifics
  - Implementation details
  - Correctness verification strategy

────────────────────────────────────────────────────────────────────────────────
METRIC 1: grad_norm_ratio_first_last — Gradient Flow Through Depth
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
Diagnoses: Vanishing/exploding gradients THROUGH the network depth
Why needed: grad_norm_total tells you overall gradient magnitude but NOT whether
            gradients survive from output layers back to input layers. A model
            can have healthy grad_norm_total but catastrophic vanishing at early
            layers if later layers have compensatingly large gradients.

 1.5.1  Theory (invariant — same for all architectures)
        ═══════════════════════════════════════════════
        For a network with L layers, define:
          g_first = ||∂L/∂W_0||₂  (gradient norm of first transformer layer)
          g_last  = ||∂L/∂W_{L-1}||₂  (gradient norm of last transformer layer)

        grad_norm_ratio_first_last = g_first / (g_last + ε)

        Interpretation:
          ratio ≈ 1.0  → healthy gradient flow (gradients propagate evenly)
          ratio → 0    → vanishing through depth (early layers not learning)
          ratio >> 1   → exploding through depth (rare but indicates instability)

        This is the discrete analogue of the "gradient flow" diagnostic from
        Pascanu et al. (2013) "On the difficulty of training RNNs", adapted
        for transformer architectures.

        Theoretical grounding: In a well-initialized transformer with residual
        connections, the Jacobian ∂h_l/∂h_{l-1} ≈ I + small perturbation,
        so gradient norms should be approximately preserved across layers.
        Deviation from ratio ≈ 1 indicates broken residual paths or
        pathological weight scaling.

 1.5.2  Encoder specifics (BERT-style)
        - g_first = grad norm of layer 0 parameters (discovered via inspector)
        - g_last = grad norm of layer (num_layers-1) parameters
        - BERT models with well-tuned LR schedule typically show ratio 0.5–2.0
        - Pre-LayerNorm vs Post-LayerNorm affects expected ratio range

 1.5.3  Decoder specifics (GPT-style)
        - Same computation, but decoder layers are numbered differently
        - Causal models sometimes show stronger gradient attenuation due to
          the triangular mask reducing effective gradient paths
        - Expected healthy range may be slightly lower (0.3–1.5)

 1.5.4  Implementation
        In GradientMetrics._compute_gradient_norms():
        a) Already computing per-group grad norms: grad_norm_layer{i}_attention
        b) After the per-group loop, extract:
           first_layer_norm = metrics.get('grad_norm_layer0_attention', 0)
           last_idx = self.inspector.num_layers - 1
           last_layer_norm = metrics.get(f'grad_norm_layer{last_idx}_attention', 0)
        c) Compute: grad_norm_ratio_first_last = first_layer_norm / (last_layer_norm + 1e-12)
        d) Add to metrics dict

 1.5.5  Correctness verification
        Test: test_grad_norm_ratio_first_last
        a) Create a tiny BERT model, run forward+backward
        b) Assert 'grad_norm_ratio_first_last' in result
        c) Assert result is a finite positive float
        d) Assert it equals manual computation:
           - Extract grad norms of first and last layer params manually
           - Compare to reported ratio within tolerance 1e-6
        e) Sanity check: for a healthy randomly-initialized model,
           ratio should be in [0.01, 100] (not exactly 0 or inf)

────────────────────────────────────────────────────────────────────────────────
METRIC 2: grad_cosine_successive — Gradient Direction Stability
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
Diagnoses: Oscillation, bad learning rate, sharp loss landscape
Why needed: A model can have stable gradient norms but wildly oscillating
            gradient DIRECTIONS. This causes the optimizer to "jitter" without
            making progress. Current metrics miss this entirely.

 1.5.6  Theory (invariant)
        ═════════════════
        Let g_t = flatten(all gradients at step t) ∈ ℝ^N
        Let g_{t-1} = flatten(all gradients at step t-1) ∈ ℝ^N

        grad_cosine_successive = cos(g_t, g_{t-1})
                               = (g_t · g_{t-1}) / (||g_t|| · ||g_{t-1}|| + ε)

        Interpretation:
          cos ≈ 1.0   → gradient direction is stable (smooth optimization)
          cos ≈ 0.0   → gradient direction is random (high-curvature region)
          cos < 0     → gradient direction is REVERSING (oscillation, too-high LR)

        Theoretical grounding: This is the "gradient interference" measure from
        multi-task learning (Yu et al., 2020 "Gradient Surgery for Multi-Task
        Learning"), applied to successive steps. It's also related to the
        gradient noise scale (McCandlish et al., 2018) — high noise scale
        implies low cosine similarity between steps.

        Important: This requires storing the previous step's gradient vector.
        Memory cost = O(num_parameters) on CPU in float32.

 1.5.7  Encoder vs Decoder
        - Identical computation for both architectures
        - Decoders with autoregressive loss may show slightly lower cosine
          due to position-dependent loss contribution
        - Both should show cos > 0.3 during stable training

 1.5.8  Implementation
        a) Add self._previous_grad: Optional[torch.Tensor] = None to __init__
        b) In _compute_gradient_norms(), after computing all grad norms:
           - Flatten all param.grad into a single vector (CPU, float32)
           - If self._previous_grad is not None:
             cosine = F.cosine_similarity(current_grad.unsqueeze(0),
                                           self._previous_grad.unsqueeze(0)).item()
             metrics['grad_cosine_successive'] = cosine
           - Else: metrics['grad_cosine_successive'] = 0.0
           - self._previous_grad = current_grad_flat.clone()

 1.5.9  Correctness verification
        Test: test_grad_cosine_successive
        a) Run 2 forward+backward passes on the same model
        b) First call: assert 'grad_cosine_successive' == 0.0 (no previous)
        c) Second call: assert it's a finite float in [-1, 1]
        d) Determinism check: same data on 2 consecutive steps should yield
           cosine close to 1.0 (identical inputs → identical gradients)
        e) Direction-flip check: negate all labels → cosine should drop
           significantly (different loss → different gradient direction)

────────────────────────────────────────────────────────────────────────────────
METRIC 3: dead_neuron_frac — Dead Neuron Detection in FFN
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/structural.py (StructuralMetrics)
Diagnoses: Dying ReLU / dead FFN units
Why needed: A neuron whose activation is near-zero for ALL inputs in a batch
            contributes nothing to the model's computation. This is a distinct
            fault from weight collapse — weights may be non-zero but the
            activation function (GELU/ReLU) gates them to zero. Current metrics
            track variance ratios but NOT whether individual neurons are dead.

 1.5.10 Theory (invariant)
        ════════════════════
        For layer l, let FFN have intermediate dimension d_ff.
        Let a_l ∈ ℝ^{B×S×d_ff} be the intermediate activations after the
        activation function (GELU/ReLU) across all B*S tokens in the batch.

        For each neuron j ∈ [0, d_ff):
          max_activation_j = max_{b,s} |a_l[b, s, j]|

        dead_neuron_frac_l{i} = (# neurons where max_activation_j < τ) / d_ff

        where τ = dead_neuron_threshold (default: 1e-6)

        Aggregated: dead_neuron_frac_mean = mean across sampled layers

        Interpretation:
          frac ≈ 0     → all neurons active (healthy)
          frac > 0.1   → 10%+ neurons are dead (concerning)
          frac > 0.3   → severe dead neuron problem (model is wasting capacity)

        Theoretical grounding: Dying ReLU problem (Lu et al., 2019 "Dying ReLU
        and Initialization"). GELU partially mitigates this but does not
        eliminate it — neurons can still effectively die if their input
        distribution shifts entirely negative.

 1.5.11 Encoder specifics
        - BERT uses GELU activation in FFN → fewer truly dead neurons than ReLU
          but still possible when weights push pre-activation to very negative
        - Intermediate dimension is typically 4× hidden_size
        - Use inspector.get_ffn_module(layer_idx) to locate the FFN
        - Hook the intermediate activation (output of first Linear + activation fn)

 1.5.12 Decoder specifics
        - GPT-2 uses GELU in MLP; GPT-NeoX may use SiLU
        - Same computation, but FFN module path differs (mlp vs intermediate)
        - Some decoder FFNs use gated architecture (SwiGLU) — in that case,
          hook after the gating multiplication

 1.5.13 Implementation
        a) Add dead_neuron_threshold to ExtractionConfig (default: 1e-6)
        b) In StructuralMetrics.collect(), after the per-layer loop:
           For each sampled layer (using inspector.get_sampled_layer_indices()):
           - Get FFN module via inspector.get_ffn_module(layer_idx)
           - Register a forward hook to capture intermediate activations
           - After the hook fires (from the existing forward pass):
             intermediate_acts = captured activation  # [batch, seq, d_ff]
             max_per_neuron = intermediate_acts.abs().amax(dim=(0, 1))  # [d_ff]
             dead = (max_per_neuron < threshold).float().mean().item()
             metrics[f'dead_neuron_frac_l{layer_idx}'] = dead
           - Remove hook after use
        c) Aggregate: dead_neuron_frac_mean = mean across sampled layers

        NOTE ON RESIDUAL DELTA APPROXIMATION (DO NOT USE):
        An earlier draft suggested approximating dead neurons via
        hidden_states[i+1] - hidden_states[i]. This is IMPRECISE because
        the delta includes attention + FFN + residual combined. A dimension
        with zero delta could mean the FFN was alive but attention canceled
        its contribution — giving a false positive. USE THE HOOK-BASED
        method above for the 3 sampled layers (3 hooks = negligible cost).

 1.5.14 Correctness verification
        Test: test_dead_neuron_frac
        a) Normal model: assert dead_neuron_frac_mean is in [0, 1]
        b) Normal model: assert dead_neuron_frac_mean < 0.5
           (randomly initialized model should NOT have >50% dead neurons)
        c) Manual verification: for a sampled layer, manually compute:
           - Get hidden_states[i] and hidden_states[i+1]
           - Compute delta = h_out - h_in
           - Count dimensions where delta.abs().max(dim=0) < threshold
           - Compare to reported metric
        d) Stress test: create a model with zeroed-out FFN weights →
           dead_neuron_frac should be close to 1.0

────────────────────────────────────────────────────────────────────────────────
METRIC 4: representation_rank — Effective Rank of Hidden States
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/structural.py (StructuralMetrics)
Diagnoses: Rank collapse (representations occupy a low-dimensional subspace)
Why needed: ffn_var_ratio and cosine similarity tell you about relative changes
            between layers, but NOT about the absolute dimensionality of the
            representation space. A model can have healthy variance ratios but
            all representations lie on a 3D subspace of a 768D space.

 1.5.15 Theory (invariant)
        ════════════════════
        For hidden state matrix H ∈ ℝ^{N×d} (N = batch×seq tokens, d = hidden):
        Compute SVD: H = UΣV^T, where σ_1 ≥ σ_2 ≥ ... ≥ σ_d ≥ 0

        Normalize singular values to a probability distribution:
          p_i = σ_i / Σ_j σ_j

        Effective rank (Roy & Vetterli, 2007):
          erank(H) = exp(-Σ_i p_i log(p_i)) = exp(Shannon entropy of p)

        Properties:
          - erank ∈ [1, d]: always between 1 and the ambient dimension
          - erank = d when all singular values are equal (full rank)
          - erank = 1 when a single singular value dominates (rank-1 collapse)

        representation_rank_l{i} = erank(H_l) / d
          Normalized to [0, 1] for comparability across model sizes.

        Aggregated: representation_rank_mean = mean across sampled layers

        Interpretation:
          rank ≈ 1.0   → full rank representations (healthy)
          rank < 0.3   → severe rank collapse (most capacity unused)
          rank < 0.1   → catastrophic collapse

        Theoretical grounding: Rank collapse in transformers is documented in
        Dong et al. (2021) "Attention is Not All You Need" — they show that
        without residual connections, self-attention converges to rank-1.
        Our metric directly measures this phenomenon.

 1.5.16 Encoder vs Decoder
        - Identical computation for both
        - BERT-style encoders with bidirectional attention tend to maintain
          higher rank than GPT-style decoders due to richer token interactions
        - Expected healthy range: encoder 0.3–0.8, decoder 0.2–0.6

 1.5.17 Implementation
        In StructuralMetrics.collect(), for each sampled layer:
        a) H = hidden_states[layer_idx].reshape(-1, hidden_size)  # [N, d]
        b) If N > probe_tokens: subsample to probe_tokens rows
        c) Center: H = H - H.mean(dim=0)
        d) Compute SVD: _, S, _ = torch.linalg.svd(H, full_matrices=False)
        e) Normalize: p = S / (S.sum() + 1e-12)
        f) Remove zeros: p = p[p > 1e-12]
        g) entropy = -(p * p.log()).sum().item()
        h) erank = math.exp(entropy)
        i) metrics[f'representation_rank_l{layer_idx}'] = erank / hidden_size
        j) Collect across layers → representation_rank_mean

        Performance note: SVD on a (256 × d) matrix is fast — the bottleneck
        is the matrix size, and we already limit to probe_tokens=256 rows.
        For hidden_size=768, this is a 256×768 SVD → ~1ms on CPU.

 1.5.18 Correctness verification
        Test: test_representation_rank
        a) Normal model: assert representation_rank_mean in (0, 1]
        b) Construct pathological case:
           - Create hidden states where all tokens are identical →
             representation_rank should be near 1/d (≈0.001)
        c) Construct full-rank case:
           - Create hidden states from random Gaussian →
             representation_rank should be near 1.0
        d) Verify formula: manually compute SVD, entropy, exp(entropy)/d
           and compare to reported metric within tolerance 1e-4

────────────────────────────────────────────────────────────────────────────────
METRIC 5: token_isotropy — Representation Degeneration Measure
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/structural.py (StructuralMetrics)
Diagnoses: Anisotropy / representation degeneration
Why needed: Distinct from rank — a representation can be high-rank but
            anisotropic (all vectors pointing in similar directions). This is
            the "representation degeneration problem" documented in Ethayarajh
            (2019) for contextual embeddings. residual_cos measures layer-to-
            layer similarity; this measures WITHIN-layer token diversity.

 1.5.19 Theory (invariant)
        ════════════════════
        For hidden state matrix H ∈ ℝ^{N×d}:
        Sample M pairs of token vectors (h_i, h_j) where i ≠ j.

        token_isotropy_l{i} = (1/M) Σ_{(i,j)} cos(h_i, h_j)

        This is the average pairwise cosine similarity between token
        representations within a single layer.

        Interpretation:
          isotropy ≈ 0     → isotropic (uniformly distributed directions — healthy)
          isotropy → 1.0   → anisotropic (all tokens similar — degenerated)
          isotropy < 0     → unusual but possible (tokens are anti-correlated)

        Theoretical grounding: Ethayarajh (2019) "How Contextual are
        Contextualized Word Representations?" showed that BERT's upper layers
        produce highly anisotropic representations (avg cosine > 0.95),
        concentrating in a narrow cone. This limits the model's expressive
        power and correlates with poor downstream performance.

        Note: We compute AVERAGE pairwise cosine, not full pairwise matrix.
        Sampling M=min(1000, N*(N-1)/2) pairs is sufficient for a stable estimate.

 1.5.20 Encoder vs Decoder
        - Encoder: all tokens attend to all others, so isotropy measures
          how much the bidirectional context "homogenizes" representations
        - Decoder: causal mask means early tokens see less context, so
          early tokens tend to be more similar → higher isotropy expected
        - Both architectures: isotropy typically INCREASES with depth
          (deeper layers produce more anisotropic representations)

 1.5.21 Implementation
        In StructuralMetrics.collect(), for each sampled layer:
        a) H = hidden_states[layer_idx].reshape(-1, hidden_size)  # [N, d]
        b) If N > probe_tokens: subsample
        c) Normalize: H_norm = F.normalize(H, dim=-1)
        d) Compute pairwise cosine via matrix multiply: C = H_norm @ H_norm.T
        e) Extract upper triangle (exclude diagonal):
           mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
           avg_cosine = C[mask].mean().item()
        f) metrics[f'token_isotropy_l{layer_idx}'] = avg_cosine

        If N is too large for N×N matrix: sample 100 random pairs instead.

        Aggregated: token_isotropy_mean = mean across sampled layers

 1.5.22 Correctness verification
        Test: test_token_isotropy
        a) Normal model: assert token_isotropy_mean in [-1, 1]
        b) Identical tokens: create hidden states where all tokens are the
           same vector → isotropy should be exactly 1.0
        c) Orthogonal tokens: create hidden states from orthonormal basis →
           isotropy should be near 0.0
        d) Random Gaussian tokens: isotropy should be near 0.0
           (high-dimensional random vectors are approximately orthogonal)
        e) Verify against manual computation: sample 10 pairs, compute
           cosine manually, compare to the mean

────────────────────────────────────────────────────────────────────────────────
METRIC 6: attention_sink_score — Attention Sink Pathology
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/attention.py (AttentionMetrics)
Diagnoses: Attention sink pattern (pathological attention to position 0)
Why needed: attention_entropy captures overall attention distribution shape
            but NOT whether attention is concentrated on a SPECIFIC position.
            A head can have low entropy (peaked) and be perfectly healthy if
            it's attending to relevant tokens. But if it's always attending
            to position 0 regardless of content, that's a pathology.

 1.5.23 Theory (invariant)
        ════════════════════
        For attention weights A ∈ ℝ^{heads×seq_q×seq_k}:

        attention_sink_score = mean over heads of: A[:, :, 0].mean()
          = average attention mass on key position 0 across all queries

        More precisely, for each head h and each query position q:
          sink_mass_h_q = A[h, q, 0]
        Then: attention_sink_score = mean_{h,q} sink_mass_h_q

        For uniform attention over seq_len positions:
          expected_score = 1 / seq_len ≈ 0.008 for seq_len=128

        Interpretation:
          score ≈ 1/seq_len  → no sink (healthy, uniform-ish attention)
          score > 5/seq_len  → mild sink (5× expected, some heads fixated)
          score > 0.3        → severe sink (30%+ of ALL attention goes to pos 0)

        Theoretical grounding: Xiao et al. (2024) "Efficient Streaming Language
        Models with Attention Sinks" documented that transformer LMs learn to
        dump attention on the first token as a "garbage collector" for unused
        attention budget. This is harmless during normal inference but indicates
        a model that hasn't learned meaningful attention patterns — critical for
        fault diagnosis.

 1.5.24 Encoder specifics
        - Position 0 is typically [CLS] token
        - Some attention to [CLS] is EXPECTED and healthy (CLS is designed to
          aggregate information)
        - The metric should compare against the special token's expected mass
        - Sink becomes pathological when ALL heads concentrate on [CLS],
          not just the aggregation heads

 1.5.25 Decoder specifics
        - Position 0 is typically BOS or the first real token
        - Due to causal masking, position 0 key is visible to ALL queries
          but later positions are only visible to subsequent queries
        - This creates a natural bias toward position 0 that must be
          accounted for: normalize by the expected mass under uniform
          causal attention = Σ_q (1/(q+1)) / seq_len

 1.5.26 Implementation
        In AttentionMetrics._compute_layer_metrics():
        a) After computing attn = attention_weights.detach().float():
           # attn shape: [batch, heads, seq_q, seq_k]
           sink_mass = attn[:, :, :, 0]  # [batch, heads, seq_q]
           sink_score = sink_mass.mean().item()
           metrics['attention_sink_score'] = sink_score
        b) In the global alias aggregation (collect method):
           Accumulate across sampled layers → take mean for global alias

 1.5.27 Correctness verification
        Test: test_attention_sink_score
        a) Normal model: assert 'attention_sink_score' in result
        b) Assert value is in [0, 1]
        c) Construct uniform attention: all weights = 1/seq_len →
           sink_score should equal 1/seq_len within tolerance
        d) Construct sink attention: set A[:, :, :, 0] = 0.99 →
           sink_score should be ≈ 0.99
        e) Per-layer verification: check L{i}_attention_sink_score exists
           for each sampled layer

────────────────────────────────────────────────────────────────────────────────
METRIC 7: dead_head_count — Dead Attention Head Detection
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/attention.py (AttentionMetrics)
Diagnoses: Attention heads that contribute nothing to computation
Why needed: head_similarity tells you if heads are REDUNDANT (computing the
            same thing). dead_head_count tells you if heads are INACTIVE
            (computing nothing at all). A dead head has near-uniform attention
            (maximum entropy) — it attends equally everywhere, which is
            equivalent to averaging all values with equal weights, producing
            the same output regardless of content.

 1.5.28 Theory (invariant)
        ════════════════════
        For each attention head h in layer l:
        Let H(h) = -Σ_k A[h, :, k] log A[h, :, k], averaged over queries
        (this is the per-head entropy we already compute)

        max_entropy = log(seq_len)

        A head is "dead" if: H(h) > α × max_entropy
        where α = dead_head_entropy_threshold (default: 0.95)

        dead_head_count = # heads where H(h) > α × max_entropy
        dead_head_frac = dead_head_count / num_heads

        Interpretation:
          frac ≈ 0     → all heads are selectively attending (healthy)
          frac > 0.25  → >25% of heads are uniform (wasting capacity)
          frac > 0.5   → majority of heads are dead (severe)

        Note on "uniform as dead": A truly uniform head computes:
          output = (1/N) Σ_k V_k = mean of all value vectors
        This is a constant function of the query, providing no position-
        specific information. It's equivalent to removing the head entirely.

        There's a subtle distinction: some heads are INTENTIONALLY broad
        (attending to many positions). These have high entropy but not
        MAXIMUM entropy. The threshold α=0.95 catches only near-uniform heads.

 1.5.29 Encoder vs Decoder
        - Encoder: max_entropy = log(seq_len) for all heads
        - Decoder: due to causal mask, max_entropy for head at query position q
          is log(q+1), which varies by position. Must compute per-position
          max entropy and average: effective_max_entropy = mean_q log(q+1)
        - In practice, using log(seq_len) for both is a reasonable
          approximation since most queries see most of the sequence

 1.5.30 Implementation
        In AttentionMetrics._compute_layer_metrics():
        a) Already computed: head_entropy = per-head mean entropy [num_heads]
        b) max_ent = math.log(attn.size(-1))  # log(seq_len)
        c) threshold = 0.95 * max_ent
        d) dead_count = (head_entropy > threshold).sum().item()
        e) metrics['dead_head_count'] = dead_count
           metrics['dead_head_frac'] = dead_count / attn.size(1)
        f) In global aliases: accumulate across sampled layers → take max

 1.5.31 Correctness verification
        Test: test_dead_head_count
        a) Normal model: assert dead_head_count >= 0
        b) Normal model: assert dead_head_frac in [0, 1]
        c) Construct uniform attention: set A = 1/seq_len everywhere →
           ALL heads should be detected as dead → dead_head_frac = 1.0
        d) Construct peaked attention: set A[:, :, :, 0] = 1.0, rest = 0 →
           NO heads should be dead → dead_head_frac = 0.0
        e) Verify threshold: create attention where one head is 96%
           uniform and one is 90% uniform → only the 96% one is counted

────────────────────────────────────────────────────────────────────────────────
METRIC 8: loss_spike_ratio — Training Instability Detection
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/training.py (TrainingMetrics)
Diagnoses: Sudden loss spikes during training
Why needed: Raw loss value tells you the current loss but NOT whether it's
            abnormally high relative to recent history. A loss of 2.5 could
            be perfectly normal at epoch 1 but catastrophic at epoch 10 if
            the model was at 0.3. Current metrics have no mechanism to detect
            sudden regressions.

 1.5.32 Theory (invariant)
        ════════════════════
        Maintain an exponential moving average (EMA) of the loss:
          EMA_t = β × EMA_{t-1} + (1-β) × loss_t
          where β = 0.99 (smoothing factor)

        loss_spike_ratio = loss_t / (EMA_t + ε)

        Interpretation:
          ratio ≈ 1.0   → loss is tracking its recent average (stable)
          ratio > 2.0   → loss is 2× its recent average (mild spike)
          ratio > 5.0   → loss is 5× its recent average (severe spike)
          ratio < 0.5   → loss dropped sharply (suspicious — possible bug)

        Theoretical grounding: Loss spikes are a common failure mode in
        transformer training, documented in Chowdhery et al. (2022) "PaLM"
        and Zhang et al. (2022) "OPT". They occur due to:
        - Pathological batches (outlier data)
        - Numerical instability (fp16 overflow)
        - Learning rate too high for current loss landscape
        - Gradient accumulation bugs

        The EMA approach is preferred over a fixed-window average because:
        1. It's O(1) memory (just one scalar)
        2. It naturally adapts to the current loss scale
        3. β=0.99 gives an effective window of ~100 steps

 1.5.33 Encoder vs Decoder
        - Identical computation for both
        - Decoder LM loss tends to be higher (cross-entropy over large vocab)
          but the RATIO is scale-independent, making it comparable
        - Both should show ratio < 2.0 during stable training

 1.5.34 Implementation
        a) Add self._loss_ema: Optional[float] = None to TrainingMetrics.__init__
           Add self._loss_ema_beta: float = 0.99
        b) In TrainingMetrics.collect(), after computing train_loss:
           if self._loss_ema is None:
               self._loss_ema = metrics['train_loss']
               metrics['loss_spike_ratio'] = 1.0
           else:
               self._loss_ema = self._loss_ema_beta * self._loss_ema + \
                                (1 - self._loss_ema_beta) * metrics['train_loss']
               metrics['loss_spike_ratio'] = metrics['train_loss'] / (self._loss_ema + 1e-12)
        c) No config changes needed (beta is hardcoded, reasonable default)

 1.5.35 Correctness verification
        Test: test_loss_spike_ratio
        a) First step: assert loss_spike_ratio == 1.0
        b) Constant loss: feed loss=1.0 for 10 steps →
           ratio should converge to 1.0 (within 1e-4)
        c) Spike test: feed loss=1.0 for 50 steps, then loss=10.0 →
           ratio should be approximately 10.0 / EMA ≈ 10.0 / 1.0 ≈ 10.0
        d) Recovery test: after spike, feed loss=1.0 again for 20 steps →
           ratio should return toward 1.0 (EMA adapts)
        e) Scale invariance: feed loss=1000.0 constantly →
           ratio should still be ≈ 1.0 (ratio is relative)

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1.5 GATE TEST — INITIAL (for Metrics 1-8 only)                       │
│ NOTE: This gate test was SUPERSEDED by the FINAL gate test below            │
│ (after Metrics 9-15 were added). See "PHASE 1.5 GATE TEST — FINAL"         │
│ for the complete 58-test suite. This section retained for traceability.     │
│                                                                              │
│ tests/test_phase1_5_gate.py:                                                 │
│                                                                              │
│ Gradient metrics (2 new):                                                    │
│  T1.5.1  grad_norm_ratio_first_last exists and is finite positive float     │
│  T1.5.2  grad_norm_ratio_first_last matches manual computation              │
│  T1.5.3  grad_cosine_successive: first call returns 0.0                     │
│  T1.5.4  grad_cosine_successive: second call returns value in [-1, 1]       │
│  T1.5.5  grad_cosine_successive: identical input → cosine near 1.0          │
│                                                                              │
│ Structural metrics (3 new):                                                  │
│  T1.5.6  dead_neuron_frac_mean exists and is in [0, 1]                      │
│  T1.5.7  dead_neuron_frac: healthy model has frac < 0.5                     │
│  T1.5.8  representation_rank_mean exists and is in (0, 1]                   │
│  T1.5.9  representation_rank: random Gaussian → rank near 1.0               │
│  T1.5.10 representation_rank: identical tokens → rank near 0                │
│  T1.5.11 token_isotropy_mean exists and is in [-1, 1]                       │
│  T1.5.12 token_isotropy: identical tokens → isotropy = 1.0                  │
│  T1.5.13 token_isotropy: random Gaussian → isotropy near 0.0                │
│                                                                              │
│ Attention metrics (2 new):                                                   │
│  T1.5.14 attention_sink_score exists and is in [0, 1]                       │
│  T1.5.15 attention_sink_score: uniform attn → score = 1/seq_len             │
│  T1.5.16 dead_head_count exists and is non-negative integer                 │
│  T1.5.17 dead_head_frac: uniform attn → frac = 1.0                         │
│  T1.5.18 dead_head_frac: peaked attn → frac = 0.0                          │
│                                                                              │
│ Training metrics (1 new):                                                    │
│  T1.5.19 loss_spike_ratio: first step = 1.0                                │
│  T1.5.20 loss_spike_ratio: constant loss → ratio ≈ 1.0                     │
│  T1.5.21 loss_spike_ratio: 10× spike detected (ratio ≈ 10)                 │
│                                                                              │
│ Integration:                                                                 │
│  T1.5.22 All 8 new metrics present in MetricCollector output               │
│  T1.5.23 No new metric produces NaN under normal input                      │
│  T1.5.24 Phase 1 gate tests STILL PASS (no regression)                      │
│                                                                              │
│ Run: pytest tests/test_phase1_5_gate.py tests/test_phase1_gate.py -v         │
│ All must PASS.                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────
METRICS 9-12: DEFault COVERAGE GAP METRICS
────────────────────────────────────────────────────────────────────────────────
Context: DEFault (the original system for MLP/CNN/RNN) extracts 28 dynamic
         features at runtime. After systematic mapping (see analysis in
         conversation), 24 of those features are either directly covered by
         DEFault++ or SUPERSEDED by richer continuous metrics (e.g., windowed
         slopes supersede binary trend flags, update_ratio supersedes weight-
         constant binary flags, gradient noise scale supersedes HVP curvature).

         However, 4 features represent genuine diagnostic gaps that are ALSO
         relevant to transformers. These are adapted below for transformer
         architectures.

         Features deliberately NOT ported (with reasoning):
         - saturated_activation: transformers use GELU/SiLU, not sigmoid/tanh.
           GELU has no hard saturation plateau. Dead neuron frac already
           captures the failure mode (neurons stuck in GELU's near-zero region).
         - HVP (5 features): 7× training overhead. For DEFault's ~10K param
           MLPs this is fine; for 100M+ param transformers it's prohibitive.
           Gradient noise scale is a first-order proxy for curvature
           (McCandlish et al. 2018).
         - cons_mean_weight_count / cons_std_weight_count: binary flags for
           "weights stopped changing". update_ratio_total == 0 captures this
           with continuous granularity.
         - decrease_acc_count / increase_loss_count: binary trend flags.
           Windowed slope features (early/mid/late _slope) are richer.

────────────────────────────────────────────────────────────────────────────────
METRIC 9: nan_weight_count — NaN in Model Weights
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
        (placed here because it iterates model.named_parameters() alongside
         gradient computation, so no extra parameter iteration needed)
Diagnoses: Numerical instability causing NaN corruption in weights
Why needed: NaN in weights is the most catastrophic training failure. Once a
            single weight becomes NaN, it propagates through matrix multiplies
            to corrupt ALL downstream computations within 1-2 forward passes.
            DEFault++ currently detects NaN in LOGITS (logit_nan_ratio) but by
            that point the model is already destroyed. Catching NaN at the
            weight level gives EARLIER warning — potentially before the first
            NaN logit even appears.

            DEFault original: `nan_weight_count = np.isnan(weights).sum()`
            (Static_Feature_Extraction.py was only for architecture; this is
             from CustomCallback.py line 102)

 1.5.36 Theory (invariant — same for all architectures)
        ═══════════════════════════════════════════════════
        For a model with parameters θ = {W_1, W_2, ..., W_K}:

        nan_weight_count = Σ_k Σ_{i,j} 𝟙[isNaN(W_k[i,j])]

        This is simply the total count of NaN entries across all weight
        tensors. It is:
          - Zero during healthy training (NaN weights should NEVER occur)
          - Non-zero only under numerical pathology
          - Monotonically non-decreasing once NaN appears (NaN + anything = NaN,
            so NaN weights create more NaN weights on subsequent updates)

        Why NaN weights happen in transformers specifically:
        1. Mixed precision (fp16): half-precision has max value 65504.
           Any intermediate value > 65504 overflows to Inf, and Inf - Inf = NaN.
           Common in: attention score computation (Q·K^T with large hidden dim),
           LayerNorm (division by std ≈ 0 when variance collapses)
        2. Loss scaling underflow/overflow: gradient scaling in AMP can produce
           Inf gradients → Inf weight updates → NaN on next step
        3. Learning rate too high: weight update Δw = lr × grad, if
           lr × ||grad|| > ||w|| by orders of magnitude, floating point
           precision is lost → eventual NaN

        Companion metric: inf_weight_count (same computation for Inf values)
        Inf often PRECEDES NaN (Inf × 0 = NaN), so detecting Inf gives even
        earlier warning.

 1.5.37 Encoder specifics (BERT-style)
        - Most vulnerable component: LayerNorm. If hidden states have zero
          variance (all dimensions identical), LN divides by ~0 → Inf → NaN
        - Embedding layer: large vocabulary (30K+) × hidden_size weights.
          Rarely NaN but contributes the most parameters to count.
        - Pre-softmax attention scores: Q·K^T / √d_k can overflow in fp16
          when d_k is small and Q/K values are large. This creates NaN in
          attention weights → NaN in value projection → NaN in output →
          NaN in subsequent LayerNorm → NaN in FFN weights after update.
        - Expected: nan_weight_count = 0 always. Any non-zero value is a
          critical failure indicator.

 1.5.38 Decoder specifics (GPT-style)
        - Same vulnerability profile as encoders, plus:
        - KV-cache accumulation: if using past_key_values, a single NaN key
          vector persists across all future steps, corrupting attention for
          the entire remaining sequence
        - Vocabulary projection (lm_head): maps hidden_size → vocab_size
          (often 50K+). This is the largest weight matrix and most likely
          to develop NaN if gradient explosion hits the output layer.
        - Rotary position embeddings (RoPE): sin/cos computations are
          numerically stable in theory, but custom implementations can
          produce NaN for very long sequences if position indices overflow

 1.5.39 Implementation
        In GradientMetrics._compute_gradient_norms(), which already iterates
        model.named_parameters():
        a) Add counters before the parameter loop:
           nan_w_count = 0
           inf_w_count = 0
        b) Inside the loop, for each parameter (not just those with gradients):
           nan_w_count += int(torch.isnan(param.data).sum().item())
           inf_w_count += int(torch.isinf(param.data).sum().item())
        c) After the loop:
           metrics['nan_weight_count'] = float(nan_w_count)
           metrics['inf_weight_count'] = float(inf_w_count)

        Performance: O(num_parameters) — one pass over all weight tensors.
        This is already paid by the gradient norm computation, so the
        additional cost is just the isnan/isinf checks (negligible — these
        are single-pass element-wise operations that PyTorch vectorizes).

        No config additions needed. No thresholds (any NaN is a failure).

 1.5.40 Correctness verification
        Test: test_nan_weight_count
        Strategy: Since we cannot expect a normally-initialized model to
        produce NaN weights, we test both the "healthy" case (should be 0)
        and an INJECTED fault case (manually set a weight to NaN).

        a) Healthy model test:
           - Create a tiny BERT, run forward+backward
           - Assert nan_weight_count == 0.0
           - Assert inf_weight_count == 0.0
           - Rationale: randomly initialized models with standard optimizers
             should never produce NaN weights in a single step

        b) NaN injection test:
           - Create a tiny BERT model
           - Manually set one weight tensor to contain NaN:
             with torch.no_grad():
                 model.bert.encoder.layer[0].attention.self.query.weight[0, 0] = float('nan')
           - Run GradientMetrics.collect()
           - Assert nan_weight_count >= 1.0
           - Rationale: we injected exactly 1 NaN, counter should detect it

        c) Inf injection test:
           - Same as (b) but inject float('inf') instead
           - Assert inf_weight_count >= 1.0

        d) Count accuracy test:
           - Inject exactly 5 NaN values across 2 different weight tensors
           - Assert nan_weight_count == 5.0
           - Rationale: counter should be EXACT, not approximate

        e) Decoder test:
           - Same test suite on tiny GPT-2 model
           - Assert nan_weight_count == 0.0 for healthy model
           - Assert detection works after injection

────────────────────────────────────────────────────────────────────────────────
METRIC 10: nan_gradient_count — NaN in Gradients
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
Diagnoses: Numerical instability in gradient computation
Why needed: NaN gradients are the PRECURSOR to NaN weights. The causal chain
            is: loss instability → NaN gradient → NaN weight update → NaN
            weights → NaN outputs. Detecting NaN at the gradient level gives
            one step of LEAD TIME to intervene (e.g., skip the update, reduce
            LR) before weights are corrupted.

            DEFault++ currently has gradient_vanish (norm < 1e-4) and
            gradient_explode (norm > 100), but both assume gradients are
            FINITE. A NaN gradient has undefined norm — it won't trigger
            either detector, silently corrupting the model.

            DEFault original: `nan_gradients_count = sum([np.isnan(g).sum()
            for g in epoch_gradients[0]])` (CustomCallback.py line 103)

 1.5.41 Theory (invariant)
        ═════════════════════
        For gradients g_k = ∂L/∂W_k for each parameter tensor W_k:

        nan_gradient_count = Σ_k Σ_{i,j} 𝟙[isNaN(g_k[i,j])]

        Only counted for parameters where grad is not None (parameters
        that didn't participate in the forward pass have no gradient).

        Why NaN gradients happen:
        1. Division by zero in loss: cross_entropy with probability = 0
           produces -log(0) = Inf, and ∂Inf/∂w = NaN
        2. Log of negative number: if softmax produces negative values
           (shouldn't but can with numerical errors), log(negative) = NaN
        3. 0/0 in normalization: LayerNorm computes (x - μ) / σ.
           If σ = 0 (all dimensions identical), gradient of this is 0/0 = NaN
        4. Inf × 0 in gradient chain: if a forward activation is Inf and
           the upstream gradient is 0, the product is NaN
        5. Gradient clipping interaction: if grad is Inf, then
           grad * (max_norm / grad_norm) = Inf * 0 = NaN when grad_norm = Inf

        Companion metric: inf_gradient_count
        Same rationale — Inf gradients cause NaN on the next step via
        weight_new = weight_old - lr × Inf = -Inf, then -Inf + anything
        may NaN.

 1.5.42 Encoder specifics (BERT-style)
        - Gradient NaN most commonly originates from:
          a) Attention softmax: if all scores are -Inf (bad masking),
             softmax produces 0/0 = NaN, gradient is NaN
          b) LayerNorm backward: requires variance > 0 for numerical stability.
             If an entire batch has identical hidden states, variance = 0,
             gradient = NaN
          c) Cross-entropy loss: if label is out of range or logits contain
             NaN, loss gradient is NaN
        - Detection point: check param.grad after loss.backward(), before
          optimizer.step()

 1.5.43 Decoder specifics (GPT-style)
        - Same sources as encoder, plus:
          a) Causal mask numerical issues: some implementations add -1e9 or
             -Inf to masked positions. If the mask is wrong, ALL positions
             get -Inf → softmax NaN → gradient NaN
          b) Token prediction loss: decoder predicts over vocab_size (often
             50K+). With that many classes, numerical precision in
             log_softmax is more challenging → higher NaN risk
          c) Sequence length: longer sequences mean more gradient
             accumulation steps within a single loss.backward(), increasing
             the chance of numerical instability compounding

 1.5.44 Implementation
        In GradientMetrics._compute_gradient_norms(), which already has a
        `for name, param in model.named_parameters()` loop that checks
        `param.grad is not None`:

        a) Add counters before the loop:
           nan_g_count = 0
           inf_g_count = 0
        b) Inside the existing `if param.grad is None: continue` block,
           after accessing `grad = param.grad.data`:
           nan_g_count += int(torch.isnan(grad).sum().item())
           inf_g_count += int(torch.isinf(grad).sum().item())
        c) After the loop:
           metrics['nan_gradient_count'] = float(nan_g_count)
           metrics['inf_gradient_count'] = float(inf_g_count)

        Performance: Same as nan_weight_count — piggybacking on existing
        parameter iteration. The isnan/isinf calls are O(num_gradient_elements)
        which we're already iterating to compute norms.

        Important interaction with existing metrics:
        - If nan_gradient_count > 0, grad_norm_total will be NaN
        - The existing gradient_vanish / gradient_explode checks compare
          grad_norm_total against thresholds — NaN fails BOTH comparisons
          (NaN < x is False, NaN > x is False), so neither flag fires
        - This is why nan_gradient_count is needed as a SEPARATE check:
          the existing metrics have a NaN-shaped blind spot

 1.5.45 Correctness verification
        Test: test_nan_gradient_count
        Strategy: Similar to weight NaN tests — healthy case + injection.

        a) Healthy model test:
           - Create tiny BERT, run forward+backward with valid inputs
           - Assert nan_gradient_count == 0.0
           - Assert inf_gradient_count == 0.0
           - Rationale: standard training produces finite gradients

        b) NaN gradient injection test:
           - Create tiny BERT, run forward+backward
           - Before calling collect(), manually inject NaN into a gradient:
             param = list(model.parameters())[0]
             param.grad.data[0, 0] = float('nan')
           - Call GradientMetrics.collect()
           - Assert nan_gradient_count >= 1.0

        c) Inf gradient injection test:
           - Same as (b) but inject float('inf')
           - Assert inf_gradient_count >= 1.0

        d) NaN-causes-NaN-norm verification:
           - After injecting NaN gradient, verify:
             assert math.isnan(result['grad_norm_total']) or \
                    result['nan_gradient_count'] > 0
           - Rationale: proves that without nan_gradient_count, the NaN
             would go undetected (grad_norm_total becomes NaN, which
             doesn't trigger vanish or explode flags)

        e) Decoder test:
           - Run same suite on tiny GPT-2
           - Verify detection works for decoder architecture

────────────────────────────────────────────────────────────────────────────────
METRIC 11: weight_norm_max — Maximum Weight Matrix Norm
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
        (iterates model parameters — same loop as gradient norms)
Diagnoses: Weight magnitude explosion / large weight pathology
Why needed: DEFault tracks `large_weight_count` (count of weights > 10.0).
            This concept is valid for transformers but the implementation
            needs adaptation. A fixed threshold (10.0) is arbitrary:
            - Embedding weights in BERT are often > 10 by design
            - Attention projection weights are typically < 1
            - A threshold that catches embeddings as "large" is useless

            Instead, we track the MAXIMUM Frobenius norm across all weight
            matrices. This is:
            - Scale-aware (compares matrices, not individual elements)
            - Sensitive to weight explosion (which manifests as one matrix
              growing disproportionately large)
            - Architecture-agnostic (works for any transformer)

            Additionally, we track weight_norm_mean for baseline comparison:
            weight_norm_max >> weight_norm_mean indicates one component is
            diverging from the rest — a localized weight explosion.

 1.5.46 Theory (invariant)
        ═════════════════════
        For model parameter tensors {W_1, W_2, ..., W_K}:
        (only weight matrices, not biases — biases are 1D and have
         incomparable norms)

        weight_norm_per_matrix_k = ||W_k||_F = √(Σ_{i,j} W_k[i,j]²)
          (Frobenius norm — the L2 norm of the flattened weight tensor)

        weight_norm_max = max_k ||W_k||_F
        weight_norm_mean = mean_k ||W_k||_F

        Interpretation:
          Both metrics track absolute weight magnitude, not relative change
          (that's update_ratio's job). They answer: "how big are the weights
          RIGHT NOW?"

          weight_norm_max is the early warning for weight explosion:
          - If one matrix's norm is growing while others stay constant,
            weight_norm_max increases while weight_norm_mean stays stable
          - Ratio: weight_norm_max / weight_norm_mean >> 1 indicates
            localized explosion

        Why Frobenius norm (not spectral norm):
          - Spectral norm (largest singular value) is more theoretically
            grounded for controlling Lipschitz constants, but requires SVD
            which is expensive for large matrices
          - Frobenius norm is O(num_elements) — just sum of squares
          - For diagnosis (not training regularization), Frobenius is
            sufficient: if any singular value explodes, Frobenius explodes too
          - The approximation quality: ||W||_σ ≤ ||W||_F ≤ √rank × ||W||_σ
            For transformer weight matrices with rank ~ hidden_size, the
            gap is bounded by √hidden_size ≈ 28 for hidden_size=768.
            This is a constant factor — exploding weights are still detectable.

 1.5.47 Encoder specifics (BERT-style)
        - Embedding matrix: vocab_size × hidden_size (e.g., 30K × 768).
          This is the LARGEST matrix and typically has the largest norm.
          It should be excluded from weight_norm_max OR tracked separately,
          because it's always large by design and doesn't indicate a fault.
        - Attention Q/K/V projections: hidden_size × hidden_size.
          These should have norm ∝ √hidden_size under standard initialization.
        - FFN intermediate: hidden_size × 4×hidden_size.
          Largest non-embedding matrix. Weight explosion often manifests here.
        - Classifier head: hidden_size × num_labels.
          Small matrix, unlikely to dominate norm.

        Strategy: Track weight_norm_max over all LAYER parameters (not
        embedding). Track embedding_weight_norm separately.

 1.5.48 Decoder specifics (GPT-style)
        - Embedding matrix (wte): vocab_size × hidden_size.
          In GPT-2, this is often TIED with the output projection (lm_head).
          If they share weights, don't double-count.
        - Position embedding (wpe): max_position × hidden_size.
          Also large, also expected to be large.
        - Attention: same as encoder (Q/K/V projections)
        - MLP: hidden_size × 4×hidden_size (same vulnerability as encoder FFN)
        - lm_head: hidden_size × vocab_size.
          This is the output projection — gradient explosion from the loss
          hits this matrix FIRST (it's closest to the loss in the compute
          graph). If any matrix explodes, it's often this one.

 1.5.49 Implementation
        In GradientMetrics._compute_gradient_norms(), within the existing
        parameter loop:

        a) Add collection lists before the loop:
           weight_norms = []
        b) Inside the loop, for each parameter with dim >= 2 (weight matrices,
           not biases):
           if param.data.dim() >= 2:
               w_norm = param.data.norm(2).item()  # Frobenius norm
               weight_norms.append(w_norm)
        c) After the loop:
           if weight_norms:
               metrics['weight_norm_max'] = float(max(weight_norms))
               metrics['weight_norm_mean'] = float(sum(weight_norms) / len(weight_norms))
           else:
               metrics['weight_norm_max'] = 0.0
               metrics['weight_norm_mean'] = 0.0

        Performance: O(num_parameters) — computing Frobenius norm is just
        sqrt(sum of squares), which is what torch.norm(2) does on the
        flattened tensor. This is already the same cost as computing
        gradient norms (which we already do).

        No config additions needed.

 1.5.50 Correctness verification
        Test: test_weight_norm_max
        Strategy: Verify against manual computation, and verify sensitivity
        to weight scaling.

        a) Existence and type:
           - Create tiny BERT, run forward+backward, call collect()
           - Assert 'weight_norm_max' in result
           - Assert 'weight_norm_mean' in result
           - Assert both are finite positive floats

        b) Manual verification:
           - Manually compute Frobenius norm of every param with dim >= 2:
             manual_norms = [p.data.norm(2).item()
                             for p in model.parameters() if p.data.dim() >= 2]
             manual_max = max(manual_norms)
             manual_mean = sum(manual_norms) / len(manual_norms)
           - Assert result['weight_norm_max'] == manual_max (within 1e-6)
           - Assert result['weight_norm_mean'] ≈ manual_mean (within 1e-6)

        c) Sensitivity to weight scaling:
           - Record initial weight_norm_max
           - Multiply one weight matrix by 100:
             with torch.no_grad():
                 param = list(model.parameters())[0]
                 param.data *= 100
           - Call collect() again
           - Assert new weight_norm_max > old weight_norm_max * 50
             (should be ~100× larger, allow margin for other matrices)
           - Rationale: proves the metric is sensitive to weight explosion

        d) Ratio interpretation test:
           - For a healthy model: assert weight_norm_max / weight_norm_mean < 20
             (no single matrix should dominate by more than 20×)
           - After scaling one matrix by 1000×:
             assert weight_norm_max / weight_norm_mean > 50
             (the scaled matrix now dominates)
           - Rationale: validates the diagnostic interpretation

        e) Decoder test:
           - Run (a) and (b) on tiny GPT-2
           - Verify metric exists and matches manual computation

────────────────────────────────────────────────────────────────────────────────
METRIC 12: gradient_median — Median Gradient Norm Across Parameter Groups
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/gradient.py (GradientMetrics)
Diagnoses: Gradient distribution health, outlier sensitivity
Why needed: DEFault tracks `gradient_median` as a complement to
            `mean_gradient`. This is a valid diagnostic principle:
            - mean gradient norm is SENSITIVE to outliers (one exploding
              layer can dominate the mean)
            - median gradient norm is ROBUST to outliers
            - If mean >> median, it indicates a few layers have
              disproportionately large gradients (localized explosion)
            - If mean ≈ median, gradient magnitudes are relatively uniform
              across layers (healthy)

            DEFault++ has grad_norm_total (global L2 norm) and per-group
            grad_norm_{group}, but NOT a summary statistic across groups
            that is robust to outliers.

 1.5.51 Theory (invariant)
        ═════════════════════
        Given per-layer gradient norms: g_0, g_1, ..., g_{L-1}
        (one norm per transformer layer, computed from the layer's
         attention + FFN + LN parameters combined)

        gradient_median = median(g_0, g_1, ..., g_{L-1})

        Properties:
          - Breakdown point = 50%: up to half the layers can have
            pathological gradients without affecting the median
          - If gradient_median ≈ grad_norm_total / √L, gradients are
            uniformly distributed (healthy)
          - If gradient_median << grad_norm_total / √L, a few layers
            dominate the total norm (localized issue)

        Companion metric: gradient_mean_per_layer
          gradient_mean_per_layer = mean(g_0, g_1, ..., g_{L-1})
          Ratio: gradient_mean_per_layer / gradient_median
            ≈ 1.0 → symmetric distribution (healthy)
            >> 1.0 → right-skewed (a few layers have very large gradients)

 1.5.52 Encoder vs Decoder
        - Identical computation for both architectures
        - In well-trained transformers with residual connections, gradient
          norms should be approximately uniform across layers (ratio ≈ 1)
        - Pre-LayerNorm architectures tend to have more uniform gradients
          than Post-LayerNorm (Pre-LN is better conditioned)
        - Decoders may show slightly higher gradients in final layers
          (closer to loss) — gradient_median < gradient_mean is expected

 1.5.53 Implementation
        In GradientMetrics._compute_gradient_norms(), after the per-group
        loop where grad_norm_{group} metrics are already computed:

        a) Collect per-layer total gradient norms:
           layer_norms = []
           for i in range(self.inspector.num_layers):
               attn_key = f'grad_norm_layer{i}_attention'
               ffn_key = f'grad_norm_layer{i}_ffn'
               # Combine attention + FFN norms for total layer norm
               attn_norm = metrics.get(attn_key, 0.0)
               ffn_norm = metrics.get(ffn_key, 0.0)
               layer_norm = math.sqrt(attn_norm**2 + ffn_norm**2)
               layer_norms.append(layer_norm)

        b) Compute statistics:
           import numpy as np
           if layer_norms:
               metrics['gradient_median'] = float(np.median(layer_norms))
               metrics['gradient_mean_per_layer'] = float(np.mean(layer_norms))
           else:
               metrics['gradient_median'] = 0.0
               metrics['gradient_mean_per_layer'] = 0.0

        Performance: O(num_layers) — just a median over L values (typically
        6-48). Negligible compared to the gradient norm computation itself.

 1.5.54 Correctness verification
        Test: test_gradient_median
        Strategy: Verify against manual computation and verify robustness
        property.

        a) Existence and range:
           - Create tiny BERT, run forward+backward, call collect()
           - Assert 'gradient_median' in result
           - Assert 'gradient_mean_per_layer' in result
           - Assert both are finite non-negative floats

        b) Manual verification:
           - Manually compute per-layer gradient norms:
             For each layer, sum gradient norms of all params in that layer
           - Compute numpy.median of these per-layer norms
           - Assert result['gradient_median'] matches within tolerance 1e-6

        c) Uniform gradient test:
           - For a normally initialized model after one step, gradients
             should be roughly uniform across layers
           - Assert gradient_mean_per_layer / gradient_median < 3.0
             (ratio should be near 1 for healthy models)

        d) Outlier robustness test:
           - Manually scale gradients of layer 0 by 1000×:
             for name, param in model.named_parameters():
                 if 'layer.0' in name and param.grad is not None:
                     param.grad.data *= 1000
           - Call collect()
           - Assert gradient_mean_per_layer >> gradient_median
             (mean is dominated by outlier, median is not)
           - Assert gradient_median has NOT changed dramatically from
             pre-scaling value (within 2×)
           - Rationale: proves median's robustness to outliers

        e) Decoder test:
           - Run (a) and (b) on tiny GPT-2
           - Verify metric matches manual computation

────────────────────────────────────────────────────────────────────────────────
METRICS 13-14: APPROXIMATE CURVATURE — Feasible Sharpness for Transformers
────────────────────────────────────────────────────────────────────────────────
Context: DEFault computes full Hessian-vector products (HVP) via nested
         autodiff — 5 features (mean_hvp, hvp_std, hvp_max, hvp_min,
         hvp_median). This costs ~2× a gradient computation PER BATCH,
         and DEFault samples 3 mini-batches per step → 7× total overhead.

         For DEFault's target models (~10K-1M params), this is acceptable.
         For transformers (100M+ params), 7× overhead is prohibitive.

         However, the DIAGNOSTIC SIGNAL is valuable: loss landscape
         curvature tells us whether the model is in a sharp minimum
         (poor generalization, fragile) or a flat minimum (good
         generalization, robust). This is exactly what a fault diagnosis
         system should detect.

         Our approach: TWO complementary approximations with STRATEGIC
         scheduling — compute curvature cheaply and infrequently, not
         expensively and constantly.

         Key insight: curvature changes SLOWLY relative to gradients.
         The loss landscape geometry evolves over epochs, not steps. So
         computing curvature every step is wasteful even if it were free.

────────────────────────────────────────────────────────────────────────────────
METRIC 13: sharpness_sam — SAM-style Directional Sharpness
────────────────────────────────────────────────────────────────────────────────
Module: NEW file — extraction/metrics/curvature.py (CurvatureMetrics)
Diagnoses: Loss landscape sharpness → generalization quality
Why a new module: Curvature metrics are fundamentally different from the
                  other 7 modules — they require weight perturbation (modifying
                  the model temporarily) and strategic scheduling (not every
                  step). A separate module keeps this complexity isolated.

 1.5.55 Theory (invariant — architecture-independent)
        ═══════════════════════════════════════════════════
        Sharpness-Aware Minimization (Foret et al., 2021 "Sharpness-Aware
        Minimization for Efficiently Improving Generalization"):

        The sharpness of the loss landscape at weight point w is defined as
        the maximum increase in loss within an ε-neighborhood:

          sharpness(w) = max_{||δ|| ≤ ε} L(w + δ) - L(w)

        Computing this exactly requires solving an inner optimization.
        The SAM approximation uses the gradient direction as the
        worst-case perturbation (first-order Taylor expansion):

          δ* ≈ ε · ∇L(w) / ||∇L(w)||

        This gives:
          sharpness_sam = L(w + δ*) - L(w)

        Where:
          ε = perturbation radius (default: 0.05, following SAM paper)
          ∇L(w) = gradient at current weights (ALREADY COMPUTED — free)
          L(w) = current loss (ALREADY COMPUTED — free)
          L(w + δ*) = loss at perturbed weights (ONE extra forward pass)

        Total cost: ONE forward pass. No backward pass needed.

        Interpretation:
          sharpness ≈ 0     → flat minimum (good generalization)
          sharpness > 0.5   → moderately sharp (some generalization risk)
          sharpness > 2.0   → very sharp (poor generalization expected)
          sharpness < 0     → shouldn't happen (indicates numerical issues)

        Why this works for diagnosis:
          - Sharp minima = model is in a fragile region where small weight
            perturbations cause large loss increases
          - This correlates with: overfitting, sensitivity to hyperparameters,
            poor transfer, and training instability
          - Keskar et al. (2017) showed large-batch training finds sharper
            minima → our metric can detect this problem
          - Hochreiter & Schmidhuber (1997) showed flat minima have better
            generalization bounds via minimum description length

        Normalized variant for cross-model comparison:
          sharpness_sam_normalized = sharpness_sam / (L(w) + ε_denom)
          This gives the RELATIVE sharpness — a sharpness of 0.5 means "loss
          increases by 50% of its current value under perturbation".
          This is comparable across models with different loss scales.

 1.5.56 Encoder specifics (BERT-style)
        - BERT fine-tuning is known to be sensitive to sharpness: Hao et al.
          (2019) "Visualizing and Understanding the Effectiveness of BERT"
          showed that BERT fine-tuning lands in sharp minima more often than
          training from scratch.
        - The perturbation ε = 0.05 is appropriate for BERT-scale models
          (hidden_size 768, ~110M params). For smaller models like DistilBERT,
          ε = 0.05 is still reasonable.
        - Classification loss (cross-entropy with few classes) has smoother
          landscape than LM loss — sharpness values tend to be lower.
        - The perturbation direction (gradient) is computed from ALL parameters
          including embeddings. For classification fine-tuning, most gradient
          mass is in the classifier head + last few layers, so the perturbation
          primarily tests sharpness in those regions.

 1.5.57 Decoder specifics (GPT-style)
        - Language modeling loss (cross-entropy over vocab_size=50K+) creates
          a naturally sharper landscape than classification — more output
          dimensions = more directions in which loss can increase.
        - Expected healthy sharpness values are higher for decoders than
          encoders (0.2–1.0 vs 0.05–0.3 for encoders).
        - The perturbation should use ε scaled by model norm:
          ε_effective = ε × ||w|| / √num_params
          This ensures the perturbation is a constant FRACTION of the weight
          magnitude, regardless of model size. Without this, ε=0.05 means
          very different things for a 124M GPT-2 vs a 1.5B GPT-2.
        - Tied embeddings: if wte == lm_head, the perturbation affects both
          input and output projections simultaneously. This is correct
          behavior (both are perturbed together as they would be in training).

 1.5.58 Implementation
        a) Create new file: src/defaultplusplus/extraction/metrics/curvature.py

        b) Class: CurvatureMetrics(MetricModule)
           __init__(self, inspector, config=None):
             self.rho = 0.05  # SAM perturbation radius
             self.compute_interval = 50  # compute every N steps
             self.step_counter = 0
             self._last_sharpness = 0.0  # cache between computations

        c) collect() method:
           self.step_counter += 1

           # Strategic scheduling: only compute every N steps
           if self.step_counter % self.compute_interval != 0:
               return {
                   'sharpness_sam': self._last_sharpness,
                   'sharpness_sam_normalized': self._last_sharpness_norm,
               }

           # Need: model (for weights), loss (current), the batch data
           if model is None or loss is None:
               return {'sharpness_sam': 0.0, 'sharpness_sam_normalized': 0.0}

           current_loss = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

           # Step 1: Compute perturbation direction from existing gradients
           grad_norm = 0.0
           for p in model.parameters():
               if p.grad is not None:
                   grad_norm += p.grad.data.norm(2).item() ** 2
           grad_norm = math.sqrt(grad_norm)

           if grad_norm < 1e-12:
               return {'sharpness_sam': 0.0, 'sharpness_sam_normalized': 0.0}

           # Step 2: Perturb weights (in-place, then restore)
           #   δ = ρ × grad / ||grad||
           old_params = {}
           scale = self.rho / grad_norm
           for name, p in model.named_parameters():
               if p.grad is not None:
                   old_params[name] = p.data.clone()
                   p.data.add_(p.grad.data, alpha=scale)  # w + δ

           # Step 3: Forward pass at perturbed weights (no grad needed)
           model.eval()
           with torch.no_grad():
               perturbed_outputs = model(**batch_inputs)
               perturbed_loss = perturbed_outputs.loss.item()
           model.train()

           # Step 4: Restore original weights
           for name, p in model.named_parameters():
               if name in old_params:
                   p.data.copy_(old_params[name])

           # Step 5: Compute sharpness
           sharpness = perturbed_loss - current_loss
           sharpness_norm = sharpness / (abs(current_loss) + 1e-8)

           self._last_sharpness = sharpness
           self._last_sharpness_norm = sharpness_norm

           return {
               'sharpness_sam': float(sharpness),
               'sharpness_sam_normalized': float(sharpness_norm),
           }

        CRITICAL IMPLEMENTATION NOTES:
        - Weight perturbation is IN-PLACE then RESTORED. This avoids
          allocating a full model copy. The restore MUST happen even if
          the forward pass fails (use try/finally).
        - model.eval() during perturbed forward: disables dropout and
          batch norm updates, so the perturbed loss is deterministic.
        - No backward pass at perturbed weights — we only need the loss
          VALUE, not gradients.
        - The batch inputs (input_ids, attention_mask, labels) must be
          passed to collect() via the existing kwargs. The collector
          already passes these.

 1.5.59 Strategic scheduling design
        ═══════════════════════════════
        The compute_interval parameter controls how often sharpness is
        computed. Between computations, the cached value is returned.

        Default: compute_interval = 50 (every 50 steps)
        - For a typical transformer fine-tuning run of 3 epochs × 500
          steps/epoch = 1500 steps, this gives 30 sharpness measurements
        - Overhead: 30 extra forward passes / 1500 total forward passes
          = 2% additional training time
        - Compare to DEFault's 7× overhead: this is 350× cheaper

        Adaptive scheduling (future enhancement):
        - When loss_spike_ratio > 2.0 → compute immediately
        - When gradient_variance is high → compute next step
        - During first epoch → compute every 10 steps (more volatile)
        - During later epochs → compute every 100 steps (more stable)

        Config additions to ExtractionConfig:
        - sharpness_rho: float = 0.05  # perturbation radius
        - sharpness_interval: int = 50  # steps between computations

────────────────────────────────────────────────────────────────────────────────
METRIC 14: hutchinson_trace — Hessian Trace via Stochastic Estimation
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/curvature.py (CurvatureMetrics)
Diagnoses: Average curvature of the loss landscape
Why needed: SAM-sharpness measures curvature in ONE direction (the gradient
            direction). Hutchinson trace measures AVERAGE curvature across ALL
            directions. These give complementary information:
            - High SAM-sharpness + low trace → sharp in gradient direction
              but flat overall (narrow valley, common in fine-tuning)
            - High SAM-sharpness + high trace → sharp everywhere
              (uniformly bad landscape, severe generalization issue)
            - Low SAM-sharpness + high trace → flat in gradient direction
              but sharp in other directions (saddle point region)

 1.5.60 Theory (invariant)
        ═════════════════════
        Hutchinson's Trace Estimator (Hutchinson, 1990):

        For a symmetric matrix H ∈ ℝ^{n×n} (the Hessian):
          tr(H) = E_v[v^T H v]

        where v is a random vector with E[vv^T] = I.
        Common choices: v ~ Rademacher (each entry ±1 with equal probability)
        or v ~ N(0, I).

        One sample gives an unbiased estimate:
          tr(H) ≈ v^T H v = v^T (Hv)

        The Hessian-vector product Hv can be computed WITHOUT forming H:
          Hv = ∂/∂w (∇L(w)^T · v)

        This is ONE backward pass through the gradient (not through the
        model twice). PyTorch autograd supports this directly:
          Hv = torch.autograd.grad(
                   outputs=grad.dot(v),   # scalar: ∇L^T · v
                   inputs=params,          # differentiate w.r.t. params
                   create_graph=False       # we don't need second-order grads
               )

        Normalized trace:
          hessian_trace_normalized = tr(H) / num_params
          = average eigenvalue of the Hessian
          = average curvature per parameter

        Interpretation:
          trace > 0    → model is in a region of positive curvature
                          (near a minimum — expected during training)
          trace ≈ 0    → flat region (plateau or saddle point)
          trace < 0    → negative curvature (saddle point — rare in
                          over-parameterized models)
          trace >> 0   → very sharp curvature (generalization risk)

        Variance reduction:
        - Single sample: variance ~ O(||H||_F^2 / n)
        - K samples: variance ~ O(||H||_F^2 / (K·n))
        - For diagnosis, K=1 is sufficient — we want the ORDER OF MAGNITUDE,
          not a precise value. K=1 gives ±50% relative accuracy, which is
          fine for detecting "sharp vs flat" (which differ by 10-100×).

 1.5.61 Encoder specifics (BERT-style)
        - BERT models have ~110M parameters. The Hv computation costs
          ONE backward pass through the gradient computation graph.
        - IMPORTANT: We compute Hv on a PARAMETER SUBSET, not all params.
          Strategy: only include the attention Q/K/V projections and FFN
          weights of the SAMPLED layers (3 layers out of 12).
          This reduces the Hv computation from ~110M params to ~15M params
          (7× cheaper), while still capturing the curvature of the most
          diagnostically important components.
        - For BERT fine-tuning: the classifier head contributes
          disproportionate curvature (small matrix, high gradient). Include
          it in the subset to capture task-head sharpness.

 1.5.62 Decoder specifics (GPT-style)
        - Same Hv computation, same parameter subset strategy
        - Decoder loss (cross-entropy over 50K vocab) typically has higher
          curvature than encoder classification loss — expected trace values
          are ~10× higher
        - The lm_head projection (hidden_size × vocab_size) is the largest
          contributor to curvature. Including it in the subset is critical.
        - For GPT-2 (124M params): parameter subset of 3 sampled layers
          + lm_head ≈ 20M params. Hv cost ≈ 0.15× a full backward pass.

 1.5.63 Implementation
        In CurvatureMetrics (same file as Metric 13):

        a) Add to __init__:
           self.trace_interval = 0  # 0 = epoch-boundary only
           self._last_trace = 0.0
           self._last_trace_normalized = 0.0

        b) New method: _compute_hutchinson_trace(model, loss, params_subset):
           """Compute Hessian trace estimate via Hutchinson's method."""

           # Step 1: Get gradients (already computed from loss.backward())
           grads = [p.grad for p in params_subset if p.grad is not None]
           params_with_grad = [p for p in params_subset if p.grad is not None]

           if not grads:
               return 0.0, 0.0

           # Step 2: Generate random Rademacher vector
           vs = [torch.randint_like(g, 0, 2).float() * 2 - 1 for g in grads]

           # Step 3: Compute grad^T · v (scalar)
           grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vs))

           # Step 4: Compute Hv = d/dw (grad^T · v)
           #   This requires the gradient computation graph to still exist.
           #   IMPORTANT: loss.backward(create_graph=True) must have been
           #   called for this to work. If create_graph=False (default),
           #   we fall back to finite-difference approximation.
           try:
               Hv = torch.autograd.grad(
                   grad_dot_v, params_with_grad,
                   retain_graph=False, create_graph=False
               )
               # Step 5: trace ≈ v^T Hv = Σ_i v_i · (Hv)_i
               trace_estimate = sum(
                   (v * hv).sum().item() for v, hv in zip(vs, Hv)
               )
           except RuntimeError:
               # Fallback: finite-difference approximation
               # Hv ≈ (∇L(w + δv) - ∇L(w)) / δ
               trace_estimate = self._finite_diff_trace(
                   model, params_with_grad, grads, vs
               )

           num_params = sum(p.numel() for p in params_with_grad)
           trace_normalized = trace_estimate / max(num_params, 1)

           return float(trace_estimate), float(trace_normalized)

        c) Finite-difference fallback method:
           def _finite_diff_trace(self, model, params, grads, vs, delta=1e-4):
               """Finite-diff Hv when autograd graph is unavailable."""
               # This requires one extra forward+backward pass
               # Perturb: w → w + δ·v
               old_data = [p.data.clone() for p in params]
               for p, v in zip(params, vs):
                   p.data.add_(v, alpha=delta)

               # Recompute gradients at perturbed point
               # (need to re-run forward pass — expensive)
               # Only use this fallback for epoch-boundary computations
               ... (implementation mirrors SAM perturbation pattern)

               # Restore weights
               for p, old in zip(params, old_data):
                   p.data.copy_(old)

               # Hv ≈ (grad_perturbed - grad_original) / delta
               # trace ≈ v^T Hv
               trace = sum(
                   (v * (gp - go) / delta).sum().item()
                   for v, gp, go in zip(vs, perturbed_grads, grads)
               )
               return trace

        d) Integration into collect():
           # Hutchinson trace: compute at epoch boundaries only (default)
           # or at specified interval, or on trigger
           compute_trace = False
           if self.trace_interval > 0:
               compute_trace = (self.step_counter % self.trace_interval == 0)
           # Also compute on trigger: loss spike detected
           if kwargs.get('_loss_spike_detected', False):
               compute_trace = True

           if compute_trace and model is not None:
               # Select parameter subset: sampled layers + head
               params_subset = self._get_curvature_params_subset(model)
               trace, trace_norm = self._compute_hutchinson_trace(
                   model, loss, params_subset
               )
               self._last_trace = trace
               self._last_trace_normalized = trace_norm

           metrics['hessian_trace'] = self._last_trace
           metrics['hessian_trace_normalized'] = self._last_trace_normalized

        e) Parameter subset selection:
           def _get_curvature_params_subset(self, model):
               """Select parameters for curvature estimation."""
               subset = []
               sampled = self.inspector.get_sampled_layer_indices()
               for idx in sampled:
                   attn = self.inspector.get_attention_module(idx)
                   ffn = self.inspector.get_ffn_module(idx)
                   if attn:
                       subset.extend(attn.parameters())
                   if ffn:
                       subset.extend(ffn.parameters())
               # Add classifier/lm_head (high curvature component)
               if self.inspector.classifier is not None:
                   subset.extend(self.inspector.classifier.parameters())
               return subset

 1.5.64 Strategic scheduling design
        ═══════════════════════════════
        Hutchinson trace is MORE expensive than SAM-sharpness (requires
        gradient computation graph or an extra forward+backward pass).
        Schedule it MORE conservatively:

        Default: trace_interval = 0 (epoch boundaries only)
        - Typical fine-tuning: 10 epochs → 10 trace measurements
        - Each measurement on a parameter subset (~15% of params)
        - BEST CASE (create_graph=True in user's loss.backward()):
          Per-measurement cost: ~0.15× a backward pass (autograd on subset)
          Total overhead: 10 × 0.15 / 1500 steps ≈ 0.1%
        - WORST CASE (create_graph=False, default — finite-difference fallback):
          Per-measurement cost: ~1× a full forward+backward pass
          Total overhead: 10 × 1.0 / 1500 steps ≈ 0.7%
        - Document: users who want cheap trace should pass
          create_graph=True in their loss.backward() call

        Triggered computation (supplements interval):
        - When loss_spike_ratio > 3.0 → compute trace on NEXT step
          (sharpness spike may indicate transition to sharp region)
        - When sharpness_sam crosses a threshold (|Δsharpness| > 2× mean)
          → compute trace to confirm whether landscape changed globally
          or only in the gradient direction

        The collector orchestrates this: it passes a _loss_spike_detected
        flag to CurvatureMetrics when loss_spike_ratio exceeds threshold.

        Config additions to ExtractionConfig:
        - trace_interval: int = 0  # 0 = epoch-only; N = every N steps
        - curvature_subset_fraction: float = 0.15  # param subset size

 1.5.65 Cost analysis — DEFault++ vs DEFault
        ═══════════════════════════════════════
        DEFault (MLP/CNN/RNN, ~100K params):
          - 3 mini-batches × (1 forward + 1 backward + 1 Hessian backward)
          - = 9 extra passes per BATCH STEP
          - Overhead: ~700% per step

        DEFault++ (transformer, ~100M params):
          SAM-sharpness:
          - 1 extra forward pass every 50 steps
          - Overhead: 1/50 × (forward_cost / total_step_cost) ≈ 0.7%
          Hutchinson trace:
          - Best: 0.15× backward on subset at epoch boundaries → ≈ 0.1%
          - Worst: 1× full step at epoch boundaries → ≈ 0.7%
          Combined: ≈ 0.8–1.4% overhead (vs DEFault's 700%)

        This is ~500-875× cheaper than DEFault's approach while capturing
        the same two diagnostic dimensions:
          1. Directional sharpness (SAM → DEFault's HVP direction)
          2. Average curvature (Hutchinson trace → DEFault's mean HVP)

 1.5.66 Correctness verification — SAM sharpness
        Test: test_sharpness_sam
        Strategy: Verify mathematical properties, not exact values
        (curvature is stochastic and model-dependent).

        a) Existence and non-negative:
           - Run forward+backward on tiny BERT
           - Call CurvatureMetrics.collect() at step=compute_interval
           - Assert 'sharpness_sam' in result
           - Assert 'sharpness_sam_normalized' in result
           - Assert sharpness_sam >= -0.01
             (should be ≥ 0 for convex-ish regions; small negative allowed
              due to numerical precision)

        b) Weight restoration verification (CRITICAL):
           - Record all model weights before collect()
           - Call collect() (which perturbs and restores weights)
           - Assert ALL weights are identical to pre-call values
             (within 1e-7 tolerance for floating point)
           - Rationale: if weights aren't restored, we've corrupted the
             model during diagnosis. This is a safety-critical test.

        c) Scaling test:
           - Compute sharpness on a normal model → S_normal
           - Multiply all weights by 0.01 (very small weights → flat region)
           - Compute sharpness → S_flat
           - Assert S_flat < S_normal
           - Rationale: near-zero weights = near-zero loss surface = flat

        d) Caching test:
           - Call collect() at step != compute_interval
           - Assert returned value equals the cached value (no recomputation)
           - Call collect() at step == compute_interval
           - Assert returned value MAY differ (recomputation happened)

        e) Decoder test:
           - Run same tests on tiny GPT-2
           - Verify metric is computed correctly

 1.5.67 Correctness verification — Hutchinson trace
        Test: test_hutchinson_trace
        Strategy: Verify against known-curvature cases.

        a) Existence:
           - Assert 'hessian_trace' in result
           - Assert 'hessian_trace_normalized' in result

        b) Positive curvature near minimum:
           - Train a tiny model for a few steps (so it's near a minimum)
           - Assert hessian_trace > 0
             (near a minimum, curvature should be positive)

        c) Quadratic loss verification (synthetic):
           - Create a trivial 2-parameter model: y = ax + b
           - Loss = (y - target)^2 (quadratic in a, b)
           - The Hessian of a quadratic is CONSTANT:
             H = 2 × [[x^2, x], [x, 1]] (for a single data point)
             tr(H) = 2 × (x^2 + 1)
           - Compute Hutchinson estimate with K=100 samples
           - Assert estimate is within 20% of true trace
           - Rationale: for a quadratic loss, Hutchinson should converge
             to the exact trace. This validates the implementation.

        d) Parameter subset test:
           - Verify that _get_curvature_params_subset returns params from
             sampled layers + classifier head
           - Assert subset size < total model params
           - Assert subset is non-empty

        e) Triggered computation test:
           - Set trace_interval=0 (epoch-only)
           - Call collect() at normal steps → assert cached value returned
           - Call collect() with _loss_spike_detected=True →
             assert fresh computation happens (value may change)

        f) Decoder test:
           - Run on tiny GPT-2
           - Verify trace is computed from sampled layers + lm_head

────────────────────────────────────────────────────────────────────────────────
METRIC 15: activation_magnitude — Absolute Activation Scale Monitoring
────────────────────────────────────────────────────────────────────────────────
Module: extraction/metrics/structural.py (StructuralMetrics)
Diagnoses: Activation magnitude drift, fp16 overflow risk, uniform scaling
           pathologies that relative metrics cannot detect
Why needed: All existing structural metrics are RELATIVE:
              - ffn_delta_mean: difference between layers (delta)
              - residual_cos_mean: angle between layers (cosine)
              - ffn_var_ratio_mean: ratio of variances (ratio)
              - ln_std_mean: std of output (normalized by LN)

            If all hidden states uniformly drift to large values (e.g.,
            every layer outputs values around 10,000), these relative
            metrics look perfectly healthy:
              - delta between 10,000-valued layers can still be small
              - cosine between 10,000-valued vectors is unaffected by scale
              - variance ratio of large values can still be ~1.0
              - LayerNorm normalizes away the absolute scale

            But the model is in serious trouble:
              - fp16 max value is 65,504. Values > 10,000 are 1 multiplication
                away from overflow → Inf → NaN cascade
              - Attention scores Q·K^T scale with hidden state magnitude.
                If hidden states are 100× larger than expected, attention
                scores are 10,000× larger → softmax saturation → gradient
                vanishing through attention
              - Optimizer state (Adam's second moment) tracks magnitude².
                Large activations → large gradients → large v_t → small
                effective LR → training stalls

            DEFault (ICSE 2025 version) captures this via mean_activation /
            std_activation averaged across layers. We adapt this for
            transformers with per-layer granularity and fp16-aware thresholds.

 1.5.68 Theory (invariant — same for all architectures)
        ═══════════════════════════════════════════════════
        For hidden state tensor H_l ∈ ℝ^{B×S×d} at layer l:

        activation_magnitude_l = mean_{b,s,j} |H_l[b, s, j]|
          = the mean absolute value of all hidden state entries

        This is the L1-norm averaged over all elements:
          activation_magnitude_l = ||H_l||_1 / (B × S × d)

        Aggregated across sampled layers:
          activation_magnitude_mean = mean_l activation_magnitude_l
          activation_magnitude_max  = max_l activation_magnitude_l

        Additionally, track the STANDARD DEVIATION of activation values
        to capture spread:
          activation_std_mean = mean_l std(H_l)

        Together these answer three questions:
          1. How large are the activations? (magnitude_mean)
          2. Is any layer disproportionately large? (magnitude_max)
          3. How spread out are the values? (std_mean)

        Interpretation:
          magnitude_mean ≈ 0.5–5.0    → normal range for most transformers
          magnitude_mean > 100         → danger zone for fp16
          magnitude_mean > 10,000      → imminent overflow
          magnitude_max >> magnitude_mean → one layer is diverging
          std_mean ≈ magnitude_mean    → values are spread (healthy)
          std_mean << magnitude_mean   → values are clustered (collapse)

        Why L1-mean instead of L2-norm:
          - L1-mean is bounded by the actual value range (interpretable:
            "average activation is 3.2" makes sense)
          - L2-norm grows with √(B×S×d) and is not comparable across
            different batch sizes or sequence lengths
          - L1-mean is O(n) and trivially cheap

        Relationship to other metrics:
          - embedding_norm_mean: tracks embedding layer specifically
          - ln_std_mean: tracks post-LayerNorm statistics (normalized)
          - activation_magnitude: tracks pre-normalization raw values
            (the ones that actually overflow)

        Why this is NOT redundant with ln_std_mean:
          LayerNorm normalizes: LN(x) = (x - μ) / σ × γ + β
          ln_std_mean captures the statistics of LN OUTPUT (post-norm).
          But the OVERFLOW happens in LN INPUT (pre-norm):
            - Computing μ = mean(x): if x values are huge, μ is huge
              but the subtraction x - μ can lose precision
            - Computing σ = std(x): if x values are huge but similar,
              σ can underflow to 0 → division by zero → NaN
          activation_magnitude captures the PRE-norm values that cause
          these failures. It's the early warning before LN breaks.

 1.5.69 Encoder specifics (BERT-style)
        ═════════════════════════════════
        Hidden state flow in BERT:
          embedding → LN → [attention → residual → LN → FFN → residual → LN] × L

        Where activation magnitude matters:
        a) Post-embedding, pre-first-LN:
           - Embeddings are learned vectors, typically magnitude 1–10
           - If embedding table has been corrupted (NaN weights, bad init),
             this is where it shows up first
           - hidden_states[0] captures this

        b) Post-FFN, pre-residual-add (between layers):
           - FFN output can grow if GELU doesn't adequately gate
           - Residual add: h_out = h_in + FFN(LN(h_in))
             If FFN output is 1000× larger than h_in, the residual
             connection is dominated by FFN → defeats the purpose of
             skip connections
           - hidden_states[l] for l > 0 captures this

        c) Post-LN, pre-attention:
           - LN should normalize magnitudes, but with learned γ (scale),
             the output can still be large: LN(x) = normalized × γ
             If γ is large (which can happen during training), LN output
             is large → Q, K projections produce large values →
             Q·K^T / √d_k can overflow
           - This is the most common fp16 overflow path in BERT

        Expected ranges for BERT-base (hidden_size=768):
          - Healthy: activation_magnitude_mean ∈ [0.3, 5.0]
          - Concerning: > 20 (approaching fp16 precision limits)
          - Critical: > 500 (overflow imminent)

        Implementation: Use hidden_states from model output
        (output_hidden_states=True), which are the POST-layer hidden states.
        These are the inputs to the NEXT layer's LayerNorm — exactly the
        values where overflow matters.

 1.5.70 Decoder specifics (GPT-style)
        ═════════════════════════════════
        Hidden state flow in GPT-2:
          embedding + position_embedding → [LN → attention → residual → LN → MLP → residual] × L → LN

        Key differences from encoder:
        a) Pre-LN architecture (GPT-2):
           GPT-2 applies LayerNorm BEFORE attention/MLP, not after.
           This means hidden_states[l] = post-residual, PRE-LayerNorm.
           These values are NOT normalized — they accumulate residual
           contributions from all previous layers.

           Consequence: activation_magnitude naturally GROWS with depth
           in GPT-2 (each layer adds its output to the residual stream).
           A monotonically increasing activation_magnitude_l across layers
           is NORMAL for pre-LN architectures, not a fault.

           Diagnostic: the RATE of growth matters.
           - Linear growth: magnitude_l ∝ l → healthy (each layer adds O(1))
           - Exponential growth: magnitude_l ∝ 2^l → pathological

        b) Post-LN architecture (some newer decoders):
           Same as encoder — LN after each sublayer, hidden_states are
           post-LN. Magnitudes should be relatively stable across layers.

        c) Position embedding addition:
           h_0 = word_embedding + position_embedding
           Both can be large independently. If position embeddings are
           poorly initialized (e.g., for long sequences they haven't seen),
           h_0 can be anomalously large → affects all downstream layers.

        Expected ranges for GPT-2 (hidden_size=768):
          - Healthy: activation_magnitude_mean ∈ [0.5, 10.0]
          - Concerning: > 50
          - Critical: > 1000
          (Slightly higher than BERT due to pre-LN residual accumulation)

 1.5.71 Implementation
        In StructuralMetrics.collect(), within the existing per-layer loop
        that already iterates over hidden_states:

        a) Add collection lists before the layer loop:
           activation_magnitudes = []
           activation_stds = []

        b) Inside the loop, for each layer_idx (already iterating):
           # Already have: h_in = hs_list[layer_idx]
           # Use h_in directly (these are the raw hidden states)
           flat = h_in.reshape(-1)  # flatten all dimensions
           if flat.numel() > 0:
               act_mag = flat.abs().mean().item()
               act_std = flat.std().item()
               metrics[f'activation_magnitude_l{layer_idx}'] = float(act_mag)
               metrics[f'activation_std_l{layer_idx}'] = float(act_std)
               activation_magnitudes.append(act_mag)
               activation_stds.append(act_std)

        c) After the layer loop, compute aggregates:
           if activation_magnitudes:
               metrics['activation_magnitude_mean'] = float(
                   np.mean(activation_magnitudes))
               metrics['activation_magnitude_max'] = float(
                   max(activation_magnitudes))
               metrics['activation_std_mean'] = float(
                   np.mean(activation_stds))

        Performance: O(B × S × d) per layer — one pass to compute
        abs().mean() and std(). This is cheaper than the FFN delta
        computation we already do (which requires TWO hidden states
        and a subtraction). The hidden states are already in memory
        (we iterate them for ffn_delta, residual_cos, etc.), so there
        is zero additional memory cost.

        No config additions needed. No thresholds (the values are
        continuous and interpreted by the diagnosis model).

 1.5.72 Correctness verification
        Test: test_activation_magnitude
        Strategy: Verify against manual computation, verify sensitivity
        to scaling, and verify the fp16-overflow detection use case.

        a) Existence and type:
           - Create tiny BERT, run forward with output_hidden_states=True
           - Call StructuralMetrics.collect() with hidden_states
           - Assert 'activation_magnitude_mean' in result
           - Assert 'activation_magnitude_max' in result
           - Assert 'activation_std_mean' in result
           - Assert all three are finite positive floats

        b) Manual verification:
           - For hidden_states[0] (embedding output):
             manual_mag = hidden_states[0].abs().mean().item()
           - Assert result['activation_magnitude_l0'] ≈ manual_mag (±1e-5)
           - Compute manual mean across all layers
           - Assert result['activation_magnitude_mean'] matches (±1e-5)

        c) Normal range check:
           - For a randomly initialized tiny BERT:
             Assert 0.01 < activation_magnitude_mean < 50.0
             (random init should produce values in a reasonable range)

        d) Scaling sensitivity test:
           - Record activation_magnitude_mean with normal model
           - Multiply ALL embedding weights by 100:
             with torch.no_grad():
                 inspector.embedding.weight.data *= 100
           - Re-run forward pass, call collect()
           - Assert new activation_magnitude_mean > old × 10
             (100× embedding scale → hidden states should be much larger)
           - Rationale: proves the metric detects activation drift

        e) Relative metrics DON'T catch what this catches:
           - After the 100× scaling in (d):
             Assert residual_cos_mean is still reasonable (> 0.5)
             (cosine is scale-invariant — it DOESN'T detect the problem)
             Assert activation_magnitude_mean has changed dramatically
             (magnitude IS scale-sensitive — it DOES detect the problem)
           - This is the KEY test: proves activation_magnitude catches
             failures that relative metrics miss

        f) Per-layer granularity test:
           - Assert activation_magnitude_l0 exists (first layer)
           - For a model with N layers, assert N per-layer values exist
           - Assert activation_magnitude_max == max of per-layer values

        g) Decoder test:
           - Run (a), (b), (c) on tiny GPT-2
           - Verify metric exists and matches manual computation
           - Verify activation_magnitude grows with depth for pre-LN GPT-2
             (this is expected behavior, not a fault):
             Assert activation_magnitude_l{last} >= activation_magnitude_l0
             (may not always hold for tiny random models, so use >= not >)

        h) fp16 overflow simulation (synthetic):
           - Create hidden_states tensor with values = 60,000
             (just below fp16 max of 65,504)
           - Call collect() with these synthetic hidden_states
           - Assert activation_magnitude_mean > 50,000
           - Rationale: if a user is training in fp16 and sees
             activation_magnitude_mean approaching 60K, they know
             overflow is imminent. No other metric tells them this.

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1.5 GATE TEST — FINAL (includes Metrics 1-15)                        │
│                                                                              │
│ tests/test_phase1_5_gate.py:                                                 │
│                                                                              │
│ Gradient metrics (2 new — from original Phase 1.5):                         │
│  T1.5.1  grad_norm_ratio_first_last exists and is finite positive float     │
│  T1.5.2  grad_norm_ratio_first_last matches manual computation              │
│  T1.5.3  grad_cosine_successive: first call returns 0.0                     │
│  T1.5.4  grad_cosine_successive: second call returns value in [-1, 1]       │
│  T1.5.5  grad_cosine_successive: identical input → cosine near 1.0          │
│                                                                              │
│ Structural metrics (3 new):                                                  │
│  T1.5.6  dead_neuron_frac_mean exists and is in [0, 1]                      │
│  T1.5.7  dead_neuron_frac: healthy model has frac < 0.5                     │
│  T1.5.8  representation_rank_mean exists and is in (0, 1]                   │
│  T1.5.9  representation_rank: random Gaussian → rank near 1.0               │
│  T1.5.10 representation_rank: identical tokens → rank near 0                │
│  T1.5.11 token_isotropy_mean exists and is in [-1, 1]                       │
│  T1.5.12 token_isotropy: identical tokens → isotropy = 1.0                  │
│  T1.5.13 token_isotropy: random Gaussian → isotropy near 0.0                │
│                                                                              │
│ Attention metrics (2 new):                                                   │
│  T1.5.14 attention_sink_score exists and is in [0, 1]                       │
│  T1.5.15 attention_sink_score: uniform attn → score = 1/seq_len             │
│  T1.5.16 dead_head_count exists and is non-negative integer                 │
│  T1.5.17 dead_head_frac: uniform attn → frac = 1.0                         │
│  T1.5.18 dead_head_frac: peaked attn → frac = 0.0                          │
│                                                                              │
│ Training metrics (1 new):                                                    │
│  T1.5.19 loss_spike_ratio: first step = 1.0                                │
│  T1.5.20 loss_spike_ratio: constant loss → ratio ≈ 1.0                     │
│  T1.5.21 loss_spike_ratio: 10× spike detected (ratio ≈ 10)                 │
│                                                                              │
│ NaN/Inf detection (Metrics 9-10):                                           │
│  T1.5.25 nan_weight_count: healthy model → 0                               │
│  T1.5.26 nan_weight_count: injected NaN → count >= 1                       │
│  T1.5.27 inf_weight_count: injected Inf → count >= 1                       │
│  T1.5.28 nan_weight_count: exact count matches injected count (5 NaN → 5)  │
│  T1.5.29 nan_gradient_count: healthy model → 0                             │
│  T1.5.30 nan_gradient_count: injected NaN grad → count >= 1                │
│  T1.5.31 inf_gradient_count: injected Inf grad → count >= 1                │
│  T1.5.32 nan_gradient_count: proves NaN blind spot in vanish/explode flags  │
│                                                                              │
│ Weight norm (Metric 11):                                                     │
│  T1.5.33 weight_norm_max exists and is finite positive                      │
│  T1.5.34 weight_norm_max matches manual Frobenius norm computation          │
│  T1.5.35 weight_norm_max: 100× scaling detected (norm increases ~100×)     │
│  T1.5.36 weight_norm_max/mean ratio: healthy < 20, scaled > 50             │
│                                                                              │
│ Gradient median (Metric 12):                                                │
│  T1.5.37 gradient_median exists and is finite non-negative                  │
│  T1.5.38 gradient_median matches manual median computation                  │
│  T1.5.39 gradient_median robust to outlier (1000× one layer scaling)       │
│  T1.5.40 gradient_mean_per_layer / gradient_median ≈ 1 for healthy model   │
│                                                                              │
│ Curvature metrics (Metrics 13-14):                                          │
│  T1.5.42 sharpness_sam exists and is >= -0.01                               │
│  T1.5.43 sharpness_sam: weights RESTORED after perturbation (safety)        │
│  T1.5.44 sharpness_sam: flat model (tiny weights) < normal model            │
│  T1.5.45 sharpness_sam: caching works (non-compute step returns cache)      │
│  T1.5.46 hessian_trace exists after epoch-boundary call                     │
│  T1.5.47 hessian_trace > 0 near minimum (trained a few steps)              │
│  T1.5.48 hessian_trace: quadratic loss → estimate ≈ true trace (±20%)      │
│  T1.5.49 hessian_trace: param subset < total params                         │
│  T1.5.50 hessian_trace: triggered by _loss_spike_detected flag             │
│                                                                              │
│ Activation magnitude (Metric 15):                                           │
│  T1.5.51 activation_magnitude_mean exists and is finite positive            │
│  T1.5.52 activation_magnitude_mean matches manual abs().mean() computation  │
│  T1.5.53 activation_magnitude_mean in normal range (0.01–50) for tiny BERT │
│  T1.5.54 100× embedding scaling → magnitude increases dramatically         │
│  T1.5.55 100× scaling: residual_cos DOESN'T catch it, magnitude DOES       │
│  T1.5.56 Per-layer values exist; max == max(per-layer values)              │
│  T1.5.57 Decoder: metric exists and matches manual computation             │
│  T1.5.58 Synthetic fp16 overflow: values=60000 → magnitude > 50000         │
│                                                                              │
│ Integration:                                                                 │
│  T1.5.22 All 15 new metric groups present in MetricCollector output        │
│  T1.5.23 No new metric produces NaN under normal input                      │
│  T1.5.24 Phase 1 gate tests STILL PASS (no regression)                      │
│  T1.5.41 Encoder (BERT) + Decoder (GPT-2) both produce all metrics        │
│                                                                              │
│ Run: pytest tests/test_phase1_5_gate.py tests/test_phase1_gate.py -v         │
│ All 58 tests must PASS.                                                      │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 1.6: AUDIT FIXES — BUGS, MISSING VARIANTS, NORMALIZATION
================================================================================
Goal: Address all issues discovered during the comprehensive metric audit.
      These are NOT new diagnostic metrics — they are corrections to existing
      code, missing statistical variants, normalization for cross-model
      comparability, and two small high-value additions.
Dependencies: Phase 1 (existing implementations) and Phase 1.5 (planned metrics)
Note: Phase 1.6 tasks can be done alongside Phase 1.5 implementation.

Prioritization:
  MUST FIX   = bugs that produce wrong/missing/crashing output
  SHOULD ADD = high diagnostic value, zero/low cost
  NORMALIZE  = cross-model comparability improvements

────────────────────────────────────────────────────────────────────────────────
BUGFIX 1: Division by zero in training.py — step_time guard
────────────────────────────────────────────────────────────────────────────────
Priority: MUST FIX
File: src/defaultplusplus/extraction/metrics/training.py
Line: 44 (approximately)

 Problem:
   The current code:
     if step_time is not None and step_time > 0:
         metrics['runtime_step_time'] = float(step_time)
         metrics['runtime_steps_per_sec'] = 1.0 / step_time

   The condition `step_time > 0` guards against negative values but NOT
   against step_time = 0.0 exactly. In practice, `time.time()` difference
   can return 0.0 on fast operations (sub-microsecond resolution on some
   systems). If step_time = 0.0:
     - `step_time > 0` is False → block is skipped → no crash
   So this is NOT actually a crash bug in the current code.

   HOWEVER, the real issue is: when step_time is None (not provided),
   runtime_step_time and runtime_steps_per_sec are ABSENT from the output
   dict. This means:
     - EpochAggregator never sees these keys → no _mean/_var at epoch level
     - Windowed features never include them
     - Downstream models that expect these features get KeyError

 Fix:
   Always emit both metrics. Use 0.0 as default when step_time unavailable:

   if step_time is not None and step_time > 0:
       metrics['runtime_step_time'] = float(step_time)
       metrics['runtime_steps_per_sec'] = 1.0 / step_time
   else:
       metrics['runtime_step_time'] = 0.0
       metrics['runtime_steps_per_sec'] = 0.0

 Test: test_bugfix_step_time_guard
   a) step_time=None → assert both keys exist and equal 0.0
   b) step_time=0.0 → assert both keys exist and equal 0.0 (no crash)
   c) step_time=0.5 → assert runtime_steps_per_sec == 2.0
   d) step_time=-1.0 → assert both keys exist and equal 0.0 (negative guard)

────────────────────────────────────────────────────────────────────────────────
BUGFIX 2: cache_nll_divergence never computed — remove or implement
────────────────────────────────────────────────────────────────────────────────
Priority: MUST FIX
File: src/defaultplusplus/extraction/metrics/cache.py

 Problem:
   cache_nll_divergence is initialized to 0.0 (line 28) but no code ever
   computes or updates it. It always returns 0.0, providing zero diagnostic
   signal while consuming a feature slot.

   The INTENT was: compare loss with KV-cache vs. loss without cache to
   detect caching-related divergence. But this requires running the model
   TWICE with different cache settings — expensive.

 Decision: REMOVE the metric entirely.
   Reasoning:
   - Computing it properly requires 2× forward passes (prohibitive)
   - The diagnostic intent (cache corruption) is partially captured by
     cache_hidden_sim (high similarity = repetitive/stuck generation)
   - A never-computed metric that always returns 0.0 is worse than absent:
     it gives false confidence that "divergence is zero" when in reality
     it was never measured

 Fix:
   Remove 'cache_nll_divergence' from the output dict in CacheMetrics.collect()

 Test: test_bugfix_cache_nll_removed
   a) Assert 'cache_nll_divergence' NOT in CacheMetrics.collect() output
   b) Assert 'cache_hidden_sim' still present and correctly computed
   c) Assert no regression in MetricCollector output for decoder models

────────────────────────────────────────────────────────────────────────────────
BUGFIX 3: Duplicate 'loss' alias creates redundant windowed features
────────────────────────────────────────────────────────────────────────────────
Priority: MUST FIX
File: src/defaultplusplus/extraction/metrics/training.py

 Problem:
   TrainingMetrics emits both 'train_loss' and 'loss' with identical values.
   When EpochAggregator processes these, it creates:
     train_loss_mean, train_loss_var, train_loss_count
     loss_mean, loss_var, loss_count           ← DUPLICATE
   And windowed features:
     train_loss_early_mean, train_loss_early_slope, ...
     loss_early_mean, loss_early_slope, ...     ← DUPLICATE

   This doubles the feature count for loss-related features, wastes
   computation, and confuses practitioners ("which loss do I look at?").

 Fix:
   Remove the 'loss' alias from TrainingMetrics.collect(). Only emit
   'train_loss'. If a top-level 'loss' alias is needed for the public
   API (e.g., monitor.step(loss=...)), handle it in the collector or
   core.py, not in the metric module.

   Delete the line:
     metrics['loss'] = metrics['train_loss']

 Test: test_bugfix_no_loss_alias
   a) Call TrainingMetrics.collect(loss=1.5)
   b) Assert 'train_loss' in result and result['train_loss'] == 1.5
   c) Assert 'loss' NOT in result
   d) Run full collector + finalize_epoch → assert no 'loss_mean' key
      (only 'train_loss_mean')

────────────────────────────────────────────────────────────────────────────────
BUGFIX 4: Pre-softmax stats fail silently for fused QKV (GPT-2)
────────────────────────────────────────────────────────────────────────────────
Priority: MUST FIX
File: src/defaultplusplus/extraction/metrics/attention.py

 Problem:
   _compute_pre_softmax_stats() checks:
     if qkv.qkv_style != 'separate': return {}

   This means GPT-2 (which uses fused c_attn) gets NO pre-softmax
   statistics at all — 4 features silently missing. These features are
   diagnostically important:
     - pre_softmax_score_mean: detects attention score magnitude issues
     - pre_softmax_score_var: detects score distribution pathologies
     - pre_softmax_score_skew/kurt: distributional shape

   GPT-2's c_attn is a single Linear(hidden_size, 3*hidden_size) where:
     output[:, :, 0:hidden_size] = Q
     output[:, :, hidden_size:2*hidden_size] = K
     output[:, :, 2*hidden_size:3*hidden_size] = V

 Fix:
   Add fused QKV handling in _compute_pre_softmax_stats():

   if qkv.qkv_style == 'fused' and len(qkv.qkv_names) == 1:
       fused_name = qkv.qkv_names[0]  # 'c_attn'
       fused_mod = self.inspector._find_submodule(attn_module, fused_name)
       if fused_mod is None:
           return {}
       with torch.no_grad():
           hidden = layer_input.detach()
           batch_size, seq_len, hidden_size = hidden.shape
           n_heads = self.inspector.num_heads or 1
           dim_per_head = hidden_size // n_heads
           # Fused projection: [B, S, 3*H] → split into Q, K, V
           qkv_out = fused_mod(hidden)  # [B, S, 3*hidden_size]
           q, k, _ = qkv_out.split(hidden_size, dim=-1)
           # Reshape to multi-head
           q = q.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)
           k = k.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)
           scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dim_per_head)
           # ... same masking and stats as separate path ...

 Test: test_bugfix_fused_qkv_pre_softmax
   a) Create tiny GPT-2 model (uses fused c_attn)
   b) Run AttentionMetrics.collect() with attention_weights + hidden_states
   c) Assert 'pre_softmax_score_mean' in per-layer result (was previously
      missing for GPT-2)
   d) Assert value is finite and non-zero
   e) Verify: manual QK^T computation on GPT-2 matches reported score_mean
      - Get GPT-2's c_attn weight, split into Q/K/V thirds
      - Compute Q·K^T/√d manually
      - Compare mean to reported pre_softmax_score_mean (within 1e-4)

────────────────────────────────────────────────────────────────────────────────
BUGFIX 5: Inconsistent global attention alias aggregation
────────────────────────────────────────────────────────────────────────────────
Priority: MUST FIX (naming/semantics, not crash)
File: src/defaultplusplus/extraction/metrics/attention.py

 Problem:
   Global aliases use inconsistent aggregation and unintuitive names:

   Current:
     mass_pad      = MAX across layers
     mass_leak     = MAX across layers
     head_similarity_mean = MAX across layers  ← confusing: "mean" in name but MAX in aggregation

   The word "mean" in "head_similarity_mean" suggests averaging, but the
   global value is actually the MAX of per-layer means. A practitioner
   seeing head_similarity_mean=0.95 would think "average head similarity
   is 0.95" when it actually means "the WORST layer has average head
   similarity 0.95".

 Fix (two parts):

   Part A — Rename global aliases to reflect aggregation method:
   | Old Name | New Name | Aggregation |
   |---|---|---|
   | mass_pad | attention_pad_mass_max | max |
   | mass_leak | attention_leak_mass_max | max |
   | cross_example_attention | attention_cross_leak_max | max |
   | attention_mass_future | attention_future_mass_max | max |
   | head_similarity_mean | attention_head_sim_layer_max | max of per-layer means |
   | head_similarity_max | attention_head_sim_pair_max | max of per-layer maxes |

   Part B — Also emit MEAN variants for metrics that use MAX:
   For each MAX alias, also emit the MEAN across layers. This gives both
   "worst case" (max) and "typical case" (mean) perspectives.
     attention_pad_mass_mean = mean across layers (in addition to max)
     attention_head_sim_layer_mean = mean across layers (in addition to max)

 Test: test_bugfix_attention_alias_names
   a) Assert old names (mass_pad, mass_leak, etc.) are NOT in output
   b) Assert new names (attention_pad_mass_max, etc.) ARE in output
   c) For a model with 3 sampled layers:
      - Manually compute max and mean of per-layer attention_mass_pad_mean
      - Assert attention_pad_mass_max == max of the 3 per-layer values
      - Assert attention_pad_mass_mean == mean of the 3 per-layer values
   d) Run on both BERT and GPT-2 → verify renamed keys present

────────────────────────────────────────────────────────────────────────────────
ADDITION 1: Structural metric std/min/max variants
────────────────────────────────────────────────────────────────────────────────
Priority: SHOULD ADD
File: src/defaultplusplus/extraction/metrics/structural.py

 Problem:
   We compute 7 aggregated structural metrics as MEAN across layers:
     ffn_delta_mean, residual_cos_mean, ffn_var_ratio_mean,
     ln_std_mean, ln_mean_abs_mean, ffn_active_dim_frac_mean, ffn_out_skew_mean

   But we do NOT compute std/min/max variants. This means we cannot
   distinguish:
     Scenario A: ffn_delta_mean=5.0, all layers ≈ 5.0 (uniform, healthy)
     Scenario B: ffn_delta_mean=5.0, one layer=49.5, rest≈0.05 (localized fault)

   These scenarios require completely different diagnoses but produce
   identical mean values.

 Theory (invariant):
   For any metric m with per-layer values {m_0, m_1, ..., m_{L-1}}:
     m_mean = (1/L) Σ m_i                          (central tendency)
     m_std  = sqrt((1/L) Σ (m_i - m_mean)²)        (dispersion)
     m_min  = min_i m_i                             (best-case layer)
     m_max  = max_i m_i                             (worst-case layer)

   The ratio m_std / (|m_mean| + ε) is the coefficient of variation:
     CV ≈ 0   → uniform across layers (healthy)
     CV > 1   → one layer dominates (localized fault)

   We do NOT add CV as a separate metric (it's a ratio of existing
   metrics, computable at diagnosis time). We add the raw std/min/max
   so the diagnosis model can learn whatever pattern is relevant.

 Implementation:
   In StructuralMetrics.collect(), after the per-layer loop where we
   already have lists: delta_means, cos_means, var_ratios, ln_std_means,
   ln_mean_abs_means, active_fracs, skew_vals.

   For each list, add:
     if delta_means:
         metrics['ffn_delta_mean'] = float(np.mean(delta_means))
         metrics['ffn_delta_std'] = float(np.std(delta_means))      # NEW
         metrics['ffn_delta_min'] = float(min(delta_means))          # NEW
         metrics['ffn_delta_max'] = float(max(delta_means))          # NEW
     # ... same pattern for all 7 aggregated metrics ...

   Cost: Zero. We already have the lists in memory.

   New features (7 metrics × 3 variants = 21 new features):
     ffn_delta_std, ffn_delta_min, ffn_delta_max
     residual_cos_std, residual_cos_min, residual_cos_max
     ffn_var_ratio_std, ffn_var_ratio_min, ffn_var_ratio_max
     ln_std_std, ln_std_min, ln_std_max
     ln_mean_abs_std, ln_mean_abs_min, ln_mean_abs_max
     ffn_active_dim_frac_std, ffn_active_dim_frac_min, ffn_active_dim_frac_max
     ffn_out_skew_std, ffn_out_skew_min, ffn_out_skew_max

 Test: test_structural_std_min_max
   a) Assert all 21 new keys present in output
   b) For ffn_delta: assert min <= mean <= max
   c) For a model with 5 layers: manually compute std/min/max of
      [ffn_delta_l0_mean, ..., ffn_delta_l4_mean] and compare
   d) Uniform case: if all per-layer values are identical,
      assert std ≈ 0.0 and min == max == mean
   e) Outlier case: manually set hidden_states[3] to huge values →
      assert ffn_delta_max >> ffn_delta_mean and ffn_delta_std > 0

────────────────────────────────────────────────────────────────────────────────
ADDITION 2: attention_entropy_normalized — Cross-length comparability
────────────────────────────────────────────────────────────────────────────────
Priority: SHOULD ADD
File: src/defaultplusplus/extraction/metrics/attention.py

 Problem:
   attention_entropy_mean is in nats, range [0, log(seq_len)].
   For seq_len=64: max entropy = 4.16
   For seq_len=512: max entropy = 6.24

   The same model producing entropy=4.0 is "nearly uniform" at seq_len=64
   but "moderately peaked" at seq_len=512. Practitioners comparing across
   different datasets (different sequence lengths) will be misled.

 Theory (invariant):
   Normalized entropy:
     H_norm = H / H_max = (-Σ p_i log p_i) / log(N)

   where N = number of positions (key length).

   Properties:
     H_norm ∈ [0, 1] for all sequence lengths
     H_norm = 1.0 → uniform attention (maximum uncertainty)
     H_norm = 0.0 → fully concentrated on one position
     H_norm is DIRECTLY COMPARABLE across different sequence lengths

   This is also called the "efficiency" or "evenness" of a distribution
   in information theory (Pielou, 1966).

 Encoder vs Decoder:
   - Encoder: N = seq_len for all heads (bidirectional)
   - Decoder: N = position+1 for causal attention at each query position.
     For the global metric, use N = seq_len (effective key length for
     the last query position, which sees the full sequence). This is
     an approximation but consistent across comparisons.

 Implementation:
   In AttentionMetrics._compute_layer_metrics(), after computing
   head_entropy and metrics['attention_entropy_mean']:

     max_entropy = math.log(max(attn.size(-1), 1))
     if max_entropy > 0:
         metrics['attention_entropy_normalized'] = (
             metrics['attention_entropy_mean'] / max_entropy
         )
     else:
         metrics['attention_entropy_normalized'] = 0.0

   In the global alias section of collect():
     attention_entropy_normalized = mean across sampled layers

   Cost: One division. Negligible.

 Test: test_attention_entropy_normalized
   a) Assert value in [0, 1]
   b) Uniform attention (all weights = 1/seq_len):
      assert normalized ≈ 1.0 (within 0.01)
   c) Peaked attention (one position = 1.0, rest = 0):
      assert normalized ≈ 0.0 (within 0.01)
   d) Cross-length invariance:
      - Create attention with same relative distribution at seq_len=32 and 128
      - Assert normalized values are equal (within 0.05)
      - Assert RAW entropy values are NOT equal (proving the need)

────────────────────────────────────────────────────────────────────────────────
ADDITION 3: effective_step_size — LR-gradient interaction
────────────────────────────────────────────────────────────────────────────────
Priority: SHOULD ADD
File: src/defaultplusplus/extraction/metrics/gradient.py

 Problem:
   We track learning rate (training.py) and gradient norm (gradient.py)
   separately. But the ACTUAL step size that determines training dynamics
   is their product:

     Δw = -lr × ∇L(w)     (for vanilla SGD)
     ||Δw|| ≈ lr × ||∇L||  (approximate for Adam)

   A high LR with small gradients produces the same step as a low LR with
   large gradients. Neither metric alone tells you if the optimizer is
   taking reasonable steps.

   This replaces DEFault's `adjusted_lr` with something more diagnostic:
   DEFault's adjusted_lr was an INTERVENTION (multiplying LR by 0.9 each
   epoch). Our effective_step_size is a DIAGNOSTIC (are the optimizer steps
   the right size?).

 Theory (invariant):
   effective_step_size = learning_rate × grad_norm_total

   For Adam optimizer (most common for transformers):
     actual_step ≈ lr × grad / (sqrt(v_t) + eps)
   where v_t is the second moment estimate. The true effective step is
   harder to compute (requires v_t), but lr × grad_norm is a useful
   UPPER BOUND and scales proportionally with the actual step.

   Interpretation:
     effective_step_size << 1e-6  → training is effectively frozen
     effective_step_size ≈ 1e-4 to 1e-2  → typical healthy range
     effective_step_size > 1.0    → steps are too large (instability)

   For diagnosis: a model with gradient_vanish=False but
   effective_step_size < 1e-8 has a "silent stall" — gradients exist
   but LR is so small that weights aren't actually changing. This is
   a common fault after aggressive LR decay.

 Encoder vs Decoder:
   - Identical computation
   - Encoder fine-tuning typically: lr=2e-5, grad_norm≈1.0 → step≈2e-5
   - Decoder training: lr=3e-4, grad_norm≈0.5 → step≈1.5e-4
   - Both are in the healthy range

 Implementation:
   In GradientMetrics.collect(), after computing gradient norms:

   # Need optimizer to get LR — passed as kwarg
   lr = 0.0
   if optimizer is not None:
       lr = optimizer.param_groups[0]['lr']
   metrics['effective_step_size'] = lr * metrics.get('grad_norm_total', 0.0)

   Note: optimizer is already passed to collect() but not currently used
   in GradientMetrics. Add it to the kwargs usage.

   Cost: One multiplication. Negligible.

 Test: test_effective_step_size
   a) Assert key exists and is finite non-negative
   b) lr=0.01, grad_norm_total=5.0 → assert effective_step_size ≈ 0.05
   c) lr=0.0 → assert effective_step_size == 0.0 (frozen training)
   d) No optimizer → assert effective_step_size == 0.0

────────────────────────────────────────────────────────────────────────────────
ADDITION 4: loss_batch_std, loss_batch_max — Per-example loss distribution
────────────────────────────────────────────────────────────────────────────────
Priority: SHOULD ADD
File: src/defaultplusplus/extraction/metrics/logit.py

 Problem:
   We compute a single scalar loss (NLL) for the entire batch. This
   hides the DISTRIBUTION of losses across examples:

   Scenario A: batch of 32 examples, all with loss ≈ 0.5 → NLL=0.5
   Scenario B: batch of 32 examples, 31 with loss=0.1, 1 with loss=12.9
               → NLL≈0.5 (dominated by outlier)

   These are completely different situations:
     A: model is uniformly mediocre (needs more training)
     B: model is good but one outlier is corrupting the gradient
        (data quality issue or distribution shift)

   This is particularly important for transformers because:
   - Large batch sizes amplify the outlier-hiding effect
   - Sequence-level tasks have high loss variance by nature
     (some sequences are hard, some are easy)
   - Mixed-precision training can cause single-example overflow that
     goes unnoticed in mean loss but corrupts gradients

 Theory (invariant):
   For a batch of B examples with per-example losses {l_1, ..., l_B}:

   loss_batch_std = std(l_1, ..., l_B)
   loss_batch_max = max(l_1, ..., l_B)

   Interpretation:
     loss_batch_std / NLL ≈ coefficient of variation
       CV ≈ 0   → all examples have similar difficulty (homogeneous batch)
       CV > 1   → losses vary wildly (heterogeneous or outlier-heavy)
     loss_batch_max >> NLL → outlier examples present
     loss_batch_max = Inf → fp16 overflow on specific examples

 Encoder vs Decoder:
   Encoder (classification):
     - Per-example loss via F.cross_entropy(..., reduction='none')
     - Returns [batch_size] tensor
     - Filter out labels == -100

   Decoder (language modeling):
     - Per-example loss via F.cross_entropy(..., reduction='none')
       on reshaped [B*S, V] logits → [B*S] losses
     - Reshape back to [B, S] and mean over sequence dimension per example
       → [B] per-example losses (excluding -100 tokens)
     - This gives per-SEQUENCE loss, not per-token loss

 Implementation:
   In LogitMetrics._compute_performance(), after computing NLL:

   # Per-example loss distribution
   with torch.no_grad():
       if predictions.dim() == 3:  # Decoder
           b, s, v = predictions.shape
           per_token_loss = F.cross_entropy(
               predictions.view(b * s, v), labels.view(b * s),
               ignore_index=-100, reduction='none'
           )  # [B*S]
           per_token_loss = per_token_loss.view(b, s)
           # Mean over sequence, per example
           valid_mask = (labels != -100).float()
           valid_counts = valid_mask.sum(dim=1).clamp(min=1)
           per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_counts
       else:  # Encoder
           per_example_loss = F.cross_entropy(
               predictions, labels, ignore_index=-100, reduction='none'
           )  # [B]

       # Filter out examples where all labels were -100
       valid_examples = per_example_loss[per_example_loss.isfinite()]
       if valid_examples.numel() >= 2:
           metrics['loss_batch_std'] = float(valid_examples.std().item())
           metrics['loss_batch_max'] = float(valid_examples.max().item())
       else:
           metrics['loss_batch_std'] = 0.0
           metrics['loss_batch_max'] = metrics.get('nll', 0.0)

   Cost: One cross_entropy with reduction='none'. Same compute as the
   existing NLL, just without the final mean(). The per_token_loss tensor
   is [B*S] which is small. Negligible cost.

 Test: test_loss_batch_distribution
   a) Assert both keys exist and are finite non-negative
   b) Uniform batch: all examples have same label → loss_batch_std ≈ 0
   c) Manual verification:
      - Compute per-example CE loss manually (reduction='none')
      - Assert loss_batch_std matches numpy.std(per_example_losses)
      - Assert loss_batch_max matches max(per_example_losses)
   d) Outlier detection: create batch where 1 example has random labels,
      rest have correct labels → assert loss_batch_max >> nll
   e) Decoder test: run on GPT-2, assert per-sequence loss stats computed
   f) Edge case: batch_size=1 → assert loss_batch_std == 0.0

────────────────────────────────────────────────────────────────────────────────
ADDITION 5: Normalized structural metrics — cross-model comparability
────────────────────────────────────────────────────────────────────────────────
Priority: NORMALIZE
File: src/defaultplusplus/extraction/metrics/structural.py

 Problem:
   ffn_delta and embedding_norm use L2 norms that scale with √hidden_size:
     - hidden_size=128: random vector norm ≈ √128 ≈ 11.3
     - hidden_size=768: random vector norm ≈ √768 ≈ 27.7
     - hidden_size=1024: random vector norm ≈ √1024 = 32.0

   A ffn_delta of 15.0 is "large relative to random" for hidden_size=128
   but "small relative to random" for hidden_size=1024.

   For DEFault++ to work across different model sizes (which is a stated
   requirement — BERT-tiny to BERT-large), these metrics need normalized
   variants.

 Theory (invariant):
   For a random vector x ~ N(0, 1)^d:
     E[||x||₂] = √d × √(2/π) × Γ((d+1)/2) / Γ(d/2) ≈ √d for large d

   So the expected L2 norm of a random d-dimensional vector is ≈ √d.
   Dividing by √d gives a unit-free metric:

     ffn_delta_normalized = ffn_delta_mean / sqrt(hidden_size)
     embedding_norm_normalized = embedding_norm_mean / sqrt(hidden_size)
     h1_delta_norm_normalized = h1_delta_norm_mean / sqrt(hidden_size)

   These normalized metrics have the interpretation:
     value ≈ 1.0 → magnitude comparable to random initialization
     value >> 1.0 → much larger than random (potential explosion)
     value << 1.0 → much smaller than random (potential collapse)

 Implementation:
   In StructuralMetrics.collect(), after computing the aggregated metrics:

   hidden_size = self.inspector.hidden_size or 1
   sqrt_d = math.sqrt(hidden_size)
   if 'ffn_delta_mean' in metrics:
       metrics['ffn_delta_normalized'] = metrics['ffn_delta_mean'] / sqrt_d
   if 'embedding_norm_mean' in metrics:
       metrics['embedding_norm_normalized'] = metrics['embedding_norm_mean'] / sqrt_d
   if 'h1_delta_norm_mean' in metrics:
       metrics['h1_delta_norm_normalized'] = metrics['h1_delta_norm_mean'] / sqrt_d

   Cost: Three divisions. Negligible.

 Test: test_normalized_structural_metrics
   a) Assert all 3 normalized keys present
   b) For hidden_size=32 model:
      assert ffn_delta_normalized == ffn_delta_mean / sqrt(32)
   c) Cross-model comparison:
      - Create BERT with hidden_size=32 and hidden_size=128
      - Run identical inputs (same random seed)
      - Assert RAW ffn_delta_mean values differ significantly
      - Assert NORMALIZED ffn_delta_normalized values are closer
        (should be within 2× of each other for randomly initialized models)

────────────────────────────────────────────────────────────────────────────────
MIGRATION: Update Phase 1 gate tests for renamed keys
────────────────────────────────────────────────────────────────────────────────
Priority: MUST DO (after implementing Bugfixes 3 and 5)
File: tests/test_phase1_gate.py

 Problem:
   Phase 1.6 Bugfix 3 removes the 'loss' alias and Bugfix 5 renames
   global attention aliases. The existing Phase 1 gate tests (32 tests,
   all passing) assert the OLD names. After implementing the bugfixes,
   these tests will FAIL unless updated.

 Changes needed in test_phase1_gate.py:
   a) test_training_metrics (T1.8):
      - Remove assertion for 'loss' key (only 'train_loss' now)
   b) Any test that checks 'mass_pad', 'mass_leak', 'cross_example_attention',
      'head_similarity_mean', 'head_similarity_max':
      - Replace with new names from Bugfix 5 renaming table
      - Current Phase 1 tests don't directly assert these global aliases
        (they test per-layer keys), so impact may be limited
   c) test_collector_many_metrics (T1.16):
      - The ">50 metrics" threshold should still pass (we added metrics,
        only removed 1 alias). Verify count is still above threshold.
   d) test_no_nan_metrics (T1.15):
      - Iterates ALL output keys — should work with renamed keys
        (checks isinstance(v, float) and not NaN, name-agnostic)

 Test: Run pytest tests/test_phase1_gate.py after applying Bugfixes 3 and 5
       All 32 tests must still PASS.

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1.6 GATE TEST                                                          │
│                                                                              │
│ tests/test_phase1_6_gate.py:                                                 │
│                                                                              │
│ Bugfixes:                                                                    │
│  T1.6.1  step_time=None → both runtime keys exist and equal 0.0            │
│  T1.6.2  step_time=0.0 → no crash, keys equal 0.0                          │
│  T1.6.3  step_time=0.5 → runtime_steps_per_sec == 2.0                      │
│  T1.6.4  step_time=-1.0 → keys equal 0.0                                   │
│  T1.6.5  cache_nll_divergence NOT in CacheMetrics output                    │
│  T1.6.6  cache_hidden_sim still correctly computed                          │
│  T1.6.7  'loss' NOT in TrainingMetrics output (only 'train_loss')           │
│  T1.6.8  No 'loss_mean' in finalize_epoch (only 'train_loss_mean')         │
│  T1.6.9  GPT-2 pre_softmax_score_mean present (fused QKV fix)              │
│  T1.6.10 GPT-2 pre_softmax matches manual QK^T computation                 │
│  T1.6.11 Old attention alias names (mass_pad etc.) NOT in output            │
│  T1.6.12 New alias names (attention_pad_mass_max etc.) ARE in output        │
│  T1.6.13 attention_pad_mass_max == max of per-layer values                  │
│  T1.6.14 attention_pad_mass_mean == mean of per-layer values                │
│                                                                              │
│ Structural std/min/max:                                                      │
│  T1.6.15 All 21 new std/min/max keys present                               │
│  T1.6.16 For each metric: min <= mean <= max                                │
│  T1.6.17 Manual std/min/max matches numpy on per-layer values              │
│  T1.6.18 Uniform layers: std ≈ 0, min == max == mean                       │
│                                                                              │
│ Normalized entropy:                                                          │
│  T1.6.19 attention_entropy_normalized in [0, 1]                             │
│  T1.6.20 Uniform attention → normalized ≈ 1.0                              │
│  T1.6.21 Peaked attention → normalized ≈ 0.0                               │
│  T1.6.22 Cross-length invariance: same distribution at seq=32 and 128      │
│           → normalized values equal, raw values different                   │
│                                                                              │
│ Effective step size:                                                         │
│  T1.6.23 Key exists and is finite non-negative                              │
│  T1.6.24 lr=0.01, grad_norm=5.0 → value ≈ 0.05                            │
│  T1.6.25 lr=0.0 → value == 0.0                                             │
│  T1.6.26 No optimizer → value == 0.0                                        │
│                                                                              │
│ Loss batch distribution:                                                     │
│  T1.6.27 loss_batch_std and loss_batch_max exist                            │
│  T1.6.28 Manual verification matches numpy computation                      │
│  T1.6.29 Outlier batch: loss_batch_max >> nll                               │
│  T1.6.30 Decoder test: per-sequence loss stats on GPT-2                     │
│  T1.6.31 batch_size=1 → loss_batch_std == 0.0                              │
│                                                                              │
│ Normalized structural:                                                       │
│  T1.6.32 ffn_delta_normalized == ffn_delta_mean / sqrt(hidden_size)         │
│  T1.6.33 embedding_norm_normalized == embedding_norm_mean / sqrt(hidden)    │
│  T1.6.34 Cross-model: raw values differ, normalized values closer           │
│                                                                              │
│ Regression:                                                                  │
│  T1.6.35 Phase 1 gate tests STILL PASS (account for renamed keys)          │
│  T1.6.36 Encoder + Decoder produce all fixed/added metrics                  │
│                                                                              │
│ Run: pytest tests/test_phase1_6_gate.py -v                                   │
│ All 36 tests must PASS.                                                      │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 2: PROCESSING + DIAGNOSIS PIPELINE
================================================================================
Goal: Port existing processing/diagnosis code into the new package and
      build the end-to-end inference pipeline.
Dependencies: Phase 0 (directory structure exists)
Note: Phase 2 is INDEPENDENT of Phase 1 — can be done in parallel.
      Phase 1 produces feature vectors; Phase 2 consumes them.

────────────────────────────────────────────────────────────────────────────────
STEP 12.5: aggregator.py — Add Proportional Window Mode
────────────────────────────────────────────────────────────────────────────────
Source: MODIFY existing extraction/aggregator.py
Size: ~30 lines of additions
Why here: The hardcoded WINDOW_DEFINITION = {early: (1,3), mid: (4,7), late: (8,10)}
          assumes 10-epoch training. For 3-epoch fine-tuning (common for
          transformers), mid and late windows have no data. For 100-epoch
          training, 90% of training is ignored.

 12.5.1 Add proportional window computation
        def _proportional_windows(total_epochs):
            third = max(1, total_epochs // 3)
            return {
                'early': (1, third),
                'mid': (third + 1, 2 * third),
                'late': (2 * third + 1, total_epochs),
            }

 12.5.2 Update compute_window_features() signature
        Add window_mode parameter:
        def compute_window_features(metric_history, total_epochs,
                                     window_mode='fixed'):
            if window_mode == 'proportional':
                windows = _proportional_windows(total_epochs)
            else:
                windows = WINDOW_DEFINITION

 12.5.3 Default to 'proportional' in MetricCollector
        The collector should use proportional by default since DEFault++
        targets transformer training which varies from 3 to 100+ epochs.
        Keep 'fixed' available for backward compatibility with existing
        CSV data trained with 10-epoch assumption.

 12.5.4 Test: test_proportional_windows
        a) 3 epochs: early=(1,1), mid=(2,2), late=(3,3) — each window has data
        b) 10 epochs: early=(1,3), mid=(4,6), late=(7,10) — similar to fixed
        c) 100 epochs: early=(1,33), mid=(34,66), late=(67,100) — full coverage
        d) 1 epoch: early=(1,1), mid=(2,2) empty, late=(2,1) empty — graceful

────────────────────────────────────────────────────────────────────────────────
STEP 13: processing/pipeline.py — FeatureProcessor
────────────────────────────────────────────────────────────────────────────────
Source: src/data/feature_processor.py (428 lines, already working)
Size: ~450-480 lines (adds save/load)
Action: Refactor (add serialization, keep all 6 processing steps)

 13.1 Copy src/data/feature_processor.py → src/defaultplusplus/processing/pipeline.py
      - Change import paths: from src.data.feature_groups → from ..processing.groups

 13.2 Add save() method
      Sub-steps:
      a) Serialize fitted state to pickle:
         - self._drop_mask (step 1)
         - self._log_columns (step 2)
         - self._layer_agg_map (step 3)
         - self._medians (step 4)
         - self._cv_mask (step 5)
         - self._final_feature_names (step 6)
         - self._group_indices (step 6)
         - self.arch
      b) Use joblib.dump() for compatibility with sklearn objects

 13.3 Add load() classmethod
      Sub-steps:
      a) @classmethod def load(cls, path) -> FeatureProcessor
      b) Load pickle, reconstruct FeatureProcessor with fitted state
      c) Mark as fitted (self._is_fitted = True)
      d) Validate loaded state has all required attributes

 13.4 Add transform() method (inference-only mode)
      Sub-steps:
      a) Check self._is_fitted, raise if not
      b) Apply all 6 steps using saved state (no re-fitting)
      c) Return (X_processed, feature_names, group_indices)

 13.5 Verify: existing fit_transform() still works identically
      - Run on encoder_v1_killed_binary.csv
      - Compare output shape and feature names with original

 13.6 FEATURE ALIGNMENT STRATEGY (CRITICAL)
      ═══════════════════════════════════════
      The pretrained diagnosis model was trained on features from the OLD
      base_metrics.py pipeline. The NEW extraction system (Phase 1 + 1.5 + 1.6)
      produces a DIFFERENT feature set: new metrics, renamed keys, per-layer
      keys that depend on num_layers. This MUST be reconciled.

      Strategy (3 tiers):

      TIER 1 — Pretrained model compatibility (ship with v0.2.0):
        The pretrained model expects a FIXED feature set (from the training CSVs).
        The FeatureProcessor.transform() must:
        a) Accept the new extraction output (which has MORE features)
        b) DROP features the pretrained model doesn't know about
           (Phase 1.5/1.6 additions like dead_neuron_frac, sharpness_sam)
        c) FILL features the pretrained model expects but extraction doesn't
           produce with 0.0 defaults (e.g., old kernel_fault_* features
           that were fault-injection-specific)
        d) RENAME features where Phase 1.6 changed names (mass_pad →
           attention_pad_mass_max) — maintain a LEGACY_NAME_MAP dict
        Add this mapping logic to FeatureProcessor.transform() as a
        feature alignment step that runs BEFORE the 6-step processing.

      TIER 2 — Retrained model (v0.3.0):
        After the library is working end-to-end, retrain the diagnosis model
        using the NEW feature set (which includes Phase 1.5/1.6 metrics).
        This gives better diagnosis because more features = richer signal.
        Ship new weights. Drop the legacy name mapping.

      TIER 3 — User-trained model:
        Users who train their own diagnosis model (custom faults, custom
        model types) use the NEW feature set directly. No alignment needed.
        The FeatureProcessor.fit_transform() learns the feature set from
        the training data.

      Implementation in FeatureProcessor:
        Add a _align_features(feature_dict, expected_features) method:
          - For each expected feature: look up in feature_dict or LEGACY_NAME_MAP
          - If found: use it. If not: fill with 0.0 and log warning.
          - Drop features not in expected_features list.
          - Return aligned feature vector in the correct order.

      Test: test_feature_alignment
        a) Create a feature dict with Phase 1.5 names → assert alignment
           produces the pretrained model's expected feature vector
        b) Create a feature dict with OLD names (mass_pad) → assert
           LEGACY_NAME_MAP translates correctly
        c) Create a feature dict MISSING some features → assert 0.0 fill
           and warning logged
        d) Verify aligned vector length == pretrained model's input_dim

────────────────────────────────────────────────────────────────────────────────
STEP 14: processing/groups.py — Feature Group Mapping
────────────────────────────────────────────────────────────────────────────────
Source: src/data/feature_groups.py (151 lines)
Size: 151 lines (direct move)
Action: Move (no changes needed)

 14.1 Copy src/data/feature_groups.py → src/defaultplusplus/processing/groups.py
      - No internal imports to fix (uses only stdlib `re`)
      - Verify: assign_feature_to_group, build_group_indices, get_group_sizes all work

 14.2 Verify group coverage for decoder features
      - Check that cache_hidden_sim maps to 'cache_diagnostics'
      - Check that all feature names from Phase 1 map to exactly one group
      - Any unmapped features = bug (either in groups.py or feature naming)

────────────────────────────────────────────────────────────────────────────────
STEP 15: diagnosis/fpg.py — Fault Propagation Graph
────────────────────────────────────────────────────────────────────────────────
Source: src/data/fundamental_fpg.py (470 lines)
Size: 470 lines (direct move)
Action: Move (fix imports only)

 15.1 Copy src/data/fundamental_fpg.py → src/defaultplusplus/diagnosis/fpg.py
      - No internal imports to fix (uses only stdlib + numpy)
      - Public API: build_fundamental_fpg(arch), fundamental_to_feature_group_adjacency(arch)

 15.2 Verify: adjacency matrices for "encoder" and "decoder" arch types
      - Check shapes: (13, 13) for feature group adjacency
      - Check component filtering by scope (encoder/decoder/both)

────────────────────────────────────────────────────────────────────────────────
STEP 16: diagnosis/encoder.py — GroupEncoder + GraphAggregator
────────────────────────────────────────────────────────────────────────────────
Source: src/models/group_encoder.py (270 lines)
Size: 270 lines (move + fix imports)
Action: Move

 16.1 Copy src/models/group_encoder.py → src/defaultplusplus/diagnosis/encoder.py
      - Fix import: remove sys.path.insert if any
      - Classes to preserve: GroupEncoder, FlatEncoder, GraphAggregator, ProtoClassifier
      - Dependencies: torch, numpy only

 16.2 Verify: GroupEncoder forward pass works with dummy data
      - Create dummy group_indices, input tensor
      - Check output shape: (batch, n_groups, hidden_dim)

────────────────────────────────────────────────────────────────────────────────
STEP 17: diagnosis/model.py — HierarchicalDiagnosisModel
────────────────────────────────────────────────────────────────────────────────
Source: hierarchical_graph_category_rootcause/model.py (286 lines)
Size: ~290 lines (move + fix imports)
Action: Move + update imports

 17.1 Copy model.py → src/defaultplusplus/diagnosis/model.py
      Fix imports:
      - from src.models.group_encoder → from .encoder
      - Remove sys.path.insert() hack
      - Remove pathlib manipulations for import resolution

 17.2 Verify: model instantiation with dummy config
      - Create HierarchicalDiagnosisModel with test config
      - Check: encode(), detect(), categorize(), diagnose_proto(), explain_diagnosis()
      - Verify shapes at each stage

────────────────────────────────────────────────────────────────────────────────
STEP 18: diagnosis/inference.py — DiagnosisPipeline (NEW)
────────────────────────────────────────────────────────────────────────────────
Source: NEW (assembles Steps 13-17 into end-to-end pipeline)
Size: ~150-180 lines

 18.1 Define DiagnosisResult dataclass
      Fields:
      - detection: str ("clean" or "faulty")
      - detection_confidence: float
      - category: Optional[str] (fault category name)
      - category_confidence: Optional[float]
      - root_cause: Optional[str]
      - root_cause_distance: Optional[float]
      - explanation: Optional[Dict[str, float]] (group_name → fraction)
      - all_distances: Optional[Dict[str, float]] (per root-cause distances)

 18.2 Implement DiagnosisPipeline.__init__(arch_family, weights_dir)
      Sub-steps:
      a) Load FeatureProcessor from weights_dir / f"{arch}_processor.pkl"
      b) Load scaler from weights_dir / f"{arch}_scaler.pkl"
      c) Load HierarchicalDiagnosisModel state_dict from weights_dir / f"{arch}_model.pt"
      d) Load category names + root cause names from weights metadata
      e) Build group_indices from processor's feature names
      f) Build FPG adjacency for this arch

 18.3 Implement predict(feature_vector, feature_names) -> DiagnosisResult
      Sub-steps (match the 6-step pipeline from original plan):
      a) Validate: len(feature_vector) matches expected, feature_names align
      b) Step 1: Feature processing via self.processor.transform()
      c) Step 2: Scaling via self.scaler.transform()
      d) Step 3: Convert to torch tensor
      e) Step 4: Encode — z, h_groups = model.encode(x, group_indices)
      f) Step 5: Detection — det_logits = model.detect(z)
         If clean (argmax == 0): return DiagnosisResult(detection="clean", confidence=softmax[0])
      g) Step 6: Categorization — cat_logits = model.categorize(z)
      h) Step 7: Root cause — preds, distances, group_dists = model.diagnose_proto(h_groups, cat_name)
      i) Step 8: Explanation — explanation = model.explain_diagnosis(h_groups, cat_name, preds[0])
      j) Package into DiagnosisResult and return

 18.4 Add batch prediction support
      - predict_batch(feature_matrix, feature_names) -> List[DiagnosisResult]
      - Vectorized version for multiple samples at once
      - Useful for offline analysis of CSV data

────────────────────────────────────────────────────────────────────────────────
STEP 19: diagnosis/explanation.py — Explanation Formatting (NEW)
────────────────────────────────────────────────────────────────────────────────
Source: NEW (formats DiagnosisResult explanations for display)
Size: ~80-100 lines

 19.1 Implement format_explanation(result: DiagnosisResult) -> str
      Sub-steps:
      a) Header: "Fault detected: {category} → {root_cause}"
      b) Confidence section: detection %, category %, root cause distance
      c) Group breakdown: sorted bar chart of group contributions
         "attention    ████████████░░░  42.3%"
         "ffn          ██████░░░░░░░░░  21.8%"
         "layernorm    ████░░░░░░░░░░░  14.1%"
         ...
      d) FPG path: trace fault propagation from root cause through graph

 19.2 Implement get_fpg_path(result, fpg_adjacency) -> List[str]
      Sub-steps:
      a) Starting from the top-contributing group, follow FPG edges
      b) Return ordered list of group names in propagation order
      c) Used by report.py and visualization.py for highlighting

 19.3 Implement explanation_to_dict(result) -> Dict
      - Structured version for JSON export
      - Includes: detection, category, root_cause, confidence, group_contributions

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 GATE TEST — Must pass before Phase 2 is considered complete         │
│                                                                              │
│ tests/test_phase2_gate.py:                                                   │
│                                                                              │
│ Processing tests (use existing CSV data):                                    │
│  T2.1  FeatureProcessor.fit_transform() on encoder CSV produces valid output│
│  T2.2  FeatureProcessor.fit_transform() on decoder CSV produces valid output│
│  T2.3  Output feature count > 50 (not overly aggressive filtering)          │
│  T2.4  All output features map to exactly one FPG group (no orphans)        │
│  T2.5  save() → load() roundtrip: transform() on same data produces        │
│         identical output (max |diff| < 1e-6)                                │
│  T2.6  load() then transform() without fit raises NO error (inference mode) │
│  T2.7  transform() on wrong-shaped input raises clear ValueError            │
│                                                                              │
│ Feature groups tests:                                                        │
│  T2.8  assign_feature_to_group('attention_entropy_mean') → 'attention'      │
│  T2.9  assign_feature_to_group('grad_norm_total') → 'training_dynamics'     │
│  T2.10 build_group_indices returns dict with 13 groups                      │
│  T2.11 Every feature from CSV data maps to a group (no None returns)        │
│                                                                              │
│ FPG tests:                                                                   │
│  T2.12 build_fundamental_fpg('encoder') returns adjacency with shape (N,N)  │
│  T2.13 fundamental_to_feature_group_adjacency('encoder') → (13, 13)         │
│  T2.14 fundamental_to_feature_group_adjacency('decoder') → (13, 13)         │
│  T2.15 Adjacency is NOT symmetric (directed graph)                          │
│                                                                              │
│ Diagnosis model tests (dummy weights, no pretrained):                        │
│  T2.16 HierarchicalDiagnosisModel instantiates with test config             │
│  T2.17 model.encode(x, group_indices) → z shape (batch, embed_dim)         │
│  T2.18 model.encode(x, group_indices) → h_groups shape (batch, 13, hidden) │
│  T2.19 model.detect(z) → logits shape (batch, 2)                           │
│  T2.20 model.categorize(z) → logits shape (batch, n_categories)            │
│  T2.21 model.explain_diagnosis() returns dict with group fractions          │
│  T2.22 Group fractions sum to ~1.0 (within 0.01 tolerance)                 │
│                                                                              │
│ Inference pipeline tests (with dummy/random weights):                        │
│  T2.23 DiagnosisPipeline.predict() returns DiagnosisResult                  │
│  T2.24 DiagnosisResult has all required fields (detection, confidence, etc.) │
│  T2.25 Clean sample → detection="clean" (with mocked clean-biased weights)  │
│  T2.26 format_explanation() produces non-empty string                       │
│                                                                              │
│ Run: pytest tests/test_phase2_gate.py -v                                     │
│ All 26 must PASS. If any fail → fix before proceeding to Phase 3.            │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 3: CORE API + REAL-TIME UI
================================================================================
Goal: Build the user-facing API (DEFaultPP, DEFaultPPCallback) and
      Rich terminal output.
Dependencies: Phase 1 (extraction) + Phase 2 (diagnosis) must both be complete.

────────────────────────────────────────────────────────────────────────────────
STEP 20: config.py — Configuration Dataclasses
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~60-80 lines

 20.1 EXTEND existing ExtractionConfig dataclass (already created in Phase 0)
      ════════════════════════════════════════════════════════════════════
      IMPORTANT: src/defaultplusplus/config.py already exists from Phase 0
      with ExtractionConfig containing: grad_vanish_threshold, grad_explode_threshold,
      grad_activity_threshold, attention_leak_threshold, position_cutoff, ece_num_bins,
      ffn_probe_tokens, ffn_var_activity_threshold, activation_interval, gradient_window,
      representation_epochs, representation_tokens, and special token IDs.

      DO NOT recreate this file. EXTEND it with:
      - sharpness_rho: float = 0.05     # SAM perturbation radius (Phase 1.5)
      - sharpness_interval: int = 50    # Steps between SAM computations (Phase 1.5)
      - trace_interval: int = 0         # 0 = epoch-only Hutchinson trace (Phase 1.5)
      - dead_neuron_threshold: float = 1e-6  # (Phase 1.5)

      Use EXISTING field names (activation_interval NOT log_every,
      ece_num_bins NOT ece_bins). Changing names breaks Phase 1 imports.

 20.2 Define UIConfig dataclass
      Fields:
      - quiet: bool = False          # Suppress all terminal output
      - show_warnings: bool = True   # Show real-time warnings
      - show_progress: bool = True   # Show progress bar
      - show_epoch_table: bool = True # Show epoch summary table
      - warn_on_nan: bool = True
      - warn_on_gradient_vanish: bool = True
      - warn_on_gradient_explode: bool = True

 20.3 Define DiagnosisConfig dataclass
      Fields:
      - weights_dir: Optional[Path] = None  # Custom weights location
      - use_pretrained: bool = True          # Use shipped weights
      - confidence_threshold: float = 0.5    # Min confidence for diagnosis

 20.4 Define DEFaultPPConfig (combines all three)
      Fields: extraction, ui, diagnosis sub-configs
      + convenience factory: DEFaultPPConfig.default()

────────────────────────────────────────────────────────────────────────────────
STEP 21: core.py — DEFaultPP Main Class
────────────────────────────────────────────────────────────────────────────────
Source: NEW (orchestrator — connects extraction, diagnosis, UI)
Size: ~150-180 lines

 21.1 Implement DEFaultPP.__init__(model, optimizer, *, config=None, tokenizer=None, **kwargs)
      Sub-steps:
      a) Create config from kwargs or passed DEFaultPPConfig
      b) Create ModelInspector(model) → self.inspector
      c) Handle attention implementation compatibility:
         - If model.config._attn_implementation != 'eager':
           Switch to 'eager' and log warning:
           "DEFault++ requires output_attentions=True, which is incompatible
            with '{attn_impl}' attention. Switching to 'eager'. This may
            reduce training throughput by ~10-20%. To avoid this, load your
            model with attn_implementation='eager'."
         - Set model.config.output_attentions = True
      d) Set model.config.output_hidden_states = True
      e) Create MetricCollector(inspector, config.extraction) → self.collector
      f) Create Console(config.ui) → self.console
      g) Store weak references: weakref.ref(model), weakref.ref(optimizer)
      h) Initialize: epoch_history=[], current_epoch=0, step_count=0
      i) Call self.console.show_banner(self.inspector) — display detected arch

 21.2 Implement step(*, loss, outputs=None, labels=None, step_time=None)
      Sub-steps:
      a) Get model/optimizer from weak refs
      b) Extract attention_weights from outputs (if available)
      c) Extract hidden_states from outputs (if available)
      d) Call collector.collect_step(loss=loss, model=model, optimizer=optimizer,
             outputs=outputs, labels=labels, attention_weights=attention_weights,
             hidden_states=hidden_states, batch_idx=self.step_count,
             epoch=self.current_epoch, step_time=step_time)
      e) Call console.update_step(metrics) for live display
      f) Increment step_count

 21.3 Implement end_epoch(val_metrics=None)
      Sub-steps:
      a) epoch_summary = collector.finalize_epoch(self.current_epoch)
      b) If val_metrics: epoch_summary.update({f"val_{k}": v for k, v in val_metrics.items()})
      c) Append to epoch_history
      d) Call console.show_epoch_summary(epoch_summary)
      e) Increment current_epoch

 21.4 Implement diagnose() -> DiagnosisReport
      Sub-steps:
      a) Compute final features: features, names = collector.get_final_features(epoch_history)
      b) Create DiagnosisPipeline(inspector.arch_family, config.diagnosis.weights_dir)
      c) result = pipeline.predict(features, names)
      d) Return DiagnosisReport(result, epoch_history, inspector)

 21.5 Implement context manager API for cleaner PyTorch ergonomics
      Sub-steps:
      a) monitor.epoch() context manager:
         - __enter__: record epoch start time, reset step counter
         - __exit__: auto-call end_epoch(), handle exceptions gracefully
         - Returns self so `with monitor.epoch() as ep:` works
      b) monitor.step() as BOTH a function AND context manager:
         - When called normally: monitor.step(loss=loss) — works as before
         - When used as context: `with monitor.step(loss=loss):` — measures step_time
           automatically via __enter__/__exit__ timing
         - Use a StepContext helper class that implements __enter__/__exit__
      c) This gives users two equivalent PyTorch patterns:
         # Pattern A (explicit):
         monitor.step(loss=loss, outputs=outputs)
         optimizer.step()

         # Pattern B (context manager, auto-timed):
         with monitor.step(loss=loss, outputs=outputs):
             optimizer.step()

 21.6 Implement convenience properties
      - arch_family → inspector.arch_family
      - model_type → inspector.model_type
      - num_epochs → len(epoch_history)
      - feature_names → collector.feature_names

────────────────────────────────────────────────────────────────────────────────
STEP 22: callback.py — Framework-Specific Callbacks
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~100-120 lines

 IMPORTANT DESIGN NOTE:
 PyTorch itself has NO callback system. The primary API for PyTorch users is:
   Option 1: monitor.step() (explicit calls) — Step 21.2
   Option 2: monitor.epoch() + monitor.step() context managers — Step 21.5

 This file provides callbacks ONLY for frameworks that have callback systems:
   - HuggingFace Trainer (transformers.TrainerCallback)
   - PyTorch Lightning (pytorch_lightning.Callback) — optional/stretch

 22.1 Implement DEFaultPPCallback(TrainerCallback)
      NOTE: This wraps the DEFaultPP core class. It does NOT duplicate logic.
      The callback just translates HF Trainer events → monitor.step()/end_epoch().

      Sub-steps:
      a) __init__(**kwargs): store kwargs, _monitor = None
      b) on_train_begin(args, state, control, model, **kw):
         - Extract optimizer from kw (HF Trainer passes it as kwarg)
         - self._monitor = DEFaultPP(model, optimizer, **self._kwargs)
         - EDGE CASE: if optimizer is None (not always passed by Trainer),
           log warning and proceed — gradient metrics will be limited
      c) on_step_end(args, state, control, model, **kw):
         - loss = state.log_history[-1].get('loss') if state.log_history else None
         - self._monitor.step(loss=loss)
         - NOTE: HF Trainer does NOT expose raw outputs/labels in callbacks.
           We get loss from logs. For full feature extraction (attention, hidden
           states), users should use the manual PyTorch API instead.
      d) on_epoch_end(args, state, control, metrics, **kw):
         - self._monitor.end_epoch(val_metrics=metrics)
      e) diagnose() -> DiagnosisReport:
         - return self._monitor.diagnose()

 22.2 Document callback limitations vs manual API
      The HF Trainer callback has inherent limitations:
      - No direct access to model outputs (logits, attention, hidden states)
        → logit metrics, attention metrics, structural metrics UNAVAILABLE
      - Only loss + LR + gradient metrics are fully available
      - For FULL feature extraction: use manual PyTorch API (Option 1/2)
      - Callback is best for: quick monitoring + gradient/training diagnostics
      Add these as warnings in the callback docstring + banner output

 22.3 Handle edge cases
      Sub-steps:
      a) If optimizer not available in on_train_begin → log warning, create without optimizer
      b) If model changes during training (e.g., gradient checkpointing) → re-inspect
      c) If training is resumed → handle epoch offset
      d) If Trainer uses gradient accumulation → adjust step counting

 22.4 (STRETCH) PyTorch Lightning callback
      - DEFaultPPLightningCallback(pl.Callback)
      - Lightning DOES pass outputs in on_train_batch_end → full features available
      - on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx):
           self._monitor.step(loss=outputs['loss'], outputs=outputs)
      - on_train_epoch_end(trainer, pl_module):
           self._monitor.end_epoch()
      - Only implement if transformers users actually request it

────────────────────────────────────────────────────────────────────────────────
STEP 23: ui/console.py — Rich Terminal Output
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~120-150 lines
Dependencies: rich library

 23.1 Implement Console.__init__(config: UIConfig)
      Sub-steps:
      a) from rich.console import Console as RichConsole
      b) from rich.table import Table
      c) from rich.panel import Panel
      d) from rich.progress import Progress
      e) If config.quiet: self._console = None (no-op mode)
      f) Initialize warnings list, step counter

 23.2 Implement show_banner(inspector)
      Display at training start:
      ┌─────────────────────────────────────────┐
      │  DEFault++ Fault Diagnosis Monitor       │
      │  Architecture: bert (encoder)            │
      │  Layers: 12 | Heads: 12 | Hidden: 768   │
      │  Monitoring: 6 metric modules active     │
      └─────────────────────────────────────────┘

 23.3 Implement update_step(metrics)
      Sub-steps:
      a) Check warning conditions:
         - gradient_vanish == 1 → warn "Vanishing gradients"
         - gradient_explode == 1 → warn "Exploding gradients"
         - Any NaN in metrics → warn "NaN detected in {key}"
         - attention_mass_leak > threshold → warn "Attention leaking to padding"
      b) Update progress bar (if show_progress)
      c) Append warnings to self.warnings list with timestamp

 23.4 Implement show_epoch_summary(summary)
      Display epoch table:
      ┌────────┬────────┬──────────┬──────────┬───────────┬──────────┐
      │ Epoch  │  Loss  │ Accuracy │ Grad Norm│ LR        │ Warnings │
      ├────────┼────────┼──────────┼──────────┼───────────┼──────────┤
      │   1    │ 2.341  │  0.312   │  1.23    │ 5.0e-5    │ 0        │
      │   2    │ 1.876  │  0.542   │  0.89    │ 4.5e-5    │ 1 ⚠      │
      └────────┴────────┴──────────┴──────────┴───────────┴──────────┘

 23.5 Implement _warn(msg, severity='warning')
      - Use rich.console.print with colored formatting
      - severity levels: 'info' (blue), 'warning' (yellow), 'critical' (red)
      - Store in self.warnings for report inclusion

────────────────────────────────────────────────────────────────────────────────
STEP 24: ui/report.py — DiagnosisReport
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~150-180 lines

 24.1 Implement DiagnosisReport.__init__(result, epoch_history, inspector)
      Store: result (DiagnosisResult), epoch_history, inspector, timestamp

 24.2 Implement show()
      Sub-steps:
      a) Use rich panels for each diagnosis stage:
         Panel 1: Detection — "FAULTY (98.2% confidence)" in red, or "CLEAN" in green
         Panel 2: Category — "data_pipeline (87.3% confidence)"
         Panel 3: Root Cause — "E2.1: Zero Query Projection (distance: 0.342)"
         Panel 4: Explanation — bar chart of group contributions (from explanation.py)
      b) Panel 5: Training Warnings — list of all warnings during training
      c) Panel 6: Recommended Actions — based on root cause category

 24.3 Implement save(path="diagnosis_report.json")
      Sub-steps:
      a) Convert to dict via to_dict()
      b) json.dump with indent=2
      c) Include: result, epoch_history summary, inspector info, timestamp

 24.4 Implement to_dict() -> Dict
      Structured output:
      {
          "detection": {"label": "faulty", "confidence": 0.982},
          "category": {"label": "data_pipeline", "confidence": 0.873},
          "root_cause": {"label": "E2.1", "name": "...", "distance": 0.342},
          "explanation": {"attention": 0.423, "ffn": 0.218, ...},
          "training_summary": {"epochs": 10, "final_loss": 0.15, ...},
          "warnings": [...],
          "architecture": {"family": "encoder", "type": "bert", ...}
      }

 24.5 Implement plot() — delegates to visualization.py
      - Calls visualization functions, collects figures
      - Optional save_dir to save PNG files
      - Returns list of matplotlib Figure objects

────────────────────────────────────────────────────────────────────────────────
STEP 25: ui/visualization.py — Matplotlib Plots
────────────────────────────────────────────────────────────────────────────────
Source: Partially from hierarchical_.../plotting.py (adapt style)
Size: ~200-250 lines

 25.1 Implement plot_training_curves(epoch_history) -> Figure
      Sub-steps:
      a) 2x2 subplot grid: loss, accuracy, grad_norm, learning_rate
      b) Plot each epoch metric as line with markers
      c) Overlay warning markers (red dots) where anomalies detected
      d) Style: clean, publication-quality (from plotting.py aesthetic)

 25.2 Implement plot_fpg(result, fpg_adjacency) -> Figure
      Sub-steps:
      a) Draw FPG as directed graph (networkx layout)
      b) Highlight fault path in red
      c) Node size proportional to group contribution
      d) Edge thickness proportional to propagation strength
      e) Color nodes: green=clean, yellow=mild, red=severe

 25.3 Implement plot_explanation(result) -> Figure
      Sub-steps:
      a) Horizontal bar chart of group contributions
      b) Sorted by contribution fraction (descending)
      c) Color gradient from blue (low) to red (high)
      d) Add percentage labels on bars
      e) Title: "Fault Explanation: {root_cause_name}"

 25.4 Implement plot_prototype_distances(result) -> Figure
      Sub-steps:
      a) Heatmap: rows = root causes in category, columns = FPG groups
      b) Cell value = per-group distance to that root cause's prototype
      c) Highlight predicted root cause row
      d) Colormap: viridis (low distance = dark, high = bright)

────────────────────────────────────────────────────────────────────────────────
STEP 26: __init__.py — Public API Exports
────────────────────────────────────────────────────────────────────────────────
Source: NEW (wiring only)
Size: ~20-30 lines

 26.1 Update src/defaultplusplus/__init__.py
      Exports:
      from .core import DEFaultPP
      from .callback import DEFaultPPCallback
      from .config import DEFaultPPConfig, ExtractionConfig, UIConfig, DiagnosisConfig
      from .ui.report import DiagnosisReport
      from .diagnosis.inference import DiagnosisPipeline, DiagnosisResult

      __version__ = "0.2.0"
      __all__ = ["DEFaultPP", "DEFaultPPCallback", "DEFaultPPConfig", ...]

 26.2 Update sub-package __init__.py files
      a) extraction/__init__.py: export ModelInspector, MetricCollector
      b) processing/__init__.py: export FeatureProcessor
      c) diagnosis/__init__.py: export HierarchicalDiagnosisModel, DiagnosisPipeline
      d) ui/__init__.py: export DiagnosisReport, Console

 26.3 Verify: `from defaultplusplus import DEFaultPP` works
      - Run smoke test with BERT-tiny model

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 GATE TEST — Must pass before Phase 3 is considered complete         │
│                                                                              │
│ tests/test_phase3_gate.py:                                                   │
│                                                                              │
│ Core API tests (REAL model, REAL forward pass, 1-2 batches):                 │
│  T3.1  DEFaultPP(bert_model, optimizer) initializes without error           │
│  T3.2  DEFaultPP(gpt2_model, optimizer) initializes without error           │
│  T3.3  monitor.step(loss=loss, outputs=outputs) runs without error          │
│  T3.4  monitor.step() returns or stores >50 metric values                   │
│  T3.5  monitor.end_epoch() produces epoch summary dict                      │
│  T3.6  After 2 epochs: monitor.epoch_history has 2 entries                  │
│  T3.7  monitor.arch_family returns 'encoder' or 'decoder' correctly         │
│  T3.8  monitor.feature_names returns non-empty list                         │
│                                                                              │
│ Context manager tests:                                                       │
│  T3.9  `with monitor.epoch():` auto-calls end_epoch on exit                │
│  T3.10 `with monitor.step(loss=loss):` measures step_time automatically     │
│  T3.11 Exception inside `with monitor.epoch():` is re-raised, not swallowed│
│  T3.12 Nested `with monitor.epoch() / with monitor.step()` works correctly │
│                                                                              │
│ Error handling tests:                                                        │
│  T3.13 DEFaultPP(non_hf_model, opt) raises clear error (e.g., ResNet)      │
│  T3.14 monitor.step() without loss= raises TypeError                        │
│  T3.15 monitor.diagnose() with 0 epochs raises ValueError("no epochs")     │
│  T3.16 monitor.diagnose() with 1 epoch logs warning (< 3 recommended)      │
│  T3.17 Passing NaN loss → warning logged, no crash                          │
│  T3.18 Model deleted (weakref dead) → clear error on next step()            │
│                                                                              │
│ Callback tests (HF Trainer, if transformers installed):                      │
│  T3.19 DEFaultPPCallback() instantiates without error                       │
│  T3.20 Callback + Trainer(model, callbacks=[cb]).train() completes          │
│  T3.21 callback.diagnose() returns DiagnosisReport                          │
│  T3.22 Callback logs limitation warning about missing outputs               │
│                                                                              │
│ UI tests (non-visual, check no crashes):                                     │
│  T3.23 Console(quiet=True) suppresses all output                            │
│  T3.24 Console(quiet=False).show_banner(inspector) prints without error     │
│  T3.25 Console.update_step({}) does not crash on empty metrics              │
│  T3.26 Console.update_step({'gradient_vanish': 1.0}) triggers warning       │
│  T3.27 DiagnosisReport.show() prints without error                          │
│  T3.28 DiagnosisReport.save('/tmp/test.json') creates valid JSON file       │
│  T3.29 DiagnosisReport.to_dict() returns dict with required keys            │
│  T3.30 DiagnosisReport.plot() returns list of matplotlib Figures            │
│                                                                              │
│ Import tests:                                                                │
│  T3.31 `from defaultplusplus import DEFaultPP` works                        │
│  T3.32 `from defaultplusplus import DEFaultPPCallback` works                │
│  T3.33 `from defaultplusplus import DEFaultPPConfig` works                  │
│  T3.34 `from defaultplusplus import DiagnosisReport` works                  │
│                                                                              │
│ Run: pytest tests/test_phase3_gate.py -v --timeout=180                       │
│ All 34 must PASS. If any fail → fix before proceeding to Phase 4.            │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
PHASE 4: PRETRAINED WEIGHTS + POLISH
================================================================================
Goal: Generate pretrained weights, build download system, write tests.
Dependencies: All previous phases complete.

────────────────────────────────────────────────────────────────────────────────
STEP 27: Generate Pretrained Weights
────────────────────────────────────────────────────────────────────────────────
Size: Script modifications only (~50 lines of changes)

 27.1 Modify hierarchical_graph_category_rootcause/train.py to save artifacts
      After best-fold training completes, add:
      a) torch.save(best_model.state_dict(), f"{arch}_model.pt")
      b) joblib.dump(scaler, f"{arch}_scaler.pkl")
      c) processor.save(f"{arch}_processor.pkl")
      d) Save metadata JSON: feature_names, category_names, root_cause_names per category

 27.2 Run training for encoder
      - python -m hierarchical_graph_category_rootcause.train --arch encoder
      - Save: encoder_model.pt, encoder_scaler.pkl, encoder_processor.pkl

 27.3 Run training for decoder
      - python -m hierarchical_graph_category_rootcause.train --arch decoder
      - Save: decoder_model.pt, decoder_scaler.pkl, decoder_processor.pkl

 27.4 Validate saved weights
      - Load model, run inference on test fold
      - Verify accuracy matches training results
      - Verify feature names in processor match collector.feature_names

────────────────────────────────────────────────────────────────────────────────
STEP 28: pretrained/registry.py — Weight Loading
────────────────────────────────────────────────────────────────────────────────
Source: NEW
Size: ~80-100 lines

 28.1 Implement _get_cache_dir() -> Path
      - Returns ~/.cache/defaultplusplus/
      - Creates directory if not exists

 28.2 Implement _ensure_downloaded(arch_family) -> Path
      Sub-steps:
      a) Check if weights exist in cache dir
      b) If not, download from GitHub release URL
      c) Use urllib.request or requests (keep deps minimal)
      d) Extract tar.gz to cache dir
      e) Return path to extracted directory

 28.3 Implement load_pretrained(arch_family) -> Dict
      Sub-steps:
      a) weights_dir = _ensure_downloaded(arch_family)
      b) Load model state dict: torch.load(weights_dir / f"{arch}_model.pt", map_location='cpu')
      c) Load scaler: joblib.load(weights_dir / f"{arch}_scaler.pkl")
      d) Load processor: FeatureProcessor.load(weights_dir / f"{arch}_processor.pkl")
      e) Load metadata: json.load(weights_dir / f"{arch}_metadata.json")
      f) Return dict with all components

 28.4 Implement PRETRAINED_URLS dict + version handling
      - URLs point to GitHub release assets
      - Version embedded in URL path
      - Check for newer version on download (optional)

────────────────────────────────────────────────────────────────────────────────
STEP 29: Update pyproject.toml — Final Dependencies
────────────────────────────────────────────────────────────────────────────────
Size: ~10 lines of changes

 29.1 Add all dependencies
      - rich>=13.0
      - joblib>=1.3
      - transformers>=4.30  (for TrainerCallback, model config types)
      - scipy>=1.10  (for attention entropy, skew, kurtosis)
      - networkx>=3.0  (for FPG visualization)
      Dependencies already present: torch, numpy, pandas, scikit-learn, matplotlib

 29.2 Add console_scripts entry point (optional)
      [project.scripts]
      defaultplusplus = "defaultplusplus.cli:main"  # optional CLI tool

 29.3 Add package data
      [tool.setuptools.package-data]
      defaultplusplus = ["pretrained/weights/*.pt", "pretrained/weights/*.pkl"]

────────────────────────────────────────────────────────────────────────────────
STEP 30: Final Integration Tests + Regression Suite
────────────────────────────────────────────────────────────────────────────────
NOTE: Most tests are already covered by Phase Gate Tests (T0-T3 above).
      This step adds: shared fixtures, end-to-end regression, and the
      final acceptance tests that use REAL pretrained weights from Step 27.

 30.1 tests/conftest.py — Shared fixtures (used by ALL gate + final tests)
      Sub-steps:
      a) @fixture bert_model: load prajjwal1/bert-tiny for fast testing
      b) @fixture gpt2_model: load sshleifer/tiny-gpt2
      c) @fixture distilbert_model: load distilbert-base-uncased
      d) @fixture dummy_optimizer: Adam with lr=1e-4
      e) @fixture dummy_batch_encoder: {input_ids, attention_mask, labels} (batch=4, seq=32)
      f) @fixture dummy_batch_decoder: {input_ids, attention_mask, labels} (batch=4, seq=32)
      g) @fixture sample_feature_vector: load row from encoder_v1_killed_binary.csv
      h) @fixture pretrained_weights_dir: path to Step 27 generated weights
         (skip tests if weights not generated yet)

 30.2 tests/test_e2e_pretrained.py — End-to-end with REAL pretrained weights
      These are the ACCEPTANCE TESTS — they prove the whole system works:
      a) test_bert_full_pipeline:
         - Load bert-tiny, create optimizer
         - DEFaultPP(model, optimizer)
         - Train 3 epochs on dummy data (random labels, ~30s)
         - diagnose() → DiagnosisReport
         - report.show() prints without error
         - report.save('/tmp/test_report.json') creates valid JSON
         - report.to_dict() has all required keys
      b) test_gpt2_full_pipeline: same for decoder architecture
      c) test_context_manager_pipeline:
         - Use `with monitor.epoch()` / `with monitor.step()` pattern
         - Verify identical results to explicit pattern
      d) test_callback_pipeline:
         - DEFaultPPCallback() + HF Trainer
         - Verify: callback.diagnose() returns valid report
         - Verify: limitations warning was logged
      e) test_feature_alignment_pretrained:
         - Extract features from live training
         - Verify feature names match pretrained processor's expected features
         - THIS IS THE CRITICAL ALIGNMENT TEST

 30.3 tests/test_error_handling.py — Comprehensive error/edge case tests
      a) test_unsupported_model: DEFaultPP(torchvision.models.resnet18(), opt)
         → raises ValueError with message naming supported architectures
      b) test_no_optimizer: DEFaultPP(model, None)
         → logs warning, still works for non-gradient metrics
      c) test_zero_epochs_diagnose: monitor.diagnose() with no data
         → raises ValueError("No epoch data collected")
      d) test_single_epoch_warning: monitor.diagnose() after 1 epoch
         → warns "Less than 3 epochs: windowed features may be unreliable"
      e) test_nan_loss: monitor.step(loss=float('nan'))
         → warning logged, no crash, metric stored as NaN
      f) test_inf_loss: monitor.step(loss=float('inf'))
         → warning logged, no crash
      g) test_empty_batch: monitor.step(loss=0.5, outputs=None, labels=None)
         → only training metrics collected, no crash
      h) test_model_garbage_collected:
         - del model → weakref dies → next step() raises clear error
      i) test_mismatched_features:
         - Feed wrong-length feature vector to DiagnosisPipeline
         → raises ValueError with expected vs actual feature count
      j) test_missing_pretrained_weights:
         - DiagnosisPipeline with nonexistent weights_dir
         → raises FileNotFoundError with helpful message
      k) test_corrupted_pretrained_weights:
         - Write garbage to a .pt file, try to load
         → raises RuntimeError with "corrupted weights" message
      l) test_concurrent_monitors:
         - Two DEFaultPP instances on same model
         → both work independently (no shared state pollution)

 30.4 tests/test_regression.py — Regression tests against known CSV data
      a) test_processor_output_matches_original:
         - Run FeatureProcessor on encoder CSV via NEW code path
         - Run FeatureProcessor on encoder CSV via OLD code path (src/data/)
         - Assert outputs match within tolerance
      b) test_diagnosis_output_matches_original:
         - Run HierarchicalDiagnosisModel on known test fold
         - Compare accuracy with published results (within 2%)
      c) test_fpg_adjacency_matches_original:
         - Compare NEW fpg.py output with OLD fundamental_fpg.py output
         - Must be identical (no tolerance)

────────────────────────────────────────────────────────────────────────────────
STEP 31: Update README
────────────────────────────────────────────────────────────────────────────────
Size: ~100 lines of additions

 31.1 Add "Library Usage" section to README.md
      Sub-steps:
      a) Installation: pip install -e .
      b) Quick Start: minimal code example (5 lines)
      c) Manual Training Loop: full example
      d) HuggingFace Trainer Callback: full example
      e) Understanding the Report: explanation of output sections
      f) Supported Architectures: table of model categories

 31.2 Add "Architecture Coverage" documentation
      Create table of:
      | Category    | Models Covered                                    |
      |-------------|---------------------------------------------------|
      | BERT-style  | bert, roberta, distilbert, albert, electra, etc.  |
      | GPT-style   | gpt2, gpt_neo, gpt_neox, distilgpt2, opt, etc.   |

┌──────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4 FINAL GATE — The library is DONE when all of these pass             │
│                                                                              │
│ Run: pytest tests/ -v --timeout=300                                          │
│                                                                              │
│ Gate test summary (total across all phases):                                 │
│   Phase 0 gate:     5 tests  (scaffolding)                                  │
│   Phase 1 gate:    32 tests  (extraction — implemented)                     │
│   Phase 1.5 gate:  58 tests  (enhanced metrics — 15 new metric groups)     │
│   Phase 1.6 gate:  36 tests  (audit fixes + additions)                     │
│   Phase 2 gate:    26 tests  (processing + diagnosis)                       │
│   Phase 3 gate:    34 tests  (API + UI)                                     │
│   Step 30 final:  ~25 tests  (e2e + error handling + regression)            │
│   ─────────────────────────────────────────                                  │
│   TOTAL:         ~216 tests                                                  │
│                                                                              │
│ Final acceptance criteria:                                                    │
│  FA.1  `pip install -e .` succeeds cleanly                                  │
│  FA.2  `from defaultplusplus import DEFaultPP` works                        │
│  FA.3  pytest tests/ passes 100% (0 failures, 0 errors)                     │
│  FA.4  BERT end-to-end: train 3 epochs → diagnose → report.show()           │
│  FA.5  GPT-2 end-to-end: train 3 epochs → diagnose → report.show()         │
│  FA.6  Feature alignment: extracted names == pretrained model's expected     │
│  FA.7  Regression: diagnosis accuracy on test fold within 2% of published   │
│  FA.8  No import warnings or deprecation errors in clean install            │
│  FA.9  README has working code examples for all 3 API patterns              │
│  FA.10 Pretrained weights download + cache works on fresh machine           │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
EXECUTION DEPENDENCY GRAPH
================================================================================

Phase 0 ──────────────────────────────────────────────→ (DONE)
    │
    ├──→ Phase 1 (Steps 1-12)     ─── SEQUENTIAL within phase ──→ ─┐
    │       Step 1 (inspector) ← everything depends on this        │
    │       Step 2 (base.py) ← all metric modules depend on this   │
    │       Steps 3-9 (metrics) ← can be PARALLEL after Step 2     │
    │       Steps 10-12 (collector, aggregator, export)             │
    │       Step 12.5 (proportional windows)                        │
    │       │                                                       │
    │       ├──→ Phase 1.5 (Metrics 1-15)  ─── NEEDS Phase 1 ──→ ─┤
    │       │       Metrics 1-2 (gradient additions)               │
    │       │       Metrics 3-5 (structural additions)             │
    │       │       Metrics 6-7 (attention additions)              │
    │       │       Metric 8 (training addition)                   │
    │       │       Metrics 9-12 (DEFault coverage gaps)           │
    │       │       Metrics 13-14 (curvature — new module)         │
    │       │       Metric 15 (activation magnitude)               │
    │       │                                                       │
    │       └──→ Phase 1.6 (Bugfixes + Additions) ─ PARALLEL w/1.5 ┤
    │               Bugfixes 1-5 (can be done independently)       │
    │               Additions 1-5 (can be done independently)      │
    │               Update Phase 1 gate tests for renamed keys     │
    │                                                               │
    ├──→ Phase 2 (Steps 13-19)    ─── PARALLEL with Phase 1  ──→ ─┤
    │       Step 13 (processing + feature alignment) ← independent │
    │       Steps 14-17 (groups, fpg, encoder, model) ← independent│
    │       Steps 18-19 (inference + explanation) ← needs 13-17    │
    │                                                               │
    └──→ Phase 3 (Steps 20-26)    ─── NEEDS Phase 1* + Phase 2 ──→┤
            Step 20 (EXTEND config) ← can start early             │
            Steps 21-22 (core + callback) ← needs P1+P1.5+P1.6+P2│
            Steps 23-25 (UI) ← can parallel with 21-22            │
            Step 26 (init exports) ← needs everything              │
                                                                    │
         Phase 4 (Steps 27-31)    ─── NEEDS Phase 3 ──────────→ DONE
            Step 27 (weights) ← needs working training pipeline
            Step 28 (registry) ← needs weights
            Steps 29-31 (polish) ← needs everything

    * Phase 3 depends on Phase 1 + 1.5 + 1.6 (the complete extraction
      system with all fixes applied). Phase 1.6 bugfixes change key names
      that Phase 3's core.py and callback.py must match.

PARALLELIZATION OPPORTUNITIES:
  - Phase 1 Steps 3-9 (metric modules) can all be coded in parallel
  - Phase 1.5 and Phase 1.6 can be done in parallel with each other
  - Phase 2 Steps 13-17 (move/refactor) can all be done in parallel
  - Phase 2 is fully independent of Phase 1.5/1.6
  - Phase 3 Steps 20, 23-25 can start before 21-22
  - Phase 1 and Phase 2 are fully independent until Phase 3 merges them

================================================================================
TOTAL ESTIMATES
================================================================================

 New code to write:     ~2,800-3,200 lines (includes Phase 1.5/1.6 additions + curvature.py)
 Code to port/refactor: ~1,600-1,800 lines (from base_metrics + metric_collector + statistics)
 Code to move:          ~1,160 lines (feature_groups + feature_processor + fpg + encoder + model)
 Test code:             ~2,000-2,400 lines (gate tests across 6 phases + e2e + regression)
 Documentation:         ~650 lines (FEATURE_REFERENCE.md)
 ─────────────────────────────────────────
 Total:                 ~8,200-9,050 lines

 Files to create:       ~38 files (29 source + 1 doc + 8 test files)
 Files to modify:       ~10 existing files (pyproject.toml, README.md, .gitignore,
                        + Phase 1 modules modified by Phase 1.5/1.6)
 Sub-tasks:             ~160 granular sub-tasks across 31 steps + 3 sub-phases
 Test count:            ~216 tests across 6 phase gates + final suite

================================================================================
CODE REUSE MAP (unchanged from original)
================================================================================

 ┌──────────────────────────┬────────────────────────────────────────────────────────┬─────────────────────┐
 │       New location       │                         Source                         │       Action        │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ extraction/metrics/*     │ last_project_phd/.../base_metrics.py (1050 lines)      │ Port + refactor     │
 │                          │                                                        │ into modules        │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ extraction/collector.py  │ last_project_phd/.../metric_collector.py               │ Port + generalize   │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ extraction/aggregator.py │ last_project_phd/.../statistics.py + running_metrics.py│ Port                │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ processing/pipeline.py   │ src/data/feature_processor.py                          │ Refactor (add       │
 │                          │                                                        │ save/load)          │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ processing/groups.py     │ src/data/feature_groups.py                             │ Move                │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ diagnosis/fpg.py         │ src/data/fundamental_fpg.py                            │ Move                │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ diagnosis/encoder.py     │ src/models/group_encoder.py                            │ Move                │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ diagnosis/model.py       │ hierarchical_graph_category_rootcause/model.py         │ Move                │
 ├──────────────────────────┼────────────────────────────────────────────────────────┼─────────────────────┤
 │ All other files          │ —                                                      │ New code            │
 └──────────────────────────┴────────────────────────────────────────────────────────┴─────────────────────┘

================================================================================
KEY RISKS (unchanged + expanded mitigations)
================================================================================

 1. Feature alignment: Features extracted from live training MUST match the column names/order used in the
 pretrained model. Mitigation: canonical feature name registry with strict validation. Set rules for features
 and their variants (temporal, statistical version etc.)
 → EXPANDED: Step 2.2 defines CANONICAL_FEATURES dict. Step 10.5 validates feature_names property.
   Step 18.3a validates alignment at inference time.

 2. Architecture generalization: Different HF models have different module hierarchies. Mitigation:
 Auto-discovery approach — probe model structure at runtime instead of maintaining a per-model
 registry. Two structural categories (BERT-style encoder, GPT-style decoder) defined by their
 module tree patterns, not by model name. Any model following the same structure auto-works.
 → EXPANDED: Step 1.1-1.8 implements discovery. Step 1.14 tests with unknown-but-compatible model.

 3. Attention weight access: Some models don't return attention weights by default. Mitigation: Set
 output_attentions=True in model config, or use forward hooks.
 → EXPANDED: Step 1.8 implements both strategies (config flag + hooks fallback).

 4. Memory overhead: Collecting 400+ features every N steps adds overhead. Mitigation: configurable log_every,
 lazy computation, dynamic layer sampling (early/mid/late).
 → EXPANDED: Step 1.9 implements dynamic sampling. Step 10.2 gates expensive metrics behind log_every.
   Step 20.1 makes all intervals configurable.

 5. Pretrained model mismatch: Feature processing is fold-specific (fitted on training fold). Mitigation: Use
 the best-fold processor, validate feature count matches at inference time.
 → EXPANDED: Step 27.1 saves best-fold artifacts. Step 13.3 loads them. Step 18.3a validates.

================================================================================
VERIFICATION CHECKLIST
================================================================================

 1. Unit test: ModelInspector auto-discovers both categories (encoder/decoder)
    → Step 1.14, Phase 1 Gate T1.1-T1.7b

 2. Integration test: DEFaultPP(bert_model/gpt2_model, optimizer).step(loss=...) collects metrics
    → Step 30.4, Step 30.7a-b

 3. Feature alignment test: Extracted feature names match pretrained model's expected features
    → Step 30.4d, Step 30.5d

 4. End-to-end test: Train BERT/GPT2 for 3 epochs with DEFaultPP → diagnose() returns valid report
    → Step 30.7a-b

 5. Callback test: HuggingFace Trainer with DEFaultPPCallback completes training + diagnosis
    → Step 30.7c

 6. Round-trip test: FeatureProcessor save → load → transform produces identical results
    → Step 30.5b

 7. Export test: report.save() produces valid JSON, report.plot() produces Figures
    → Step 30.7d-f
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
