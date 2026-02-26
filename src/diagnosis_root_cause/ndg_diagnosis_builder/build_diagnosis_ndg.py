"""Root-cause diagnosis and remediation.

Builds a reasoning layer on top of detection/categorization results.
Inspired by MODE (state differential analysis) and ATTNChecker (fault propagation tracing).

Three analysis components:
  A) Differential Signature Analysis -- per-family behavioral fingerprints by contrasting
     faulty vs. baseline metric distributions (MODE-style heat maps adapted for telemetry)
  B) Fault Propagation Profiling -- maps how each fault type impacts different model subsystems
     (attention, FFN, normalization, etc.) to trace primary/secondary/tertiary effects
  C) Per-sample diagnostic reasoning -- given categorization predictions, generates structured
     diagnostic reports with evidence chains, severity assessment, and remediation guidance

Outputs (in results/):
  enc_diagnosis.json / dec_diagnosis.json

Usage:
  python build_diagnosis_ndg.py              # full run
  python build_diagnosis_ndg.py --arch enc   # encoder only
"""
import argparse, json, re, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
VARIANT_DIR = ROOT / "src" / "detection_categorization_xai" / "data"
ORIG_DIR = ROOT / "src" / "diagnosis_root_cause" / "data"
RESULTS_DIR = ROOT / "results" / "diagnosis"
N_SPLITS = 5
RNG = 42

# ── METRIC GROUPING: maps abs_ features to model subsystems ──────────────
# Analogous to ATTNChecker's per-GEMM segmentation; groups metrics by the
# transformer component they probe.

# ── Subsystem Mapping (authoritative) ─────────────────────────────────────
# We map *feature variants* (e.g., abs_ffn_delta_l1_mean_final) to a canonical
# subsystem using feature_core_map.md as the source of truth.
#
# If a feature is not found in the map, we fall back to a conservative
# token-based heuristic. This preserves robustness while keeping the mapping
# aligned with the thesis artifact.

FEATURE_CORE_TO_SUBSYSTEM = None  # populated at runtime

_CORE_SECTION_TO_SUBSYSTEM = {
    "activation": "ffn",
    "attention": "attention",
    "diagnostic (logit)": "output_logits",
    "diagnostic (logits)": "output_logits",
    "embedding": "embedding",
    "ffn": "ffn",
    "gradient": "gradient",
    "layernorm": "layernorm",
    "residual": "residual",
    "positional": "positional",
    "runtime": "runtime",
    "kv cache": "kv_cache",
    "representation": "representation",
    "performance": "performance",
    "structural": "structural",
}

_TOKEN_RULES = [
    (r"(?:^|_)kv_cache(?:_|$)|(?:^|_)cache_|cache_hidden|cache_nll", "kv_cache"),
    (r"peak_mem|mem_|step_time|latency|runtime|kernel", "runtime"),
    (r"(?:^|_)ffn(?:_|$)|mlp|activation", "ffn"),
    (r"(?:^|_)ln(?:_|$)|layernorm", "layernorm"),
    (r"residual", "residual"),
    (r"(?:^|_)pos(?:_|$)|position|positional", "positional"),
    (r"(?:^|_)qkv(?:_|$)|presoftmax|head_similarity|attn|mass_|entropy", "attention"),
    (r"logit|margin|ece|calib", "output_logits"),
    (r"loss|accuracy|f1|perplexity|nll|precision|recall", "performance"),
    (r"grad|update_ratio|weight_|gns", "gradient"),
    (r"(?:^|_)repr(?:_|$)|h1_|drift|cos", "representation"),
    (r"(?:^|_)emb(?:_|$)|embedding", "embedding"),
    (r"severity|layer_idx|arch_", "structural"),
]

def _subsystem_fallback_token(feat_name: str) -> str:
    c = feat_name.lower()
    for pat, sub in _TOKEN_RULES:
        if re.search(pat, c):
            return sub
    return "other"

def load_feature_core_map_md(path: Path) -> dict:
    """Parse feature_core_map.md into core_feature -> subsystem mapping."""
    mapping = {}
    current_section = None
    sec_re = re.compile(r"^##\s+(.+?)(?:\s+\(|$)")
    row_re = re.compile(r"^\|\s*`([^`]+)`\s*\|")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        m = sec_re.match(line.strip())
        if m:
            current_section = m.group(1).strip().lower()
            # normalize
            if current_section.startswith("diagnostic"):
                current_section = "diagnostic (logit)"
            continue
        m = row_re.match(line)
        if m:
            core = m.group(1).strip()
            if not core or core.lower() == "core feature":
                continue
            sub = None
            if current_section:
                sub = _CORE_SECTION_TO_SUBSYSTEM.get(current_section)
            if not sub:
                sub = _subsystem_fallback_token(core)
            mapping[core] = sub
    return mapping

_STAT_SUFFIXES = [
    "_early_mean", "_early_slope", "_mid_mean", "_mid_slope", "_final",
    "_finalwin", "_mean_finalwin", "_finalwin_mean"
]

def _strip_stat_suffix(name: str) -> str:
    for suf in _STAT_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def _candidate_core_keys(feat_name: str) -> list:
    """
    Generate candidate core keys from a feature variant name.
    Tries progressively more aggressive normalizations.
    """
    f = feat_name.strip()
    cand = []

    # 1) exact
    cand.append(f)

    # 2) strip stat suffix
    f2 = _strip_stat_suffix(f)
    if f2 != f:
        cand.append(f2)

    # 3) if per-layer pattern exists: keep up to layer id token
    # e.g., abs_ffn_delta_l1_mean_final -> abs_ffn_delta_l1
    m = re.match(r"^(abs_[a-z0-9_]+_l\d+)", f2)
    if m:
        cand.append(m.group(1))

    # 4) drop per-layer index entirely: abs_ffn_delta_l1 -> abs_ffn_delta
    f3 = re.sub(r"_l\d+$", "", cand[-1])
    if f3 not in cand:
        cand.append(f3)

    # 5) remove trailing metric-specific fragments that are often variants
    # e.g., abs_step_time_mean_finalwin -> abs_step_time_mean
    f4 = re.sub(r"_(mean|std|var|skew|kurt)$", "", f2)
    if f4 not in cand:
        cand.append(f4)

    # 6) final fallback: keep abs_ prefix and first 3 tokens
    toks = f2.split("_")
    if len(toks) >= 3:
        cand.append("_".join(toks[:3]))

    # unique preserving order
    seen = set()
    out = []
    for c in cand:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def feature_to_subsystem(feat_name):
    """
    Map a feature variant name to its subsystem using feature_core_map.md when available.
    """
    global FEATURE_CORE_TO_SUBSYSTEM
    if FEATURE_CORE_TO_SUBSYSTEM:
        for ck in _candidate_core_keys(feat_name):
            if ck in FEATURE_CORE_TO_SUBSYSTEM:
                return FEATURE_CORE_TO_SUBSYSTEM[ck]
    return _subsystem_fallback_token(feat_name)


# ── DIAGNOSTIC KNOWLEDGE BASE ────────────────────────────────────────────
DIAGNOSTIC_KB = {
    "qkv": {
        "description": "QKV projection fault -- corrupted query, key, or value weight matrices",
        "primary_subsystem": "attention_score",
        "secondary_subsystems": ["attention_pattern", "output_logits"],
        "propagation": "QKV corruption -> attention score distortion -> attention pattern collapse/divergence -> logit degradation",
        "signature": "Attention distribution collapse or divergence; head desynchronization",
        "subcategory_hints": {
            "freeze_qkv": "QKV weights frozen -- attention patterns static across epochs. Look for zero update ratios at affected layer.",
            "swapped_qk": "Q and K matrices exchanged -- attention asymmetry. Softmax receives transposed scores, breaking causal patterns.",
            "tie_heads": "Attention heads tied -- reduced head diversity. Head similarity metric will spike to 1.0.",
            "wrong_head_dim": "Incorrect head dimension -- attention scale mismatch. Pre-softmax variance will be abnormally high/low.",
            "zero_key": "Key matrix zeroed -- attention becomes uniform. Entropy will be maximal (= log(seq_len)).",
            "zero_query": "Query matrix zeroed -- attention becomes data-independent. All positions attend identically.",
            "zero_value": "Value matrix zeroed -- attention output near zero. Residual stream dominates; FFN compensates.",
        },
        "remediation_steps": [
            "1. Check QKV projection weight shapes match model config (d_model -> 3*d_model or separate Q,K,V)",
            "2. Verify Q, K, V are distinct parameters (not accidentally shared references)",
            "3. Inspect head dimension: d_k = d_model / n_heads must be exact",
            "4. Compare attention entropy distribution against known-good baseline",
            "5. If entropy is uniform: key/query likely zeroed. If collapsed: scaling issue.",
        ],
    },
    "score": {
        "description": "Attention score computation fault -- scaling, dropout, or type casting error",
        "primary_subsystem": "attention_score",
        "secondary_subsystems": ["attention_pattern", "performance"],
        "propagation": "Score computation error -> softmax input distortion -> attention weight collapse/explosion -> downstream degradation",
        "signature": "Softmax scaling breakdown; attention scores explode or collapse",
        "subcategory_hints": {
            "misplaced_dropout": "Dropout at wrong stage -- stochastic corruption of attention scores. Loss variance will increase across seeds.",
            "missing_scaling": "Missing 1/sqrt(d_k) -- pre-softmax values too large, causing softmax saturation. Attention becomes one-hot.",
            "unsafe_type_cast": "Precision loss in computation. Subtle numerical errors accumulate; check for float16/bfloat16 overflow.",
            "wrong_scaling_factor": "Incorrect scaling constant -- attention temperature wrong. Pre-softmax variance diagnostic is key.",
        },
        "remediation_steps": [
            "1. Verify scaling factor = 1/sqrt(d_k) in attention score computation",
            "2. Check dropout placement: should be AFTER softmax, not before",
            "3. Inspect pre-softmax score distribution: should have variance ~1.0 with correct scaling",
            "4. Verify numerical precision: attention computation should use float32 minimum",
        ],
    },
    "positional": {
        "description": "Positional encoding fault -- position information corrupted or missing",
        "primary_subsystem": "positional",
        "secondary_subsystems": ["performance", "attention_pattern"],
        "propagation": "Position signal corruption -> loss of sequential ordering -> position-dependent accuracy drops -> model treats input as bag-of-tokens",
        "signature": "Position-dependent accuracy degrades; model loses sequential ordering",
        "subcategory_hints": {
            "double_position": "Positional encoding applied twice -- doubled position signal. Model may still learn but with shifted attention patterns.",
            "missing_positional": "No positional encoding -- model is bag-of-tokens. Position inversion metric will be ~0.5 (random).",
            "off_by_one": "Position indices shifted by one -- boundary effects. First/last token handling will be most affected.",
            "truncate_positions": "Positions truncated beyond max_length -- long sequences lose position info beyond cutoff.",
        },
        "remediation_steps": [
            "1. Verify positional encoding is applied exactly once (check forward() call chain)",
            "2. Check position indices: should be 0-indexed, length-matched to input",
            "3. Verify max_position_embeddings >= max input sequence length",
            "4. Compare position inversion metric: should be >> 0.5 for correct position encoding",
        ],
    },
    "masking": {
        "description": "Attention masking fault -- causal or padding mask corrupted",
        "primary_subsystem": "attention_pattern",
        "secondary_subsystems": ["attention_score", "performance"],
        "propagation": "Mask corruption -> unauthorized attention to masked positions -> information leakage -> degraded generalization",
        "signature": "Causal mask breach or padding leak; cross-example attention detected",
        "subcategory_hints": {
            "inverted_mask": "Mask logic inverted -- attending to masked positions, ignoring valid ones. Accuracy will drop catastrophically.",
            "wrong_mask_broadcast": "Mask shape mismatch -- some heads or positions unmasked. Partial information leakage.",
            "zero_mask": "All-zero mask -- no masking applied. All tokens attend everywhere. Encoder-like behavior in decoder.",
            "break_causal_mask": "Causal mask broken -- future tokens visible (decoder only). Training loss will be artificially low.",
            "pad_masking_error": "Padding mask incorrect -- padded tokens influence attention. Batch-size-dependent behavior.",
        },
        "remediation_steps": [
            "1. Verify mask shape matches (batch, 1, seq_len, seq_len) for causal or (batch, 1, 1, seq_len) for padding",
            "2. Check mask values: -inf or large negative for masked, 0 for unmasked (additive); 0/1 for multiplicative",
            "3. For causal mask: verify lower-triangular structure",
            "4. Check cross-example attention metric: should be ~0 with correct padding mask",
        ],
    },
    "ffn": {
        "description": "Feed-forward network fault -- activation or weight corruption in MLP layers",
        "primary_subsystem": "ffn",
        "secondary_subsystems": ["gradient", "performance"],
        "propagation": "FFN activation/weight error -> intermediate representation distortion -> residual stream contamination -> output degradation",
        "signature": "FFN activation or gradient anomaly; update ratios deviate",
        "subcategory_hints": {
            "activation_distortion": "Activation function corrupted -- wrong nonlinearity. FFN output distribution will differ from expected GELU/ReLU pattern.",
            "ffn_neuron_drop": "FFN neurons dropped -- reduced MLP capacity. Model may partially compensate via attention.",
            "ffn_weight_scaling": "FFN weights scaled incorrectly -- output magnitude wrong. Gradient norms will be abnormal.",
        },
        "remediation_steps": [
            "1. Verify activation function type matches config (GELU for BERT-family, ReLU for GPT-2)",
            "2. Check FFN intermediate dimension: typically 4 * d_model",
            "3. Inspect FFN output variance ratio per layer for anomalies",
            "4. Compare update ratios: FFN layers should have similar magnitude across layers",
        ],
    },
    "layernorm": {
        "description": "Layer normalization fault -- normalization statistics or affine parameters corrupted",
        "primary_subsystem": "layernorm",
        "secondary_subsystems": ["residual", "gradient"],
        "propagation": "LayerNorm corruption -> residual stream scale/shift distortion -> gradient flow disruption -> training instability",
        "signature": "LayerNorm scale/shift collapse; normalization destabilizes residual stream",
        "subcategory_hints": {
            "ln_beta_fault": "LayerNorm bias (beta) corrupted -- shifted output distribution. Mean of normalized output will be non-zero.",
            "ln_gamma_fault": "LayerNorm scale (gamma) corrupted -- variance distortion. Layer outputs will have wrong scale.",
            "ln_stats_fault": "Running statistics corrupted -- mean/variance estimates wrong. Pre-LN vs post-LN architecture matters here.",
        },
        "remediation_steps": [
            "1. Verify gamma initialized to 1.0, beta to 0.0",
            "2. Check eps value (typically 1e-5 or 1e-6)",
            "3. Verify normalization axis (should be last dimension for LayerNorm)",
            "4. Compare ln_std and ln_mean_abs across layers: should be smooth, not spiky",
        ],
    },
    "residual": {
        "description": "Residual connection fault -- skip connection corrupted or missing",
        "primary_subsystem": "residual",
        "secondary_subsystems": ["gradient", "performance"],
        "propagation": "Residual corruption -> gradient highway broken -> deep layer gradients vanish -> catastrophic training failure",
        "signature": "Residual cosine similarity drops; gradient flow disrupted",
        "subcategory_hints": {
            "residual_drop": "Residual connection dropped -- no skip path. Gradient flow completely severed at that layer.",
            "residual_noise": "Noise injected into residual stream. Gradual degradation proportional to noise magnitude.",
            "residual_scale": "Residual scaling wrong -- skip connection magnitude incorrect. Layer output scale drifts across depth.",
        },
        "remediation_steps": [
            "1. Verify x + sublayer(x) is computed correctly (not x * sublayer(x) or sublayer(x) alone)",
            "2. Check residual cosine similarity: should be high (>0.9) for early layers, moderately high for later layers",
            "3. If residual_cos drops sharply at layer L: fault is at or before layer L",
            "4. Compare gradient norms before and after suspected layer",
        ],
    },
    "embedding": {
        "description": "Input embedding fault -- token or type embeddings corrupted",
        "primary_subsystem": "embedding",
        "secondary_subsystems": ["output_logits", "performance"],
        "propagation": "Embedding corruption -> all downstream representations degraded from layer 0 -> global performance collapse",
        "signature": "Embedding norm drifts; token representations degrade",
        "subcategory_hints": {
            "embedding_swap": "Embedding vectors swapped -- some tokens map to wrong representations. Partial accuracy loss on affected tokens.",
            "embedding_zero": "Embeddings zeroed out -- no input signal. Model receives zero input; only positional info survives.",
            "type_embedding_drop": "Type/segment embeddings dropped -- sentence distinction lost. NLI/paraphrase tasks most affected.",
        },
        "remediation_steps": [
            "1. Verify embedding table shape: (vocab_size, d_model)",
            "2. Check embedding norm distribution: should be roughly uniform across vocabulary",
            "3. For type embeddings: verify num_token_types matches task (2 for sentence pairs)",
            "4. If embedding norm is near-zero: initialization or zeroing bug",
        ],
    },
    "output": {
        "description": "Output projection fault -- final linear layer or logit computation corrupted",
        "primary_subsystem": "output_logits",
        "secondary_subsystems": ["performance"],
        "propagation": "Output projection error -> logit distribution distortion -> calibration loss -> decision boundary shift",
        "signature": "ECE spike; calibration loss; decision boundary shifted",
        "subcategory_hints": {
            "out_noise": "Noise in output projection -- logit perturbation. ECE will increase; accuracy less affected for high-confidence predictions.",
            "out_row_drop": "Output rows dropped -- some classes unreachable. Specific class recall = 0.",
            "out_scale": "Output scaling wrong -- logit magnitude incorrect. Softmax temperature effectively changed.",
        },
        "remediation_steps": [
            "1. Verify output projection shape: (d_model, num_classes)",
            "2. Check per-class logit distributions: all classes should be reachable",
            "3. If ECE is high but accuracy is reasonable: scaling/calibration issue",
            "4. Check if any output neurons have zero gradient (dead classes)",
        ],
    },
    "kernel": {
        "description": "Kernel/backend fault -- computational backend or memory management issue",
        "primary_subsystem": "runtime",
        "secondary_subsystems": ["performance"],
        "propagation": "Backend fault -> runtime/memory anomaly -> indirect performance impact (if any). Often no accuracy impact.",
        "signature": "Runtime spike or memory pressure; non-behavioral but operational impact",
        "subcategory_hints": {
            "force_unoptimized": "Forced unoptimized kernel -- slower but correct computation. Accuracy should be unaffected.",
            "inconsistent_dropout": "Dropout inconsistent between train/eval modes. Eval performance will differ from expected.",
            "wrong_layout": "Tensor memory layout wrong -- transposition overhead. Step time increases; accuracy unaffected.",
        },
        "remediation_steps": [
            "1. Compare step_time against baseline: >2x suggests kernel fallback or layout issue",
            "2. Verify model.eval() is called during evaluation (dropout/batchnorm mode)",
            "3. Check tensor contiguity: call .contiguous() before operations requiring it",
            "4. For memory issues: check peak_mem vs expected for batch size and model size",
        ],
    },
    "kv_cache": {
        "description": "KV cache fault (decoder only) -- cached key/value states corrupted",
        "primary_subsystem": "kv_cache",
        "secondary_subsystems": ["attention_score", "performance"],
        "propagation": "Cache corruption -> stale/misaligned K/V -> attention computed on wrong context -> generation quality degrades over sequence length",
        "signature": "Cached hidden state diverges from fresh computation; NLL increases over sequence",
        "subcategory_hints": {
            "cross_request_cache_leak": "Cache not cleared between requests -- context contamination across batches.",
            "off_by_one_index": "Cache indexing off by one -- shifted K/V alignment. Subtle but cumulative error.",
            "stale_cache": "Cache not updated -- using outdated K/V pairs. Quality degrades monotonically with sequence length.",
            "truncated_cache": "Cache truncated prematurely -- context lost for long sequences.",
        },
        "remediation_steps": [
            "1. Verify cache is cleared between forward passes (not just between batches)",
            "2. Check cache index alignment: past_key_value length should equal current position - 1",
            "3. Compare generation quality at position 1 vs position N: divergence suggests cache issue",
            "4. Check cache_hidden_similarity metric: should be > 0.99 for correct caching",
        ],
    },
    "variant": {
        "description": "Attention variant fault -- alternative attention mechanism misconfigured",
        "primary_subsystem": "attention_score",
        "secondary_subsystems": ["attention_pattern", "performance"],
        "propagation": "Wrong attention variant -> fundamentally different attention computation -> model behavior diverges from expected",
        "signature": "Entropy distortion from wrong attention variant; head similarity anomaly",
        "subcategory_hints": {
            "causal_in_noncausal": "Causal masking in bidirectional context -- half the context lost. Encoder accuracy drops ~50%.",
            "wrong_variant": "Wrong attention variant (e.g., linear instead of softmax). Attention pattern will be qualitatively different.",
        },
        "remediation_steps": [
            "1. Verify attention type matches architecture: softmax for standard, linear for efficient variants",
            "2. For encoder models: verify NO causal mask is applied",
            "3. For decoder models: verify causal mask IS applied",
            "4. Check attention pattern visualization: should match expected pattern for architecture type",
        ],
    },
}


def _jdef(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)): return round(float(o), 6)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(f"{type(o)}")


def confidence_level(prob):
    if prob >= 0.8: return "high"
    if prob >= 0.5: return "moderate"
    return "low"


def group_zscore(X_raw, norm_groups, n_abs, train_idx):
    X = X_raw.copy()
    for g in np.unique(norm_groups):
        g_all = np.where(norm_groups == g)[0]
        g_train = np.intersect1d(g_all, train_idx)
        if len(g_train) == 0:
            continue
        mu = X_raw[g_train, :n_abs].mean(axis=0)
        std = X_raw[g_train, :n_abs].std(axis=0)
        std[std == 0] = 1
        X[g_all, :n_abs] = (X_raw[g_all, :n_abs] - mu) / std
    return X


def _es_split(tr_idx, frac=0.2):
    rng = np.random.RandomState(RNG)
    perm = rng.permutation(len(tr_idx))
    n_es = max(int(frac * len(tr_idx)), 1)
    return tr_idx[perm[n_es:]], tr_idx[perm[:n_es]]


# ── A: DIFFERENTIAL SIGNATURE ANALYSIS ───────────────────────────────────
# MODE-style: contrast faulty vs. baseline metric distributions per family.

def compute_differential_signatures(df_baseline, df_faulty, feat_cols, categories):
    """For each fault family, compute a behavioral fingerprint by comparing
    faulty vs. baseline distributions across all features.

    Returns dict mapping family -> {
        per_feature: [{feature, subsystem, effect_size, p_value, direction, z_baseline_mean, z_faulty_mean}],
        primary_subsystem_impact: {subsystem: mean_abs_effect_size},
        propagation_profile: ordered list of affected subsystems,
    }
    """
    X_base = df_baseline[feat_cols].values.astype(np.float64)
    base_means = np.nanmean(X_base, axis=0)
    base_stds = np.nanstd(X_base, axis=0)
    base_stds[base_stds == 0] = 1.0

    families = sorted(set(categories))
    signatures = {}

    for fam in families:
        fam_mask = categories == fam
        X_fam = df_faulty.loc[fam_mask, feat_cols].values.astype(np.float64)
        if len(X_fam) < 5:
            continue

        fam_means = np.nanmean(X_fam, axis=0)
        # Cohen's d effect size: (faulty_mean - baseline_mean) / baseline_std
        effect_sizes = (fam_means - base_means) / base_stds

        per_feature = []
        subsystem_effects = defaultdict(list)
        for i, feat in enumerate(feat_cols):
            es = float(effect_sizes[i])
            subsys = feature_to_subsystem(feat)

            # Welch's t-test (unequal variances)
            f_vals = X_fam[:, i]
            b_vals = X_base[:, i]
            f_vals_clean = f_vals[~np.isnan(f_vals)]
            b_vals_clean = b_vals[~np.isnan(b_vals)]
            if len(f_vals_clean) < 3 or len(b_vals_clean) < 3:
                pval = 1.0
            else:
                _, pval = stats.ttest_ind(f_vals_clean, b_vals_clean, equal_var=False)
                if np.isnan(pval):
                    pval = 1.0

            per_feature.append({
                "feature": feat,
                "subsystem": subsys,
                "effect_size": round(es, 4),
                "p_value": round(float(pval), 6),
                "direction": "increased" if es > 0 else "decreased",
                "baseline_mean": round(float(base_means[i]), 6),
                "faulty_mean": round(float(fam_means[i]), 6),
            })
            subsystem_effects[subsys].append(abs(es))

        # Sort by absolute effect size
        per_feature.sort(key=lambda x: -abs(x["effect_size"]))

        # Subsystem impact summary
        subsystem_impact = {s: round(float(np.mean(v)), 4)
                           for s, v in sorted(subsystem_effects.items(),
                                              key=lambda x: -np.mean(x[1]))}

        # Propagation profile: ordered list of affected subsystems (> 0.2 mean |d|)
        propagation = [s for s, d in subsystem_impact.items() if d > 0.2]

        # KB propagation path
        kb = DIAGNOSTIC_KB.get(fam, {})

        signatures[fam] = {
            "n_samples": int(fam_mask.sum()),
            "top20_differential_features": per_feature[:20],
            "subsystem_impact": subsystem_impact,
            "propagation_profile": propagation,
            "kb_propagation": kb.get("propagation", ""),
            "kb_primary_subsystem": kb.get("primary_subsystem", ""),
        }

    return signatures


# ── B: DIFFERENTIAL DIAGNOSIS ─────────────────────────────────────────────
# For each family, identify confusable families and the features that best
# discriminate between them. This directly answers: "the model says qkv, but
# how do I confirm it's not score?" -- the practical question a practitioner has.

def compute_differential_diagnosis(df_faulty, df_baseline, abs_cols, categories, subcategories, cat_label_names, signatures):
    """For each family, find:
    1. Top confusable families (most similar *effect-size* profiles from Part A)
    2. Discriminative features that separate each confusable pair
    3. Concrete decision rules: IF feat_X > T THEN family_A (not B)
    """
    families = sorted(set(categories))
    # Baseline stats for effect-size computation
    X_base = df_baseline[abs_cols].values.astype(np.float64)
    base_means = np.nanmean(X_base, axis=0)
    base_stds = np.nanstd(X_base, axis=0)
    base_stds[base_stds == 0] = 1.0

    # Pre-compute per-family means, stds, and full effect-size vectors
    fam_stats = {}
    for fam in families:
        mask = categories == fam
        X_fam = df_faulty.loc[mask, abs_cols].values.astype(np.float64)
        if len(X_fam) < 5:
            continue
        fam_means = np.nanmean(X_fam, axis=0)
        # Full Cohen's d vector: (fam_mean - baseline_mean) / baseline_std
        es_vec = (fam_means - base_means) / base_stds
        fam_stats[fam] = {
            "mean": fam_means,
            "std": np.nanstd(X_fam, axis=0),
            "n": len(X_fam),
            "X": X_fam,
            "effect_sizes": es_vec,
        }

    diagnosis = {}
    for fam_a in families:
        if fam_a not in fam_stats:
            continue
        sa = fam_stats[fam_a]

        # Rank by cosine similarity of full effect-size profiles
        # (mean_fam - mean_baseline) / std_baseline -- computed per family vs global baseline
        # Two families are confusable when they deviate from baseline in similar ways
        va = np.nan_to_num(sa["effect_sizes"], nan=0.0)
        sims = []
        for fam_b in families:
            if fam_b == fam_a or fam_b not in fam_stats:
                continue
            vb = np.nan_to_num(fam_stats[fam_b]["effect_sizes"], nan=0.0)
            dot = np.dot(va, vb)
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            cos = float(dot / max(norm_a * norm_b, 1e-12))
            sims.append((fam_b, cos))
        sims.sort(key=lambda x: -x[1])
        confusable = sims[:3]  # top-3 most confusable by effect-size profile

        # For each confusable pair, find discriminative features
        pair_guides = []
        for fam_b, sim in confusable:
            sb = fam_stats[fam_b]
            # Welch's t-test between family A and family B (not vs baseline)
            n_a, n_b = sa["n"], sb["n"]
            pooled_std = np.sqrt((sa["std"] ** 2 / max(n_a, 1)) + (sb["std"] ** 2 / max(n_b, 1)))
            pooled_std[pooled_std == 0] = 1.0
            t_scores = (sa["mean"] - sb["mean"]) / pooled_std
            # Compute actual p-values for top features
            abs_t = np.abs(t_scores)
            top_idx = np.argsort(abs_t)[::-1][:10]

            rules = []
            for idx in top_idx:
                feat = abs_cols[idx]
                t_val = float(t_scores[idx])
                a_vals = sa["X"][:, idx]
                b_vals = sb["X"][:, idx]
                a_clean = a_vals[~np.isnan(a_vals)]
                b_clean = b_vals[~np.isnan(b_vals)]
                if len(a_clean) < 3 or len(b_clean) < 3:
                    continue
                _, pval = stats.ttest_ind(a_clean, b_clean, equal_var=False)
                if np.isnan(pval):
                    pval = 1.0
                # Relax threshold for small families (< 200 samples)
                p_thresh = 0.05 if min(sa["n"], sb["n"]) < 200 else 0.01
                if pval > p_thresh:
                    continue
                # Build a decision threshold (midpoint of means)
                a_mean, b_mean = float(np.mean(a_clean)), float(np.mean(b_clean))
                threshold = (a_mean + b_mean) / 2
                direction = "higher" if a_mean > b_mean else "lower"
                # Discrimination accuracy: how well does this single feature separate?
                if a_mean > b_mean:
                    acc = (np.sum(a_clean > threshold) + np.sum(b_clean <= threshold)) / (len(a_clean) + len(b_clean))
                else:
                    acc = (np.sum(a_clean < threshold) + np.sum(b_clean >= threshold)) / (len(a_clean) + len(b_clean))

                rules.append({
                    "feature": feat,
                    "subsystem": feature_to_subsystem(feat),
                    "rule": f"If {feat} is {direction} (threshold ~{threshold:.4g}), "
                            f"more likely {fam_a} than {fam_b}",
                    "discrimination_accuracy": round(float(acc), 4),
                    "mean_this_family": round(a_mean, 6),
                    "mean_confusable": round(b_mean, 6),
                    "p_value": round(float(pval), 8),
                })
                if len(rules) >= 5:
                    break

            pair_guides.append({
                "confusable_family": fam_b,
                "cosine_similarity": round(sim, 4),
                "discriminative_rules": rules,
            })

        # ── Intra-family: subcategory differential diagnosis ──
        # This is the core root-cause reasoning: "It's a QKV fault, but which one?"
        fam_mask = categories == fam_a
        sub_vals = subcategories[fam_mask]
        unique_subs = sorted(set(sub_vals))
        subcategory_guides = []

        if len(unique_subs) >= 2:
            X_fam = df_faulty.loc[fam_mask, abs_cols].values.astype(np.float64)
            sub_stats = {}
            for sc in unique_subs:
                sc_mask = sub_vals == sc
                X_sc = X_fam[sc_mask]
                if len(X_sc) < 3:
                    continue
                sub_stats[sc] = {
                    "mean": np.nanmean(X_sc, axis=0),
                    "std": np.nanstd(X_sc, axis=0),
                    "n": len(X_sc),
                    "X": X_sc,
                }

            for sc_a in sorted(sub_stats.keys()):
                ssa = sub_stats[sc_a]
                # Find most confusable subcategory within same family
                sc_sims = []
                for sc_b in sorted(sub_stats.keys()):
                    if sc_b == sc_a:
                        continue
                    ssb = sub_stats[sc_b]
                    # Cosine of mean vectors (within-family these are more comparable)
                    va_sc = np.nan_to_num(ssa["mean"], nan=0.0)
                    vb_sc = np.nan_to_num(ssb["mean"], nan=0.0)
                    dot = np.dot(va_sc, vb_sc)
                    na_sc = np.linalg.norm(va_sc)
                    nb_sc = np.linalg.norm(vb_sc)
                    cos = float(dot / max(na_sc * nb_sc, 1e-12))
                    sc_sims.append((sc_b, cos))
                sc_sims.sort(key=lambda x: -x[1])
                top_confusable_sc = sc_sims[:2]

                sc_rules = []
                for sc_b, sc_sim in top_confusable_sc:
                    ssb = sub_stats[sc_b]
                    n_a, n_b = ssa["n"], ssb["n"]
                    pooled = np.sqrt((ssa["std"] ** 2 / max(n_a, 1)) + (ssb["std"] ** 2 / max(n_b, 1)))
                    pooled[pooled == 0] = 1.0
                    t_sc = (ssa["mean"] - ssb["mean"]) / pooled
                    top_sc_idx = np.argsort(np.abs(t_sc))[::-1][:8]
                    for idx in top_sc_idx:
                        feat = abs_cols[idx]
                        a_v = ssa["X"][:, idx]
                        b_v = ssb["X"][:, idx]
                        a_c = a_v[~np.isnan(a_v)]
                        b_c = b_v[~np.isnan(b_v)]
                        if len(a_c) < 3 or len(b_c) < 3:
                            continue
                        _, pv = stats.ttest_ind(a_c, b_c, equal_var=False)
                        if np.isnan(pv):
                            pv = 1.0
                        p_thresh = 0.05 if min(n_a, n_b) < 50 else 0.01
                        if pv > p_thresh:
                            continue
                        a_m, b_m = float(np.mean(a_c)), float(np.mean(b_c))
                        thr = (a_m + b_m) / 2
                        direction = "higher" if a_m > b_m else "lower"
                        if a_m > b_m:
                            acc = (np.sum(a_c > thr) + np.sum(b_c <= thr)) / (len(a_c) + len(b_c))
                        else:
                            acc = (np.sum(a_c < thr) + np.sum(b_c >= thr)) / (len(a_c) + len(b_c))
                        sc_rules.append({
                            "vs_subcategory": sc_b,
                            "feature": feat,
                            "subsystem": feature_to_subsystem(feat),
                            "rule": f"If {feat} is {direction} (threshold ~{thr:.4g}), "
                                    f"more likely {sc_a} than {sc_b}",
                            "discrimination_accuracy": round(float(acc), 4),
                            "p_value": round(float(pv), 8),
                        })
                        if sum(1 for r in sc_rules if r["vs_subcategory"] == sc_b) >= 3:
                            break

                kb_hint = DIAGNOSTIC_KB.get(fam_a, {}).get("subcategory_hints", {}).get(sc_a, "")
                subcategory_guides.append({
                    "subcategory": sc_a,
                    "n_samples": ssa["n"],
                    "kb_explanation": kb_hint,
                    "confusable_with": [{"subcategory": s, "cosine": round(c, 4)} for s, c in top_confusable_sc],
                    "discriminative_rules": sc_rules,
                })

        # KB propagation for context
        kb = DIAGNOSTIC_KB.get(fam_a, {})
        diagnosis[fam_a] = {
            "n_samples": int((categories == fam_a).sum()),
            "n_subcategories": len(unique_subs),
            "subcategory_names": unique_subs,
            "description": kb.get("description", ""),
            "expected_primary_subsystem": kb.get("primary_subsystem", ""),
            "propagation_path": kb.get("propagation", ""),
            "confusable_families": pair_guides,
            "subcategory_diagnosis": subcategory_guides,
        }

    return diagnosis


# ── C: PER-SAMPLE DIAGNOSTIC REASONING ───────────────────────────────────

def load_data(arch_prefix):
    """Load detection pickle (all rows) + killed labels from original CSV."""
    # Detection pickle has both baseline + faulty
    det_pkl = VARIANT_DIR / f"{arch_prefix}_v1_detection.pkl"
    with open(det_pkl, "rb") as f:
        det_data = pickle.load(f)

    # Categorization pickle has faulty-only
    cat_pkl = VARIANT_DIR / f"{arch_prefix}_v1_categorization.pkl"
    with open(cat_pkl, "rb") as f:
        cat_data = pickle.load(f)

    arch_name = "encoder" if arch_prefix == "enc" else "decoder"
    orig_csv = ORIG_DIR / f"{arch_name}_absolute_filled_labeled.csv"
    orig = pd.read_csv(orig_csv, low_memory=False)

    return det_data, cat_data, orig


def run_diagnosis(arch_prefix):
    arch_name = "encoder" if arch_prefix == "enc" else "decoder"
    print(f"\n{'=' * 70}")
    print(f"  STAGE 3: {arch_name.upper()} -- ROOT-CAUSE DIAGNOSIS & REMEDIATION")
    print(f"{'=' * 70}")

    det_data, cat_data, orig_df = load_data(arch_prefix)
    feat_names = cat_data["feature_names"]
    tier_map = cat_data["tier_map"]
    cat_label_names = cat_data["label_names"]
    n_cat = len(cat_label_names)

    # Separate baseline and faulty
    baselines = orig_df[orig_df["is_faulty"] == 0]
    faulty = orig_df[orig_df["is_faulty"] == 1].reset_index(drop=True)
    abs_cols = [c for c in orig_df.columns if c.startswith("abs_")]

    # Faulty metadata
    categories = faulty["fault_category"].values
    subcategories = faulty["fault_subcategory"].values
    killed = faulty["killed"].values.astype(int)
    identifiers = faulty["Identifier"].values
    layer_idx_arr = pd.to_numeric(faulty["layer_idx"], errors="coerce").values
    severity_arr = faulty["severity_params"].values

    print(f"  Baselines: {len(baselines)}, Faulty: {len(faulty)}, "
          f"Features: {len(abs_cols)}, Killed: {killed.sum()}/{len(killed)}")

    # ── Part A: Differential Signatures ──
    print(f"\n  --- Part A: Differential Signature Analysis ---")
    t0 = time.time()
    signatures = compute_differential_signatures(baselines, faulty, abs_cols, categories)
    print(f"  Computed signatures for {len(signatures)} families ({time.time()-t0:.1f}s)")

    for fam, sig in sorted(signatures.items()):
        top3 = sig["top20_differential_features"][:3]
        top_str = ", ".join(f"{f['feature']}(d={f['effect_size']:.2f})" for f in top3)
        print(f"    {fam:15s}: n={sig['n_samples']:5d}, "
              f"propagation=[{', '.join(sig['propagation_profile'][:3])}], "
              f"top: {top_str}")

    # ── Part B: Differential Diagnosis ──
    print(f"\n  --- Part B: Differential Diagnosis ---")
    diff_diag = compute_differential_diagnosis(faulty, baselines, abs_cols, categories, subcategories, cat_label_names, signatures)
    for fam, dd in sorted(diff_diag.items()):
        confusable_str = ", ".join(
            f"{c['confusable_family']}(cos={c['cosine_similarity']:.3f})"
            for c in dd["confusable_families"]
        )
        n_inter = sum(len(c["discriminative_rules"]) for c in dd["confusable_families"])
        n_intra = sum(len(s["discriminative_rules"]) for s in dd.get("subcategory_diagnosis", []))
        n_subs = dd.get("n_subcategories", 0)
        print(f"    {fam:15s}: {n_subs} subcats, {n_inter} inter-family rules, "
              f"{n_intra} intra-family rules, confusable=[{confusable_str}]")

    # ── Part C: Per-Sample Diagnostic Reasoning ──
    print(f"\n  --- Part C: Per-Sample Diagnostic Reasoning ---")

    X_raw = cat_data["X"].astype(np.float64)
    y_cat = cat_data["y"]
    cv_groups = cat_data["cv_groups"]
    norm_groups = cat_data.get("norm_groups")
    n_abs = cat_data.get("n_abs_features", X_raw.shape[1])

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_raw, y_cat, cv_groups))

    if norm_groups is not None:
        X_folds = [group_zscore(X_raw, norm_groups, n_abs, tr) for tr, te in splits]
    else:
        X_folds = [X_raw] * N_SPLITS

    xgb_base = {"tree_method": "hist", "random_state": RNG, "n_jobs": -1,
                "n_estimators": 2000, "verbosity": 0, "subsample": 0.8,
                "colsample_bytree": 0.8, "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "max_depth": 3, "min_child_weight": 5, "learning_rate": 0.1}
    xgb_kill_base = {"tree_method": "hist", "random_state": RNG, "n_jobs": -1,
                     "n_estimators": 2000, "verbosity": 0, "subsample": 0.8,
                     "colsample_bytree": 0.8, "objective": "binary:logistic",
                     "eval_metric": "aucpr",
                     "max_depth": 3, "min_child_weight": 5, "learning_rate": 0.1}
    neg_k, pos_k = np.bincount(killed)
    xgb_kill_base["scale_pos_weight"] = neg_k / max(pos_k, 1)

    all_reports = []
    all_true_cat, all_pred_cat = [], []
    all_true_sc, all_pred_sc = [], []
    all_true_kill, all_pred_kill = [], []

    for fi, (tri, tei) in enumerate(splits):
        print(f"    Fold {fi}: {len(tei)} test...", end=" ", flush=True)
        Xf = X_folds[fi]
        Xtr, Xte = Xf[tri], Xf[tei]

        # Fit scaler for z-score evidence
        sc = StandardScaler().fit(Xtr)
        Xte_scaled = sc.transform(Xte)

        # Categorization model
        fit_c, es_c = _es_split(tri)
        clf_cat = XGBClassifier(**xgb_base)
        sw_c = compute_sample_weight("balanced", y_cat[fit_c])
        clf_cat.fit(Xf[fit_c], y_cat[fit_c], eval_set=[(Xf[es_c], y_cat[es_c])],
                    sample_weight=sw_c, verbose=False)
        cat_probs = clf_cat.predict_proba(Xte)
        cat_preds = clf_cat.predict(Xte)
        if cat_probs.shape[1] < n_cat:
            full = np.zeros((len(Xte), n_cat))
            for ci, c in enumerate(clf_cat.classes_):
                full[:, c] = cat_probs[:, ci]
            cat_probs = full

        # Kill model
        fit_k, es_k = _es_split(tri)
        clf_kill = XGBClassifier(**xgb_kill_base)
        clf_kill.fit(Xf[fit_k], killed[fit_k], eval_set=[(Xf[es_k], killed[es_k])], verbose=False)
        kill_probs = clf_kill.predict_proba(Xte)
        kill_preds = clf_kill.predict(Xte)

        # Per-family root-cause models
        rc_models, rc_encoders = {}, {}
        for fam in cat_label_names:
            fam_mask_tr = categories[tri] == fam
            sub_vals = subcategories[tri][fam_mask_tr]
            unique_sc = np.unique(sub_vals)
            if len(unique_sc) < 2 or fam_mask_tr.sum() < 30:
                continue
            le_sc = LabelEncoder()
            y_rc = le_sc.fit_transform(sub_vals)
            X_rc = Xtr[fam_mask_tr]
            n_rc = len(le_sc.classes_)
            xgb_rc = {**xgb_base, "num_class": n_rc}
            fit_r, es_r = _es_split(np.arange(len(X_rc)))
            clf_rc = XGBClassifier(**xgb_rc)
            if len(fit_r) >= 5 and len(es_r) >= 2:
                clf_rc.fit(X_rc[fit_r], y_rc[fit_r],
                           eval_set=[(X_rc[es_r], y_rc[es_r])],
                           sample_weight=compute_sample_weight("balanced", y_rc[fit_r]),
                           verbose=False)
            else:
                clf_rc.fit(X_rc, y_rc, sample_weight=compute_sample_weight("balanced", y_rc),
                           verbose=False)
            rc_models[fam] = clf_rc
            rc_encoders[fam] = le_sc

        # Permutation importance for feature evidence
        pi = permutation_importance(clf_cat, Xte, y_cat[tei],
                                    scoring="f1_macro", n_repeats=5,
                                    random_state=RNG, n_jobs=-1)
        fold_perm = {feat_names[i]: float(pi.importances_mean[i]) for i in range(len(feat_names))}

        # Generate per-sample reports
        for i in range(len(tei)):
            idx = tei[i]
            true_cat_name = categories[idx]
            true_sc = subcategories[idx]
            true_kill = int(killed[idx])
            pred_cat_idx = int(cat_preds[i])
            pred_cat_name = cat_label_names[pred_cat_idx]

            # Family prediction with top-3
            top3_idx = np.argsort(cat_probs[i])[::-1][:3]
            family_prob = float(cat_probs[i, top3_idx[0]])

            # Root cause prediction
            rc_pred, rc_prob, rc_alts, rc_explanation = "unknown", 0.0, [], ""
            if pred_cat_name in rc_models:
                clf_rc = rc_models[pred_cat_name]
                le_sc = rc_encoders[pred_cat_name]
                rc_proba = clf_rc.predict_proba(Xte[i:i + 1])[0]
                rc_classes = le_sc.classes_.tolist()
                rc_top = np.argsort(rc_proba)[::-1]
                rc_pred = rc_classes[rc_top[0]]
                rc_prob = float(rc_proba[rc_top[0]])
                rc_alts = [{"subcategory": rc_classes[j], "probability": round(float(rc_proba[j]), 4)}
                           for j in rc_top[1:3] if rc_proba[j] > 0.05]
                kb = DIAGNOSTIC_KB.get(pred_cat_name, {})
                rc_explanation = kb.get("subcategory_hints", {}).get(
                    rc_pred, f"Subcategory '{rc_pred}' within {pred_cat_name} family")

            # Kill prediction
            kill_prob_val = float(kill_probs[i, 1]) if kill_probs.shape[1] > 1 else float(kill_preds[i])
            kill_pred_val = bool(kill_preds[i])

            # Evidence: top anomalous features (z-score weighted by permutation importance)
            abnormality = np.abs(Xte_scaled[i])
            evidence_features = []
            for fi_idx, fn in enumerate(feat_names):
                if fn in ("layer_idx_num", "severity_scalar") or fn.startswith("arch_"):
                    continue
                imp = fold_perm.get(fn, 0.0)
                score = imp * abnormality[fi_idx] if imp > 0 else abnormality[fi_idx] * 1e-6
                if score > 0.001:
                    evidence_features.append({
                        "feature": fn,
                        "subsystem": feature_to_subsystem(fn),
                        "z_score": round(float(Xte_scaled[i, fi_idx]), 3),
                        "importance": round(imp, 5),
                        "anomaly_score": round(float(score), 5),
                        "tier": tier_map.get(fn, "?"),
                    })
            evidence_features.sort(key=lambda x: -x["anomaly_score"])

            # Differential context: which subsystems are most anomalous for this sample?
            subsys_z = defaultdict(list)
            for ef in evidence_features[:30]:
                subsys_z[ef["subsystem"]].append(abs(ef["z_score"]))
            subsys_anomaly = {s: round(float(np.mean(v)), 3)
                             for s, v in sorted(subsys_z.items(), key=lambda x: -np.mean(x[1]))}

            # Layer and severity
            layer = layer_idx_arr[idx]
            layer_val = int(layer) if not np.isnan(layer) else None
            sev_raw = severity_arr[idx]
            sev_str = str(sev_raw) if pd.notna(sev_raw) and str(sev_raw) not in ("nan", "") else None

            # Build KB-driven reasoning
            kb = DIAGNOSTIC_KB.get(pred_cat_name, {})
            severity = "critical" if kill_prob_val > 0.8 else ("moderate" if kill_prob_val > 0.5 else "low")

            # Differential signature for predicted family
            fam_sig = signatures.get(pred_cat_name, {})
            fam_top_diff = fam_sig.get("top20_differential_features", [])[:5] if fam_sig else []

            report = {
                "sample_id": str(identifiers[idx]),
                "family": {
                    "predicted": pred_cat_name,
                    "probability": round(family_prob, 4),
                    "confidence": confidence_level(family_prob),
                    "top3": [{"family": cat_label_names[j], "probability": round(float(cat_probs[i, j]), 4)}
                             for j in top3_idx],
                },
                "root_cause": {
                    "predicted": rc_pred,
                    "probability": round(rc_prob, 4),
                    "confidence": confidence_level(rc_prob),
                    "explanation": rc_explanation,
                    "alternatives": rc_alts,
                },
                "impact": {
                    "killed": kill_pred_val,
                    "kill_probability": round(kill_prob_val, 4),
                    "severity": severity,
                },
                "evidence": {
                    "top_anomalous_features": evidence_features[:8],
                    "subsystem_anomaly_profile": subsys_anomaly,
                    "layer_idx": layer_val,
                    "severity_params": sev_str,
                },
                "differential_diagnosis": {
                    "how_to_confirm": (diff_diag.get(pred_cat_name, {})
                                       .get("confusable_families", [])[:2]),
                    "propagation_path": kb.get("propagation", ""),
                    "family_signature_top5": fam_top_diff,
                },
                "remediation": {
                    "family_guidance": kb.get("description", ""),
                    "subcategory_guidance": rc_explanation,
                    "steps": kb.get("remediation_steps", []),
                    "priority": "HIGH" if kill_pred_val else "MEDIUM",
                },
                "ground_truth": {
                    "fault_category": true_cat_name,
                    "fault_subcategory": true_sc,
                    "killed": true_kill,
                },
                "fold": fi,
            }
            all_reports.append(report)
            all_true_cat.append(true_cat_name)
            all_pred_cat.append(pred_cat_name)
            all_true_sc.append(true_sc)
            all_pred_sc.append(rc_pred)
            all_true_kill.append(true_kill)
            all_pred_kill.append(int(kill_pred_val))

        print("done")

    # ── Aggregate Metrics ──
    tc, pc = np.array(all_true_cat), np.array(all_pred_cat)
    tsc, psc = np.array(all_true_sc), np.array(all_pred_sc)
    tk, pk = np.array(all_true_kill), np.array(all_pred_kill)

    family_acc = float((tc == pc).mean())
    correct_fam = tc == pc
    valid_rc = (psc != "unknown") & correct_fam
    rc_acc = float((tsc[valid_rc] == psc[valid_rc]).mean()) if valid_rc.any() else 0.0
    cascade_e2e = family_acc * rc_acc
    kill_acc = float((tk == pk).mean())

    from sklearn.metrics import roc_auc_score
    kill_probs_oof = np.array([r["impact"]["kill_probability"] for r in all_reports])
    try:
        kill_auroc = round(roc_auc_score(tk, kill_probs_oof), 4)
    except ValueError:
        kill_auroc = None

    # Confidence calibration
    conf_buckets = {"high": {"correct": 0, "total": 0},
                    "moderate": {"correct": 0, "total": 0},
                    "low": {"correct": 0, "total": 0}}
    for r in all_reports:
        conf = r["family"]["confidence"]
        conf_buckets[conf]["total"] += 1
        if r["family"]["predicted"] == r["ground_truth"]["fault_category"]:
            conf_buckets[conf]["correct"] += 1
    for lvl in conf_buckets:
        t = conf_buckets[lvl]["total"]
        conf_buckets[lvl]["accuracy"] = round(conf_buckets[lvl]["correct"] / t, 4) if t > 0 else 0.0

    # Per-family diagnostic quality
    per_fam_quality = {}
    for fam in cat_label_names:
        fam_idx_all = [j for j in range(len(tc)) if tc[j] == fam]
        if not fam_idx_all:
            continue
        fam_correct = sum(1 for j in fam_idx_all if pc[j] == fam)
        fam_rc_correct = sum(1 for j in fam_idx_all if pc[j] == fam and psc[j] != "unknown" and tsc[j] == psc[j])
        fam_rc_total = sum(1 for j in fam_idx_all if pc[j] == fam and psc[j] != "unknown")
        fam_kill_correct = sum(1 for j in fam_idx_all if pk[j] == tk[j])
        per_fam_quality[fam] = {
            "n_samples": len(fam_idx_all),
            "family_accuracy": round(fam_correct / len(fam_idx_all), 4),
            "root_cause_accuracy": round(fam_rc_correct / max(fam_rc_total, 1), 4),
            "root_cause_evaluated": fam_rc_total,
            "kill_accuracy": round(fam_kill_correct / len(fam_idx_all), 4),
        }

    print(f"\n  DIAGNOSIS RESULTS ({arch_name})")
    print(f"  {'=' * 50}")
    print(f"  Family accuracy:       {family_acc:.4f}")
    print(f"  Root cause accuracy:   {rc_acc:.4f} (given correct family)")
    print(f"  Cascade E2E:           {cascade_e2e:.4f}")
    print(f"  Kill accuracy:         {kill_acc:.4f}")
    print(f"  Kill AUROC:            {kill_auroc}")
    print(f"  Confidence calibration:")
    for lvl in ["high", "moderate", "low"]:
        b = conf_buckets[lvl]
        print(f"    {lvl}: {b['correct']}/{b['total']} = {b['accuracy']:.3f}")

    # Build final output
    output = {
        "architecture": arch_name,
        "n_reports": len(all_reports),
        "n_baselines": len(baselines),
        "n_faulty": len(faulty),

        "differential_signatures": signatures,
        "differential_diagnosis": diff_diag,

        "cascade_metrics": {
            "family_accuracy": round(family_acc, 4),
            "root_cause_accuracy_given_correct_family": round(rc_acc, 4),
            "cascade_end_to_end": round(cascade_e2e, 4),
            "kill_accuracy": round(kill_acc, 4),
            "kill_auroc": kill_auroc,
        },
        "confidence_calibration": conf_buckets,
        "per_family_quality": per_fam_quality,
        "remediation_coverage": round(
            sum(1 for r in all_reports if r["root_cause"]["predicted"] != "unknown")
            / max(len(all_reports), 1), 4),

        "example_reports": [r for r in all_reports
                           if r["ground_truth"]["fault_category"] != r["family"]["predicted"]][:10]
                          + [r for r in all_reports
                             if r["ground_truth"]["fault_category"] == r["family"]["predicted"]][:15],
    }

    return output


def main():
    p = argparse.ArgumentParser(description="Stage 3: Root-Cause Diagnosis & Remediation")
    p.add_argument("--arch", choices=["enc", "dec", "both"], default="both")
    p.add_argument(
        "--feature_core_map",
        type=Path,
        default=(ROOT / "src" / "diagnosis_root_cause" / "ndg_graph" / "feature_core_map.md"),
        help="Path to feature_core_map.md for authoritative subsystem mapping.",
    )
    args = p.parse_args()

    # Load authoritative subsystem mapping if available
    global FEATURE_CORE_TO_SUBSYSTEM
    if args.feature_core_map and Path(args.feature_core_map).exists():
        FEATURE_CORE_TO_SUBSYSTEM = load_feature_core_map_md(Path(args.feature_core_map))
    else:
        FEATURE_CORE_TO_SUBSYSTEM = None


    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    archs = ["enc", "dec"] if args.arch == "both" else [args.arch]

    for arch in archs:
        t0 = time.time()
        output = run_diagnosis(arch)

        out_path = RESULTS_DIR / f"{arch}_diagnosis.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=_jdef)
        print(f"  Saved: {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
