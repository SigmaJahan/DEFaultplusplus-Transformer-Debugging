"""Feature-group decomposition for DEFault++ diagnosis.

Each diagnostic feature column is mapped to one of 12 (encoder) or 13
(decoder) feature groups. Groups split into two roles:

  Structural groups: each maps to a transformer subsystem and receives
  message-passing edges from connected groups in the Fault Propagation
  Graph (FPG).

    attention        attention masking, weights, output
    score            pre-softmax attention score
    ffn_output       feed-forward output norm
    layernorm        LayerNorm scale parameter and post-norm distribution
    residual_stream  residual cosine similarity
    qkv_alignment    Q-K, Q-V, K-V cosine similarity
    embedding        embedding norm and token-level variance
    positional       positional sensitivity
    output           prediction confidence, output entropy, margin
    cache            KV cache hidden similarity / distribution divergence
                     (decoder only)

  Non-structural groups: model-wide context with no neighbor aggregation
  in the FPG (self-loop only). They still pass through the same encoder
  and projection.

    representation_drift  cross-layer CKA similarity
    training_dynamics     loss trajectory, gradient noise scale, step time,
                          peak memory, and component-level gradient stats
                          that are not appended to a structural group
    validation_perf       task accuracy / perplexity, calibration error

Component-level gradient/update statistics (gradient norm, update ratio,
update activity) are routed to the structural group of the originating
component rather than to a separate gradient group: gradient metrics about
attention parameters land in `attention`, gradient metrics about FFN
parameters land in `ffn_output`, and so on.
"""
import re

# Group names exposed to the encoder, classifier, and explanation code.
STRUCTURAL_GROUPS = [
    "attention",
    "score",
    "ffn_output",
    "layernorm",
    "residual_stream",
    "qkv_alignment",
    "embedding",
    "positional",
    "output",
    "cache",            # decoder only
]

NON_STRUCTURAL_GROUPS = [
    "representation_drift",
    "training_dynamics",
    "validation_perf",
]

SUBSYSTEM_GROUPS = STRUCTURAL_GROUPS + NON_STRUCTURAL_GROUPS

# Component-level gradient/update metrics: the component name (attn, ffn, etc.)
# appears after the metric-type prefix. These are appended to the structural
# group of the originating component rather than collected in
# training_dynamics, so each structural group's embedding sees the gradient
# behavior of its own parameters.
_COMPONENT_MAP = {
    "attn": "attention",
    "attention": "attention",
    "ffn": "ffn_output",
    "layernorm": "layernorm",
    "qkv": "qkv_alignment",
    "emb": "embedding",
    "embedding": "embedding",
}

# Match both short layer prefixes (l3_) and long ones (layer3_), and both
# short component names (attn) and long ones (attention/embedding).
_COMPONENT_RE = re.compile(
    r"(?:grad_norm_(?:agg_)?(?:l(?:ayer)?\d+_)?"
    r"|update_ratio_(?:agg_)?(?:l(?:ayer)?\d+_)?"
    r"|update_active_(?:agg_)?(?:l(?:ayer)?\d+_)?)"
    r"(attn|attention|ffn|layernorm|qkv|emb|embedding)"
)

# Token-based rules for mapping feature column names to feature groups.
# Order matters: first match wins. Uses substring matching on cleaned names.
# Each rule list contains both the short form (e.g. attn_entropy) emitted
# by the in-process extractor and the long form (e.g. attention_entropy)
# emitted by the offline raw collector.
_TOKEN_RULES = [
    # Pre-softmax attention scores: must come before 'attention' rules so
    # 'pre_softmax_score', 'attention_score', etc. land in 'score'.
    (["score_mean", "score_var", "score_max", "score_std", "score_skew",
      "attn_score", "attention_score", "presoftmax", "pre_softmax_score"], "score"),
    # Attention mechanism: masking, weights, output.
    (["attn_entropy", "attn_pad", "attn_weight", "attn_cross", "attn_mass",
      "attn_max", "attn_sparsity",
      "attention_entropy", "attention_pad", "attention_weight",
      "attention_cross", "attention_mass", "attention_max",
      "attention_sparsity",
      "head_similarity", "head_util",
      "mass_leak", "mass_pad", "cross_example"], "attention"),
    # QKV alignment: pairwise cosine similarities of Q, K, V projections.
    (["qk_cos", "qv_cos", "kv_cos", "qkv_"], "qkv_alignment"),
    # Positional encoding effects.
    (["pos_discrim", "pos_acc", "positional", "pos_recv", "pos_inv",
      "pos_loss", "pos_margin"], "positional"),
    # FFN output norm.
    (["ffn_delta", "ffn_norm", "ffn_out", "ffn_var", "ffn_active",
      "ffn_", "activation_"], "ffn_output"),
    # LayerNorm scale parameter and post-norm distribution.
    (["ln_gamma", "ln_post", "layernorm", "ln_"], "layernorm"),
    # Residual stream cosine similarity.
    (["res_cos", "res_sim", "residual"], "residual_stream"),
    # Cross-layer representation drift (CKA).
    (["cka_", "repr_drift", "representation", "repr_l", "h1_delta"],
     "representation_drift"),
    # Embedding norm and variance.
    (["emb_norm", "emb_var", "embedding"], "embedding"),
    # KV cache diagnostics (decoder only).
    (["cache_", "kv_cache"], "cache"),
    # Output head: prediction confidence, output entropy, margin.
    (["logit_conf", "logit_entropy", "logit_margin", "logit_kl",
      "logit_nan", "logit_inf",
      "margin_gap", "margin_neg", "margin_pos"], "output"),
    # Validation performance: task accuracy / perplexity / calibration error.
    (["accuracy", "perplexity", "ece", "f1_score",
      "precision", "recall", "primary_metric", "edge_case"], "validation_perf"),
    # Training dynamics: loss trajectory, gradient noise scale, step time,
    # peak memory, and global gradient stats. Per-step training loss
    # belongs here (not validation_perf).
    (["loss", "nll", "step_time", "peak_mem", "kernel_time",
      "grad_norm", "grad_noise", "grad_abs", "grad_zero",
      "gradient_noise", "gradient_variance", "gradient_vanish",
      "gradient_explode",
      "update_ratio", "update_active", "weight_mean", "weight_std"],
     "training_dynamics"),
]

# Structural/metadata columns (not feature groups)
_STRUCTURAL_PREFIXES = ["arch_", "layer_idx", "severity_"]


def _extract_component(clean: str) -> str | None:
    """Extract transformer component from gradient/update metric names.

    Routes grad_norm_agg_attn_* -> attention, update_ratio_l3_ffn_* -> ffn_output, etc.
    Returns None for total/classifier/unrecognized suffixes.
    """
    m = _COMPONENT_RE.match(clean)
    if m:
        return _COMPONENT_MAP[m.group(1)]
    return None


def assign_feature_to_group(feature_name: str) -> str | None:
    """Map a single feature column name to its feature group.

    Returns None for structural/metadata columns that are not features.
    """
    fname = feature_name.lower()

    for prefix in _STRUCTURAL_PREFIXES:
        if fname.startswith(prefix):
            return None

    # Strip presentation prefixes that don't affect component identity
    clean = fname
    for prefix in ("abs_", "delta_", "rank_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    # Component-level gradient/update metrics: route by component
    component = _extract_component(clean)
    if component is not None:
        return component

    # Match against token rules (substring matching)
    for tokens, group in _TOKEN_RULES:
        for token in tokens:
            if token in clean:
                return group

    return "training_dynamics"


def build_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    """Build a mapping from feature group name to column indices.

    Returns dict: group_name -> list of column indices into the feature array.
    Only includes groups that have at least one feature.
    """
    group_indices: dict[str, list[int]] = {}

    for idx, name in enumerate(feature_names):
        group = assign_feature_to_group(name)
        if group is None:
            continue
        if group not in group_indices:
            group_indices[group] = []
        group_indices[group].append(idx)

    return group_indices


def get_group_sizes(feature_names: list[str]) -> dict[str, int]:
    """Get the number of features in each feature group."""
    indices = build_group_indices(feature_names)
    return {g: len(idxs) for g, idxs in indices.items()}
