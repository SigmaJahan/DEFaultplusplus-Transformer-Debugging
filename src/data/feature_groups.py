"""Feature group decomposition mapping feature columns to transformer subsystem groups.

Maps each feature column to one of 13 diagnostic subsystem groups based on
the naming conventions from the FrankenFormer metrics collection pipeline.
Handles both encoder (per-layer probes: ffn_delta_l3_mean_final) and decoder
(aggregated component metrics: grad_norm_agg_ffn_early_mean) naming patterns.
"""
import re

SUBSYSTEM_GROUPS = [
    "attention",
    "qkv",
    "score",
    "positional",
    "ffn",
    "layernorm",
    "residual",
    "representation",
    "embedding",
    "training_dynamics",
    "task_metrics",
    "kernel_timing",
    "cache_diagnostics",
]

# Component-level gradient/update metrics: the component name (attn, ffn, etc.)
# appears AFTER the metric-type prefix (grad_norm_agg_, update_ratio_l3_, etc.).
# These must be routed to the component's group, not to training_dynamics.
_COMPONENT_MAP = {
    "attn": "attention",
    "ffn": "ffn",
    "layernorm": "layernorm",
    "qkv": "qkv",
    "emb": "embedding",
}

_COMPONENT_RE = re.compile(
    r"(?:grad_norm_(?:agg_)?(?:l\d+_)?|update_ratio_(?:agg_)?(?:l\d+_)?|update_active_(?:agg_)?(?:l\d+_)?)"
    r"(attn|ffn|layernorm|qkv|emb)"
)

# Token-based rules for mapping feature column names to subsystem groups.
# Order matters: first match wins. Uses substring matching on cleaned names.
_TOKEN_RULES = [
    # Attention mechanism: entropy, sparsity, head redundancy, masking behavior
    (["attn_entropy", "attn_pad", "attn_weight", "attn_cross", "attn_mass",
      "attn_max", "attn_sparsity", "head_similarity", "head_util",
      "mass_leak", "mass_pad", "cross_example"], "attention"),
    # QKV projection alignment
    (["qk_cos", "qv_cos", "kv_cos", "qkv_"], "qkv"),
    # Pre-softmax attention scores (checked after attention to avoid attn_ clash)
    (["score_mean", "score_var", "score_max", "score_std", "score_skew",
      "attn_score", "presoftmax"], "score"),
    # Positional encoding effects
    (["pos_discrim", "pos_acc", "positional", "pos_recv", "pos_inv",
      "pos_loss", "pos_margin"], "positional"),
    # FFN block (including activation function output)
    (["ffn_delta", "ffn_norm", "ffn_out", "ffn_var", "ffn_active",
      "ffn_", "activation_"], "ffn"),
    # LayerNorm statistics
    (["ln_gamma", "ln_post", "layernorm", "ln_"], "layernorm"),
    # Residual stream similarity
    (["res_cos", "res_sim", "residual"], "residual"),
    # Representation drift (CKA, hidden-state cosine drift, hidden delta)
    (["cka_", "repr_drift", "representation", "repr_l", "h1_delta"], "representation"),
    # Embedding statistics
    (["emb_norm", "emb_var", "embedding"], "embedding"),
    # Cache diagnostics (decoder-only)
    (["cache_", "kv_cache"], "cache_diagnostics"),
    # Kernel timing / runtime
    (["step_time", "peak_mem", "kernel_time"], "kernel_timing"),
    # Task-level performance metrics
    (["accuracy", "loss", "perplexity", "ece", "nll", "f1_score",
      "precision", "recall", "primary_metric", "edge_case",
      "margin_gap", "margin_neg", "margin_pos"], "task_metrics"),
    # Training dynamics (total gradients, weight stats, logit distributions)
    (["grad_norm", "grad_noise", "grad_abs", "grad_zero",
      "update_ratio", "update_active", "weight_mean", "weight_std",
      "logit_conf", "logit_entropy", "logit_kl", "logit_margin"], "training_dynamics"),
]

# Structural/metadata columns (not feature groups)
_STRUCTURAL_PREFIXES = ["arch_", "layer_idx", "severity_"]


def _extract_component(clean: str) -> str | None:
    """Extract transformer component from gradient/update metric names.

    Routes grad_norm_agg_attn_* -> attention, update_ratio_l3_ffn_* -> ffn, etc.
    Returns None for total/classifier/unrecognized suffixes.
    """
    m = _COMPONENT_RE.match(clean)
    if m:
        return _COMPONENT_MAP[m.group(1)]
    return None


def assign_feature_to_group(feature_name: str) -> str | None:
    """Map a single feature column name to its subsystem group.

    Returns None for structural/metadata columns.
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
    """Build a mapping from subsystem group name to column indices.

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
    """Get the number of features in each subsystem group."""
    indices = build_group_indices(feature_names)
    return {g: len(idxs) for g, idxs in indices.items()}
