from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# Canonical subsystem labels used by NDG (keep stable across thesis + artifacts)
SUBSYSTEMS = {
    "attention", "output_logits", "embedding", "ffn", "layernorm", "residual",
    "positional", "gradient", "runtime", "kv_cache", "representation",
    "performance", "structural", "other"
}

_SECTION_TO_SUBSYSTEM = {
    "activation": "ffn",
    "attention": "attention",
    "diagnostic (logit)": "output_logits",
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

def subsystem_from_core_feature_name(core: str) -> str:
    c = core.lower()
    for pat, sub in _TOKEN_RULES:
        if re.search(pat, c):
            return sub
    return "other"

def parse_feature_core_map_md(path: Path) -> Dict[str, str]:
    """
    Parse feature_core_map.md into a mapping: core_feature -> subsystem.

    Strategy:
    - Track the current section header like "## Attention (42 core, 85 cols)"
    - For markdown tables under that header, parse the first column `abs_*`
    - Map based on section name when possible; fallback to token rules.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_section: Optional[str] = None
    mapping: Dict[str, str] = {}

    # Section header pattern: "## Attention (42 core, 85 cols)"
    sec_re = re.compile(r"^##\s+(.+?)(?:\s+\(|$)")
    # Table row pattern: | `abs_x` | Tier | Scope | # | ...
    row_re = re.compile(r"^\|\s*`([^`]+)`\s*\|")

    for line in text:
        m = sec_re.match(line.strip())
        if m:
            current_section = m.group(1).strip().lower()
            continue
        m = row_re.match(line)
        if m:
            core = m.group(1).strip()
            # skip header separator rows
            if core.lower() in ("core feature", "---"):
                continue
            sub = None
            if current_section:
                # normalize some section names
                sec_key = current_section
                if sec_key.startswith("diagnostic"):
                    sec_key = "diagnostic (logit)"
                sub = _SECTION_TO_SUBSYSTEM.get(sec_key)
            if not sub:
                sub = subsystem_from_core_feature_name(core)
            mapping[core] = sub
    return mapping

def resolve_subsystem(core: str, core_map: Optional[Dict[str,str]] = None) -> str:
    if core_map and core in core_map:
        return core_map[core]
    return subsystem_from_core_feature_name(core)


# ── Expanded column -> core feature resolution ──────────────────────────
_TIME_SUFFIXES = [
    "_early_mean", "_early_slope", "_mid_mean", "_mid_slope",
    "_mean_finalwin", "_final",
]
_LAYER_COMP_RE = re.compile(r"_l\d+_(attn|ffn|layernorm|qkv)_final$")
_LAYER_SOLO_RE = re.compile(r"_l\d+_final$")
_EXTRA_ALIASES = {
    "abs_repr_cos": "abs_repr",
    "abs_repr_drift": "abs_repr",
    "abs_step_time_mean": "abs_step_time",
}

def resolve_to_core(expanded: str, core_set: set) -> str:
    """Map an expanded column name back to its canonical core feature name."""
    if expanded in core_set:
        return expanded
    if expanded in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[expanded]
    for suf in _TIME_SUFFIXES:
        if expanded.endswith(suf):
            c = expanded[:-len(suf)]
            if c in core_set:
                return c
    m = _LAYER_COMP_RE.search(expanded)
    if m:
        c = expanded[:m.start()]
        if c in core_set:
            return c
    m = _LAYER_SOLO_RE.search(expanded)
    if m:
        c = expanded[:m.start()]
        if c in core_set:
            return c
    return expanded  # fallback: return as-is


# ── Human-readable display names for thesis/plots ───────────────────────
_DISPLAY_NAMES = {
    # Performance (P tier)
    "abs_accuracy": "Accuracy",
    "abs_loss": "Training Loss",
    "abs_val_loss": "Validation Loss",
    "abs_val_primary_metric": "Primary Metric (Val)",
    "abs_ece": "Calibration Error (ECE)",
    "abs_val_ece": "Calibration Error (Val)",
    "abs_f1_score": "F1 Score",
    "abs_val_f1_score": "F1 Score (Val)",
    "abs_precision": "Precision",
    "abs_val_precision": "Precision (Val)",
    "abs_recall": "Recall",
    "abs_val_recall": "Recall (Val)",
    "abs_val_accuracy": "Accuracy (Val)",
    "abs_val_accuracy_gap": "Accuracy Gap (Val)",
    "abs_val_accuracy_neg": "Accuracy Neg-class (Val)",
    "abs_val_accuracy_pos": "Accuracy Pos-class (Val)",
    "abs_nll": "Negative Log-Likelihood",
    "abs_val_nll": "NLL (Val)",
    "abs_val_perplexity": "Perplexity (Val)",
    "abs_log_val_perplexity": "Log Perplexity (Val)",
    "abs_val_edge_case_mse": "Edge-Case MSE (Val)",
    "abs_val_edge_case_mse_std": "Edge-Case MSE Std (Val)",
    "abs_cache_nll_divergence": "Cache NLL Divergence",
    "abs_val_cache_nll_divergence": "Cache NLL Divergence (Val)",
    "abs_val_pos_inv": "Position Inversion Rate (Val)",
    # Diagnostic -- Logit
    "abs_logit_entropy": "Logit Entropy",
    "abs_logit_kl_uniform": "Logit KL from Uniform",
    "abs_logit_conf": "Logit Confidence",
    "abs_logit_margin_mean": "Logit Margin (Mean)",
    "abs_logit_margin_min": "Logit Margin (Min)",
    "abs_logit_margin_p25": "Logit Margin (P25)",
    "abs_logit_margin_p50": "Logit Margin (P50)",
    "abs_logit_margin_p75": "Logit Margin (P75)",
    "abs_logit_margin_var": "Logit Margin (Var)",
    "abs_val_logit_margin_mean": "Logit Margin Mean (Val)",
    "abs_val_logit_margin_min": "Logit Margin Min (Val)",
    "abs_val_logit_margin_p25": "Logit Margin P25 (Val)",
    "abs_val_logit_margin_p50": "Logit Margin P50 (Val)",
    "abs_val_logit_margin_p75": "Logit Margin P75 (Val)",
    "abs_val_logit_margin_var": "Logit Margin Var (Val)",
    "abs_val_margin_gap": "Margin Gap (Val)",
    "abs_val_margin_neg": "Margin Neg-class (Val)",
    "abs_val_margin_pos": "Margin Pos-class (Val)",
    # Attention
    "abs_attn_entropy": "Attention Entropy",
    "abs_head_similarity_mean": "Head Similarity (Mean)",
    "abs_head_similarity_max": "Head Similarity (Max)",
    "abs_presoftmax_mean": "Pre-Softmax Mean",
    "abs_presoftmax_var": "Pre-Softmax Variance",
    "abs_presoftmax_skew": "Pre-Softmax Skewness",
    "abs_presoftmax_kurt": "Pre-Softmax Kurtosis",
    "abs_mass_leak": "Attention Mass Leak",
    "abs_mass_pad": "Attention Mass on Padding",
    "abs_cross_example_attn": "Cross-Example Attention",
    "abs_agg_attn_entropy": "Aggregated Attn Entropy",
    "abs_agg_attn_entropy_mean": "Agg Attn Entropy (Mean)",
    "abs_agg_attn_entropy_std": "Agg Attn Entropy (Std)",
    "abs_agg_attn_score_var": "Agg Attn Score Variance",
    "abs_agg_attn_score_skew": "Agg Attn Score Skewness",
    "abs_agg_attn_sparsity": "Attention Sparsity",
    "abs_agg_attn_weight_magnitude": "Attn Weight Magnitude",
    "abs_agg_attn_max_mean": "Attn Max (Mean)",
    "abs_agg_attn_max_std": "Attn Max (Std)",
    "abs_agg_attn_mass_leak": "Agg Attn Mass Leak",
    "abs_agg_attn_mass_leak_max": "Agg Attn Mass Leak (Max)",
    "abs_agg_attn_mass_pad_max": "Agg Attn Pad Mass (Max)",
    "abs_agg_attn_mass_pad_mean": "Agg Attn Pad Mass (Mean)",
    "abs_agg_attn_cross_example_leak": "Agg Cross-Example Leak",
    "abs_agg_head_similarity_mean": "Agg Head Similarity (Mean)",
    "abs_agg_head_similarity_max": "Agg Head Similarity (Max)",
    "abs_agg_head_similarity_std": "Agg Head Similarity (Std)",
    "abs_agg_presoftmax_mean": "Agg Pre-Softmax Mean",
    "abs_agg_presoftmax_var": "Agg Pre-Softmax Variance",
    "abs_agg_presoftmax_skew": "Agg Pre-Softmax Skewness",
    "abs_agg_presoftmax_kurt": "Agg Pre-Softmax Kurtosis",
    "abs_agg_pos_recv_mean": "Agg Position Recv (Mean)",
    "abs_agg_pos_recv_early": "Agg Position Recv (Early)",
    "abs_agg_pos_recv_mid": "Agg Position Recv (Mid)",
    "abs_agg_pos_recv_var": "Agg Position Recv (Var)",
    "abs_agg_pos_recv_skew": "Agg Position Recv (Skew)",
    "abs_agg_pos_recv_mid_over_early": "Agg Position Recv Mid/Early",
    "abs_update_ratio_agg_attn": "Update Ratio (Attention)",
    "abs_update_active_agg_attn": "Active Update Ratio (Attn)",
    # FFN
    "abs_ffn_delta_mean": "FFN Delta (Mean)",
    "abs_ffn_var_ratio": "FFN Variance Ratio",
    "abs_ffn_var_ratio_mean": "FFN Variance Ratio (Mean)",
    "abs_ffn_out_skew": "FFN Output Skewness",
    "abs_ffn_out_skew_mean": "FFN Output Skewness (Mean)",
    "abs_ffn_delta": "FFN Delta",
    "abs_ffn_active_dim_frac_mean": "FFN Active Dimension Frac",
    "abs_h1_delta_norm_mean": "Hidden Delta Norm (Mean)",
    "abs_activation_mean": "Activation Mean",
    "abs_activation_std": "Activation Std",
    # LayerNorm
    "abs_ln_mean_abs_mean": "LayerNorm Mean Abs (Mean)",
    "abs_ln_std_mean": "LayerNorm Std (Mean)",
    "abs_ln_mean_abs": "LayerNorm Mean Abs",
    "abs_ln_std": "LayerNorm Std",
    # Residual
    "abs_residual_cos_mean": "Residual Cosine Sim (Mean)",
    "abs_residual_cos": "Residual Cosine Sim",
    # Embedding
    "abs_emb_norm_mean": "Embedding Norm (Mean)",
    "abs_emb_norm_std": "Embedding Norm (Std)",
    # Positional
    "abs_pos_acc_delta": "Positional Accuracy Delta",
    "abs_pos_acc_startpos": "Positional Accuracy Start",
    "abs_pos_loss_startpos": "Positional Loss Start",
    "abs_pos_margin_delta": "Positional Margin Delta",
    "abs_pos_margin_startpos": "Positional Margin Start",
    "abs_val_pos_acc_delta": "Positional Accuracy Delta (Val)",
    "abs_val_pos_margin_delta": "Positional Margin Delta (Val)",
    # Representation
    "abs_repr": "Representation Drift",
    "abs_repr_cos": "Representation Cosine",
    "abs_repr_drift": "Representation Drift",
    # Gradient
    "abs_grad_abs_max": "Gradient Abs Max",
    "abs_grad_zero_ratio": "Gradient Zero Ratio",
    "abs_grad_norm_total": "Gradient Norm (Total)",
    "abs_grad_norm_total_gns": "Gradient Noise Scale (Total)",
    "abs_grad_norm_total_window_var": "Gradient Norm Var (Total)",
    "abs_grad_norm_emb": "Gradient Norm (Embedding)",
    "abs_grad_norm_emb_gns": "Gradient Noise Scale (Emb)",
    "abs_grad_norm_emb_window_var": "Gradient Norm Var (Emb)",
    "abs_grad_norm_agg_attn": "Gradient Norm (Attention)",
    "abs_grad_norm_agg_attn_gns": "Gradient Noise Scale (Attn)",
    "abs_grad_norm_agg_attn_window_var": "Gradient Norm Var (Attn)",
    "abs_grad_norm_agg_ffn": "Gradient Norm (FFN)",
    "abs_grad_norm_agg_ffn_gns": "Gradient Noise Scale (FFN)",
    "abs_grad_norm_agg_ffn_window_var": "Gradient Norm Var (FFN)",
    "abs_grad_norm_agg_layernorm": "Gradient Norm (LayerNorm)",
    "abs_grad_norm_agg_layernorm_gns": "Gradient Noise Scale (LN)",
    "abs_grad_norm_agg_layernorm_window_var": "Gradient Norm Var (LN)",
    "abs_grad_norm_agg_qkv": "Gradient Norm (QKV)",
    "abs_grad_norm_agg_qkv_gns": "Gradient Noise Scale (QKV)",
    "abs_grad_norm_agg_qkv_window_var": "Gradient Norm Var (QKV)",
    # Update Ratios
    "abs_update_ratio_total": "Update Ratio (Total)",
    "abs_update_ratio_emb": "Update Ratio (Embedding)",
    "abs_update_ratio_classifier": "Update Ratio (Classifier)",
    "abs_update_ratio_agg_ffn": "Update Ratio (FFN)",
    "abs_update_ratio_agg_layernorm": "Update Ratio (LayerNorm)",
    "abs_update_ratio_agg_qkv": "Update Ratio (QKV)",
    "abs_update_ratio": "Update Ratio (Per-Layer)",
    "abs_update_active_agg_ffn": "Active Update Ratio (FFN)",
    "abs_update_active_agg_layernorm": "Active Update Ratio (LN)",
    "abs_update_active_agg_qkv": "Active Update Ratio (QKV)",
    "abs_update_active_emb": "Active Update Ratio (Emb)",
    # Weight Stats
    "abs_weight_mean": "Weight Mean",
    "abs_weight_std": "Weight Std",
    # Runtime / Resource
    "abs_peak_mem_alloc_mb": "Peak Memory Alloc (MB)",
    "abs_peak_mem_reserved_mb": "Peak Memory Reserved (MB)",
    "abs_step_time": "Step Time",
    "abs_step_time_mean": "Step Time (Mean)",
    # KV Cache
    "abs_cache_hidden_similarity": "Cache Hidden Similarity",
    "abs_val_cache_hidden_similarity": "Cache Hidden Similarity (Val)",
}

def display_name(feature: str, core_set: Optional[set] = None) -> str:
    """Convert a raw feature name to a human-readable thesis display name."""
    # Direct lookup first
    if feature in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[feature]
    # Resolve to core, then look up
    if core_set:
        core = resolve_to_core(feature, core_set)
        if core in _DISPLAY_NAMES:
            return _DISPLAY_NAMES[core]
    # Fallback: clean up the raw name
    name = feature.replace("abs_", "").replace("_", " ").title()
    return name
