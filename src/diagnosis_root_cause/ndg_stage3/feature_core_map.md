# FrankenFormer Feature Core Map

146 core features -> 672 columns (543 encoder, 215 decoder, 86 shared)

## Tier Legend
- P = Performance (task metrics: accuracy, loss, F1, ECE, perplexity)
- D = Diagnostic (model internals: logit margins, attention, FFN, LayerNorm, residuals)
- I = Internal (infrastructure: update ratios, gradients, weights, memory, KV cache)
- S = Structural (arch_enc, layer_idx, severity_scalar)
- Suffix * = extended tier for arch-specific features (not in the 86 shared)

## Time-Window Stat Variants
Most core features expand into 5 columns via training-phase statistics:
- `early_mean`: mean over early training phase
- `early_slope`: linear trend during early phase
- `mid_mean`: mean over middle training phase
- `mid_slope`: linear trend during middle phase
- `final`: end-of-training snapshot

## Activation (2 core, 10 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_activation_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_activation_std` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Attention (42 core, 85 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_agg_attn_cross_example_leak` | D* | E-only | 1 | final |
| `abs_agg_attn_entropy` | D* | E-only | 1 | final |
| `abs_agg_attn_entropy_mean` | D* | E-only | 2 | mid_mean, mid_slope |
| `abs_agg_attn_entropy_std` | D* | E-only | 1 | final |
| `abs_agg_attn_mass_leak` | D* | E-only | 1 | final |
| `abs_agg_attn_mass_leak_max` | D* | E-only | 1 | final |
| `abs_agg_attn_mass_pad_max` | D* | E-only | 1 | final |
| `abs_agg_attn_mass_pad_mean` | D* | E-only | 1 | final |
| `abs_agg_attn_max_mean` | D* | E-only | 1 | final |
| `abs_agg_attn_max_std` | D* | E-only | 1 | final |
| `abs_agg_attn_score_skew` | D* | E-only | 1 | final |
| `abs_agg_attn_score_var` | D* | E-only | 1 | final |
| `abs_agg_attn_sparsity` | D* | E-only | 1 | final |
| `abs_agg_attn_weight_magnitude` | D* | E-only | 1 | final |
| `abs_agg_head_similarity_max` | D* | E-only | 1 | final |
| `abs_agg_head_similarity_mean` | D* | E-only | 1 | final |
| `abs_agg_head_similarity_std` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_early` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_mean` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_mid` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_mid_over_early` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_skew` | D* | E-only | 1 | final |
| `abs_agg_pos_recv_var` | D* | E-only | 1 | final |
| `abs_agg_presoftmax_kurt` | D* | E-only | 1 | final |
| `abs_agg_presoftmax_mean` | D* | E-only | 1 | final |
| `abs_agg_presoftmax_skew` | D* | E-only | 1 | final |
| `abs_agg_presoftmax_var` | D* | E-only | 1 | final |
| `abs_attn_entropy` | D* | E-only | 3 | early_mean, early_slope, final |
| `abs_cross_example_attn` | D* | E-only | 1 | final |
| `abs_grad_norm_agg_attn` | D* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_attn_gns` | D* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_attn_window_var` | D* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_head_similarity_max` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_head_similarity_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_mass_leak` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_mass_pad` | D* | E-only | 1 | final |
| `abs_presoftmax_kurt` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_presoftmax_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_presoftmax_skew` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_presoftmax_var` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_update_active_agg_attn` | D* | D-only | 1 | final |
| `abs_update_ratio_agg_attn` | I | shared | 1 | final |

## Diagnostic (logit) (18 core, 90 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_logit_conf` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_entropy` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_kl_uniform` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_mean` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_min` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_p25` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_p50` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_p75` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_logit_margin_var` | D | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_min` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_p25` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_p50` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_p75` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_logit_margin_var` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_margin_gap` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_margin_neg` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_margin_pos` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Embedding (2 core, 8 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_emb_norm_mean` | I* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_emb_norm_std` | I* | E-only | 3 | early_slope, final, mid_slope |

## FFN (8 core, 57 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_ffn_active_dim_frac_mean` | D* | E-only | 1 | final |
| `abs_ffn_delta` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_ffn_delta_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_ffn_out_skew` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_ffn_out_skew_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_ffn_var_ratio` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_ffn_var_ratio_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_h1_delta_norm_mean` | I* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Gradient (17 core, 84 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_grad_abs_max` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_ffn` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_ffn_gns` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_ffn_window_var` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_layernorm` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_layernorm_gns` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_layernorm_window_var` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_qkv` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_qkv_gns` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_agg_qkv_window_var` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_emb` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_emb_gns` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_emb_window_var` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_total` | I* | D-only | 4 | early_mean, early_slope, mid_mean, mid_slope |
| `abs_grad_norm_total_gns` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_norm_total_window_var` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_grad_zero_ratio` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## KV Cache (2 core, 6 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_cache_hidden_similarity` | I* | D-only | 1 | (single) |
| `abs_val_cache_hidden_similarity` | I* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## LayerNorm (4 core, 34 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_ln_mean_abs` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_ln_mean_abs_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_ln_std` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_ln_std_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Performance (25 core, 119 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_accuracy` | P | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_cache_nll_divergence` | P* | D-only | 1 | (single) |
| `abs_ece` | P | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_f1_score` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_log_val_perplexity` | P* | D-only | 3 | early_mean, final, mid_mean |
| `abs_loss` | P | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_nll` | P* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_precision` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_recall` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_accuracy` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_accuracy_gap` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_accuracy_neg` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_accuracy_pos` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_cache_nll_divergence` | P* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_ece` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_edge_case_mse` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_edge_case_mse_std` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_f1_score` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_loss` | P | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_nll` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_perplexity` | P* | D-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_pos_inv` | P | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_precision` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_primary_metric` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_recall` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Positional (7 core, 35 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_pos_acc_delta` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_pos_acc_startpos` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_pos_loss_startpos` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_pos_margin_delta` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_pos_margin_startpos` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_pos_acc_delta` | P* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_val_pos_margin_delta` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Representation (1 core, 54 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_repr` | D* | E-only | 54 | 12L x 8suf **[PER-LAYER]** |

## Residual (2 core, 17 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_residual_cos` | D* | E-only | 12 | 12L x 1suf **[PER-LAYER]** |
| `abs_residual_cos_mean` | D* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Resource (3 core, 3 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_peak_mem_alloc_mb` | I | shared | 1 | (single) |
| `abs_peak_mem_reserved_mb` | I | shared | 1 | (single) |
| `abs_step_time` | I | shared | 1 | mean_finalwin |

## Update Ratio (11 core, 64 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_update_active_agg_ffn` | I* | D-only | 1 | final |
| `abs_update_active_agg_layernorm` | I* | D-only | 1 | final |
| `abs_update_active_agg_qkv` | I* | D-only | 1 | final |
| `abs_update_active_emb` | I* | D-only | 1 | final |
| `abs_update_ratio` | I* | E-only | 42 | 12L x 4suf **[PER-LAYER]** |
| `abs_update_ratio_agg_ffn` | I | shared | 1 | final |
| `abs_update_ratio_agg_layernorm` | I | shared | 1 | final |
| `abs_update_ratio_agg_qkv` | I | shared | 1 | final |
| `abs_update_ratio_classifier` | I* | E-only | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_update_ratio_emb` | I | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |
| `abs_update_ratio_total` | I | shared | 5 | early_mean, early_slope, final, mid_mean, mid_slope |

## Weight Stats (2 core, 6 cols)

| Core Feature | Tier | Scope | # | Stat Variants |
|---|---|---|---:|---|
| `abs_weight_mean` | I* | E-only | 3 | early_slope, final, mid_slope |
| `abs_weight_std` | I* | E-only | 3 | early_slope, final, mid_slope |

## Summary

### By Tier
| Tier | Core Features | Columns |
|---|---:|---:|
| P | 29 | 139 |
| D | 78 | 356 |
| I | 39 | 177 |
| **Total** | **146** | **672** |

### By Scope
| Scope | Core Features | Columns |
|---|---:|---:|
| Shared (E+D) | 23 | 87 |
| Encoder-only | 91 | 456 |
| Decoder-only | 32 | 129 |
| **Total** | **146** | **672** |

### Per-Layer Features
8 core features expand into 168 per-layer columns (all encoder-only).

### Notable Redundancy
- logit_margin quantiles: 6 cores (mean/min/p25/p50/p75/var) x 5 stats = 30 shared + 30 encoder val_ = 60 cols
- Representation drift (abs_repr): 1 core -> 54 cols (12 layers x up to 8 suffixes)
- Per-layer update_ratio: 1 core -> 42 cols (12 layers x 4 component types)
- Gradient norms (decoder): 17 cores -> 84 cols (6 components x {raw, GNS, window_var} x 5 stats)
- Per-layer FFN stats: 3 cores -> 36 cols (12 layers each for delta, out_skew, var_ratio)
