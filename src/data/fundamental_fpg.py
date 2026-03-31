"""Fundamental Fault Propagation Graph (FPG).

Deterministic fault propagation rules derived from the mathematical structure
of transformer architectures. These rules hold for ANY transformer model
regardless of specific architecture variant, dataset, or task.

Mathematical foundations:
  - Forward pass: h = LN(x + MHA(x)), y = LN(h + FFN(h))
  - Attention:    Attn(Q,K,V) = softmax(QK^T/sqrt(d_k) + M) V
  - Projections:  Q = xW^Q, K = xW^K, V = xW^V
  - FFN:          FFN(x) = W_2 sigma(W_1 x + b_1) + b_2

From these equations, we derive 8 deterministic propagation rules that define
how a fault in any component affects all other components. The rules capture:
  - Sequential propagation (forward data flow via chain rule)
  - Simultaneous propagation (multiple outputs from one component)
  - Residual bypass (skip connections guarantee unit-gain propagation)
  - Cross-layer propagation (transitive through residual stream)
  - Gradient-mediated propagation (backward pass during training)
  - Resource coupling (shared GPU kernels / numerical precision)
  - Multi-component spanning (faults that ARE in multiple components)
  - Temporal coupling (KV cache across generation steps)
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PropagationType(str, Enum):
    """How the fault effect travels between components."""
    FORWARD_SEQUENTIAL = "forward_sequential"
    FORWARD_SIMULTANEOUS = "forward_simultaneous"
    RESIDUAL_BYPASS = "residual_bypass"
    CROSS_LAYER = "cross_layer"
    BACKWARD_GRADIENT = "backward_gradient"
    RESOURCE_SHARED = "resource_shared"
    MULTI_COMPONENT = "multi_component"
    TEMPORAL_COUPLING = "temporal_coupling"


class Bottleneck(str, Enum):
    """Nonlinear operations that bound propagation magnitude."""
    NONE = "none"
    SOFTMAX = "softmax"       # bounds to [0,1], sum-to-1
    LAYERNORM = "layernorm"   # rescales to zero-mean unit-variance
    ACTIVATION = "activation"  # ReLU/GELU clips or reshapes


@dataclass(frozen=True)
class Component:
    """A transformer component (node in the FPG)."""
    name: str
    math: str  # mathematical expression
    scope: str  # "encoder", "decoder", or "both"


@dataclass(frozen=True)
class PropagationRule:
    """A deterministic fault propagation rule (edge in the FPG).

    Derived from the mathematical structure of the transformer.
    """
    source: str
    target: str
    prop_type: PropagationType
    rule_id: int
    bottleneck: Bottleneck = Bottleneck.NONE
    justification: str = ""
    scope: str = "both"  # "encoder", "decoder", or "both"


# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

COMPONENTS = [
    Component("embedding",        "x = E[token] + P[pos]",            "both"),
    Component("positional",       "P[pos] (sinusoidal/learned/RoPE)",  "both"),
    Component("qkv_projection",   "Q=xW^Q, K=xW^K, V=xW^V",          "both"),
    Component("score_computation","S = QK^T / sqrt(d_k)",              "both"),
    Component("attention_masking","S' = S + M",                        "both"),
    Component("attention_weights","A = softmax(S')",                   "both"),
    Component("attention_output", "O = A V",                           "both"),
    Component("residual_post_attn","h = x + O",                       "both"),
    Component("layernorm_post_attn","h' = LN(h)",                     "both"),
    Component("ffn",              "FFN(h') = W2 sigma(W1 h' + b1)+b2","both"),
    Component("residual_post_ffn","y = h' + FFN(h')",                 "both"),
    Component("layernorm_post_ffn","y' = LN(y)",                      "both"),
    Component("output_head",      "logits = y' W_out + b_out",        "both"),
    Component("kv_cache",         "K_cache[t], V_cache[t]",           "decoder"),
    Component("kernel_runtime",   "GPU kernel execution context",     "both"),
]


# ============================================================================
# THE 8 DETERMINISTIC PROPAGATION RULES
# ============================================================================

RULES = [
    # -----------------------------------------------------------------------
    # RULE 1: Forward Sequential Propagation (Chain Rule)
    # If B = f(A), then dB = (df/dA) dA. Perturbation propagates forward.
    # -----------------------------------------------------------------------
    PropagationRule("embedding", "qkv_projection",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="Q=xW^Q: perturbation dx -> dQ=dx W^Q (linear)"),

    PropagationRule("positional", "qkv_projection",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="Position info modifies x before projection: x=E+P"),

    PropagationRule("qkv_projection", "score_computation",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="S=QK^T/sqrt(d_k): dS = dQ K^T + Q dK^T (bilinear in Q,K)"),

    PropagationRule("score_computation", "attention_masking",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="S'=S+M: scores receive mask additively"),

    PropagationRule("attention_masking", "attention_weights",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.SOFTMAX,
        justification="A=softmax(S'): softmax bounds weights to [0,1] sum-to-1, "
                       "compressing large perturbations (Jacobian has bounded spectral norm)"),

    PropagationRule("attention_weights", "attention_output",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="O=AV: dO = dA V (perturbation in weights changes output)"),

    PropagationRule("attention_output", "residual_post_attn",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="h=x+O: attention output enters residual path"),

    PropagationRule("residual_post_attn", "layernorm_post_attn",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.LAYERNORM,
        justification="LN rescales to zero-mean unit-var, partially absorbing "
                       "magnitude perturbations but preserving directional shifts"),

    PropagationRule("layernorm_post_attn", "ffn",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.ACTIVATION,
        justification="FFN(h')=W2 sigma(W1 h'+b1)+b2: nonlinear activation "
                       "can amplify or suppress perturbations depending on operating point"),

    PropagationRule("ffn", "residual_post_ffn",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="y=h'+FFN(h'): FFN output enters second residual path"),

    PropagationRule("residual_post_ffn", "layernorm_post_ffn",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.LAYERNORM,
        justification="Second LayerNorm provides normalization bottleneck"),

    PropagationRule("layernorm_post_ffn", "output_head",
        PropagationType.FORWARD_SEQUENTIAL, rule_id=1,
        justification="logits = y' W_out: final representation projected to output"),

    # -----------------------------------------------------------------------
    # RULE 2: Simultaneous Propagation (One source, multiple targets)
    # A single component feeds multiple downstream paths at once.
    # -----------------------------------------------------------------------
    PropagationRule("qkv_projection", "attention_output",
        PropagationType.FORWARD_SIMULTANEOUS, rule_id=2,
        justification="V enters attention output O=AV simultaneously with "
                       "Q,K entering score computation. A QKV fault perturbs "
                       "scores (via Q,K) AND values (via V) at the same time"),

    PropagationRule("qkv_projection", "kv_cache",
        PropagationType.FORWARD_SIMULTANEOUS, rule_id=2, scope="decoder",
        justification="K,V are simultaneously used in current attention "
                       "AND stored in cache for future steps"),

    # -----------------------------------------------------------------------
    # RULE 3: Residual Bypass (Skip connections guarantee unit-gain propagation)
    # In y = x + f(x), perturbation dx appears in y with gain >= 1.
    # The skip connection ensures faults CANNOT be absorbed by sublayers.
    # -----------------------------------------------------------------------
    PropagationRule("embedding", "residual_post_attn",
        PropagationType.RESIDUAL_BYPASS, rule_id=3,
        justification="h=x+MHA(x): input x bypasses entire attention mechanism. "
                       "Embedding perturbation dx appears directly in h with unit gain, "
                       "regardless of what attention does. This is mathematically guaranteed "
                       "by the additive skip connection."),

    PropagationRule("layernorm_post_attn", "residual_post_ffn",
        PropagationType.RESIDUAL_BYPASS, rule_id=3,
        justification="y=h'+FFN(h'): pre-FFN representation bypasses FFN. "
                       "A perturbation in the residual stream propagates through "
                       "the FFN block with gain >= 1."),

    # -----------------------------------------------------------------------
    # RULE 4: Cross-Layer Propagation (Transitive via residual stream)
    # Output of block L feeds into block L+1. The residual stream acts as
    # a highway: perturbations propagate with at least unit gain per layer.
    # After N layers, the perturbation is present in all intermediate
    # representations.
    # -----------------------------------------------------------------------
    PropagationRule("layernorm_post_ffn", "qkv_projection",
        PropagationType.CROSS_LAYER, rule_id=4,
        justification="Block L output becomes block L+1 input. Through the "
                       "residual stream, a fault in layer L propagates to ALL "
                       "subsequent layers L+1,...,L+N. The residual highway "
                       "guarantees at least unit-gain propagation per layer."),

    # -----------------------------------------------------------------------
    # RULE 5: Gradient-Mediated Propagation (Backward pass)
    # Any fault that changes the loss dL also changes the gradient
    # dL/d(theta_i) for EVERY parameter theta_i. During training, this
    # means every fault affects weight updates of every component.
    # -----------------------------------------------------------------------
    PropagationRule("output_head", "qkv_projection",
        PropagationType.BACKWARD_GRADIENT, rule_id=5,
        justification="dL/dW^Q exists by chain rule. Any fault that changes "
                       "logits changes dL, which changes all parameter gradients."),

    PropagationRule("output_head", "ffn",
        PropagationType.BACKWARD_GRADIENT, rule_id=5,
        justification="dL/dW1, dL/dW2 exist. Loss perturbation propagates "
                       "backward to all FFN parameters."),

    PropagationRule("output_head", "layernorm_post_attn",
        PropagationType.BACKWARD_GRADIENT, rule_id=5,
        justification="dL/d(gamma), dL/d(beta) exist. Loss perturbation "
                       "affects LayerNorm parameter updates."),

    PropagationRule("output_head", "embedding",
        PropagationType.BACKWARD_GRADIENT, rule_id=5,
        justification="dL/dE exists. Loss perturbation affects embedding "
                       "updates during fine-tuning."),

    # -----------------------------------------------------------------------
    # RULE 6: Resource Coupling (Shared GPU kernel / numerical precision)
    # Kernel configuration (precision, scheduling) affects ALL operations
    # that execute through that kernel. This is not data flow — it's a
    # shared execution environment.
    # -----------------------------------------------------------------------
    PropagationRule("kernel_runtime", "score_computation",
        PropagationType.RESOURCE_SHARED, rule_id=6,
        justification="Attention score matmul uses GPU kernel. Kernel fallback "
                       "changes float16/32 precision for QK^T computation."),

    PropagationRule("kernel_runtime", "attention_weights",
        PropagationType.RESOURCE_SHARED, rule_id=6,
        justification="Softmax uses GPU kernel. Precision changes affect "
                       "numerical stability of exp() in softmax."),

    PropagationRule("kernel_runtime", "ffn",
        PropagationType.RESOURCE_SHARED, rule_id=6,
        justification="FFN matmuls use GPU kernels. Precision changes affect "
                       "W1x and W2x computations."),

    PropagationRule("kernel_runtime", "qkv_projection",
        PropagationType.RESOURCE_SHARED, rule_id=6,
        justification="QKV projection matmuls use GPU kernels."),

    # -----------------------------------------------------------------------
    # RULE 7: Multi-Component Spanning
    # Some fault types ARE inherently multi-component. A variant fault
    # (e.g., single-head instead of multi-head) changes the attention
    # mechanism's structure itself, affecting computation, capacity,
    # parameter count, and kernel dispatch simultaneously.
    # -----------------------------------------------------------------------
    PropagationRule("attention_weights", "kernel_runtime",
        PropagationType.MULTI_COMPONENT, rule_id=7,
        justification="A variant fault (single-head vs multi-head) changes "
                       "the number of attention heads, which simultaneously "
                       "changes: (a) attention computation pattern, "
                       "(b) parameter count/shape, (c) GPU kernel dispatch. "
                       "These are not sequential — they are ONE fault manifesting "
                       "across multiple components."),

    PropagationRule("attention_weights", "qkv_projection",
        PropagationType.MULTI_COMPONENT, rule_id=7,
        justification="Variant fault changes head structure, which changes "
                       "how Q,K,V projections are partitioned across heads."),

    # -----------------------------------------------------------------------
    # RULE 8: Temporal Coupling (KV Cache, decoder only)
    # Cache stores K,V across generation steps. A cache fault affects
    # current AND future tokens, potentially across different sequences.
    # This is unique to autoregressive decoding.
    # -----------------------------------------------------------------------
    PropagationRule("kv_cache", "score_computation",
        PropagationType.TEMPORAL_COUPLING, rule_id=8, scope="decoder",
        justification="Cached K used in score: S[t]=Q[t] K_cache[:t]^T. "
                       "A stale or corrupted cache K affects scores for ALL "
                       "future tokens, not just the current one."),

    PropagationRule("kv_cache", "attention_output",
        PropagationType.TEMPORAL_COUPLING, rule_id=8, scope="decoder",
        justification="Cached V used in output: O[t]=A[t] V_cache[:t]. "
                       "Cache corruption affects the values used in weighted sum."),
]


def build_fundamental_fpg(arch: str = "encoder") -> tuple[list[str], np.ndarray, dict]:
    """Build the fundamental FPG for a given architecture.

    Returns:
        component_names: ordered list of component names
        adjacency: weighted adjacency matrix where A[i,j] > 0 means
                   a fault in component i DETERMINISTICALLY affects component j.
                   Weight encodes propagation type (for visualization).
        metadata: dict with rules, bottlenecks, and justifications
    """
    # Filter components and rules by architecture
    if arch == "encoder":
        comps = [c for c in COMPONENTS if c.scope in ("both", "encoder")]
        rules = [r for r in RULES if r.scope in ("both", "encoder")]
    elif arch == "decoder":
        comps = [c for c in COMPONENTS if c.scope in ("both", "decoder")]
        rules = [r for r in RULES if r.scope in ("both", "decoder")]
    else:
        comps = COMPONENTS
        rules = RULES

    names = [c.name for c in comps]
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    # Propagation type -> weight (for visualization distinctness)
    type_weights = {
        PropagationType.FORWARD_SEQUENTIAL: 1.0,
        PropagationType.FORWARD_SIMULTANEOUS: 0.9,
        PropagationType.RESIDUAL_BYPASS: 0.8,
        PropagationType.CROSS_LAYER: 0.7,
        PropagationType.BACKWARD_GRADIENT: 0.5,
        PropagationType.RESOURCE_SHARED: 0.6,
        PropagationType.MULTI_COMPONENT: 0.85,
        PropagationType.TEMPORAL_COUPLING: 0.75,
    }

    adj = np.zeros((n, n), dtype=np.float32)
    edge_types = {}  # (i,j) -> list of (type, bottleneck, justification)

    for rule in rules:
        if rule.source in name_to_idx and rule.target in name_to_idx:
            i = name_to_idx[rule.source]
            j = name_to_idx[rule.target]
            w = type_weights[rule.prop_type]
            adj[i, j] = max(adj[i, j], w)

            key = (i, j)
            if key not in edge_types:
                edge_types[key] = []
            edge_types[key].append({
                "type": rule.prop_type.value,
                "rule_id": rule.rule_id,
                "bottleneck": rule.bottleneck.value,
                "justification": rule.justification,
            })

    metadata = {
        "architecture": arch,
        "components": [
            {"name": c.name, "math": c.math, "scope": c.scope}
            for c in comps
        ],
        "rules": [
            {
                "source": r.source, "target": r.target,
                "type": r.prop_type.value, "rule_id": r.rule_id,
                "bottleneck": r.bottleneck.value,
                "justification": r.justification,
            }
            for r in rules
        ],
        "edge_details": {
            f"{names[i]}->{names[j]}": details
            for (i, j), details in edge_types.items()
        },
        "rule_summary": {
            "Rule 1 (Forward Sequential)": "B=f(A) => dB=(df/dA)dA by chain rule",
            "Rule 2 (Simultaneous)": "One source feeds multiple targets at once",
            "Rule 3 (Residual Bypass)": "y=x+f(x) => perturbation dx has gain >= 1",
            "Rule 4 (Cross-Layer)": "Block L output -> Block L+1 via residual highway",
            "Rule 5 (Gradient-Mediated)": "dL/d(theta_i) exists for all parameters",
            "Rule 6 (Resource Shared)": "GPU kernel config affects all compute ops",
            "Rule 7 (Multi-Component)": "Fault IS in multiple components (variant)",
            "Rule 8 (Temporal Coupling)": "KV cache spans generation steps (decoder)",
        },
        "bottleneck_summary": {
            "softmax": "Bounds outputs to [0,1] sum-to-1, compresses large perturbations",
            "layernorm": "Rescales to zero-mean unit-var, absorbs magnitude shifts",
            "activation": "ReLU/GELU can amplify or suppress depending on operating point",
        },
        "propagation_types": {
            "forward_sequential": "A -> B because B = f(A)",
            "forward_simultaneous": "A -> {B,C} because B=f1(A) and C=f2(A) at same time",
            "residual_bypass": "A -> C directly via skip connection, bypassing B",
            "cross_layer": "Layer L -> Layer L+1 through residual stream",
            "backward_gradient": "All components connected through gradient flow",
            "resource_shared": "Shared GPU kernel / numerical precision",
            "multi_component": "Fault simultaneously affects multiple components",
            "temporal_coupling": "KV cache links across generation timesteps",
        },
    }

    return names, adj, metadata


def fundamental_to_feature_group_adjacency(
    arch: str = "encoder",
) -> tuple[list[str], np.ndarray, dict]:
    """Collapse the component-level FPG to feature-group-level adjacency.

    This is the adjacency matrix used by the model. Multiple components
    map to the same feature group (e.g., score_computation and attention_masking
    both map to the "attention" group at the feature level).
    """
    comp_names, comp_adj, metadata = build_fundamental_fpg(arch)
    comps = metadata["components"]

    # Component -> feature group mapping
    _COMP_TO_GROUP = {
        "embedding": "embedding",
        "positional": "positional",
        "qkv_projection": "qkv",
        "score_computation": "score",
        "attention_masking": "attention",
        "attention_weights": "attention",
        "attention_output": "attention",
        "residual_post_attn": "residual",
        "layernorm_post_attn": "layernorm",
        "ffn": "ffn",
        "residual_post_ffn": "residual",
        "layernorm_post_ffn": "layernorm",
        "output_head": "task_metrics",
        "kv_cache": "cache_diagnostics",
        "kernel_runtime": "kernel_timing",
    }

    # Unique groups present in this architecture
    present_groups = []
    seen = set()
    for c in comp_names:
        g = _COMP_TO_GROUP.get(c, "training_dynamics")
        if g not in seen:
            present_groups.append(g)
            seen.add(g)
    # Always include training_dynamics (gradient flow affects it)
    if "training_dynamics" not in seen:
        present_groups.append("training_dynamics")

    group_to_idx = {g: i for i, g in enumerate(present_groups)}
    n_groups = len(present_groups)
    group_adj = np.zeros((n_groups, n_groups), dtype=np.float32)

    # Collapse: if any comp in group_i connects to any comp in group_j
    for i, ci in enumerate(comp_names):
        for j, cj in enumerate(comp_names):
            if comp_adj[i, j] > 0:
                gi = _COMP_TO_GROUP.get(ci, "training_dynamics")
                gj = _COMP_TO_GROUP.get(cj, "training_dynamics")
                if gi in group_to_idx and gj in group_to_idx:
                    ii, jj = group_to_idx[gi], group_to_idx[gj]
                    group_adj[ii, jj] = max(group_adj[ii, jj], comp_adj[i, j])

    # Add self-loops
    np.fill_diagonal(group_adj, 1.0)

    # Add gradient flow: everything -> training_dynamics
    if "training_dynamics" in group_to_idx:
        td_idx = group_to_idx["training_dynamics"]
        for i in range(n_groups):
            group_adj[i, td_idx] = max(group_adj[i, td_idx], 0.5)

    return present_groups, group_adj, metadata
