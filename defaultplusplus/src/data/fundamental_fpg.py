"""Fault Propagation Graph (FPG) construction.

Defines the component-level directed graph G = (V, E) over transformer
components and collapses it to the group-level adjacency Â consumed by
the diagnostic model.

A fault in a component can reach another component through one of seven
dependency mechanisms; each edge in the component-level FPG carries
exactly one mechanism class.

  M1  Forward sequential propagation. Chain rule along the data flow:
      if B = f(A) then a perturbation δA produces δB = (∂f/∂A) δA.
  M2  Simultaneous propagation. A single component feeds multiple
      downstream components in the same forward pass (e.g. QKV → score
      via Q,K and QKV → attention output via V).
  M3  Residual bypass. In y = x + f(x), perturbations in x propagate
      through the skip path with unit gain regardless of f.
  M4  Cross-layer propagation. The residual stream provides repeated
      identity paths across stacked layers; a fault in layer ℓ can
      reach all later layers.
  M5  Backward gradient propagation. Any fault that changes the loss L
      changes ∂L/∂θ_i for every parameter that contributes to that loss
      term, coupling all components through training updates.
  M6  Architecture-wide intervention. A single architectural change
      (e.g. single-head instead of multi-head attention) alters several
      components jointly rather than propagating from one to another.
  M7  Cache-time propagation (decoder only). The KV cache stores K, V
      across generation steps; a cache fault at step t affects step t
      and all later steps.

Transformer reference equations used to derive the rules below:

    Forward block:  h = LayerNorm(x + MHA(x))
                    y = LayerNorm(h + FFN(h))
    Attention:      Attn(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V
    Projections:    Q = x W^Q,  K = x W^K,  V = x W^V

The group-level Â produced by ``fundamental_to_feature_group_adjacency``
is used by the message-passing layer in the diagnostic model.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PropagationType(str, Enum):
    """Dependency mechanism class for an FPG edge.

    Each edge in the component-level FPG is annotated with exactly one
    mechanism class describing how a fault propagates along it.
    """
    M1_FORWARD_SEQUENTIAL = "m1_forward_sequential"
    M2_SIMULTANEOUS = "m2_simultaneous"
    M3_RESIDUAL_BYPASS = "m3_residual_bypass"
    M4_CROSS_LAYER = "m4_cross_layer"
    M5_BACKWARD_GRADIENT = "m5_backward_gradient"
    M6_ARCH_INTERVENTION = "m6_arch_intervention"
    M7_CACHE_TIME = "m7_cache_time"


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
    Component("embedding",          "x = E[token] + P[pos]",             "both"),
    Component("positional",         "P[pos] (sinusoidal/learned/RoPE)",  "both"),
    Component("qkv_projection",     "Q=xW^Q, K=xW^K, V=xW^V",            "both"),
    Component("score_computation",  "S = QK^T / sqrt(d_k)",              "both"),
    Component("attention_masking",  "S' = S + M",                        "both"),
    Component("attention_weights",  "A = softmax(S')",                   "both"),
    Component("attention_output",   "O = A V",                           "both"),
    Component("residual_post_attn", "h = x + O",                         "both"),
    Component("layernorm_post_attn","h' = LN(h)",                        "both"),
    Component("ffn",                "FFN(h') = W2 sigma(W1 h' + b1)+b2", "both"),
    Component("residual_post_ffn",  "y = h' + FFN(h')",                  "both"),
    Component("layernorm_post_ffn", "y' = LN(y)",                        "both"),
    Component("output_head",        "logits = y' W_out + b_out",         "both"),
    Component("kv_cache",           "K_cache[t], V_cache[t]",            "decoder"),
]


# ============================================================================
# THE 8 DETERMINISTIC PROPAGATION RULES
# ============================================================================

RULES = [
    # ───────────────────────────────────────────────────────────────────────
    # M1: Forward sequential propagation. Chain rule along the data flow.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("embedding", "qkv_projection",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="Q=xW^Q: perturbation dx -> dQ=dx W^Q (linear)"),

    PropagationRule("positional", "qkv_projection",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="Position info modifies x before projection: x=E+P"),

    PropagationRule("qkv_projection", "score_computation",
        PropagationType.M2_SIMULTANEOUS, rule_id=2,
        justification="S=QK^T/sqrt(d_k): Q,K from the projection feed score "
                       "computation while V feeds attention output simultaneously"),

    PropagationRule("score_computation", "attention_masking",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="S'=S+M: scores receive mask additively"),

    PropagationRule("attention_masking", "attention_weights",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.SOFTMAX,
        justification="A=softmax(S'): softmax bounds weights to [0,1] sum-to-1, "
                       "compressing large perturbations"),

    PropagationRule("attention_weights", "attention_output",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="O=AV: dO = dA V"),

    PropagationRule("attention_output", "residual_post_attn",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="h=x+O: attention output enters residual path"),

    PropagationRule("residual_post_attn", "layernorm_post_attn",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.LAYERNORM,
        justification="LN rescales to zero-mean unit-var, absorbing "
                       "additive shifts"),

    PropagationRule("layernorm_post_attn", "ffn",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.ACTIVATION,
        justification="FFN nonlinearity can suppress changes in saturation regions "
                       "and pass them in linear regions"),

    PropagationRule("ffn", "residual_post_ffn",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="y=h'+FFN(h'): FFN output enters second residual path"),

    PropagationRule("residual_post_ffn", "layernorm_post_ffn",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        bottleneck=Bottleneck.LAYERNORM,
        justification="Second LayerNorm absorbs additive shifts"),

    PropagationRule("layernorm_post_ffn", "output_head",
        PropagationType.M1_FORWARD_SEQUENTIAL, rule_id=1,
        justification="Final representation enters output head"),

    # ───────────────────────────────────────────────────────────────────────
    # M2: Simultaneous propagation. QKV feeds score (via Q, K) and attention
    # output (via V) at the same time. In decoders, K and V additionally
    # enter the cache.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("qkv_projection", "attention_output",
        PropagationType.M2_SIMULTANEOUS, rule_id=2,
        justification="V enters attention output O=AV simultaneously with "
                       "Q,K entering score computation"),

    PropagationRule("qkv_projection", "kv_cache",
        PropagationType.M2_SIMULTANEOUS, rule_id=2, scope="decoder",
        justification="K,V are stored in cache at each step"),

    # ───────────────────────────────────────────────────────────────────────
    # M3: Residual bypass. In h = x + MHA(x), perturbation δx propagates
    # with unit gain through the skip path.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("embedding", "residual_post_attn",
        PropagationType.M3_RESIDUAL_BYPASS, rule_id=3,
        justification="h=x+MHA(x): δx passes through the skip path with unit gain"),

    PropagationRule("layernorm_post_attn", "residual_post_ffn",
        PropagationType.M3_RESIDUAL_BYPASS, rule_id=3,
        justification="y=h'+FFN(h'): δh' passes through the skip path with unit gain"),

    # ───────────────────────────────────────────────────────────────────────
    # M4: Cross-layer propagation. The residual stream provides repeated
    # identity paths across stacked layers, so a fault at layer ℓ can
    # reach all later layers.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("layernorm_post_ffn", "qkv_projection",
        PropagationType.M4_CROSS_LAYER, rule_id=4,
        justification="Layer L output becomes layer L+1 input via the residual stream"),

    # ───────────────────────────────────────────────────────────────────────
    # M5: Backward gradient propagation. Any fault that changes the loss L
    # also changes ∂L/∂θ_i for every parameter that contributes to that
    # loss term, coupling all components through training updates.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("output_head", "qkv_projection",
        PropagationType.M5_BACKWARD_GRADIENT, rule_id=5,
        justification="∂L/∂W^Q couples logits to QKV parameters during training"),

    PropagationRule("output_head", "ffn",
        PropagationType.M5_BACKWARD_GRADIENT, rule_id=5,
        justification="∂L/∂W1, ∂L/∂W2 couple logits to FFN parameters"),

    PropagationRule("output_head", "layernorm_post_attn",
        PropagationType.M5_BACKWARD_GRADIENT, rule_id=5,
        justification="∂L/∂γ, ∂L/∂β couple logits to LayerNorm parameters"),

    PropagationRule("output_head", "embedding",
        PropagationType.M5_BACKWARD_GRADIENT, rule_id=5,
        justification="∂L/∂E couples logits to embedding parameters"),

    # ───────────────────────────────────────────────────────────────────────
    # M6: Architecture-wide intervention. Variant faults (e.g. single-head
    # instead of multi-head) change attention computation, parameter
    # shapes, and kernel dispatch jointly rather than propagating from one
    # component to another.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("attention_weights", "qkv_projection",
        PropagationType.M6_ARCH_INTERVENTION, rule_id=6,
        justification="Variant fault changes head structure, affecting how "
                       "Q,K,V projections are partitioned across heads"),

    # ───────────────────────────────────────────────────────────────────────
    # M7: Cache-time propagation (decoder only). The KV cache stores K, V
    # across generation steps; a cache fault at step t affects the
    # current token and all later tokens.
    # ───────────────────────────────────────────────────────────────────────
    PropagationRule("kv_cache", "attention_weights",
        PropagationType.M7_CACHE_TIME, rule_id=7, scope="decoder",
        justification="Cached K,V affect future-step attention: "
                       "S[t]=Q[t]K_cache[:t]^T and O[t]=A[t]V_cache[:t]"),
]


# Mechanisms that contribute edges to the message-passing adjacency Â.
# Backward gradient coupling (M5) enters the model through gradient
# features and architecture-wide intervention (M6) through fault labels,
# so neither contributes edges to Â.
_MESSAGE_PASSING_TYPES = (
    PropagationType.M1_FORWARD_SEQUENTIAL,
    PropagationType.M2_SIMULTANEOUS,
    PropagationType.M3_RESIDUAL_BYPASS,
    PropagationType.M4_CROSS_LAYER,
    PropagationType.M7_CACHE_TIME,
)


def build_fundamental_fpg(
    arch: str = "encoder",
    prop_types: tuple = None,
) -> tuple[list[str], np.ndarray, dict]:
    """Build the fundamental FPG for a given architecture.

    Args:
        arch: ``"encoder"``, ``"decoder"``, or ``"both"``.
        prop_types: when given, keep only rules whose mechanism is in this
            set. The group-level adjacency Â uses the forward and
            structural mechanisms (M1, M2, M3, M4, M7); M5 and M6 are
            excluded there. When ``None``, all mechanisms are kept (used
            for the full edge-derivation table and visualization).

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

    if prop_types is not None:
        rules = [r for r in rules if r.prop_type in prop_types]

    names = [c.name for c in comps]
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    # Mechanism -> edge weight (used for visualization distinctness; the
    # message-passing adjacency Â is binarized then row-normalized in
    # fundamental_to_feature_group_adjacency).
    type_weights = {
        PropagationType.M1_FORWARD_SEQUENTIAL: 1.0,
        PropagationType.M2_SIMULTANEOUS:       0.9,
        PropagationType.M3_RESIDUAL_BYPASS:    0.8,
        PropagationType.M4_CROSS_LAYER:        0.7,
        PropagationType.M5_BACKWARD_GRADIENT:  0.5,
        PropagationType.M6_ARCH_INTERVENTION:  0.85,
        PropagationType.M7_CACHE_TIME:         0.75,
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
            "M1 (Forward sequential)": "B=f(A) ⇒ δB=(∂f/∂A)δA by chain rule",
            "M2 (Simultaneous)":        "One source feeds multiple targets at once",
            "M3 (Residual bypass)":     "y=x+f(x) ⇒ perturbation δx has unit gain",
            "M4 (Cross-layer)":         "Layer L output → Layer L+1 via residual stream",
            "M5 (Backward gradient)":   "∂L/∂θ_i exists for all parameters",
            "M6 (Arch intervention)":   "Variant fault alters multiple components jointly",
            "M7 (Cache time)":          "KV cache spans generation steps (decoder)",
        },
        "bottleneck_summary": {
            "softmax":   "Bounds outputs to [0,1] sum-to-1, compresses large perturbations",
            "layernorm": "Rescales to zero-mean unit-var, absorbs additive shifts",
            "activation":"ReLU/GELU can amplify or suppress depending on operating point",
        },
        "propagation_types": {
            "m1_forward_sequential": "A → B because B = f(A)",
            "m2_simultaneous":       "A → {B,C} because B=f1(A) and C=f2(A) at the same time",
            "m3_residual_bypass":    "A → C through the skip connection, bypassing the sublayer",
            "m4_cross_layer":        "Layer L → Layer L+1 via the residual stream",
            "m5_backward_gradient":  "All components connected through gradient flow during training",
            "m6_arch_intervention":  "Variant fault simultaneously affects multiple components",
            "m7_cache_time":         "KV cache links across generation timesteps (decoder)",
        },
    }

    return names, adj, metadata


# Component → feature-group mapping. Components that map to the same group
# (e.g. attention_masking, attention_weights, attention_output → attention)
# share the FPG node at the group level. Non-structural groups
# (representation_drift, training_dynamics, validation_perf) are not
# produced by this collapse; they are appended below as self-loop-only
# rows in Â.
_COMP_TO_GROUP = {
    "embedding":           "embedding",
    "positional":          "positional",
    "qkv_projection":      "qkv_alignment",
    "score_computation":   "score",
    "attention_masking":   "attention",
    "attention_weights":   "attention",
    "attention_output":    "attention",
    "residual_post_attn":  "residual_stream",
    "layernorm_post_attn": "layernorm",
    "ffn":                 "ffn_output",
    "residual_post_ffn":   "residual_stream",
    "layernorm_post_ffn":  "layernorm",
    "output_head":         "output",
    "kv_cache":            "cache",
}

# Non-structural groups: self-loop only, no neighbor aggregation. These
# undergo the same learned transform as structural groups but do not
# receive information from neighboring groups.
_NON_STRUCTURAL_GROUPS = (
    "representation_drift",
    "training_dynamics",
    "validation_perf",
)


def fundamental_to_feature_group_adjacency(
    arch: str = "encoder",
) -> tuple[list[str], np.ndarray, dict]:
    """Collapse the component-level FPG to the group-level adjacency Â.

    Â is consumed by the message-passing layer in the diagnostic model.
    Structural groups receive FPG-derived neighbor edges. Non-structural
    groups (representation_drift, training_dynamics, validation_perf)
    receive self-loops only. Â is built from the forward and structural
    mechanisms (M1, M2, M3, M4, M7); backward gradient coupling (M5)
    enters through gradient features and architecture-wide intervention
    (M6) through fault labels, so neither contributes edges here.

    Returns:
        group_names: ordered group names matching the rows/cols of Â
        group_adj:   Â ∈ R^{G×G} with self-loops included
        metadata:    dict with components, rules, and mechanism summary
    """
    comp_names, comp_adj, metadata = build_fundamental_fpg(
        arch, prop_types=_MESSAGE_PASSING_TYPES)

    # Structural groups present in this architecture, in component order.
    present_groups: list[str] = []
    seen: set[str] = set()
    for c in comp_names:
        g = _COMP_TO_GROUP.get(c)
        if g is None or g in seen:
            continue
        present_groups.append(g)
        seen.add(g)

    # Append non-structural groups in fixed Table 7.7 order.
    for g in _NON_STRUCTURAL_GROUPS:
        if g not in seen:
            present_groups.append(g)
            seen.add(g)

    group_to_idx = {g: i for i, g in enumerate(present_groups)}
    n_groups = len(present_groups)
    group_adj = np.zeros((n_groups, n_groups), dtype=np.float32)

    # Collapse component edges: a group-level edge exists if any component
    # in the source group has an edge to any component in the target group.
    for i, ci in enumerate(comp_names):
        gi = _COMP_TO_GROUP.get(ci)
        if gi is None:
            continue
        for j, cj in enumerate(comp_names):
            if comp_adj[i, j] <= 0:
                continue
            gj = _COMP_TO_GROUP.get(cj)
            if gj is None:
                continue
            ii, jj = group_to_idx[gi], group_to_idx[gj]
            group_adj[ii, jj] = max(group_adj[ii, jj], comp_adj[i, j])

    # Self-loops on every group (structural and non-structural).
    np.fill_diagonal(group_adj, 1.0)

    return present_groups, group_adj, metadata
