"""Mutation-operator catalog for DEForm.

Each operator targets one transformer component and one fault root cause.
Three-letter IDs encode the component and the action; the first letter
identifies the component:

    E = Embedding   M = Masking         Q = QKV projection
    S = Score       P = Positional      K = Kernel
    V = Variant     C = KV cache        F = FFN
    N = LayerNorm   R = Residual        O = Output

The ``search_type`` field describes how DEForm searches the operator's
parameter space when generating mutants:

    B   the operator has no parameter and is either applied or not
        (e.g. ``MZM`` zeroes the attention mask).
    EU  the operator takes a numeric parameter; DEForm runs it once at
        each value in a configured numeric grid (e.g. ``ETZ`` zeroes a
        configurable percentage of token embeddings).
    EL  the operator takes a categorical choice; DEForm runs it once at
        each item in a configured fixed set (e.g. ``FCA`` replaces the
        activation with each of {ReLU, GELU, Tanh, Sigmoid}).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class OperatorSearchType(str, Enum):
    """How DEForm chooses parameter values for an operator."""
    BINARY = "B"          # operator has no parameter
    NUMERIC_GRID = "EU"   # operator takes a numeric parameter from a grid
    CATEGORICAL = "EL"    # operator takes a choice from a fixed set


class OperatorComponent(str, Enum):
    """Transformer component each operator targets."""
    EMBEDDING = "embedding"
    MASKING = "masking"
    QKV = "qkv"
    SCORE = "score"
    POSITIONAL = "positional"
    KERNEL = "kernel"
    VARIANT = "variant"
    KV_CACHE = "kv_cache"
    FFN = "ffn"
    LAYERNORM = "layernorm"
    RESIDUAL = "residual"
    OUTPUT = "output"


@dataclass(frozen=True)
class Operator:
    """One mutation operator.

    Attributes:
        op_id:        three-letter ID (uppercase).
        component:    transformer component being mutated.
        root_cause:   short canonical name of the fault root cause; this
                      becomes the level-3 label of any mutant produced by
                      this operator.
        action:       human-readable description of what the operator does.
        search_type:  how DEForm searches the operator's parameter space.
        param_name:   name of the parameter searched over (or None for B).
        param_grid:   numeric grid or categorical choice list. Used only
                      for EU / EL operators.
        scope:        "encoder", "decoder", or "both".
    """
    op_id: str
    component: OperatorComponent
    root_cause: str
    action: str
    search_type: OperatorSearchType
    param_name: str | None = None
    param_grid: Tuple[float, ...] | Tuple[str, ...] = ()
    scope: str = "both"


# ─────────────────────────────────────────────────────────────────────────
# Attention-internal operators
# ─────────────────────────────────────────────────────────────────────────
_ATTN_OPS: list[Operator] = [
    # Masking
    Operator("MZM", OperatorComponent.MASKING, "mask_application",
             "Zero the attention mask, keeping the original mask shape.",
             OperatorSearchType.BINARY),
    Operator("MIM", OperatorComponent.MASKING, "mask_application",
             "Invert mask semantics while preserving mask shape and dtype.",
             OperatorSearchType.BINARY),
    Operator("MRM", OperatorComponent.MASKING, "mask_generation",
             "Reshape the mask incorrectly along a broadcast-compatible axis.",
             OperatorSearchType.CATEGORICAL,
             param_name="axis", param_grid=("batch", "head")),
    Operator("MCB", OperatorComponent.MASKING, "dynamic_mask",
             "Causal-mask break: unmask a fraction of future keys "
             "(decoder-only causal violation).",
             OperatorSearchType.NUMERIC_GRID,
             param_name="visibility", param_grid=(0.1, 0.3, 0.5),
             scope="decoder"),

    # QKV projection
    Operator("QZQ", OperatorComponent.QKV, "parameter_initialization",
             "Zero the existing query projection weights.",
             OperatorSearchType.BINARY),
    Operator("QZK", OperatorComponent.QKV, "parameter_initialization",
             "Zero the existing key projection weights.",
             OperatorSearchType.BINARY),
    Operator("QZV", OperatorComponent.QKV, "parameter_initialization",
             "Zero the existing value projection weights.",
             OperatorSearchType.BINARY),
    Operator("QSW", OperatorComponent.QKV, "head_interaction",
             "Swap compatible query and key projection tensors.",
             OperatorSearchType.BINARY),
    Operator("QTH", OperatorComponent.QKV, "head_interaction",
             "Tie a subset of attention heads together.",
             OperatorSearchType.CATEGORICAL,
             param_name="heads", param_grid=("first_half", "alternating", "all")),
    Operator("QFG", OperatorComponent.QKV, "dynamic_parameter_registration",
             "Freeze QKV gradients while preserving the forward path.",
             OperatorSearchType.BINARY),

    # Score
    Operator("SDS", OperatorComponent.SCORE, "normalization",
             "Drop the 1/sqrt(d_k) score scaling.",
             OperatorSearchType.BINARY),
    Operator("SPD", OperatorComponent.SCORE, "implementation",
             "Apply dropout to the score tensor before softmax.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="p", param_grid=(0.1, 0.3, 0.5)),
    Operator("SUC", OperatorComponent.SCORE, "precision_handling",
             "Cast score computation to fp16 unsafely.",
             OperatorSearchType.BINARY),

    # Positional
    Operator("POE", OperatorComponent.POSITIONAL, "indexing",
             "Omit positional embeddings while preserving hidden-state shape.",
             OperatorSearchType.BINARY),
    Operator("PSI", OperatorComponent.POSITIONAL, "relative_position",
             "Shift position indices by Delta within the supported range.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="delta", param_grid=(1, 2, 4)),
    Operator("PTL", OperatorComponent.POSITIONAL, "interpolation",
             "Truncate positional support to a smaller maximum length.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="cutoff", param_grid=(64, 128, 256)),

    # Kernel
    Operator("KSB", OperatorComponent.KERNEL, "silent_fallback",
             "Force a valid non-optimized attention backend.",
             OperatorSearchType.BINARY),
    Operator("KMD", OperatorComponent.KERNEL, "feature_constraints",
             "Mismatched dropout probabilities between training and kernel.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="p_kern", param_grid=(0.1, 0.3, 0.5)),
    Operator("KFT", OperatorComponent.KERNEL, "hardware_incompatibility",
             "Trigger a valid fallback path via dtype or layout mismatch.",
             OperatorSearchType.CATEGORICAL,
             param_name="trigger", param_grid=("dtype", "layout")),

    # Variant
    Operator("VSH", OperatorComponent.VARIANT, "variant_configuration",
             "Route, tie, or mask heads so only one effective head remains.",
             OperatorSearchType.BINARY),
    Operator("VEC", OperatorComponent.VARIANT, "dynamic_dispatch",
             "Apply a broadcast-compatible causal mask in an encoder layer.",
             OperatorSearchType.BINARY,
             scope="encoder"),

    # KV cache (decoder only)
    Operator("CST", OperatorComponent.KV_CACHE, "cache_invalidation",
             "Stale cache: serve previous-step K, V instead of current.",
             OperatorSearchType.CATEGORICAL,
             param_name="layers", param_grid=("first", "middle", "last", "all"),
             scope="decoder"),
    Operator("COB", OperatorComponent.KV_CACHE, "cache_position",
             "Off-by-one indexing into the cache.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="shift", param_grid=(-1, 1),
             scope="decoder"),
    Operator("CTR", OperatorComponent.KV_CACHE, "memory_layout",
             "Truncate the cache to a smaller retained length.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="length", param_grid=(8, 16, 32),
             scope="decoder"),
    Operator("CLK", OperatorComponent.KV_CACHE, "distributed_synchronization",
             "Cross-request leak: reuse cached states across requests.",
             OperatorSearchType.BINARY,
             scope="decoder"),
]


# ─────────────────────────────────────────────────────────────────────────
# Architecture-level operators (apply to encoder and decoder)
# ─────────────────────────────────────────────────────────────────────────
_ARCH_OPS: list[Operator] = [
    # Embedding
    Operator("ETZ", OperatorComponent.EMBEDDING, "input_initialization",
             "Zero a configurable percentage of the token embedding rows.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="percentage", param_grid=(0.1, 0.3, 0.5)),
    Operator("ESW", OperatorComponent.EMBEDDING, "input_type",
             "Swap a configurable percentage of token embedding pairs.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="percentage", param_grid=(0.1, 0.3, 0.5)),
    Operator("ESS", OperatorComponent.EMBEDDING, "input_type",
             "Scale segment / type embeddings by a multiplicative factor.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="factor", param_grid=(0.5, 2.0, 5.0)),

    # FFN
    Operator("FSW", OperatorComponent.FFN, "weight_scaling",
             "Scale W1, W2 of the FFN by a multiplicative factor.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="factor", param_grid=(0.5, 2.0, 5.0)),
    Operator("FDN", OperatorComponent.FFN, "neuron_dropout",
             "Permanently drop a fraction of hidden FFN neurons.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="percentage", param_grid=(0.1, 0.3, 0.5)),
    Operator("FCA", OperatorComponent.FFN, "activation_function",
             "Replace the activation function.",
             OperatorSearchType.CATEGORICAL,
             param_name="activation",
             param_grid=("ReLU", "GELU", "Tanh", "Sigmoid")),
    Operator("FRG", OperatorComponent.FFN, "regularization",
             "Replace weight decay / L2 regularization scheme.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="scheme", param_grid=(0.0, 1e-2, 1e-1)),
    Operator("FWI", OperatorComponent.FFN, "weight_initialization",
             "Replace the weight initializer for W1, W2.",
             OperatorSearchType.CATEGORICAL,
             param_name="init", param_grid=("zeros", "uniform", "constant_one")),

    # LayerNorm
    Operator("NSG", OperatorComponent.LAYERNORM, "scale_parameter",
             "Scale the LayerNorm gamma parameter by a multiplicative factor.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="factor", param_grid=(0.5, 2.0, 5.0)),
    Operator("NZG", OperatorComponent.LAYERNORM, "scale_parameter",
             "Zero the LayerNorm gamma parameter while preserving shape.",
             OperatorSearchType.BINARY),
    Operator("NSB", OperatorComponent.LAYERNORM, "bias_parameter",
             "Add an additive shift to the LayerNorm beta parameter.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="shift", param_grid=(0.5, 1.0, 2.0)),
    Operator("NCE", OperatorComponent.LAYERNORM, "stability_parameter",
             "Replace the LayerNorm epsilon stability term.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="value", param_grid=(1e-4, 1e-2, 1e0)),

    # Residual
    Operator("RRS", OperatorComponent.RESIDUAL, "skip_connection",
             "Zero / bypass the residual branch while preserving sublayer "
             "output shape.",
             OperatorSearchType.BINARY),
    Operator("RSR", OperatorComponent.RESIDUAL, "residual_scaling",
             "Multiplicative scale on the residual path.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="factor", param_grid=(0.5, 2.0, 5.0)),
    Operator("RIN", OperatorComponent.RESIDUAL, "residual_path",
             "Inject Gaussian noise of a given std into the residual path.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="sigma", param_grid=(0.01, 0.1, 1.0)),
    Operator("RGC", OperatorComponent.RESIDUAL, "gradient_clipping",
             "Replace the global max-norm gradient clip threshold.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="value", param_grid=(0.1, 1.0, 10.0)),

    # Output
    Operator("OSL", OperatorComponent.OUTPUT, "output_scaling",
             "Multiplicative scale on output logits.",
             OperatorSearchType.NUMERIC_GRID,
             param_name="factor", param_grid=(0.5, 2.0, 5.0)),
    Operator("OZR", OperatorComponent.OUTPUT, "output_dimension",
             "Zero a subset of output rows for selected classes.",
             OperatorSearchType.CATEGORICAL,
             param_name="classes", param_grid=("rare", "balanced", "all")),
    Operator("ORI", OperatorComponent.OUTPUT, "output_type",
             "Reinitialize the output projection.",
             OperatorSearchType.CATEGORICAL,
             param_name="init", param_grid=("zeros", "xavier", "kaiming")),
    Operator("OOD", OperatorComponent.OUTPUT, "output_dimension",
             "Change the output interface (dimension or mapping).",
             OperatorSearchType.NUMERIC_GRID,
             param_name="dim_delta", param_grid=(-1, 1, 2)),
]


OPERATORS: dict[str, Operator] = {op.op_id: op for op in _ATTN_OPS + _ARCH_OPS}


def list_operators(scope: str | None = None) -> list[Operator]:
    """Return all operators, optionally filtered by scope.

    Args:
        scope: ``"encoder"`` keeps operators whose scope is ``"encoder"``
               or ``"both"``; ``"decoder"`` keeps ``"decoder"`` or
               ``"both"``; ``None`` returns the full catalog.
    """
    if scope is None:
        return list(OPERATORS.values())
    if scope not in ("encoder", "decoder"):
        raise ValueError(f"scope must be 'encoder', 'decoder', or None; got {scope!r}")
    return [op for op in OPERATORS.values() if op.scope in (scope, "both")]


def operators_for_component(component: OperatorComponent,
                            scope: str | None = None) -> list[Operator]:
    """Return operators that target the given component (optionally scoped)."""
    return [op for op in list_operators(scope=scope) if op.component == component]
