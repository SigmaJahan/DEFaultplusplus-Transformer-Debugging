"""End-to-end smoke tests for the DEFault++ pipeline.

These tests are deliberately small. They run on CPU in seconds and
exercise every module that touches the training-time pipeline at
minimum scale: imports, FPG construction, feature-group routing,
static and dynamic injectors on a tiny model, the sign-flip
permutation test, the feature-construction aggregation, the
graph aggregator, and one forward pass through the diagnostic model.

The tests do not depend on DEFault-bench. They use synthetic data so
they pass in a freshly cloned repository before the benchmark has been
built.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────
def test_imports() -> None:
    modules = [
        "src.data.feature_groups",
        "src.data.feature_processor",
        "src.data.fundamental_fpg",
        "src.models.group_encoder",
        "src.defaultplusplus.deform",
        "src.defaultplusplus.deform.operators",
        "src.defaultplusplus.deform.fault_config",
        "src.defaultplusplus.deform.injection",
        "src.defaultplusplus.deform.validation",
        "src.defaultplusplus.benchmark",
        "src.defaultplusplus.benchmark.config_grid",
        "src.defaultplusplus.benchmark.runner",
        "src.defaultplusplus.benchmark.dataset_writer",
        "src.defaultplusplus.extraction.feature_construction",
        "hierarchical_graph_category_rootcause.model",
        "hierarchical_graph_category_rootcause.losses",
    ]
    for m in modules:
        importlib.import_module(m)


# ─────────────────────────────────────────────────────────────────────────
# Feature groups + FPG
# ─────────────────────────────────────────────────────────────────────────
def test_feature_group_routing() -> None:
    from src.data.feature_groups import build_group_indices, STRUCTURAL_GROUPS

    sample_names = [
        "attn_entropy_mean", "attn_pad_mass", "qk_cos", "qv_cos",
        "score_mean", "ffn_norm", "ln_gamma", "res_cos", "cka_l3_l4",
        "emb_norm", "logit_conf", "accuracy", "loss", "step_time",
        "grad_norm_attn", "cache_hidden_sim",
    ]
    g = build_group_indices(sample_names)
    # Each structural group that has matching tokens should appear.
    for expected in ("attention", "qkv_alignment", "score", "ffn_output",
                     "layernorm", "residual_stream", "embedding",
                     "representation_drift", "output", "cache",
                     "training_dynamics", "validation_perf"):
        assert expected in g, f"missing group: {expected}"
    # The structural-group canonical list is intact.
    assert "attention" in STRUCTURAL_GROUPS
    assert "cache" in STRUCTURAL_GROUPS


def test_fpg_construction() -> None:
    from src.data.fundamental_fpg import (
        fundamental_to_feature_group_adjacency,
    )

    for arch in ("encoder", "decoder"):
        names, adj, meta = fundamental_to_feature_group_adjacency(arch)
        assert adj.shape == (len(names), len(names))
        # Self-loops on every group.
        assert np.all(np.diag(adj) > 0)
        # Decoder must include the cache group; encoder must not.
        if arch == "decoder":
            assert "cache" in names
        else:
            assert "cache" not in names
        # Non-structural groups always present.
        for n in ("representation_drift", "training_dynamics", "validation_perf"):
            assert n in names


# ─────────────────────────────────────────────────────────────────────────
# DEForm injectors
# ─────────────────────────────────────────────────────────────────────────
class _TinyModel(nn.Module):
    """Smallest model that has both 'static' params and a 'forward' to wrap."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.attn = nn.MultiheadAttention(4, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        out, _ = self.attn(h, h, h)
        return out


def test_static_fault_backup_and_restore() -> None:
    from src.defaultplusplus.deform.injection import StaticFault

    class _ZeroLinear(StaticFault):
        def parameters_to_mutate(self, model):
            return [model.linear.weight, model.linear.bias]

        def mutate_parameters(self, params):
            for p in params:
                p.zero_()

    m = _TinyModel()
    w_before = m.linear.weight.detach().clone()
    b_before = m.linear.bias.detach().clone()

    with _ZeroLinear(m):
        assert torch.equal(m.linear.weight, torch.zeros_like(m.linear.weight))
        assert torch.equal(m.linear.bias, torch.zeros_like(m.linear.bias))

    # After exit: original tensors restored exactly.
    assert torch.equal(m.linear.weight, w_before)
    assert torch.equal(m.linear.bias, b_before)


def test_dynamic_fault_wraps_and_restores() -> None:
    from src.defaultplusplus.deform.injection import DynamicFault

    sentinel: dict = {"called": False, "wrapped": None}

    class _WrapAttn(DynamicFault):
        def target_modules(self, model):
            return [model.attn]

        def make_faulty_forward(self, module, original_forward):
            def faulty(*args, **kwargs):
                sentinel["called"] = True
                return original_forward(*args, **kwargs)
            sentinel["wrapped"] = faulty
            return faulty

    m = _TinyModel()
    x = torch.randn(2, 3, 4)

    with _WrapAttn(m):
        # Inside the context the faulty wrapper is bound to forward and
        # is called when the model runs.
        assert m.attn.forward is sentinel["wrapped"]
        m(x)
        assert sentinel["called"]

    # After exit the wrapper is gone: a fresh forward call must not
    # touch the faulty closure.
    sentinel["called"] = False
    m(x)
    assert sentinel["called"] is False
    assert m.attn.forward is not sentinel["wrapped"]


# ─────────────────────────────────────────────────────────────────────────
# Sign-flip permutation test
# ─────────────────────────────────────────────────────────────────────────
def test_sign_flip_kill_when_consistent_drop() -> None:
    from src.defaultplusplus.deform.validation import is_killed

    clean = [0.90, 0.91, 0.89, 0.92, 0.90]
    faulty = [0.70, 0.72, 0.68, 0.74, 0.71]
    killed, p = is_killed(clean, faulty, higher_is_better=True, alpha=0.05)
    assert killed, f"expected kill, got p={p}"
    assert p <= 0.05


def test_sign_flip_does_not_kill_no_effect() -> None:
    from src.defaultplusplus.deform.validation import is_killed

    clean = [0.90, 0.91, 0.89, 0.92, 0.90]
    faulty = [0.91, 0.90, 0.92, 0.89, 0.91]
    killed, p = is_killed(clean, faulty, higher_is_better=True, alpha=0.05)
    assert not killed
    # Floor at 1/2^n = 1/32.
    assert p >= 1 / 32


def test_sign_flip_perplexity_kill() -> None:
    """For perplexity-style metrics, lower is better."""
    from src.defaultplusplus.deform.validation import is_killed

    clean = [10.0, 10.2, 10.1, 10.3, 10.0]   # good
    faulty = [12.5, 12.7, 12.4, 12.6, 12.5]  # consistently worse
    killed, p = is_killed(clean, faulty, higher_is_better=False, alpha=0.05)
    assert killed
    assert p <= 0.05


# ─────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────
def _make_synthetic_trace(n_steps: int = 30,
                          n_layers: int = 6,
                          n_epochs: int = 3):
    from src.defaultplusplus.extraction.feature_construction import (
        TrainingTrace, LayerInternalTrace, StepTrace, EpochTrace,
    )
    rng = np.random.default_rng(0)
    layer_internal = {
        "attention_entropy": LayerInternalTrace(
            rng.normal(2.0, 0.1, size=(n_steps, n_layers))
        ),
        "ffn_output_norm": LayerInternalTrace(
            rng.normal(5.0, 0.5, size=(n_steps, n_layers))
        ),
    }
    step_level = {
        "loss": StepTrace(rng.normal(1.5, 0.2, size=n_steps)),
        "grad_norm_attn": StepTrace(rng.normal(0.5, 0.05, size=n_steps)),
    }
    epoch_level = {
        "task_accuracy": EpochTrace(rng.uniform(0.7, 0.9, size=n_epochs)),
    }
    boundaries = list(np.linspace(n_steps // n_epochs, n_steps, n_epochs).astype(int))
    return TrainingTrace(layer_internal, step_level, epoch_level, boundaries)


def test_feature_vector_has_stable_keys() -> None:
    from src.defaultplusplus.extraction.feature_construction import (
        build_feature_vector,
    )
    fv1 = build_feature_vector(_make_synthetic_trace())
    fv2 = build_feature_vector(_make_synthetic_trace())
    assert list(fv1.keys()) == list(fv2.keys()), "feature vector key order is unstable"
    # All values are finite floats.
    for k, v in fv1.items():
        assert np.isfinite(v), f"non-finite value at {k}: {v}"


def test_paired_feature_vector_runs() -> None:
    from src.defaultplusplus.extraction.feature_construction import (
        build_paired_feature_vector,
    )
    clean = [_make_synthetic_trace() for _ in range(5)]
    faulty = [_make_synthetic_trace() for _ in range(5)]
    fv = build_paired_feature_vector(clean, faulty)
    assert len(fv) > 0
    # Delta near zero on synthetic same-distribution data.
    deltas = np.array([v for v in fv.values() if np.isfinite(v)])
    assert deltas.size > 0
    assert np.abs(deltas).mean() < 5.0  # very loose; just guards against runaway values


# ─────────────────────────────────────────────────────────────────────────
# Graph aggregator (Equation 7.22 shape)
# ─────────────────────────────────────────────────────────────────────────
def test_graph_aggregator_shape_and_rows() -> None:
    from src.models.group_encoder import GraphAggregator

    g = 5
    h = 4
    adj = np.eye(g, dtype=np.float32)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0
    agg = GraphAggregator(hidden_dim=h, adjacency=adj, n_rounds=3, dropout=0.0)
    # Row-normalized buffer rows sum to 1.
    rows = agg.adj.sum(dim=1).cpu().numpy()
    assert np.allclose(rows, 1.0, atol=1e-6)

    H = torch.randn(2, g, h)
    out = agg(H)
    assert out.shape == (2, g, h)


# ─────────────────────────────────────────────────────────────────────────
# Diagnostic model forward + new loss kwargs
# ─────────────────────────────────────────────────────────────────────────
def test_hierarchical_loss_kwargs_and_forward() -> None:
    from hierarchical_graph_category_rootcause.model import HierarchicalDiagnosisModel
    from hierarchical_graph_category_rootcause.losses import hierarchical_loss

    n = 8
    embedding_dim = 16
    hidden_dim = 4
    n_groups = 5
    group_dims = {f"g{i}": 3 for i in range(n_groups)}
    adjacency = np.eye(n_groups, dtype=np.float32)

    category_names = ["catA", "catB"]
    category_sizes = {"catA": 2, "catB": 2}

    model = HierarchicalDiagnosisModel(
        group_dims=group_dims, adjacency=adjacency,
        hidden_dim=hidden_dim, embedding_dim=embedding_dim,
        n_message_passing=2, dropout=0.0,
        mode="graph_conditioned",
        n_categories=len(category_names),
        category_sizes=category_sizes,
        group_names=list(group_dims.keys()),
    )

    # Build a tiny input where features are routed by group_indices.
    total = sum(group_dims.values())
    x = torch.randn(n, total)
    group_indices = {}
    cursor = 0
    for name, d in group_dims.items():
        group_indices[name] = list(range(cursor, cursor + d))
        cursor += d

    z, h_groups = model.encode(x, group_indices)
    assert z.shape == (n, embedding_dim)
    assert h_groups.shape == (n, n_groups, hidden_dim)

    y_detect = torch.tensor([0, 1, 1, 1, 1, 1, 1, 0], dtype=torch.long)
    y_category = torch.tensor([-1, 0, 0, 1, 1, 0, 1, -1], dtype=torch.long)
    y_rootcause = torch.tensor([-1, 0, 1, 0, 1, 0, 1, -1], dtype=torch.long)
    rootcause_local_labels = {"catA": {0: 0, 1: 1}, "catB": {0: 0, 1: 1}}

    total_loss, parts = hierarchical_loss(
        model, z, h_groups, y_detect, y_category, y_rootcause,
        category_names, rootcause_local_labels,
        alpha=1.0, lambda_rc=1.0, beta=0.5, gamma=0.3,
        temperature=0.1,
    )
    assert torch.isfinite(total_loss)
    for k in ("detection", "category", "rootcause",
              "contrastive", "prototype", "separation", "total"):
        assert k in parts, f"missing loss component: {k}"


# ─────────────────────────────────────────────────────────────────────────
# Operator catalog + benchmark grid sanity
# ─────────────────────────────────────────────────────────────────────────
def test_operator_catalog() -> None:
    from src.defaultplusplus.deform.operators import (
        OPERATORS, list_operators, OperatorComponent,
    )
    assert len(OPERATORS) >= 40
    # Every operator has a non-empty action and a known component.
    for op in OPERATORS.values():
        assert op.action
        assert op.component in OperatorComponent
    # Decoder gets KV-cache operators; encoder does not.
    enc = {op.op_id for op in list_operators("encoder")}
    dec = {op.op_id for op in list_operators("decoder")}
    assert "CST" not in enc
    assert "CST" in dec


def test_benchmark_grid_enumerates() -> None:
    from src.defaultplusplus.benchmark.config_grid import (
        BenchmarkSpec, enumerate_configurations,
    )
    spec = BenchmarkSpec(
        arch="encoder",
        models=("bert-base-uncased",),
        tasks=("sst2",),
        operators=("QZQ", "FSW"),
        layer_sets=((1,),),
        severities=("low", "high"),
        seeds=(42,),
    )
    configs = list(enumerate_configurations(spec))
    # 1 model x 1 task x 2 operators x 1 layer x 2 severities x 1 seed = 4
    assert len(configs) == 4
    ids = {c.config_id() for c in configs}
    assert len(ids) == len(configs), "config IDs must be unique"
