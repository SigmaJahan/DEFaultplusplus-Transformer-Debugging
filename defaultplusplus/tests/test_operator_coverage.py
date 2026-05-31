from __future__ import annotations

import torch
import torch.nn as nn


EXPECTED_OPERATOR_IDS = [
    "MZM", "MIM", "MRM", "MCB",
    "QZQ", "QZK", "QZV", "QSW", "QTH", "QFG", "QHD",
    "SDS", "SPD", "SUC",
    "POE", "PSI", "PTL",
    "KSB", "KMD", "KFT", "KRP", "KMC",
    "VSH", "VEC",
    "CST", "CDU", "COB", "CTR", "CLK",
    "ETZ", "ESW", "ESS", "EZD",
    "FSW", "FDN", "FCA", "FRG", "FWI",
    "NSG", "NZG", "NSB", "NCE", "NWD",
    "RRS", "RSR", "RIN", "RGC", "RDR",
    "OSL", "OZR", "ORI", "OOD",
]


def test_operator_catalog_has_exact_expected_ids() -> None:
    from defaultplusplus.deform import OPERATORS

    assert list(OPERATORS.keys()) == EXPECTED_OPERATOR_IDS
    assert len(OPERATORS) == 52


def test_catalog_has_twelve_categories() -> None:
    from defaultplusplus.deform import OPERATORS, OperatorComponent

    categories = {op.component for op in OPERATORS.values()}
    assert categories == set(OperatorComponent)
    assert len(categories) == 12


def test_root_cause_label_space_matches_paper() -> None:
    """Encoders cover 40 root causes (11 categories); decoders 45 (12)."""
    from defaultplusplus.deform import root_cause_label_space

    enc = root_cause_label_space("encoder")
    dec = root_cause_label_space("decoder")

    assert sum(len(v) for v in enc.values()) == 40
    assert len(enc) == 11
    assert sum(len(v) for v in dec.values()) == 45
    assert len(dec) == 12


def test_every_operator_has_registered_injector_class() -> None:
    from defaultplusplus.deform import OPERATORS, FaultInjector, get_injector

    for op_id in EXPECTED_OPERATOR_IDS:
        injector_cls = get_injector(op_id)
        injector = injector_cls(nn.Linear(2, 2))
        assert isinstance(injector, FaultInjector), op_id

    assert set(EXPECTED_OPERATOR_IDS) == set(OPERATORS)


class _BertLikeAttention(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self = nn.Module()
        self.self.query = nn.Linear(dim, dim)
        self.self.key = nn.Linear(dim, dim)
        self.self.value = nn.Linear(dim, dim)


class _BertLikeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _BertLikeAttention()


class _BertLikeModel(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_BertLikeBlock() for _ in range(n_layers)])


def test_qsw_swaps_q_and_k_within_each_block() -> None:
    """Regression: QSW must actually swap Q with K weights and biases.

    Previously the implementation paired adjacent params from
    ``named_parameters`` (``query.weight, query.bias, ...``) which never
    have matching shapes, so QSW silently did nothing.
    """
    from defaultplusplus.deform import get_injector

    model = _BertLikeModel(n_layers=2)
    snapshots = []
    for block in model.encoder.layer:
        attn = block.attention.self
        snapshots.append({
            "qw": attn.query.weight.detach().clone(),
            "kw": attn.key.weight.detach().clone(),
            "qb": attn.query.bias.detach().clone(),
            "kb": attn.key.bias.detach().clone(),
            "vw": attn.value.weight.detach().clone(),
        })

    injector = get_injector("QSW")(model)
    with injector:
        for block, snap in zip(model.encoder.layer, snapshots):
            attn = block.attention.self
            assert torch.allclose(attn.query.weight, snap["kw"])
            assert torch.allclose(attn.key.weight, snap["qw"])
            assert torch.allclose(attn.query.bias, snap["kb"])
            assert torch.allclose(attn.key.bias, snap["qb"])
            assert torch.allclose(attn.value.weight, snap["vw"])

    for block, snap in zip(model.encoder.layer, snapshots):
        attn = block.attention.self
        assert torch.allclose(attn.query.weight, snap["qw"])
        assert torch.allclose(attn.key.weight, snap["kw"])


def test_structural_verifier_rejects_silent_static_noop() -> None:
    """A static fault that claims targets but mutates nothing must fail."""
    from defaultplusplus.deform import StructuralVerifier
    from defaultplusplus.deform.injection import StaticFault

    class _NoOpStatic(StaticFault):
        def parameters_to_mutate(self, model):
            return [model.weight]

        def mutate_parameters(self, params):
            pass  # silently do nothing

    model = nn.Linear(4, 4)
    verifier = StructuralVerifier()
    result = verifier.verify_static(model, _NoOpStatic(model), ["weight"])
    assert not result.ok
    assert "no parameter change" in result.message
