"""Enumerate the fault-configuration grid for DEFault-bench."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence

from ..deform.fault_config import FaultConfiguration
from ..deform.operators import (
    OPERATORS,
    Operator,
    OperatorSearchType,
    list_operators,
)


# Subject-model and task lists used in the benchmark. Listing them here
# keeps the grid driver self-contained; the SLURM array script does not
# need to know about HuggingFace at all.
ENCODER_MODELS = (
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "distilbert/distilroberta-base",
)
ENCODER_TASKS = ("sst2", "qnli", "rte", "mrpc", "qqp")

DECODER_MODELS = (
    "gpt2",
    "distilgpt2",
    "EleutherAI/gpt-neo-125M",
)
DECODER_TASKS = ("lambada", "ptb", "wikitext2", "openwebtext")


SEVERITY_LEVELS = ("low", "medium", "high")

# Mapping from severity label to position inside an operator's numeric
# grid. The grid in :mod:`operators.py` is always given in the order
# ``(low, medium, high)``.
_SEVERITY_INDEX = {"low": 0, "medium": 1, "high": 2}


def severity_to_param(op: Operator, severity: str):
    """Resolve the operator parameter value for a given severity level.

    For numeric-grid operators (``EU``) and categorical operators
    (``EL``), the operator's ``param_grid`` is interpreted as
    ``(low, medium, high)``. Binary operators (``B``) ignore severity
    and return ``None``.
    """
    if op.search_type == OperatorSearchType.BINARY:
        return None
    if severity not in _SEVERITY_INDEX:
        raise ValueError(f"severity must be one of {tuple(_SEVERITY_INDEX)}, "
                         f"got {severity!r}")
    grid = op.param_grid
    if not grid:
        return None
    idx = _SEVERITY_INDEX[severity]
    if idx >= len(grid):
        idx = len(grid) - 1
    return grid[idx]


@dataclass
class BenchmarkSpec:
    """Configuration grid spec for one architecture.

    Attributes:
        arch:       ``"encoder"`` or ``"decoder"``.
        models:     subject models to fine-tune.
        tasks:      downstream tasks for these models.
        operators:  operator IDs to include. Empty tuple means "all
                    operators whose scope matches ``arch``".
        layer_sets: list of layer-index tuples to mutate. Each mutant
                    instance is generated once per element. Use
                    ``(())`` to mark "operator-natural target" (e.g. the
                    embedding lookup or the output head) where layer
                    selection does not apply.
        severities: severity levels to enumerate.
        seeds:      five matched seeds shared by paired clean / faulty
                    runs.
    """
    arch: str
    models: Sequence[str]
    tasks: Sequence[str]
    operators: Sequence[str] = ()
    layer_sets: Sequence[tuple[int, ...]] = field(default_factory=lambda: ((1,), (4,), (8,), ()))
    severities: Sequence[str] = SEVERITY_LEVELS
    seeds: Sequence[int] = (42, 123, 456, 789, 101112)

    def operator_objects(self) -> list[Operator]:
        """Resolve operator IDs to Operator instances for this scope."""
        if self.operators:
            ops = []
            for op_id in self.operators:
                op = OPERATORS.get(op_id)
                if op is None:
                    raise KeyError(f"Unknown operator id: {op_id!r}")
                if op.scope not in (self.arch, "both"):
                    raise ValueError(
                        f"Operator {op_id!r} has scope {op.scope!r}, "
                        f"incompatible with arch {self.arch!r}")
                ops.append(op)
            return ops
        return list_operators(scope=self.arch)


def enumerate_configurations(spec: BenchmarkSpec) -> Iterable[FaultConfiguration]:
    """Yield every :class:`FaultConfiguration` in the grid.

    The grid is

        models x tasks x operators x layer_sets x severities x seeds

    with operator-specific parameter values resolved from severity.
    Configurations are yielded in a deterministic order so that SLURM
    array indices are reproducible.
    """
    ops = spec.operator_objects()
    for model, task, op, layers, sev, seed in product(
            spec.models, spec.tasks, ops, spec.layer_sets, spec.severities,
            spec.seeds):
        yield FaultConfiguration(
            model=model,
            task=task,
            operator_id=op.op_id,
            layers=layers,
            severity=sev,
            param_value=severity_to_param(op, sev),
            seed=seed,
        )
