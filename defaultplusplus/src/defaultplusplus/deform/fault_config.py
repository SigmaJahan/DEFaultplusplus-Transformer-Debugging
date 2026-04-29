"""Fault configuration and per-mutant result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SeverityLevel = str  # "low" | "medium" | "high"


@dataclass(frozen=True)
class FaultConfiguration:
    """One fault configuration C = (m, t, u, f, v, l, sigma).

    Attributes:
        model:        HuggingFace model name (e.g. ``"bert-base-uncased"``).
        task:         downstream task name (e.g. ``"sst2"``, ``"wikitext"``).
        operator_id:  three-letter operator ID from the DEForm catalog
                      (e.g. ``"QZQ"``, ``"FCA"``).
        layers:       tuple of layer indices to mutate (1-indexed); the
                      empty tuple means apply at the level the operator
                      naturally targets (e.g. embedding, output head).
        severity:     ``"low"``, ``"medium"``, or ``"high"``. Maps to a
                      magnitude for numeric operators and to a discrete
                      intensity for non-numeric operators (e.g. fraction
                      of unmasked future keys for ``MCB``).
        param_value:  the operator's parameter value for this run, drawn
                      from its parameter grid. ``None`` when the operator
                      takes no parameter.
        seed:         random seed shared between the paired clean and
                      faulty runs. Identical between the two so the only
                      controlled difference is the fault itself.
    """
    model: str
    task: str
    operator_id: str
    layers: tuple[int, ...]
    severity: SeverityLevel
    param_value: Any | None = None
    seed: int = 42

    def config_id(self) -> str:
        """Stable string identifier used in run logs and dataset rows."""
        layer_part = ",".join(str(i) for i in self.layers) or "all"
        param_part = "_p" + str(self.param_value) if self.param_value is not None else ""
        return (f"{self.operator_id}_l{layer_part}_s{self.severity}"
                f"{param_part}_seed{self.seed}_{self.model}_{self.task}")


@dataclass
class Mutant:
    """Outcome of one fault configuration after paired training and the
    sign-flip permutation test.

    Attributes:
        config:           the original :class:`FaultConfiguration`.
        clean_metrics:    per-seed test-set scores from the clean (no-fault)
                          runs. Accuracy for encoder classification, log
                          perplexity for decoder language modeling.
        faulty_metrics:   per-seed test-set scores from the matched faulty
                          runs.
        p_value:          one-sided sign-flip permutation p-value. The
                          floor at n=5 seeds is 1 / 2^5 ≈ 0.031.
        killed:           True if ``p_value <= alpha`` (default 0.05).
        feature_vector:   the labeled fixed-length instance produced by
                          the feature-construction pipeline (delta of
                          faulty − clean per metric, then aggregated).
                          ``None`` if the mutant was discarded by the
                          structural verifier before training.
        labels:           dict with detection / category / root-cause
                          labels for this instance (only populated when
                          ``killed=True`` and ``feature_vector`` is set).
        rejected_reason:  short string explaining why a mutant was
                          discarded by the structural verifier; empty if
                          the configuration ran successfully.
    """
    config: FaultConfiguration
    clean_metrics: tuple[float, ...] = field(default_factory=tuple)
    faulty_metrics: tuple[float, ...] = field(default_factory=tuple)
    p_value: float | None = None
    killed: bool = False
    feature_vector: dict | None = None
    labels: dict | None = None
    rejected_reason: str = ""
