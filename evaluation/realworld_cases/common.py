from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CaseMetadata:
    row_id: int
    slug: str
    title: str
    issue_url: str
    source_repo: str
    source_component: str
    fix_url: str | None
    symptom: str
    dataset: str
    fault_family: str


@dataclass(frozen=True)
class CaseResult:
    reproduced: bool
    summary: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IssueContract:
    trigger: str
    mechanism: str
    observable_symptom: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContractEvaluation:
    contract: IssueContract
    mechanism: CheckResult
    symptom: CheckResult
    buggy_vs_fixed: CheckResult

    @property
    def passed(self) -> bool:
        return self.mechanism.passed and self.symptom.passed and self.buggy_vs_fixed.passed

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class BenchmarkCase:
    metadata: CaseMetadata
    run: Callable[[], CaseResult]


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def softmax(x: Array, axis: int = -1) -> Array:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def masked_attention(q: Array, k: Array, v: Array, mask: Array | None = None) -> tuple[Array, Array]:
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def causal_mask(query_len: int, key_len: int, diagonal: int) -> Array:
    return np.tri(query_len, key_len, k=diagonal, dtype=bool)


def logsumexp(x: Array, axis: int = -1) -> Array:
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max + np.log(np.exp(x - x_max).sum(axis=axis, keepdims=True)), axis=axis)
