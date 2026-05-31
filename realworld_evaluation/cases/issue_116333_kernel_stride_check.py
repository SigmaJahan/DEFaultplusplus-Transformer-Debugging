from __future__ import annotations

import numpy as np

from realworld_evaluation.common import BenchmarkCase, CaseMetadata, CaseResult


def element_strides(x: np.ndarray) -> tuple[int, ...]:
    return tuple(int(s // x.itemsize) for s in x.strides)


def buggy_kernel_accepts(x: np.ndarray) -> bool:
    return element_strides(x)[-1] == 1


def fixed_kernel_accepts(x: np.ndarray) -> bool:
    last_size = x.shape[-1]
    last_stride = element_strides(x)[-1]
    return last_stride == 1 or last_size == 1


def run() -> CaseResult:
    x = np.array([[0.0, 1.0]], dtype=np.float32).T
    strides = list(element_strides(x))
    buggy = buggy_kernel_accepts(x)
    fixed = fixed_kernel_accepts(x)
    reproduced = strides == [1, 2] and buggy is False and fixed is True
    return CaseResult(
        reproduced=reproduced,
        summary=f"singleton-dim stride check {buggy} -> {fixed}",
        details={
            "shape": list(x.shape),
            "element_strides": strides,
            "buggy_accepts_tensor": buggy,
            "fixed_accepts_tensor": fixed,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=39,
        slug="issue_116333_kernel_stride_check",
        title="fast attention kernel rejects valid singleton-dimension stride layout",
        issue_url="https://github.com/pytorch/pytorch/issues/116333",
        source_repo="pytorch/pytorch",
        source_component="attention kernel stride validation",
        fix_url="https://github.com/pytorch/pytorch/pull/117001",
        symptom="A tensor with a singleton last dimension is rejected by the fast attention path even though that stride layout is valid.",
        dataset="synthetic tensor layout",
        fault_family="Kernel",
    ),
    run=run,
)
