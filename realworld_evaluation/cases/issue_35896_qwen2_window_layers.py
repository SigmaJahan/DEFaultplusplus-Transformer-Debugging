from __future__ import annotations

import numpy as np

from realworld_evaluation.common import BenchmarkCase, CaseMetadata, CaseResult, seeded_rng


def layer_step(x: np.ndarray, windowed: bool, scale: float) -> np.ndarray:
    if windowed:
        return np.tanh(x + scale * np.roll(x, 1, axis=1))
    return np.tanh(x + scale * x.mean(axis=1, keepdims=True))


def run_stack(x: np.ndarray, num_layers: int, max_window_layers: int, buggy: bool) -> tuple[np.ndarray, list[bool]]:
    used = []
    for layer_idx in range(num_layers):
        use_window = layer_idx >= max_window_layers if buggy else layer_idx < max_window_layers
        used.append(use_window)
        x = layer_step(x, use_window, scale=0.2 + 0.15 * layer_idx)
    return x, used


def run() -> CaseResult:
    rng = seeded_rng(44)
    x = rng.normal(size=(1, 6, 4))
    num_layers = 4
    max_window_layers = 2
    buggy_out, buggy_used = run_stack(x.copy(), num_layers, max_window_layers, buggy=True)
    fixed_out, fixed_used = run_stack(x.copy(), num_layers, max_window_layers, buggy=False)
    delta = float(np.max(np.abs(buggy_out - fixed_out)))
    reproduced = buggy_used != fixed_used and fixed_used == [True, True, False, False]
    return CaseResult(
        reproduced=reproduced and delta > 1e-3,
        summary=f"windowed layers {buggy_used} -> {fixed_used}",
        details={
            "buggy_windowed_layers": buggy_used,
            "fixed_windowed_layers": fixed_used,
            "max_abs_output_delta": delta,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=44,
        slug="issue_35896_qwen2_window_layers",
        title="Qwen2 sliding-window attention is applied to the wrong layers",
        issue_url="https://github.com/huggingface/transformers/issues/35896",
        source_repo="huggingface/transformers",
        source_component="src/transformers/models/qwen2/modular_qwen2.py",
        fix_url="https://github.com/huggingface/transformers/pull/36162",
        symptom="The implementation applies sliding-window attention to top layers instead of bottom layers.",
        dataset="synthetic token states",
        fault_family="Attention Variant",
    ),
    run=run,
)
