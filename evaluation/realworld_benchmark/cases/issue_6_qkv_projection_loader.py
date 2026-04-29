from __future__ import annotations

import numpy as np

from evaluation.realworld_benchmark.common import BenchmarkCase, CaseMetadata, CaseResult, seeded_rng


def build_checkpoint(rng: np.random.Generator, dim: int) -> dict[str, np.ndarray]:
    return {
        "atten_func.q_proj.weight": rng.normal(size=(dim, dim)),
        "atten_func.k_proj.weight": rng.normal(size=(dim, dim)),
        "atten_func.v_proj.weight": rng.normal(size=(dim, dim)),
        "atten_func.output_projection.weight": rng.normal(size=(dim, dim)),
    }


def buggy_loader(state_dict: dict[str, np.ndarray], expected_keys: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    loaded = {key: value for key, value in state_dict.items() if key in expected_keys}
    missing = [key for key in expected_keys if key not in loaded]
    return loaded, missing


def fixed_loader(state_dict: dict[str, np.ndarray], expected_keys: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    loaded = dict(state_dict)
    fused = np.concatenate(
        [
            state_dict["atten_func.q_proj.weight"],
            state_dict["atten_func.k_proj.weight"],
            state_dict["atten_func.v_proj.weight"],
        ],
        axis=0,
    )
    loaded["atten_func.qkv_projection.weight"] = fused
    missing = [key for key in expected_keys if key not in loaded]
    return loaded, missing


def run() -> CaseResult:
    rng = seeded_rng(6)
    dim = 4
    checkpoint = build_checkpoint(rng, dim)
    expected = [
        "atten_func.qkv_projection.weight",
        "atten_func.output_projection.weight",
    ]
    _, buggy_missing = buggy_loader(checkpoint, expected)
    fixed_loaded, fixed_missing = fixed_loader(checkpoint, expected)
    fused_shape = list(fixed_loaded["atten_func.qkv_projection.weight"].shape)
    reproduced = buggy_missing == ["atten_func.qkv_projection.weight"] and not fixed_missing and fused_shape == [3 * dim, dim]
    return CaseResult(
        reproduced=reproduced,
        summary=f"missing keys {len(buggy_missing)} -> {len(fixed_missing)}",
        details={
            "buggy_missing_keys": buggy_missing,
            "fixed_missing_keys": fixed_missing,
            "fused_qkv_shape": fused_shape,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=35,
        slug="issue_6_qkv_projection_loader",
        title="checkpoint loader misses fused qkv_projection weights",
        issue_url="https://github.com/google-ai-edge/ai-edge-torch/issues/6",
        source_repo="google-ai-edge/ai-edge-torch",
        source_component="loader checkpoint mapping for attention projections",
        fix_url="https://github.com/google-ai-edge/litert-torch/pull/7",
        symptom="Model loading fails with missing qkv_projection and output projection keys.",
        dataset="synthetic checkpoint tensors",
        fault_family="QKV Projection",
    ),
    run=run,
)
