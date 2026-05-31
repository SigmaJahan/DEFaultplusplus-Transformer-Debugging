from __future__ import annotations

import numpy as np

from realworld_evaluation.common import BenchmarkCase, CaseMetadata, CaseResult, seeded_rng


def project_heads(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("bld,hdf->bhlf", x, weights)


def run() -> CaseResult:
    rng = seeded_rng(10)
    batch, seq_len, dim, heads, head_dim = 1, 4, 6, 4, 3
    x = rng.normal(size=(batch, seq_len, dim))
    weights = rng.normal(size=(heads, dim, head_dim))
    rel_bias = rng.normal(size=(heads, seq_len, seq_len))
    keep = np.array([0, 2, 3])

    q = project_heads(x, weights[keep])
    k = project_heads(x, weights[keep])
    raw_scores = np.einsum("bhlf,bhmf->bhlm", q, k) / np.sqrt(head_dim)

    reference = raw_scores + rel_bias[keep][None, :, :, :]
    buggy = raw_scores + rel_bias[: len(keep)][None, :, :, :]
    fixed = raw_scores + rel_bias[keep][None, :, :, :]

    buggy_delta = float(np.max(np.abs(reference - buggy)))
    fixed_delta = float(np.max(np.abs(reference - fixed)))
    reproduced = buggy_delta > 1e-3 and fixed_delta < 1e-12
    return CaseResult(
        reproduced=reproduced,
        summary=f"pruned relative-bias misalignment {buggy_delta:.3e} -> {fixed_delta:.3e}",
        details={
            "kept_heads": keep.tolist(),
            "expected_bias_head_indices": keep.tolist(),
            "buggy_bias_head_indices": np.arange(len(keep)).tolist(),
            "fixed_bias_head_indices": keep.tolist(),
            "buggy_max_abs_delta": buggy_delta,
            "fixed_max_abs_delta": fixed_delta,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=10,
        slug="issue_17886_t5_prune_relative_bias",
        title="T5 pruning leaves relative position bias misaligned",
        issue_url="https://github.com/huggingface/transformers/issues/17886",
        source_repo="huggingface/transformers",
        source_component="src/transformers/models/t5/modeling_t5.py",
        fix_url="https://github.com/huggingface/transformers/pull/17968",
        symptom="After pruning heads, relative position bias still follows pre-pruned head order.",
        dataset="synthetic token states",
        fault_family="Positional Encoding",
    ),
    run=run,
)
