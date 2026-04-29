from __future__ import annotations

import numpy as np

from evaluation.realworld_benchmark.common import BenchmarkCase, CaseMetadata, CaseResult, masked_attention, seeded_rng


def relative_bias(query_positions: np.ndarray, key_positions: np.ndarray, table: np.ndarray) -> np.ndarray:
    max_distance = (len(table) + 1) // 2
    deltas = np.clip(query_positions[:, None] - key_positions[None, :], -(max_distance - 1), max_distance - 1)
    return table[deltas + max_distance - 1]


def run() -> CaseResult:
    rng = seeded_rng(9)
    dim = 4
    total_len = 6
    prefix_len = 4
    token_states = rng.normal(size=(1, total_len, dim))
    bias_table = np.linspace(-0.3, 0.3, 15)

    full_q = token_states[:, -2:, :]
    full_k = token_states
    full_v = token_states
    q_positions = np.arange(prefix_len, total_len)
    k_positions = np.arange(total_len)
    full_bias = relative_bias(q_positions, k_positions, bias_table)
    full_out, _ = masked_attention(full_q, full_k, full_v, np.ones((1, 2, total_len), dtype=bool))
    full_out = full_out + full_bias[None, :, :] @ full_v / total_len

    cached_q = token_states[:, -2:, :]
    cached_k = token_states
    cached_v = token_states
    buggy_bias = np.full((2, total_len), bias_table[len(bias_table) // 2])
    fixed_bias = full_bias
    buggy_out, _ = masked_attention(cached_q, cached_k, cached_v, np.ones((1, 2, total_len), dtype=bool))
    fixed_out, _ = masked_attention(cached_q, cached_k, cached_v, np.ones((1, 2, total_len), dtype=bool))
    buggy_out = buggy_out + buggy_bias[None, :, :] @ cached_v / total_len
    fixed_out = fixed_out + fixed_bias[None, :, :] @ cached_v / total_len

    buggy_delta = float(np.max(np.abs(full_out - buggy_out)))
    fixed_delta = float(np.max(np.abs(full_out - fixed_out)))
    buggy_bias_unique_count = int(np.unique(np.round(buggy_bias, 12)).size)
    fixed_bias_unique_count = int(np.unique(np.round(fixed_bias, 12)).size)
    reproduced = buggy_delta > 1e-2 and fixed_delta < 1e-12
    return CaseResult(
        reproduced=reproduced,
        summary=f"cached vs full delta {buggy_delta:.3e} -> {fixed_delta:.3e}",
        details={
            "buggy_max_abs_delta": buggy_delta,
            "fixed_max_abs_delta": fixed_delta,
            "prefix_length": prefix_len,
            "query_positions": q_positions.tolist(),
            "key_positions": k_positions.tolist(),
            "buggy_bias_unique_count": buggy_bias_unique_count,
            "fixed_bias_unique_count": fixed_bias_unique_count,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=9,
        slug="issue_19045_bert_relative_cache",
        title="relative position embedding with use_cache computes wrong distances",
        issue_url="https://github.com/huggingface/transformers/issues/19045",
        source_repo="huggingface/transformers",
        source_component="src/transformers/models/bert/modeling_bert.py",
        fix_url="https://github.com/huggingface/transformers/pull/20203",
        symptom="Cached decoding produces different outputs from full recomputation because relative distances collapse to zero.",
        dataset="synthetic token states",
        fault_family="Positional Encoding",
    ),
    run=run,
)
