from __future__ import annotations

import numpy as np

from common import BenchmarkCase, CaseMetadata, CaseResult, masked_attention, seeded_rng


def causal_window_mask(length: int, window: int) -> np.ndarray:
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        start = max(0, i - window + 1)
        mask[i, start : i + 1] = True
    return mask


def run() -> CaseResult:
    rng = seeded_rng(18)
    seq_len, dim, vocab_size, window = 6, 5, 8, 3
    embeddings = rng.normal(size=(vocab_size, dim))
    lm_head = rng.normal(size=(dim, vocab_size))
    tokens = np.array([0, 3, 1, 5, 2, 4])
    states = embeddings[tokens][None, :, :]

    full_mask = causal_window_mask(seq_len, window)[None, :, :]
    full_out, _ = masked_attention(states, states, states, full_mask)
    full_logits = full_out[0, -1] @ lm_head

    prefix = states[:, :-1, :]
    current = states[:, -1:, :]
    buggy_cache = prefix[:, -window + 1 :, :]
    fixed_cache = prefix[:, -window + 1 :, :]
    buggy_context = buggy_cache
    fixed_context = np.concatenate([fixed_cache, current], axis=1)
    buggy_mask = np.ones((1, 1, buggy_context.shape[1]), dtype=bool)
    fixed_mask = np.ones((1, 1, fixed_context.shape[1]), dtype=bool)
    buggy_out, _ = masked_attention(current, buggy_context, buggy_context, buggy_mask)
    fixed_out, _ = masked_attention(current, fixed_context, fixed_context, fixed_mask)

    buggy_logits = buggy_out[0, 0] @ lm_head
    fixed_logits = fixed_out[0, 0] @ lm_head
    buggy_delta = float(np.max(np.abs(full_logits - buggy_logits)))
    fixed_delta = float(np.max(np.abs(full_logits - fixed_logits)))
    buggy_context_token_ids = tokens[:-1][-window + 1 :].tolist()
    fixed_context_token_ids = buggy_context_token_ids + [int(tokens[-1])]
    reproduced = buggy_delta > 1e-2 and fixed_delta < 1e-10
    return CaseResult(
        reproduced=reproduced,
        summary=f"cached last-step logits delta {buggy_delta:.3e} -> {fixed_delta:.3e}",
        details={
            "window_size": window,
            "prefix_token_ids": tokens[:-1].tolist(),
            "current_token_id": int(tokens[-1]),
            "buggy_context_token_ids": buggy_context_token_ids,
            "fixed_context_token_ids": fixed_context_token_ids,
            "buggy_max_abs_delta": buggy_delta,
            "fixed_max_abs_delta": fixed_delta,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=18,
        slug="issue_20_sparse_cache_logits",
        title="native sparse attention cache produces different logits from full forward",
        issue_url="https://github.com/lucidrains/native-sparse-attention-pytorch/issues/20",
        source_repo="lucidrains/native-sparse-attention-pytorch",
        source_component="Transformer forward cache path",
        fix_url=None,
        symptom="Incremental decoding omits the current token from the sparse cache path and diverges from full-sequence logits.",
        dataset="synthetic token ids",
        fault_family="KV Cache",
    ),
    run=run,
)
