from __future__ import annotations

import numpy as np

from realworld_evaluation.common import BenchmarkCase, CaseMetadata, CaseResult, masked_attention, seeded_rng


def run() -> CaseResult:
    rng = seeded_rng(1)
    batch, query_len, key_len, dim = 2, 4, 6, 5
    q = rng.normal(size=(batch, query_len, dim))
    k = rng.normal(size=(batch, key_len, dim))
    v = rng.normal(size=(batch, key_len, dim))
    valid_lens = np.array([3, 5])

    _, buggy_weights = masked_attention(q, k, v)
    mask = np.arange(key_len)[None, None, :] < valid_lens[:, None, None]
    mask = np.broadcast_to(mask, (batch, query_len, key_len))
    _, fixed_weights = masked_attention(q, k, v, mask)

    padded = ~mask
    buggy_leak = float(buggy_weights[padded].sum())
    fixed_leak = float(fixed_weights[padded].sum())
    reproduced = buggy_leak > 0.25 and fixed_leak < 1e-9
    return CaseResult(
        reproduced=reproduced,
        summary=f"padded attention mass {buggy_leak:.3f} -> {fixed_leak:.3e}",
        details={
            "valid_lengths": valid_lens.tolist(),
            "buggy_padded_attention_mass": buggy_leak,
            "fixed_padded_attention_mass": fixed_leak,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=1,
        slug="issue_23349_jax_seq_lengths",
        title="dot_product_attention ignores key_value_seq_lengths",
        issue_url="https://github.com/jax-ml/jax/issues/23349",
        source_repo="jax-ml/jax",
        source_component="jax/_src/nn/functions.py",
        fix_url="https://github.com/jax-ml/jax/pull/23415",
        symptom="Padded tokens receive non-zero attention mass when key/value sequence lengths are provided.",
        dataset="synthetic tensors",
        fault_family="Attention Masking",
    ),
    run=run,
)
