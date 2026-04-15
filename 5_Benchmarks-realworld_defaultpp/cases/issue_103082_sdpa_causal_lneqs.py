from __future__ import annotations

import numpy as np

from common import BenchmarkCase, CaseMetadata, CaseResult, causal_mask, masked_attention, seeded_rng


def run() -> CaseResult:
    rng = seeded_rng(2)
    query_len, key_len, dim = 2, 5, 4
    q = rng.normal(size=(1, query_len, dim))
    k = rng.normal(size=(1, key_len, dim))
    v = rng.normal(size=(1, key_len, dim))

    buggy_mask = np.broadcast_to(causal_mask(query_len, key_len, diagonal=0), (1, query_len, key_len))
    fixed_mask = np.broadcast_to(causal_mask(query_len, key_len, diagonal=key_len - query_len), (1, query_len, key_len))

    buggy_out, _ = masked_attention(q, k, v, buggy_mask)
    fixed_out, _ = masked_attention(q, k, v, fixed_mask)
    delta = float(np.max(np.abs(buggy_out - fixed_out)))
    reproduced = buggy_mask[0, 0].sum() == 1 and fixed_mask[0, 0].sum() == 4 and delta > 1e-3
    return CaseResult(
        reproduced=reproduced,
        summary=f"allowed keys for first cached query {int(buggy_mask[0,0].sum())} -> {int(fixed_mask[0,0].sum())}",
        details={
            "buggy_first_query_allowed_keys": int(buggy_mask[0, 0].sum()),
            "fixed_first_query_allowed_keys": int(fixed_mask[0, 0].sum()),
            "max_abs_output_delta": delta,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=2,
        slug="issue_103082_sdpa_causal_lneqs",
        title="scaled_dot_product_attention causal mask is ambiguous when L != S",
        issue_url="https://github.com/pytorch/pytorch/issues/103082",
        source_repo="pytorch/pytorch",
        source_component="scaled_dot_product_attention causal mask construction",
        fix_url=None,
        symptom="Cached decoding queries are aligned to the left edge of the key sequence and lose legal history.",
        dataset="synthetic q/k/v tensors",
        fault_family="Attention Masking",
    ),
    run=run,
)
