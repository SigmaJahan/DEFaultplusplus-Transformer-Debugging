from __future__ import annotations

import numpy as np

from evaluation.realworld_benchmark.common import BenchmarkCase, CaseMetadata, CaseResult, logsumexp, masked_attention, seeded_rng


def run() -> CaseResult:
    rng = seeded_rng(24)
    q = rng.normal(size=(1, 3, 4))
    k = rng.normal(size=(1, 3, 4))
    v = rng.normal(size=(1, 3, 4))

    _, weights = masked_attention(q, k, v, np.ones((1, 3, 3), dtype=bool))
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    buggy_attentions = logsumexp(scores, axis=-1)
    fixed_attentions = weights

    reproduced = buggy_attentions.shape != fixed_attentions.shape and np.allclose(fixed_attentions.sum(axis=-1), 1.0)
    return CaseResult(
        reproduced=reproduced,
        summary=f"attention payload shape {buggy_attentions.shape} -> {fixed_attentions.shape}",
        details={
            "buggy_shape": list(buggy_attentions.shape),
            "fixed_shape": list(fixed_attentions.shape),
            "fixed_row_sums": fixed_attentions.sum(axis=-1).round(6).tolist(),
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=24,
        slug="issue_36096_flex_attention_weights",
        title="flex_attention returns logsumexp instead of attention weights",
        issue_url="https://github.com/huggingface/transformers/issues/36096",
        source_repo="huggingface/transformers",
        source_component="flex_attention output_attentions path",
        fix_url=None,
        symptom="The attention API returns a summary statistic with the wrong shape instead of a probability matrix.",
        dataset="synthetic q/k/v tensors",
        fault_family="Score Computation",
    ),
    run=run,
)
