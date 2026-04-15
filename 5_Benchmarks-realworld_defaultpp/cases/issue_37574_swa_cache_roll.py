from __future__ import annotations

from common import BenchmarkCase, CaseMetadata, CaseResult


def update_buggy(cache: list[int], token: int, window: int) -> list[int]:
    cache = list(cache)
    if len(cache) >= window - 1:
        cache = cache[1:]
    cache.append(token)
    return cache


def update_fixed(cache: list[int], token: int, window: int) -> list[int]:
    cache = list(cache)
    cache.append(token)
    if len(cache) > window:
        cache = cache[-window:]
    return cache


def run() -> CaseResult:
    window = 4
    prefix = [0, 1, 2]
    boundary_token = 3
    buggy = update_buggy(prefix, boundary_token, window)
    fixed = update_fixed(prefix, boundary_token, window)
    reproduced = buggy != fixed and fixed == [0, 1, 2, 3]
    return CaseResult(
        reproduced=reproduced,
        summary=f"window-boundary cache {buggy} -> {fixed}",
        details={
            "window_size": window,
            "prefix_cache": prefix,
            "buggy_cache_after_update": buggy,
            "fixed_cache_after_update": fixed,
        },
    )


CASE = BenchmarkCase(
    metadata=CaseMetadata(
        row_id=19,
        slug="issue_37574_swa_cache_roll",
        title="sliding-window cache update rolls one step too early",
        issue_url="https://github.com/huggingface/transformers/issues/37574",
        source_repo="huggingface/transformers",
        source_component="src/transformers/cache_utils.py",
        fix_url="https://github.com/huggingface/transformers/pull/38046",
        symptom="At the exact window boundary, the oldest token is dropped before it should be.",
        dataset="synthetic token stream",
        fault_family="KV Cache",
    ),
    run=run,
)
