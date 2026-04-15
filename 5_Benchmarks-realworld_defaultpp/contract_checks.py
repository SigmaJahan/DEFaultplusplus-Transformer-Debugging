from __future__ import annotations

from common import CheckResult, ContractEvaluation, IssueContract


CONTRACTS: dict[str, IssueContract] = {
    "issue_23349_jax_seq_lengths": IssueContract(
        trigger="Run attention with provided key/value valid lengths and padded trailing positions.",
        mechanism="The buggy path ignores key/value sequence-length masking, so padded keys remain visible.",
        observable_symptom="Non-zero attention mass lands on padded or invalid positions.",
    ),
    "issue_103082_sdpa_causal_lneqs": IssueContract(
        trigger="Run cached decoding with query length L smaller than key length S.",
        mechanism="The buggy path builds the causal mask without offsetting query positions into the cached prefix.",
        observable_symptom="The continuation query sees the wrong history and output values change.",
    ),
    "issue_19045_bert_relative_cache": IssueContract(
        trigger="Enable cached decoding while using relative position bias.",
        mechanism="The buggy cached path collapses relative-position distances to a constant zero-distance bias.",
        observable_symptom="Cached decoding diverges from full recomputation on the same effective sequence.",
    ),
    "issue_17886_t5_prune_relative_bias": IssueContract(
        trigger="Prune a subset of attention heads in a model with relative position bias.",
        mechanism="The buggy path keeps the pruned projection heads but reuses the wrong relative-bias head slices.",
        observable_symptom="Pruned attention scores become inconsistent because the bias tensor is misaligned.",
    ),
    "issue_6_qkv_projection_loader": IssueContract(
        trigger="Load a checkpoint with split q, k, and v weights into a model expecting fused qkv projection weights.",
        mechanism="The buggy loader fails to fuse and remap the split projection tensors into qkv_projection.",
        observable_symptom="The load path reports missing qkv projection keys.",
    ),
    "issue_20_sparse_cache_logits": IssueContract(
        trigger="Run incremental sparse decoding with a cache and compare it against full-sequence decoding.",
        mechanism="The buggy cache path omits the current token from the sparse attention context.",
        observable_symptom="Last-step cached logits differ from full forward logits.",
    ),
    "issue_37574_swa_cache_roll": IssueContract(
        trigger="Update a sliding-window cache at the first step where the cache exactly reaches the window size.",
        mechanism="The buggy path rolls the cache one step too early at the exact boundary.",
        observable_symptom="The oldest token is dropped too soon and the boundary cache contents are wrong.",
    ),
    "issue_36096_flex_attention_weights": IssueContract(
        trigger="Request attention outputs from the flex_attention path.",
        mechanism="The buggy path returns a summary statistic instead of the attention-weight matrix.",
        observable_symptom="The returned attention payload has the wrong shape and semantics.",
    ),
    "issue_116333_kernel_stride_check": IssueContract(
        trigger="Send a tensor whose last dimension is singleton but has stride greater than one to the fast attention validator.",
        mechanism="The buggy validator requires last-dimension stride 1 even when the last dimension has size 1.",
        observable_symptom="A valid tensor layout is rejected by the fast attention backend.",
    ),
    "issue_35896_qwen2_window_layers": IssueContract(
        trigger="Configure only the lower stack to use sliding-window attention and run a multi-layer decoder.",
        mechanism="The buggy predicate routes sliding-window attention to the wrong layer indices.",
        observable_symptom="The wrong subset of layers uses sliding-window attention and outputs change.",
    ),
}


def _check(passed: bool, success: str, failure: str) -> CheckResult:
    return CheckResult(passed=passed, note=success if passed else failure)


def _mechanism_check(slug: str, details: dict[str, object]) -> CheckResult:
    if slug == "issue_23349_jax_seq_lengths":
        buggy = float(details["buggy_padded_attention_mass"])
        fixed = float(details["fixed_padded_attention_mass"])
        return _check(
            buggy > 0.0 and fixed == 0.0,
            "buggy path leaves padded keys visible while fixed path masks them out",
            "padded-key visibility does not separate buggy and fixed paths",
        )
    if slug == "issue_103082_sdpa_causal_lneqs":
        buggy = int(details["buggy_first_query_allowed_keys"])
        fixed = int(details["fixed_first_query_allowed_keys"])
        return _check(
            buggy == 1 and fixed == 4,
            "buggy mask exposes only the leftmost key while fixed mask restores full cached history",
            "allowed-key counts do not match the causal-mask offset fault",
        )
    if slug == "issue_19045_bert_relative_cache":
        buggy = int(details["buggy_bias_unique_count"])
        fixed = int(details["fixed_bias_unique_count"])
        return _check(
            buggy == 1 and fixed > 1,
            "buggy cached bias collapses to a constant while fixed bias preserves relative variation",
            "relative-position bias does not show the expected cached collapse",
        )
    if slug == "issue_17886_t5_prune_relative_bias":
        expected = list(details["expected_bias_head_indices"])
        buggy = list(details["buggy_bias_head_indices"])
        fixed = list(details["fixed_bias_head_indices"])
        return _check(
            buggy != expected and fixed == expected,
            "buggy path reuses the wrong bias head indices while fixed path tracks the kept heads",
            "bias-head indexing does not expose the pruning misalignment",
        )
    if slug == "issue_6_qkv_projection_loader":
        buggy_missing = list(details["buggy_missing_keys"])
        fixed_missing = list(details["fixed_missing_keys"])
        fused_shape = list(details["fused_qkv_shape"])
        return _check(
            buggy_missing == ["atten_func.qkv_projection.weight"] and fixed_missing == [] and fused_shape[0] == 3 * fused_shape[1],
            "buggy loader fails to materialize fused qkv weights while fixed loader reconstructs them",
            "projection remapping does not show the expected fused-qkv mapping failure",
        )
    if slug == "issue_20_sparse_cache_logits":
        current = int(details["current_token_id"])
        buggy = list(details["buggy_context_token_ids"])
        fixed = list(details["fixed_context_token_ids"])
        return _check(
            current not in buggy and current in fixed,
            "buggy cache omits the current token from the sparse context while fixed cache includes it",
            "cache context does not expose the missing current-token contribution",
        )
    if slug == "issue_37574_swa_cache_roll":
        prefix = list(details["prefix_cache"])
        window = int(details["window_size"])
        buggy = list(details["buggy_cache_after_update"])
        fixed = list(details["fixed_cache_after_update"])
        return _check(
            len(prefix) == window - 1 and buggy != fixed and fixed == prefix + [window - 1],
            "buggy updater rolls at the exact boundary while fixed updater fills the window first",
            "boundary cache update does not show the early-roll mechanism",
        )
    if slug == "issue_36096_flex_attention_weights":
        buggy_shape = list(details["buggy_shape"])
        fixed_shape = list(details["fixed_shape"])
        return _check(
            len(buggy_shape) < len(fixed_shape),
            "buggy path returns a reduced summary tensor while fixed path returns a full attention matrix",
            "returned payload rank does not expose the wrong-object bug",
        )
    if slug == "issue_116333_kernel_stride_check":
        buggy = bool(details["buggy_accepts_tensor"])
        fixed = bool(details["fixed_accepts_tensor"])
        shape = list(details["shape"])
        return _check(
            shape[-1] == 1 and (not buggy) and fixed,
            "buggy validator rejects a singleton trailing dimension while fixed validator accepts it",
            "kernel validator outcome does not match the singleton-stride fault",
        )
    if slug == "issue_35896_qwen2_window_layers":
        buggy = list(details["buggy_windowed_layers"])
        fixed = list(details["fixed_windowed_layers"])
        return _check(
            buggy != fixed and fixed == [True, True, False, False],
            "buggy predicate routes sliding-window attention to the wrong layers",
            "layer routing does not expose the reversed sliding-window predicate",
        )
    raise KeyError(slug)


def _symptom_check(slug: str, details: dict[str, object]) -> CheckResult:
    if slug == "issue_23349_jax_seq_lengths":
        buggy = float(details["buggy_padded_attention_mass"])
        fixed = float(details["fixed_padded_attention_mass"])
        return _check(
            buggy > 0.25 and fixed < 1e-9,
            "padded-token attention leak is visible only in the buggy path",
            "observable padded-token leak is not strong enough or does not disappear after the fix",
        )
    if slug == "issue_103082_sdpa_causal_lneqs":
        delta = float(details["max_abs_output_delta"])
        return _check(
            delta > 1e-3,
            "wrong causal visibility changes the cached attention output",
            "output delta is too small to support the reported cached-history symptom",
        )
    if slug == "issue_19045_bert_relative_cache":
        buggy = float(details["buggy_max_abs_delta"])
        fixed = float(details["fixed_max_abs_delta"])
        return _check(
            buggy > 1e-2 and fixed < 1e-12,
            "cached decoding diverges from full recomputation only in the buggy path",
            "cached/full divergence does not match the issue symptom",
        )
    if slug == "issue_17886_t5_prune_relative_bias":
        buggy = float(details["buggy_max_abs_delta"])
        fixed = float(details["fixed_max_abs_delta"])
        return _check(
            buggy > 1e-3 and fixed < 1e-12,
            "pruned attention scores diverge only when the relative bias stays misaligned",
            "pruned-head symptom is not visible in the local reproduction",
        )
    if slug == "issue_6_qkv_projection_loader":
        buggy_missing = list(details["buggy_missing_keys"])
        fixed_missing = list(details["fixed_missing_keys"])
        return _check(
            len(buggy_missing) > 0 and len(fixed_missing) == 0,
            "buggy loader surfaces the expected missing-key failure and fixed loader clears it",
            "missing-key load symptom is not reproduced cleanly",
        )
    if slug == "issue_20_sparse_cache_logits":
        buggy = float(details["buggy_max_abs_delta"])
        fixed = float(details["fixed_max_abs_delta"])
        return _check(
            buggy > 1e-2 and fixed < 1e-10,
            "cached logits diverge from full logits only in the buggy path",
            "cached-vs-full logit divergence is not reproduced cleanly",
        )
    if slug == "issue_37574_swa_cache_roll":
        buggy = list(details["buggy_cache_after_update"])
        fixed = list(details["fixed_cache_after_update"])
        return _check(
            buggy != fixed,
            "boundary cache contents differ between buggy and fixed paths",
            "observable boundary-cache corruption is not present",
        )
    if slug == "issue_36096_flex_attention_weights":
        buggy_shape = tuple(details["buggy_shape"])
        fixed_shape = tuple(details["fixed_shape"])
        return _check(
            buggy_shape != fixed_shape,
            "attention output payload shape differs between buggy and fixed paths",
            "observable attention payload mismatch is not reproduced",
        )
    if slug == "issue_116333_kernel_stride_check":
        buggy = bool(details["buggy_accepts_tensor"])
        fixed = bool(details["fixed_accepts_tensor"])
        return _check(
            (not buggy) and fixed,
            "buggy backend rejects the layout while fixed backend accepts it",
            "backend rejection symptom is not reproduced cleanly",
        )
    if slug == "issue_35896_qwen2_window_layers":
        delta = float(details["max_abs_output_delta"])
        return _check(
            delta > 1e-3,
            "wrong layer routing changes the decoder output",
            "output change is too small to support the reported routing symptom",
        )
    raise KeyError(slug)


def _comparison_check(slug: str, details: dict[str, object]) -> CheckResult:
    if slug == "issue_23349_jax_seq_lengths":
        buggy = float(details["buggy_padded_attention_mass"])
        fixed = float(details["fixed_padded_attention_mass"])
        return _check(
            buggy > fixed,
            "fixed path removes the leak present in the buggy path",
            "buggy and fixed paths are not cleanly separated",
        )
    if slug == "issue_103082_sdpa_causal_lneqs":
        buggy = int(details["buggy_first_query_allowed_keys"])
        fixed = int(details["fixed_first_query_allowed_keys"])
        delta = float(details["max_abs_output_delta"])
        return _check(
            fixed > buggy and delta > 0.0,
            "fix restores missing history and changes the output accordingly",
            "buggy-vs-fixed comparison does not show a causal improvement",
        )
    if slug in {"issue_19045_bert_relative_cache", "issue_17886_t5_prune_relative_bias", "issue_20_sparse_cache_logits"}:
        buggy = float(details["buggy_max_abs_delta"])
        fixed = float(details["fixed_max_abs_delta"])
        return _check(
            buggy > fixed and fixed < 1e-10,
            "fixed path collapses the buggy/full mismatch to numerical noise",
            "buggy-vs-fixed delta separation is not strong enough",
        )
    if slug == "issue_6_qkv_projection_loader":
        buggy = len(list(details["buggy_missing_keys"]))
        fixed = len(list(details["fixed_missing_keys"]))
        return _check(
            buggy > fixed and fixed == 0,
            "fix removes the missing-key failure present in the buggy path",
            "buggy-vs-fixed loader comparison is not decisive",
        )
    if slug == "issue_37574_swa_cache_roll":
        buggy = list(details["buggy_cache_after_update"])
        fixed = list(details["fixed_cache_after_update"])
        return _check(
            buggy != fixed and len(fixed) > len(buggy),
            "fixed path retains the full boundary window while buggy path drops a token early",
            "buggy-vs-fixed boundary cache comparison is not decisive",
        )
    if slug == "issue_36096_flex_attention_weights":
        buggy = tuple(details["buggy_shape"])
        fixed = tuple(details["fixed_shape"])
        fixed_rows = list(details["fixed_row_sums"])
        return _check(
            buggy != fixed and all(all(abs(v - 1.0) < 1e-6 for v in row) for row in fixed_rows),
            "fixed path returns normalized attention rows instead of the buggy summary tensor",
            "buggy-vs-fixed payload comparison is not decisive",
        )
    if slug == "issue_116333_kernel_stride_check":
        buggy = bool(details["buggy_accepts_tensor"])
        fixed = bool(details["fixed_accepts_tensor"])
        return _check(
            (not buggy) and fixed,
            "fixed validator admits the tensor rejected by the buggy validator",
            "buggy-vs-fixed validator comparison is not decisive",
        )
    if slug == "issue_35896_qwen2_window_layers":
        buggy = list(details["buggy_windowed_layers"])
        fixed = list(details["fixed_windowed_layers"])
        delta = float(details["max_abs_output_delta"])
        return _check(
            buggy != fixed and delta > 0.0,
            "correcting layer routing changes both the selected layers and the resulting output",
            "buggy-vs-fixed routing comparison is not decisive",
        )
    raise KeyError(slug)


def get_contract(slug: str) -> IssueContract:
    return CONTRACTS[slug]


def evaluate_contract(slug: str, details: dict[str, object]) -> ContractEvaluation:
    contract = get_contract(slug)
    return ContractEvaluation(
        contract=contract,
        mechanism=_mechanism_check(slug, details),
        symptom=_symptom_check(slug, details),
        buggy_vs_fixed=_comparison_check(slug, details),
    )
