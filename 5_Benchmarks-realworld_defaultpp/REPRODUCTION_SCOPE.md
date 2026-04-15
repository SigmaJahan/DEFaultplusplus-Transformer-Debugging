# Reproduction Scope

This suite uses two explicit reproduction targets derived from manual analysis of the candidate spreadsheet and the public GitHub issue reports selected from it. For each active case, the source of truth is:

- the fault family assigned during spreadsheet triage
- the issue title and body
- the repro snippet or behavioral description in the issue
- the fix PR or commit when one was recoverable

The benchmark is not a historical replay suite. It is a distilled reproduction suite.

## Mechanism-level reproduction

Definition:

- A case achieves mechanism-level reproduction when the local buggy path preserves the same causal defect identified in the issue analysis.
- "Same causal defect" means the same internal logic error, not merely a superficially similar output change.
- The defect must be in the same attention subcomponent as the original issue: masking, positional indexing, cache update, score computation, qkv projection mapping, kernel eligibility, or attention-variant routing.

Operational acceptance criteria:

- The trigger condition matches the issue materially.
- The faulty branch implements the same defective rule, state update, or omitted constraint as the issue.
- The fixed branch removes that same defect, not a different workaround.
- The resulting divergence appears in the same internal object class as in the issue:
  - mask visibility
  - position-distance handling
  - cache contents
  - score or weight tensor payload
  - projection parameter mapping
  - backend acceptance predicate
  - layer-selection predicate

Interpretation:

- This is the primary validity target of the suite.
- Exact framework versions, exact model checkpoints, and exact backend selection are not required.
- A benchmark is valid at this level if it reproduces the same fault mechanism under a materially equivalent trigger.

## Issue-level observable reproduction

Definition:

- A case achieves issue-level observable reproduction when the locally reproduced mechanism yields the same externally visible symptom class reported in the GitHub issue.
- "Symptom class" means the observable failure mode, not exact textual output.

Operational acceptance criteria:

- The local run exposes the same user-visible failure category as the issue report.
- The symptom arises from the reproduced mechanism, not from unrelated scaffolding.
- The symptom remains comparable without depending on an exact historical software stack.

Typical valid symptom classes in this suite:

- masked or padded positions incorrectly influence attention
- cached decoding diverges from full recomputation
- logits differ between cached and non-cached execution
- a load path fails because expected qkv projection keys are missing
- the attention payload has the wrong structure or semantic meaning
- a kernel path rejects an otherwise valid tensor layout
- the wrong layers apply a specific attention variant

Interpretation:

- This is the secondary validity target of the suite.
- Exact exception strings, warning text, and floating-point traces are not required.
- A benchmark is valid at this level if a knowledgeable user would recognize the reproduced behavior as the same issue symptom category.

## Distinction Between The Two Levels

Mechanism-level reproduction answers:

- "Did we reproduce the same underlying bug?"

Issue-level observable reproduction answers:

- "Did that reproduced bug manifest as the same kind of externally visible problem?"

The first is about causality. The second is about manifestation.

## Active Cases Mapped To The Two Definitions

| Issue | Fault family | Mechanism-level target | Issue-level observable target |
| --- | --- | --- | --- |
| `23349` | Attention Masking | failure to enforce `key_value_seq_lengths` in masking | attention leaks onto invalid or padded positions |
| `103082` | Attention Masking | wrong causal-mask offset when `L != S` | cached query sees wrong history and outputs change |
| `19045` | Positional Encoding | cached relative-position distance handling collapses | cached decoding diverges from full recomputation |
| `17886` | Positional Encoding | head pruning leaves relative-bias path inconsistent | pruned attention behaves inconsistently |
| `6` | QKV Projection | faulty fused `qkv_projection` checkpoint mapping | load fails with missing projection keys |
| `20` | KV Cache | sparse incremental cache path updates state inconsistently | cached logits differ from full logits |
| `37574` | KV Cache | sliding-window cache rolls one step too early | cache is wrong at the first full-window boundary |
| `36096` | Score Computation | attention output path returns the wrong payload object | attention weights payload has wrong shape or meaning |
| `116333` | Kernel | stride validator rejects valid singleton-dimension layout | fast attention backend rejects otherwise valid input |
| `35896` | Attention Variant | sliding-window layer-selection predicate is reversed | wrong layers use sliding-window attention |

## Out Of Scope

The suite does not claim exact historical upstream replay by default.

That means it does not guarantee:

- exact framework-version behavior
- exact backend choice
- exact warning or exception text
- exact numerical output parity with the original environment

Those require per-issue reconstruction of the original repository version, dependency stack, runtime context, and often the original model weights.
