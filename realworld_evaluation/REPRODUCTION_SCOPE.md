# Reproduction scope

This document defines the terms each case's metadata uses under
`reproduction_scope`. Every case in this suite is a minimal, self-contained
reproduction of one real transformer fault from a public GitHub issue. The
cases run on synthetic tensors with NumPy, so they isolate the fault
mechanism without pulling in the full upstream model or training stack.

## Claim levels

Each metadata file states what its case reproduces at two levels.

- **mechanism_level_claim.** The case reproduces the specific code-level
  mechanism behind the fault (for example, a causal mask built without a
  cache offset, or a LoRA update that lands on a stale projection). The
  buggy path implements the mechanism as reported, and the fixed path
  removes it.

- **issue_level_observable_claim.** The case reproduces the symptom a user
  would observe from the issue (for example, attention mass on padded
  positions, or a model whose output does not move after an update). The
  case asserts that the symptom appears in the buggy path and disappears in
  the fixed path.

## exact_historical_replay

Every case sets `exact_historical_replay: false`. The cases reproduce the
fault mechanism and its observable symptom on a small synthetic input, not
the exact historical execution of the original library at the original
commit. They are designed to be deterministic, dependency-light, and fast,
not to pin a specific upstream version.

## How a case is structured

Each case file exposes a `CASE` (a `BenchmarkCase`) whose `run()` builds the
buggy and fixed paths, compares them, and returns a `CaseResult` with
`reproduced` and a `details` dict. `contract_checks.py` then evaluates three
checks per case from those details:

- **mechanism** — the buggy path exhibits the reported mechanism,
- **symptom** — the buggy path exhibits the observable symptom,
- **buggy_vs_fixed** — the fixed path removes both.

`run_benchmarks.py` runs every case and reports whether each contract holds.
