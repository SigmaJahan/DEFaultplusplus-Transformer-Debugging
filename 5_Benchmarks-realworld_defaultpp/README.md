# Distilled Attention Fault Benchmarks

This directory keeps only the distilled benchmark cases used for the replication package.

- `cases/` contains the runnable reproductions.
- `metadata/` keeps the issue-level provenance and local reproduction mapping for each case.
- `common.py` defines the shared benchmark dataclasses and lightweight numerical helpers.
- `contract_checks.py` encodes the mechanism-level and symptom-level acceptance checks used by the runner.
- `run_benchmarks.py` is the only entry point required to execute the suite.
- `REPRODUCTION_SCOPE.md` defines what this benchmark suite claims to reproduce.

The implementations are distilled reproductions, not vendored upstream code. Synthetic inputs are used where the underlying fault is in masking, cache logic, indexing, routing, or configuration rather than in dataset semantics.

Run all cases:

```bash
python benchmarks-realworld/run_benchmarks.py
```

Run one case:

```bash
python benchmarks-realworld/run_benchmarks.py --case issue_103082_sdpa_causal_lneqs
```

Emit JSON:

```bash
python benchmarks-realworld/run_benchmarks.py --json
```
