# Contributing

## Principles

- Keep pipelines executable from repository root.
- Preserve canonical output paths under `results/`.
- Prefer small, testable, and reviewable changes.

## Workflow

1. Create a focused branch from `main`.
2. Implement one logical change per commit.
3. Re-run affected scripts before opening a PR.
4. Update docs for any changed execution path, output location, or dependency.

## Data and Artifact Rules

- Allowed: code, configuration, and final/processed artifacts needed for replication.
- Disallowed: temporary logs, cache files, local environment artifacts, and unrelated raw dumps.

## Pull Request Checklist

- Stage scripts still run from repository root.
- `results/` layout remains consistent or is intentionally versioned.
- README/REPRODUCIBILITY documentation is updated when behavior changes.
