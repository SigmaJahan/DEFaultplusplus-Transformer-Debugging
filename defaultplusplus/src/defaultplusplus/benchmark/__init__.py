"""Benchmark construction: assemble DEFault-bench from DEForm mutants.

The pipeline runs once and produces the labeled dataset that the
diagnostic model trains on. For every fault configuration:

  1. Verify the configuration structurally (DEForm verifier).
  2. Run a clean fine-tuning pass and a paired faulty fine-tuning pass
     for each of the n=5 matched seeds, sharing all training settings.
  3. Apply the sign-flip permutation test on per-seed deltas. Killed
     mutants get the injected category and root-cause labels;
     surviving mutants are kept as level-1 negatives only.
  4. Aggregate the captured training traces into the fixed-length
     feature vector consumed by the diagnostic model.

This module wires the three steps together:

  config_grid       enumerate the configuration grid (model x task x
                    operator x layer x severity).
  runner            single-configuration driver, used by the SLURM job
                    array. Responsible for one paired-seeds run plus
                    feature aggregation, but not for distribution
                    across machines.
  dataset_writer    append a labeled instance to the on-disk dataset
                    (CSV + Parquet) and emit a per-config status JSON.
"""

from .config_grid import (
    BenchmarkSpec,
    enumerate_configurations,
    severity_to_param,
)
from .runner import run_one_configuration, RunOutcome
from .dataset_writer import DatasetWriter

__all__ = [
    "BenchmarkSpec",
    "enumerate_configurations",
    "severity_to_param",
    "run_one_configuration",
    "RunOutcome",
    "DatasetWriter",
]
