"""Append labeled instances produced by the benchmark runner.

Each instance carries three labels:

  detection_label   binary; 1 if the mutant was killed under the
                    sign-flip permutation test, 0 otherwise.
  category_label    architecture-specific fault category from the
                    operator catalog (e.g. ``"qkv"``, ``"layernorm"``).
                    Surviving mutants still carry the injected category
                    in this column so downstream tooling can audit them,
                    but level-2 training restricts to killed mutants.
  rootcause_label   the root cause associated with the operator
                    (e.g. ``"parameter_initialization"``,
                    ``"weight_scaling"``). Same rule as ``category_label``.

The writer maintains a CSV file (rolling-append, atomic) and an aligned
Parquet file for downstream tooling. Concurrent SLURM array tasks each
write their own per-task shard; a separate ``merge_shards`` script
concatenates the shards into the final dataset.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ..deform.fault_config import Mutant
from ..deform.operators import OPERATORS


class DatasetWriter:
    """Append labeled instances to a CSV shard.

    Writers are designed to be created once per SLURM array task and
    fed every mutant produced on that task. Output files are written
    atomically: each row is flushed and fsynced before returning, so a
    killed job can be restarted without partial-row recovery.

    Attributes:
        shard_path: CSV file this writer appends to. The file is
                    created on the first write with a header row.
    """

    def __init__(self,
                 shard_path: Path | str,
                 fixed_columns: Iterable[str] = ()):
        """Create a writer.

        Args:
            shard_path:    target CSV file. Parent directories are
                           created on first write.
            fixed_columns: feature column names. The header is built
                           on first write as ``identifier_columns +
                           label_columns + tuple(fixed_columns)``. Pass
                           the canonical feature-name list so all shard
                           writers share the same column order.
        """
        self.shard_path = Path(shard_path)
        self._fixed_columns = tuple(fixed_columns)
        self._lock = threading.Lock()
        self._header_written = self.shard_path.exists() and self.shard_path.stat().st_size > 0

    # ── Public API ───────────────────────────────────────────────────────
    def append(self, mutant: Mutant) -> None:
        """Append one mutant as a row in the CSV shard."""
        row = self._row_from_mutant(mutant)
        with self._lock:
            self._ensure_parent()
            mode = "a" if self._header_written else "w"
            with open(self.shard_path, mode, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames())
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(row)
                fh.flush()
                os.fsync(fh.fileno())

    def append_status(self, mutant: Mutant, status_dir: Path | str) -> None:
        """Drop a per-mutant status JSON next to the shard.

        The runner already writes a status file when configured to; this
        helper is a convenience for callers that do not pass an output
        directory directly to the runner.
        """
        out = Path(status_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(mutant.config),
            "p_value": mutant.p_value,
            "killed": mutant.killed,
            "clean_metrics": list(mutant.clean_metrics),
            "faulty_metrics": list(mutant.faulty_metrics),
            "rejected_reason": mutant.rejected_reason,
        }
        with open(out / f"{mutant.config.config_id()}.status.json", "w") as f:
            json.dump(payload, f, indent=2)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _ensure_parent(self) -> None:
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)

    def _identifier_columns(self) -> tuple[str, ...]:
        return (
            "identifier",
            "model",
            "task",
            "operator_id",
            "layers",
            "severity",
            "param_value",
            "seed",
        )

    def _label_columns(self) -> tuple[str, ...]:
        return (
            "detection_label",
            "category_label",
            "rootcause_label",
            "p_value",
        )

    def _fieldnames(self) -> list[str]:
        return list(self._identifier_columns() + self._label_columns()
                    + self._fixed_columns)

    def _row_from_mutant(self, mutant: Mutant) -> dict[str, Any]:
        op = OPERATORS.get(mutant.config.operator_id)
        category = op.component.value if op is not None else ""
        rootcause = op.root_cause if op is not None else ""

        row: dict[str, Any] = {
            "identifier": mutant.config.config_id(),
            "model": mutant.config.model,
            "task": mutant.config.task,
            "operator_id": mutant.config.operator_id,
            "layers": ",".join(str(i) for i in mutant.config.layers),
            "severity": mutant.config.severity,
            "param_value": mutant.config.param_value,
            "seed": mutant.config.seed,
            "detection_label": int(mutant.killed),
            "category_label": category,
            "rootcause_label": rootcause,
            "p_value": mutant.p_value,
        }
        if mutant.feature_vector:
            for col in self._fixed_columns:
                row[col] = mutant.feature_vector.get(col)
        return row
