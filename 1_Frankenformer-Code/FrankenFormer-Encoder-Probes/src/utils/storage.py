"""Storage utilities for metrics and labels.

Provides lightweight wrappers around HDF5 (metrics/metadata) and SQLite
(run tracking + labels) so downstream exporters can rebuild the dataset.
"""

from __future__ import annotations

import json
import sqlite3
import fcntl
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class HDF5MetricsStorage:
    """Persist per-configuration metrics/metadata in an append-friendly HDF5 file."""

    def __init__(self, filepath: str, enable_locking: bool = True, lock_timeout: float = 300.0):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._string_dtype = h5py.string_dtype(encoding="utf-8")
        self.enable_locking = enable_locking
        self.lock_timeout = lock_timeout
        self._lock_file = None
        self._lock_handle = None
        if enable_locking:
            self._lock_file = Path(str(self.filepath) + ".lock")

    def _acquire_lock(self) -> Optional[int]:
        if not self.enable_locking or self._lock_file is None:
            return None
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(self._lock_file, 'w')
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_handle = lock_handle
                return lock_handle.fileno()
            except (IOError, OSError) as e:
                elapsed = time.time() - start_time
                if elapsed >= self.lock_timeout:
                    lock_handle.close()
                    raise TimeoutError(
                        f"Failed to acquire lock on {self._lock_file} after {self.lock_timeout}s"
                    ) from e
                time.sleep(0.1)

    def _release_lock(self, lock_fd: Optional[int]) -> None:
        if lock_fd is None:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            if self._lock_handle is not None:
                self._lock_handle.close()
                self._lock_handle = None
        except Exception as e:
            logger.warning(f"Failed to release lock: {e}")

    def save_configuration_metrics(
        self,
        config_id: Any,
        epoch_metrics: List[Dict[str, Any]],
        final_metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        config_id = str(config_id)
        payload = {
            "epoch_metrics": epoch_metrics or [],
            "final_metrics": final_metrics or {},
            "metadata": metadata or {},
            "validation_metrics": validation_metrics or [],
        }
        lock_fd = None
        try:
            lock_fd = self._acquire_lock()
            with h5py.File(self.filepath, "a") as h5f:
                if config_id in h5f:
                    logger.warning(f"Overwriting existing metrics for config_id: {config_id}")
                    del h5f[config_id]
                group = h5f.create_group(config_id)
                for key, value in payload.items():
                    self._write_json_dataset(group, key, value)
            logger.info(f"Successfully saved metrics for config_id: {config_id}")
        except Exception as e:
            logger.error(f"Failed to save metrics for config_id {config_id}: {e}")
            raise
        finally:
            self._release_lock(lock_fd)

    def load_configuration_metrics(self, config_id: Any) -> Optional[Dict[str, Any]]:
        config_id = str(config_id)
        if not self.filepath.exists():
            return None
        with h5py.File(self.filepath, "r") as h5f:
            if config_id not in h5f:
                return None
            group = h5f[config_id]
            return {
                "epoch_metrics": self._read_json_dataset(group, "epoch_metrics", default=[]),
                "final_metrics": self._read_json_dataset(group, "final_metrics", default={}),
                "metadata": self._read_json_dataset(group, "metadata", default={}),
                "validation_metrics": self._read_json_dataset(group, "validation_metrics", default=[]),
            }

    def list_configurations(self) -> List[str]:
        if not self.filepath.exists():
            return []
        with h5py.File(self.filepath, "r") as h5f:
            return list(h5f.keys())

    def _write_json_dataset(self, group: h5py.Group, name: str, data: Any):
        json_blob = json.dumps(data)
        if name in group:
            del group[name]
        group.create_dataset(name, data=np.array(json_blob, dtype=self._string_dtype))

    def _read_json_dataset(self, group: h5py.Group, name: str, default: Any):
        if name not in group:
            return default
        raw = group[name][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if raw else default


class SQLiteDatabase:
    """Track configuration metadata/results and kill-function labels in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def insert_configuration(
        self,
        config_id: Any,
        seed: Optional[int],
        fault_category: str,
        fault_subcategory: str,
        is_faulty: bool,
        status: str = "pending",
        fault_id: Optional[str] = None,
        layer_idx: Optional[int] = None,
        severity_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        config_id = str(config_id)
        severity_json = json.dumps(severity_params) if severity_params is not None else None
        metadata_json = json.dumps(metadata) if metadata is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO configurations (
                    config_id, seed, fault_category, fault_subcategory,
                    fault_id, layer_idx, severity_params, is_faulty,
                    status, metadata, model_name, dataset_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_id) DO UPDATE SET
                    seed=excluded.seed,
                    fault_category=excluded.fault_category,
                    fault_subcategory=excluded.fault_subcategory,
                    fault_id=excluded.fault_id,
                    layer_idx=excluded.layer_idx,
                    severity_params=excluded.severity_params,
                    is_faulty=excluded.is_faulty,
                    status=excluded.status,
                    metadata=excluded.metadata,
                    model_name=excluded.model_name,
                    dataset_name=excluded.dataset_name,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    config_id, seed, fault_category, fault_subcategory,
                    fault_id, layer_idx, severity_json, 1 if is_faulty else 0,
                    status, metadata_json, model_name, dataset_name,
                ),
            )

    def update_configuration_results(
        self,
        config_id: Any,
        final_accuracy: float,
        final_loss: float,
        final_f1_score: float,
        best_accuracy: float,
        best_loss: float,
        status: str,
    ) -> None:
        config_id = str(config_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE configurations
                SET final_accuracy=?, final_loss=?, final_f1_score=?,
                    best_accuracy=?, best_loss=?, status=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE config_id=?
                """,
                (final_accuracy, final_loss, final_f1_score, best_accuracy, best_loss, status, config_id),
            )

    def get_configuration(self, config_id: Any) -> Optional[Dict[str, Any]]:
        config_id = str(config_id)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM configurations WHERE config_id=?", (config_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def list_configurations(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM configurations"
        params: List[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def record_kill_result(
        self,
        config_id: Any,
        killed: bool,
        decision_metric: Optional[str],
        p_value: Optional[float],
        fault_category: Optional[str],
        fault_subcategory: Optional[str],
        structural_verified: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        config_id = str(config_id)
        details_json = json.dumps(details) if details is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kill_results (
                    config_id, killed, decision_metric, p_value,
                    fault_category, fault_subcategory,
                    structural_verified, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_id) DO UPDATE SET
                    killed=excluded.killed,
                    decision_metric=excluded.decision_metric,
                    p_value=excluded.p_value,
                    fault_category=excluded.fault_category,
                    fault_subcategory=excluded.fault_subcategory,
                    structural_verified=excluded.structural_verified,
                    details=excluded.details,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    config_id, 1 if killed else 0, decision_metric, p_value,
                    fault_category, fault_subcategory, 1 if structural_verified else 0, details_json,
                ),
            )

    def get_kill_result(self, config_id: Any) -> Optional[Dict[str, Any]]:
        config_id = str(config_id)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM kill_results WHERE config_id=?", (config_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS configurations (
                    config_id TEXT PRIMARY KEY,
                    seed INTEGER,
                    fault_category TEXT,
                    fault_subcategory TEXT,
                    fault_id TEXT,
                    layer_idx INTEGER,
                    severity_params TEXT,
                    is_faulty INTEGER,
                    status TEXT,
                    final_accuracy REAL,
                    final_loss REAL,
                    final_f1_score REAL,
                    best_accuracy REAL,
                    best_loss REAL,
                    metadata TEXT,
                    model_name TEXT,
                    dataset_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._migrate_add_model_dataset_columns(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_seed ON configurations(seed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_status ON configurations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_fault_category ON configurations(fault_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_is_faulty ON configurations(is_faulty)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_fault_id ON configurations(fault_id)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(configurations)").fetchall()}
            if "model_name" in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_model_name ON configurations(model_name)")
            if "dataset_name" in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_configurations_dataset_name ON configurations(dataset_name)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kill_results (
                    config_id TEXT PRIMARY KEY,
                    killed INTEGER,
                    decision_metric TEXT,
                    p_value REAL,
                    fault_category TEXT,
                    fault_subcategory TEXT,
                    structural_verified INTEGER,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(config_id) REFERENCES configurations(config_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kill_results_killed ON kill_results(killed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kill_results_fault_category ON kill_results(fault_category)")

    def _migrate_add_model_dataset_columns(self, conn: sqlite3.Connection) -> None:
        try:
            cursor = conn.execute("PRAGMA table_info(configurations)")
            columns = [row[1] for row in cursor.fetchall()]
            needs_migration = False
            if "model_name" not in columns:
                conn.execute("ALTER TABLE configurations ADD COLUMN model_name TEXT")
                needs_migration = True
            if "dataset_name" not in columns:
                conn.execute("ALTER TABLE configurations ADD COLUMN dataset_name TEXT")
                needs_migration = True
            if needs_migration:
                rows = conn.execute(
                    "SELECT config_id FROM configurations WHERE model_name IS NULL OR dataset_name IS NULL"
                ).fetchall()
                for (config_id,) in rows:
                    parts = config_id.split("__")
                    if len(parts) >= 2:
                        conn.execute(
                            "UPDATE configurations SET model_name=?, dataset_name=? WHERE config_id=?",
                            (parts[0], parts[1], config_id)
                        )
        except Exception as e:
            logger.warning(f"Migration failed (may be normal if schema is current): {e}")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        payload = dict(row)
        if "severity_params" in payload and payload["severity_params"]:
            payload["severity_params"] = json.loads(payload["severity_params"])
        if "metadata" in payload and payload["metadata"]:
            payload["metadata"] = json.loads(payload["metadata"])
        if "details" in payload and payload["details"]:
            payload["details"] = json.loads(payload["details"])
        return payload
