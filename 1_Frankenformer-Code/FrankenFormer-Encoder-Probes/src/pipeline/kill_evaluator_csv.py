"""Kill-function evaluation runner with comprehensive CSV export for encoder models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.kill_functions.validator import KillValidator
from src.kill_functions.kill_evaluator import (
    MetricsSnapshot, CANO_METRIC_KEYS, _lookup_metric,
    _rebuild_final_metrics,
)
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase


class KillEvaluatorCSV:
    """Kill evaluation runner that exports comprehensive CSV results."""

    def __init__(
        self,
        results_dir: Path,
        output_csv: Path,
        allowed_seeds: Optional[List[int]] = None,
        alpha: float = 0.05,
    ):
        self.results_dir = results_dir
        self.output_csv = output_csv
        self.allowed_seeds = allowed_seeds
        self.alpha = alpha
        self.h5 = HDF5MetricsStorage(str(results_dir / "metrics.h5"))
        self.db = SQLiteDatabase(str(results_dir / "dataset.db"))
        self.validator = KillValidator(alpha=alpha)

    def _load_snapshot(self, config_id: str) -> Optional[MetricsSnapshot]:
        blob = self.h5.load_configuration_metrics(config_id)
        if blob is None:
            return None
        return MetricsSnapshot(
            config_id=config_id,
            epoch_metrics=blob.get("epoch_metrics") or [],
            validation_metrics=blob.get("validation_metrics") or [],
            final_metrics=blob.get("final_metrics") or {},
        )

    @staticmethod
    def _canonicalize(final_metrics: Dict[str, float]) -> Dict[str, float]:
        canonical: Dict[str, float] = {
            k: float(v) for k, v in final_metrics.items()
            if isinstance(v, (int, float))
        }
        for canonical_name, keys in CANO_METRIC_KEYS.items():
            value = _lookup_metric(final_metrics, keys)
            if value is not None:
                canonical[canonical_name] = value
        return canonical

    @staticmethod
    def _series_from_values(values: Dict[str, float]) -> Dict[str, List[float]]:
        series: Dict[str, List[float]] = {}
        for metric, value in values.items():
            series.setdefault(metric, []).append(float(value))
        return series

    @staticmethod
    def _align_series(
        clean_series: Dict[str, List[float]],
        fault_series: Dict[str, List[float]],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        shared = set(clean_series.keys()) & set(fault_series.keys())
        return ({k: clean_series[k] for k in shared}, {k: fault_series[k] for k in shared})

    @staticmethod
    def _infer_structural_status(fault_category: str, fault_name: str, faulty_values: Dict[str, float]) -> bool:
        if fault_category != "kernel":
            return True
        flag_map = {
            "force_unoptimized": "kernel_fault_force_unoptimized_active",
            "wrong_layout": "kernel_fault_wrong_layout_active",
            "inconsistent_dropout": "kernel_fault_inconsistent_dropout_active",
        }
        flag_key = flag_map.get(fault_name)
        if not flag_key:
            return True
        return faulty_values.get(flag_key, 0.0) > 0.5

    def evaluate_and_export(self) -> Dict[str, Any]:
        configs = self.db.list_configurations(status="completed")
        if self.allowed_seeds:
            configs = [cfg for cfg in configs if cfg.get("seed") in self.allowed_seeds]

        baselines: Dict[int, Dict[str, Any]] = {}
        for cfg in configs:
            if cfg.get("is_faulty"):
                continue
            seed = cfg.get("seed")
            if seed is not None:
                baselines[int(seed)] = cfg

        csv_rows: List[Dict[str, Any]] = []
        all_metric_names = set()

        print(f"Processing {len(configs)} configurations...")

        for cfg in configs:
            if not cfg.get("is_faulty"):
                continue
            seed = cfg.get("seed")
            if seed is None:
                continue
            baseline_row = baselines.get(int(seed))
            if baseline_row is None:
                continue

            clean_snapshot = self._load_snapshot(str(baseline_row["config_id"]))
            fault_snapshot = self._load_snapshot(str(cfg["config_id"]))
            if clean_snapshot is None or fault_snapshot is None:
                continue

            max_epochs = min(clean_snapshot.epochs_completed, fault_snapshot.epochs_completed)
            if max_epochs == 0:
                continue

            clean_final = clean_snapshot.build_final_metrics(max_epochs=max_epochs)
            fault_final = fault_snapshot.build_final_metrics(max_epochs=max_epochs)
            clean_values = self._canonicalize(clean_final)
            fault_values = self._canonicalize(fault_final)
            clean_series = self._series_from_values(clean_values)
            fault_series = self._series_from_values(fault_values)
            clean_series, fault_series = self._align_series(clean_series, fault_series)

            fault_category = cfg.get("fault_category") or "generic"
            fault_name = cfg.get("fault_subcategory") or cfg.get("fault_name") or str(cfg["config_id"])
            structural_ok = self._infer_structural_status(fault_category, fault_name, fault_values)

            try:
                result = self.validator.validate_fault(
                    fault_type=fault_category,
                    fault_name=str(fault_name),
                    clean_metrics=clean_series,
                    faulty_metrics=fault_series,
                    structural_verified=structural_ok,
                )
            except Exception as exc:
                print(f"[ERROR] Kill evaluation failed for {cfg['config_id']}: {exc}")
                continue

            row: Dict[str, Any] = {
                "config_id": cfg["config_id"],
                "seed": seed,
                "fault_category": fault_category,
                "fault_subcategory": fault_name,
                "fault_id": cfg.get("fault_id", ""),
                "layer_idx": cfg.get("layer_idx", ""),
                "killed": 1 if result.overall_killed else 0,
                "structural_verified": 1 if result.structural_verified else 0,
                "killed_metrics": ",".join(result.killed_metrics) if result.killed_metrics else "",
                "kill_count": result.kill_count,
            }

            metric_results = result.results.get("results", {})
            min_p_value = None
            for metric_name, metric_result in metric_results.items():
                all_metric_names.add(metric_name)
                p_value = metric_result.get("p_value")
                clean_mean = metric_result.get("clean_mean")
                faulty_mean = metric_result.get("faulty_mean")
                metric_killed = metric_result.get("killed", False)

                row[f"{metric_name}_clean"] = clean_mean
                row[f"{metric_name}_faulty"] = faulty_mean
                row[f"{metric_name}_p_value"] = p_value
                row[f"{metric_name}_killed"] = 1 if metric_killed else 0

                if p_value is not None and (min_p_value is None or p_value < min_p_value):
                    min_p_value = p_value

            row["min_p_value"] = min_p_value

            for metric_name in clean_values:
                if f"{metric_name}_clean" not in row:
                    row[f"{metric_name}_clean"] = clean_values.get(metric_name)
                    row[f"{metric_name}_faulty"] = fault_values.get(metric_name)
                    all_metric_names.add(metric_name)

            csv_rows.append(row)

            decision_metric = result.killed_metrics[0] if result.killed_metrics else None
            self.db.record_kill_result(
                config_id=cfg["config_id"],
                killed=result.overall_killed,
                decision_metric=decision_metric,
                p_value=min_p_value,
                fault_category=fault_category,
                fault_subcategory=fault_name,
                structural_verified=result.structural_verified,
                details=result.to_dict(),
            )

        self._write_csv(csv_rows, sorted(all_metric_names))

        summary = self.validator.get_summary_statistics()
        print(f"\n{'='*60}")
        print(f"Evaluated {len(csv_rows)} faulty configurations.")
        if summary.get("total_validations"):
            print(
                f"Kill rate: {summary.get('kill_rate', 0.0):.2%} "
                f"({summary.get('killed_count', 0)}/{summary.get('total_validations', 0)})"
            )
        print(f"CSV saved to: {self.output_csv}")
        print(f"{'='*60}")
        return summary

    def _write_csv(self, rows: List[Dict[str, Any]], metric_names: List[str]) -> None:
        if not rows:
            print("[WARN] No rows to write to CSV")
            return

        base_columns = [
            "config_id", "seed", "fault_category", "fault_subcategory",
            "fault_id", "layer_idx", "killed", "structural_verified",
            "killed_metrics", "kill_count", "min_p_value",
        ]
        metric_columns = []
        for metric in metric_names:
            metric_columns.extend([
                f"{metric}_clean", f"{metric}_faulty",
                f"{metric}_p_value", f"{metric}_killed",
            ])
        all_columns = base_columns + metric_columns

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate kill functions and export CSV.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--allowed-seeds", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "kill_evaluation_results.csv"
    allowed_seeds = None
    if args.allowed_seeds:
        allowed_seeds = [int(s) for s in args.allowed_seeds.replace(",", " ").split()]
    evaluator = KillEvaluatorCSV(
        results_dir=results_dir, output_csv=output_csv,
        allowed_seeds=allowed_seeds, alpha=args.alpha,
    )
    evaluator.evaluate_and_export()


if __name__ == "__main__":
    main()
