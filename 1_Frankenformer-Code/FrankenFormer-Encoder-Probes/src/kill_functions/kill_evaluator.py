"""Kill-function evaluation runner for encoder models.

Loads metrics/labels from results artifacts, aligns clean vs faulty runs
on available epochs, and executes kill criteria to compute p-values and decisions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.kill_functions.validator import KillValidator
from src.metrics.statistics import compute_window_features
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase

HEAD_SIM_PREFIXES = ["", "L0_", "L1_", "L2_", "L3_", "L4_", "L5_", "L6_", "L7_"]

CANO_METRIC_KEYS: Dict[str, List[str]] = {
    "accuracy": ["final_accuracy", "final_train_accuracy", "accuracy_final"],
    "loss": ["final_loss", "final_train_loss", "loss_final"],
    "val_accuracy": ["final_val_accuracy", "val_val_accuracy_final", "val_accuracy_final"],
    "val_loss": ["final_val_loss", "val_val_loss_final", "val_loss_final"],
    "val_f1": ["final_val_f1_score", "val_val_f1_score_final", "val_f1_final"],
    "val_precision": ["val_val_precision_final", "val_precision_final"],
    "val_recall": ["val_val_recall_final", "val_recall_final"],
    "val_positional_invariance": ["val_val_positional_invariance_final", "val_positional_invariance_final"],
    "cls_accuracy": ["cls_accuracy_final"],
    "cls_f1": ["cls_f1_final"],
    "cls_precision": ["cls_precision_final"],
    "cls_recall": ["cls_recall_final"],
    "cls_auc": ["cls_auc_final"],
    "attention_entropy": ["attention_entropy_final"],
    "grad_norm_total": ["grad_norm_total_final"],
    "update_ratio_total": ["update_ratio_total_final"],
    "pre_softmax_score_mean": ["pre_softmax_score_mean_final"],
    "pre_softmax_score_var": ["pre_softmax_score_var_final"],
    "pre_softmax_score_skew": ["pre_softmax_score_skew_final"],
    "pre_softmax_score_kurt": ["pre_softmax_score_kurt_final"],
    "logit_entropy": [
        "logit_entropy_final", "logit_entropy_late_mean",
        "logit_entropy_mid_mean", "logit_entropy_early_mean",
    ],
    "logit_kl_uniform": [
        "logit_kl_uniform_final", "logit_kl_uniform_late_mean",
        "logit_kl_uniform_mid_mean", "logit_kl_uniform_early_mean",
    ],
    "ece": ["ece_final", "val_val_ece_final", "ece_late_mean", "ece_mid_mean", "ece_early_mean"],
    "logit_nan_ratio": ["logit_nan_ratio_final"],
    "logit_inf_ratio": ["logit_inf_ratio_final"],
    "mass_pad": ["mass_pad_final"],
    "mass_leak": ["mass_leak_final"],
    "cross_example_attention": ["cross_example_attention_final"],
    "head_similarity_mean": [
        *[f"{p}head_similarity_mean_final" for p in HEAD_SIM_PREFIXES],
        *[f"{p}head_similarity_mean_late_mean" for p in HEAD_SIM_PREFIXES],
    ],
    "head_similarity_max": [
        *[f"{p}head_similarity_max_final" for p in HEAD_SIM_PREFIXES],
    ],
    "ffn_delta_mean": [
        "ffn_delta_mean_final", "ffn_delta_mean_late_mean",
        "ffn_delta_mean_mid_mean", "ffn_delta_mean_early_mean",
    ],
    "ffn_var_ratio_mean": [
        "ffn_var_ratio_mean_final", "ffn_var_ratio_mean_late_mean",
    ],
    "residual_cos_mean": [
        "residual_cos_mean_final", "residual_cos_mean_late_mean",
    ],
    "ln_std_mean": [
        "ln_std_mean_final", "ln_std_mean_late_mean",
    ],
    "ln_mean_abs_mean": [
        "ln_mean_abs_mean_final", "ln_mean_abs_mean_late_mean",
    ],
    "ffn_active_dim_frac_mean": [
        "ffn_active_dim_frac_mean_final",
    ],
    "ffn_out_skew_mean": [
        "ffn_out_skew_mean_final",
    ],
    "embedding_norm_mean": [
        "embedding_norm_mean_final", "embedding_norm_mean_late_mean",
    ],
    "embedding_subset_norm_mean": [
        "embedding_subset_norm_mean_final",
    ],
    "h1_delta_norm_mean": [
        "h1_delta_norm_mean_final",
    ],
    "positional_accuracy_delta": ["positional_accuracy_delta_final"],
    "positional_margin_delta": ["positional_margin_delta_final"],
    "positional_recv_mid_over_early": ["positional_recv_mid_over_early_final"],
    "positional_recv_late_over_early": ["positional_recv_late_over_early_final"],
    "runtime_step_time": ["runtime_step_time_final"],
    "runtime_memory_alloc_mb": ["runtime_memory_alloc_mb_final"],
    "kernel_flash_enabled": ["kernel_flash_enabled_final"],
    "kernel_mem_efficient_enabled": ["kernel_mem_efficient_enabled_final"],
    "kernel_math_enabled": ["kernel_math_enabled_final"],
    "kernel_fault_force_unoptimized_active": ["kernel_fault_force_unoptimized_active_final"],
    "kernel_fault_wrong_layout_active": ["kernel_fault_wrong_layout_active_final"],
    "kernel_fault_inconsistent_dropout_active": ["kernel_fault_inconsistent_dropout_active_final"],
    "val_accuracy_gap": ["val_val_accuracy_gap_final"],
}


def _lookup_metric(final_metrics: Dict[str, Any], candidates: Iterable[str]) -> Optional[float]:
    for key in candidates:
        if key in final_metrics:
            value = final_metrics[key]
            if value is None:
                return None
            return float(value)
    return None


def _build_epoch_history(epoch_entries: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, float]]]:
    history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for entry in epoch_entries:
        epoch_number = int(entry.get("epoch", 0)) + 1
        for key, value in entry.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            if key.endswith("_mean"):
                history[key[:-5]].append((epoch_number, float(value)))
    return history


def _build_validation_history(validation_entries: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, float]]]:
    history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for entry in validation_entries:
        epoch_number = int(entry.get("epoch", 0)) + 1
        for key, value in entry.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            history[key].append((epoch_number, float(value)))
    return history


def _rebuild_final_metrics(
    epoch_metrics: List[Dict[str, Any]],
    validation_metrics: List[Dict[str, Any]],
    max_epochs: Optional[int] = None,
) -> Dict[str, float]:
    if not epoch_metrics:
        return {}

    total_epochs = len(epoch_metrics)
    effective_epochs = min(max_epochs or total_epochs, total_epochs)
    truncated_epochs = epoch_metrics[:effective_epochs]
    truncated_val = [e for e in validation_metrics if int(e.get("epoch", 0)) < effective_epochs]

    final: Dict[str, float] = {}
    last_epoch = truncated_epochs[-1]
    final["final_train_loss"] = float(last_epoch.get("train_loss_mean", 0.0))
    final["final_train_accuracy"] = float(last_epoch.get("accuracy_mean", 0.0))
    final["final_grad_norm_total"] = float(last_epoch.get("grad_norm_total_mean", 0.0))
    final["final_f1_score"] = float(last_epoch.get("f1_score_mean", 0.0))

    acc_series = [float(e.get("accuracy_mean", 0.0)) for e in truncated_epochs if "accuracy_mean" in e]
    loss_series = [float(e.get("train_loss_mean", 0.0)) for e in truncated_epochs if "train_loss_mean" in e]

    final["best_train_accuracy"] = max(acc_series) if acc_series else 0.0
    final["best_train_loss"] = min(loss_series) if loss_series else final["final_train_loss"]
    final["best_accuracy"] = final["best_train_accuracy"]
    final["best_loss"] = final["best_train_loss"]
    final["final_loss"] = final["final_train_loss"]
    final["final_accuracy"] = final["final_train_accuracy"]

    epoch_history = _build_epoch_history(truncated_epochs)
    final.update(compute_window_features(epoch_history, effective_epochs))

    validation_history = _build_validation_history(truncated_val)
    if validation_history:
        final.update(compute_window_features(validation_history, effective_epochs))
    if truncated_val:
        last_val = truncated_val[-1]
        final["final_val_accuracy"] = float(
            _lookup_metric(last_val, ["val_val_accuracy", "val_accuracy"]) or 0.0
        )
        final["final_val_loss"] = float(_lookup_metric(last_val, ["val_val_loss", "val_loss"]) or 0.0)
        final["final_val_f1_score"] = float(_lookup_metric(last_val, ["val_val_f1_score", "val_f1_score"]) or 0.0)
        val_acc = [_lookup_metric(e, ["val_val_accuracy", "val_accuracy"]) for e in truncated_val]
        val_acc = [float(v) for v in val_acc if v is not None]
        final["best_val_accuracy"] = max(val_acc) if val_acc else final["final_val_accuracy"]
    else:
        final.setdefault("final_val_accuracy", 0.0)
        final.setdefault("final_val_loss", 0.0)
        final.setdefault("final_val_f1_score", 0.0)
        final.setdefault("best_val_accuracy", 0.0)

    return final


@dataclass
class MetricsSnapshot:
    config_id: str
    epoch_metrics: List[Dict[str, Any]]
    validation_metrics: List[Dict[str, Any]]
    final_metrics: Dict[str, Any]

    @property
    def epochs_completed(self) -> int:
        return len(self.epoch_metrics)

    def build_final_metrics(self, max_epochs: Optional[int] = None) -> Dict[str, float]:
        rebuilt = _rebuild_final_metrics(self.epoch_metrics, self.validation_metrics, max_epochs=max_epochs)
        merged = dict(rebuilt)
        merged.update(self.final_metrics or {})
        return merged


class KillEvaluationRunner:
    """Coordinates kill-function evaluation for all completed faulty configs."""

    def __init__(
        self,
        results_dir: Path,
        allowed_seeds: Optional[List[int]] = None,
        alpha: float = 0.05,
    ):
        self.results_dir = results_dir
        self.allowed_seeds = allowed_seeds
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
        canonical: Dict[str, float] = {}
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

    def evaluate(self) -> Dict[str, Any]:
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

        evaluations = []
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

            evaluations.append(result)
            decision_metric = result.killed_metrics[0] if result.killed_metrics else None
            min_p = None
            for metric_result in result.results.get("results", {}).values():
                if "p_value" in metric_result:
                    current = metric_result["p_value"]
                    if min_p is None or current < min_p:
                        min_p = current

            self.db.record_kill_result(
                config_id=cfg["config_id"],
                killed=result.overall_killed,
                decision_metric=decision_metric,
                p_value=min_p,
                fault_category=fault_category,
                fault_subcategory=fault_name,
                structural_verified=result.structural_verified,
                details=result.to_dict(),
            )

        summary = self.validator.get_summary_statistics()
        print(f"Evaluated {len(evaluations)} faulty configurations.")
        if summary.get("total_validations"):
            print(
                f"Kill rate: {summary.get('kill_rate', 0.0):.2f} "
                f"({summary.get('killed_count', 0)}/{summary.get('total_validations', 0)})"
            )
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate kill functions using collected metrics.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--allowed-seeds", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_seeds = None
    if args.allowed_seeds:
        allowed_seeds = [int(s) for s in args.allowed_seeds.replace(",", " ").split()]
    runner = KillEvaluationRunner(
        results_dir=Path(args.results_dir), allowed_seeds=allowed_seeds, alpha=args.alpha
    )
    runner.evaluate()


if __name__ == "__main__":
    main()
