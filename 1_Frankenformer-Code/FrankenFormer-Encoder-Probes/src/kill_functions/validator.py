"""Kill Function Validator for encoder models.

Coordinates fault validation by combining structural verification,
metric collection, statistical testing, and kill criteria evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from src.kill_functions.statistical_tests import exact_permutation_test, batch_permutation_test
from src.kill_functions.kill_criteria import create_kill_criteria


@dataclass
class ValidationResult:
    fault_type: str
    fault_name: str
    timestamp: str
    structural_verified: bool
    overall_killed: bool
    killed_metrics: List[str]
    kill_count: int
    total_metrics: int
    kill_rate: float
    results: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class KillValidator:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.validation_history: List[ValidationResult] = []

    def validate_fault(
        self,
        fault_type: str,
        fault_name: str,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_verified: bool = True,
        primary_metrics: Optional[List[str]] = None
    ) -> ValidationResult:
        kwargs = {}
        if primary_metrics:
            kwargs['primary_metrics'] = primary_metrics

        criteria = create_kill_criteria(fault_type, fault_name, **kwargs)
        overall_killed, details = criteria.evaluate(
            clean_metrics, faulty_metrics, structural_verified=structural_verified
        )

        result = ValidationResult(
            fault_type=fault_type,
            fault_name=fault_name,
            timestamp=datetime.now().isoformat(),
            structural_verified=structural_verified,
            overall_killed=overall_killed,
            killed_metrics=details.get('killed_metrics', []),
            kill_count=details.get('kill_count', 0),
            total_metrics=len(details.get('results', {})),
            kill_rate=details.get('kill_count', 0) / max(1, len(details.get('results', {}))),
            results=details,
            summary=self._create_summary(overall_killed, details)
        )
        self.validation_history.append(result)
        return result

    def validate_batch(
        self,
        fault_configs: List[Dict[str, Any]],
        clean_metrics: Dict[str, List[float]],
        faulty_metrics_batch: Dict[str, Dict[str, List[float]]]
    ) -> List[ValidationResult]:
        results = []
        for config in fault_configs:
            fault_name = config['fault_name']
            if fault_name in faulty_metrics_batch:
                result = self.validate_fault(
                    fault_type=config.get('fault_type', 'generic'),
                    fault_name=fault_name,
                    clean_metrics=clean_metrics,
                    faulty_metrics=faulty_metrics_batch[fault_name],
                    structural_verified=config.get('structural_verified', True),
                    primary_metrics=config.get('primary_metrics')
                )
                results.append(result)
        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {'total_validations': 0}
        total = len(self.validation_history)
        killed_count = sum(1 for r in self.validation_history if r.overall_killed)
        return {
            'total_validations': total,
            'killed_count': killed_count,
            'survived_count': total - killed_count,
            'kill_rate': killed_count / total if total > 0 else 0.0,
            'average_metrics_killed': sum(r.kill_count for r in self.validation_history) / total,
            'faults_by_type': self._group_by_type(),
        }

    def _group_by_type(self) -> Dict[str, Dict[str, int]]:
        by_type = {}
        for result in self.validation_history:
            ft = result.fault_type
            if ft not in by_type:
                by_type[ft] = {'total': 0, 'killed': 0}
            by_type[ft]['total'] += 1
            if result.overall_killed:
                by_type[ft]['killed'] += 1
        return by_type

    def _create_summary(self, killed: bool, details: Dict[str, Any]) -> str:
        if not details.get('structural_verified', True):
            return "FAIL: Structural verification failed"
        if killed:
            killed_metrics = details.get('killed_metrics', [])
            return f"KILLED: Detected via {len(killed_metrics)} metric(s): {', '.join(killed_metrics)}"
        return "SURVIVED: No statistically significant changes detected"

    def export_results(self, filepath: str) -> None:
        data = {
            'validation_history': [r.to_dict() for r in self.validation_history],
            'summary_statistics': self.get_summary_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
