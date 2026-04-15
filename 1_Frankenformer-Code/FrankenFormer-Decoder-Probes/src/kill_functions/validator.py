"""
Kill Function Validator

Coordinates fault validation by combining structural verification,
metric collection, statistical testing, and kill criteria evaluation.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from src.kill_functions.statistical_tests import exact_permutation_test, batch_permutation_test
from src.kill_functions.kill_criteria import create_kill_criteria
from src.kill_functions.kill_criteria import create_kill_criteria


@dataclass
class ValidationResult:
    """Results from kill function validation."""

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
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class KillValidator:
    """
    Validates whether faults are killed (detected) using statistical tests
    and kill criteria.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize validator.

        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
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
        """
        Validate if a fault is killed.

        CRITICAL FIX: Now routes to decoder-specific criteria for decoder faults.
        Previously always used encoder-style criteria, causing decoder_masking/kv_cache
        faults to fall back to generic metrics.

        Args:
            fault_type: Category of fault (e.g., 'masking', 'qkv', 'decoder_masking')
            fault_name: Specific fault name (e.g., 'zero_mask')
            clean_metrics: Dict of metric name -> list of values from clean runs
            faulty_metrics: Dict of metric name -> list of values from faulty runs
            structural_verified: Whether structural injection was verified
            primary_metrics: Optional list of metrics to use for generic criteria

        Returns:
            ValidationResult with kill decision and details
        """
        # Create appropriate kill criteria
        kwargs = {}
        if primary_metrics:
            kwargs['primary_metrics'] = primary_metrics

        # CRITICAL FIX: Route to decoder-specific criteria when appropriate
        # Decoder-specific fault types that require decoder criteria
        decoder_fault_types = {
            'decoder_masking',  # Decoder causal mask faults
            'kv_cache',         # KV-cache management faults
            'decoder_qkv',      # Decoder QKV faults
            'decoder_positional',  # Decoder positional faults
        }

        # Use decoder criteria factory for decoder-specific faults
        if fault_type in decoder_fault_types:
            criteria = create_kill_criteria(fault_type, fault_name, **kwargs)
        else:
            # Use encoder/generic criteria for other faults
            criteria = create_kill_criteria(fault_type, fault_name, **kwargs)

        # Evaluate kill criteria
        overall_killed, details = criteria.evaluate(
            clean_metrics,
            faulty_metrics,
            structural_verified=structural_verified
        )

        # Create validation result
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

        # Store in history
        self.validation_history.append(result)

        return result

    def validate_batch(
        self,
        fault_configs: List[Dict[str, Any]],
        clean_metrics: Dict[str, List[float]],
        faulty_metrics_batch: Dict[str, Dict[str, List[float]]]
    ) -> List[ValidationResult]:
        """
        Validate multiple faults at once.

        Args:
            fault_configs: List of dicts with 'fault_type', 'fault_name', etc.
            clean_metrics: Common clean metrics for all faults
            faulty_metrics_batch: Dict mapping fault_name to its faulty metrics

        Returns:
            List of ValidationResult objects
        """
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
        """
        Get summary statistics across all validations.

        Returns:
            Dictionary with aggregate statistics
        """
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
        """Group validation results by fault type."""
        by_type = {}

        for result in self.validation_history:
            fault_type = result.fault_type
            if fault_type not in by_type:
                by_type[fault_type] = {'total': 0, 'killed': 0}

            by_type[fault_type]['total'] += 1
            if result.overall_killed:
                by_type[fault_type]['killed'] += 1

        return by_type

    def _create_summary(self, killed: bool, details: Dict[str, Any]) -> str:
        """Create human-readable summary."""
        if not details.get('structural_verified', True):
            return "FAIL: Structural verification failed"

        if killed:
            killed_metrics = details.get('killed_metrics', [])
            return f"KILLED: Detected via {len(killed_metrics)} metric(s): {', '.join(killed_metrics)}"
        else:
            return "SURVIVED: No statistically significant changes detected"

    def export_results(self, filepath: str) -> None:
        """
        Export validation results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            'validation_history': [r.to_dict() for r in self.validation_history],
            'summary_statistics': self.get_summary_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported {len(self.validation_history)} validation results to {filepath}")