"""
Decoder-Specific Kill Criteria.

Defines kill criteria for decoder-specific faults:
- K_mask_dec: Decoder masking faults (causal mask violations)
- K_kv_dec: KV-cache management faults
"""

import logging
from typing import Dict, List, Any

import numpy as np

from src.kill_functions.statistical_tests import exact_permutation_test


class KillCriteria:
    """Base class for decoder kill criteria evaluation."""

    def __init__(self, fault_name: str):
        """
        Initialize decoder kill criteria.

        Args:
            fault_name: Specific fault name
        """
        self.fault_name = fault_name
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _ensure_list(values: Any) -> List[float]:
        """Convert metric values to a filtered list of floats."""
        if values is None:
            return []
        if isinstance(values, list):
            return [float(v) for v in values if v is not None]
        return [float(values)]

    def _test_metric(
        self,
        metric: str,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        min_samples: int = 2
    ) -> Dict[str, Any]:
        """Run statistical test for a single metric."""
        result: Dict[str, Any] = {
            'killed': False,
            'p_value': 1.0
        }

        if metric not in clean_metrics or metric not in faulty_metrics:
            result['reason'] = 'metric_not_available'
            return result

        clean_values = clean_metrics[metric]
        faulty_values = faulty_metrics[metric]

        # CRITICAL FIX: N=1 - Use threshold-based detection via _single_seed_kill_test
        if len(clean_values) == 1 and len(faulty_values) == 1:
            clean_val = float(clean_values[0])
            faulty_val = float(faulty_values[0])
            killed = self._single_seed_kill_test(metric, clean_val, faulty_val)
            result.update({
                'killed': killed,
                'reason': 'single_seed_threshold' if killed else 'single_seed_no_change',
                'clean_mean': clean_val,
                'faulty_mean': faulty_val,
                'method': 'threshold_based',
                'p_value': 0.0 if killed else 1.0
            })
            return result

        # CRITICAL FIX: N=2 - Use exact permutation but require directional agreement + minimum effect size
        if len(clean_values) == 2 and len(faulty_values) == 2:
            try:
                killed, p_value, details = exact_permutation_test(
                    clean_values,
                    faulty_values
                )

                # Require 100% directional agreement
                clean_mean = float(np.mean(clean_values))
                faulty_mean = float(np.mean(faulty_values))

                # Check if both samples agree in direction
                diff1 = clean_values[0] - faulty_values[0]
                diff2 = clean_values[1] - faulty_values[1]
                directional_agreement = (diff1 * diff2) > 0

                # Minimum effect size: at least 1% relative change
                relative_change = abs(clean_mean - faulty_mean) / (abs(clean_mean) + 1e-10)
                min_effect_size = relative_change > 0.01

                # Kill only if permutation test passes AND directional agreement AND min effect
                final_killed = killed and directional_agreement and min_effect_size

                result.update({
                    'killed': final_killed,
                    'p_value': p_value,
                    'mean_diff': details['mean_difference'],
                    'clean_mean': clean_mean,
                    'faulty_mean': faulty_mean,
                    'method': 'exact_permutation_n2',
                    'directional_agreement': directional_agreement,
                    'min_effect_size': min_effect_size,
                    'relative_change': relative_change
                })
            except Exception as exc:
                result['error'] = str(exc)
            return result

        # N < 2: Insufficient samples
        if len(clean_values) < min_samples or len(faulty_values) < min_samples:
            result.update({
                'reason': 'insufficient_samples',
                'clean_samples': len(clean_values),
                'faulty_samples': len(faulty_values),
                'clean_mean': float(np.mean(clean_values)) if clean_values else 0.0,
                'faulty_mean': float(np.mean(faulty_values)) if faulty_values else 0.0
            })
            return result

        # CRITICAL FIX: N=3-5 - Use standard exact permutation test
        try:
            killed, p_value, details = exact_permutation_test(
                clean_values,
                faulty_values
            )
            result.update({
                'killed': killed,
                'p_value': p_value,
                'mean_diff': details['mean_difference'],
                'clean_mean': float(np.mean(clean_values)),
                'faulty_mean': float(np.mean(faulty_values)),
                'method': f'exact_permutation_n{len(clean_values)}'
            })
        except Exception as exc:
            result['error'] = str(exc)

        return result

    def _single_seed_kill_test(self, metric: str, clean_val: float, faulty_val: float) -> bool:
        """
        Threshold-based kill test for single-seed experiments.

        Returns True if the metric shows a significant degradation or anomaly
        that indicates the fault was detected.
        """
        # Metrics where we expect INCREASE in faulty (bad)
        increase_metrics = {
            'mass_pad': 0.1,
            'mass_leak': 0.01,
            'cross_example_attention': 0.01,
            'loss': 0.001,
            'val_loss': 0.001,
            'eval_loss': 0.001,
            'nll': 0.01,
            'logit_nan_ratio': 0.001,
            'logit_inf_ratio': 0.001,
            'attention_entropy': 0.05,
            'pre_softmax_score_var': 0.1,
            'pre_softmax_score_kurt': 0.5,
            'pre_softmax_score_skew': 0.1,
            'runtime_step_time': 0.001,
            'runtime_memory_alloc_mb': 10,
            'logit_entropy': 0.01,
            'logit_kl_uniform': 0.01,
            'ece': 0.001,
            'head_similarity_mean': 0.01,
            # Decoder-specific metrics
            'eval_perplexity': 0.05,
            'val_perplexity': 0.05,
            'attention_mass_future': 0.01,
        }

        # Metrics where we expect DECREASE in faulty (bad)
        decrease_metrics = {
            'accuracy': 0.001,
            'val_accuracy': 0.001,
            'eval_accuracy': 0.001,
            'f1_score': 0.001,
            'val_f1_score': 0.001,
            'grad_norm_total': 0.01,
            'update_ratio_total': 0.01,
            'val_positional_invariance': 0.01,
            'kernel_flash_enabled': 0.5,
            'kernel_mem_efficient_enabled': 0.5,
        }

        # Metrics where ANY change indicates fault
        change_metrics = {
            'kernel_fault_force_unoptimized_active': 0.5,
            'kernel_fault_wrong_layout_active': 0.5,
            'kernel_fault_inconsistent_dropout_active': 0.5,
            'positional_accuracy_delta': 0.01,
            'positional_margin_delta': 0.01,
            'positional_recv_mid_over_early': 0.05,
            'positional_recv_late_over_early': 0.05,
            'embedding_norm_mean': 0.01,
            'embedding_subset_norm_mean': 0.01,
            'h1_delta_norm_mean': 0.01,
            'ffn_delta_mean': 0.01,
            'ffn_var_ratio_mean': 0.01,
            'ffn_out_skew_mean': 0.01,
            'ffn_active_dim_frac_mean': 0.01,
            'residual_cos_mean': 0.01,
            'ln_std_mean': 0.001,
            'ln_mean_abs_mean': 0.001,
        }

        # Check for increase
        if metric in increase_metrics:
            threshold = increase_metrics[metric]
            if faulty_val - clean_val > threshold:
                return True

        # Check for decrease
        if metric in decrease_metrics:
            threshold = decrease_metrics[metric]
            if metric in ['accuracy', 'val_accuracy', 'eval_accuracy', 'f1_score', 'val_f1_score']:
                if clean_val - faulty_val > threshold:
                    return True
            else:
                if clean_val - faulty_val > threshold:
                    return True

        # Check for any change
        if metric in change_metrics:
            threshold = change_metrics[metric]
            if abs(faulty_val - clean_val) > threshold:
                return True

        return False


class MaskingFaultCriteria(KillCriteria):
    """
    K_mask: Kill criteria for attention masking faults (E1).

    Covers both encoder-style (zero mask, inverted mask, broadcast error)
    and decoder-style (causal mask break, over-mask, pad error) faults.
    """

    def __init__(self, fault_name: str):
        super().__init__(fault_name)

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate if decoder masking fault is killed.

        Args:
            clean_metrics: Metrics from clean baseline runs
            faulty_metrics: Metrics from fault-injected runs
            structural_check: Whether structural condition is met

        Returns:
            Dictionary with kill evaluation results
        """
        result = {
            'fault_name': self.fault_name,
            'fault_type': 'masking',
            'structural_check': structural_check,
            'symptom_checks': {},
            'killed': False,
            'reason': ''
        }

        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        # Symptom checks for causal mask violations
        symptoms_detected = []

        # Check 1: Attention mass on future positions
        future_attn_result = self._test_metric(
            'attention_mass_future',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['attention_mass_future'] = future_attn_result
        if future_attn_result.get('killed', False):
            symptoms_detected.append('future_attention_increased')

        # Check 2: Attention mass on PAD tokens
        pad_mass_result = self._test_metric(
            'mass_pad',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['mass_pad'] = pad_mass_result
        if pad_mass_result.get('killed', False):
            symptoms_detected.append('pad_attention_increased')

        # Check 3: Perplexity degradation
        ppl_result = self._test_metric(
            'eval_perplexity',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['perplexity'] = ppl_result
        if ppl_result.get('killed', False):
            symptoms_detected.append('perplexity_degraded')

        # Check 4: Loss increase
        loss_result = self._test_metric(
            'eval_loss',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['eval_loss'] = loss_result
        if loss_result.get('killed', False):
            symptoms_detected.append('loss_increased')

        # Check 5: Positional invariance violations
        pos_inv_result = self._test_metric(
            'val_positional_invariance',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['positional_invariance'] = pos_inv_result
        if pos_inv_result.get('killed', False):
            symptoms_detected.append('positional_invariance_violated')

        # Check 6: Cross-example attention leakage (encoder broadcast faults)
        leak_result = self._test_metric(
            'cross_example_attention',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['cross_example_leak'] = leak_result
        if leak_result.get('killed', False):
            symptoms_detected.append('cross_example_leak_detected')

        # Check 7: Attention entropy shift (over-masking / mask corruption)
        entropy_result = self._test_metric(
            'attention_entropy',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['attention_entropy'] = entropy_result
        if entropy_result.get('killed', False):
            symptoms_detected.append('attention_entropy_shifted')

        # Kill if structural AND (any symptom detected)
        result['symptoms_detected'] = symptoms_detected
        result['killed'] = structural_check and len(symptoms_detected) > 0

        if result['killed']:
            result['reason'] = f"structural_check_passed_and_symptoms_{','.join(symptoms_detected)}"
        else:
            result['reason'] = 'no_symptoms_detected'

        return result


class KVCacheFaultCriteria(KillCriteria):
    """
    K_kv_dec: Kill criteria for KV-cache management faults.

    Detects cache corruption, staleness, or leakage.
    """

    def __init__(self, fault_name: str):
        super().__init__(fault_name)

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate if KV-cache fault is killed.

        Args:
            clean_metrics: Metrics from clean baseline runs
            faulty_metrics: Metrics from fault-injected runs
            structural_check: Whether structural condition is met

        Returns:
            Dictionary with kill evaluation results
        """
        result = {
            'fault_name': self.fault_name,
            'fault_type': 'kv_cache',
            'structural_check': structural_check,
            'symptom_checks': {},
            'killed': False,
            'reason': ''
        }

        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        # Symptom checks for KV-cache faults
        symptoms_detected = []

        # Check 1: Cache correctness (hidden state similarity)
        cache_corr_result = self._test_metric(
            'cache_correctness',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['cache_correctness'] = cache_corr_result

        # For cache correctness, we expect DECREASE in faulty (lower similarity = bad)
        if cache_corr_result.get('killed', False):
            clean_mean = cache_corr_result.get('clean_mean', 1.0)
            faulty_mean = cache_corr_result.get('faulty_mean', 1.0)
            if faulty_mean < clean_mean:
                symptoms_detected.append('cache_similarity_decreased')

        # Check 2: Cache NLL divergence
        cache_div_result = self._test_metric(
            'cache_nll_divergence',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['cache_nll_divergence'] = cache_div_result

        # For cache divergence, we expect INCREASE in faulty (higher divergence = bad)
        if cache_div_result.get('killed', False):
            clean_mean = cache_div_result.get('clean_mean', 0.0)
            faulty_mean = cache_div_result.get('faulty_mean', 0.0)
            if faulty_mean > clean_mean:
                symptoms_detected.append('cache_divergence_increased')

        # Check 3: Perplexity degradation
        ppl_result = self._test_metric(
            'eval_perplexity',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['perplexity'] = ppl_result
        if ppl_result.get('killed', False):
            symptoms_detected.append('perplexity_degraded')

        # Check 4: Generation quality degradation
        # Repetition increase
        rep_result = self._test_metric(
            'repetition_max_run',
            clean_metrics,
            faulty_metrics
        )
        result['symptom_checks']['repetition'] = rep_result
        if rep_result.get('killed', False):
            clean_mean = rep_result.get('clean_mean', 1.0)
            faulty_mean = rep_result.get('faulty_mean', 1.0)
            if faulty_mean > clean_mean:
                symptoms_detected.append('repetition_increased')

        # Kill if structural AND (cache symptoms OR generation degradation)
        result['symptoms_detected'] = symptoms_detected
        result['killed'] = structural_check and len(symptoms_detected) > 0

        if result['killed']:
            result['reason'] = f"structural_check_passed_and_symptoms_{','.join(symptoms_detected)}"
        else:
            result['reason'] = 'no_symptoms_detected'

        return result


class QKVFaultCriteria(KillCriteria):
    """Kill criteria for QKV faults in decoder models."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'zero_query': ['grad_norm_total', 'eval_loss'],
            'zero_key': ['grad_norm_total', 'attention_entropy', 'eval_loss'],
            'zero_value': ['grad_norm_total', 'eval_loss'],
            'swapped_qk': ['attention_entropy', 'eval_loss'],
            'tie_heads': ['head_similarity_mean', 'eval_loss'],
            'wrong_head_dim': ['grad_norm_total', 'eval_loss'],
            'freeze_qkv': ['update_ratio_total', 'grad_norm_total'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        """Evaluate if decoder QKV fault is killed."""
        result = {
            'fault_name': self.fault_name,
            'fault_type': 'decoder_qkv',
            'structural_check': structural_check,
            'symptom_checks': {},
            'killed': False
        }

        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        metric_names = self.metric_map.get(self.fault_name, ['eval_loss', 'grad_norm_total'])
        symptoms_detected = []
        for metric in metric_names:
            metric_result = self._test_metric(metric, clean_metrics, faulty_metrics)
            result['symptom_checks'][metric] = metric_result
            if metric_result.get('killed', False):
                symptoms_detected.append(metric)

        if self.fault_name == 'tie_heads':
            overall_killed = result['symptom_checks'].get('head_similarity_mean', {}).get('killed', False) \
                or result['symptom_checks'].get('eval_loss', {}).get('killed', False)
        elif self.fault_name == 'freeze_qkv':
            overall_killed = result['symptom_checks'].get('update_ratio_total', {}).get('killed', False) \
                or result['symptom_checks'].get('grad_norm_total', {}).get('killed', False)
        else:
            overall_killed = any(result['symptom_checks'][metric].get('killed', False) for metric in metric_names)

        result['symptoms_detected'] = symptoms_detected
        result['killed'] = structural_check and overall_killed
        result['reason'] = 'detected' if result['killed'] else 'no_symptoms'

        return result


class PositionalFaultCriteria(KillCriteria):
    """Kill criteria for positional faults in decoder models."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        common_metrics = [
            'positional_accuracy_delta',
            'positional_margin_delta',
            'val_positional_invariance',
            'positional_recv_mid_over_early',
            'positional_recv_late_over_early',
        ]
        self.metric_map = {
            'off_by_one': common_metrics,
            'truncate_positions': common_metrics,
            'double_position': common_metrics,
            'missing_positional': common_metrics,
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        """Evaluate if decoder positional fault is killed."""
        result = {
            'fault_name': self.fault_name,
            'fault_type': 'decoder_positional',
            'structural_check': structural_check,
            'symptom_checks': {},
            'killed': False
        }

        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        metric_names = self.metric_map.get(self.fault_name, [
            'positional_accuracy_delta',
            'val_positional_invariance'
        ])

        symptoms_detected = []
        for metric in metric_names:
            metric_result = self._test_metric(metric, clean_metrics, faulty_metrics)
            result['symptom_checks'][metric] = metric_result
            if metric_result.get('killed', False):
                symptoms_detected.append(metric)

        result['symptoms_detected'] = symptoms_detected
        result['killed'] = structural_check and len(symptoms_detected) > 0
        result['reason'] = 'detected' if result['killed'] else 'no_symptoms'

        return result


class FFNFaultCriteria(KillCriteria):
    """Kill criteria for FFN faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'ffn_weight_scaling': ['ffn_delta_mean', 'ffn_var_ratio_mean', 'residual_cos_mean'],
            'ffn_neuron_drop': ['ffn_active_dim_frac_mean', 'ffn_var_ratio_mean', 'ffn_delta_mean'],
            'activation_distortion': ['ffn_out_skew_mean', 'ffn_delta_mean', 'logit_entropy'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['ffn_delta_mean', 'residual_cos_mean'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'ffn',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class LayerNormFaultCriteria(KillCriteria):
    """Kill criteria for LayerNorm faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'ln_gamma_fault': ['ln_std_mean', 'residual_cos_mean'],
            'ln_beta_fault': ['ln_mean_abs_mean', 'ln_std_mean'],
            'ln_stats_fault': ['ln_std_mean', 'ffn_out_skew_mean'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['ln_std_mean'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'layernorm',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class ResidualFaultCriteria(KillCriteria):
    """Kill criteria for residual path faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'residual_drop': ['residual_cos_mean', 'ffn_delta_mean'],
            'residual_scale': ['residual_cos_mean', 'ffn_delta_mean'],
            'residual_noise': ['ffn_delta_mean', 'logit_entropy'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['residual_cos_mean'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'residual',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class EmbeddingFaultCriteria(KillCriteria):
    """Kill criteria for embedding faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'embedding_zero': ['embedding_norm_mean', 'embedding_subset_norm_mean', 'h1_delta_norm_mean'],
            'embedding_swap': ['embedding_norm_mean', 'h1_delta_norm_mean'],
            'type_embedding_drop': ['h1_delta_norm_mean', 'embedding_norm_mean'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['embedding_norm_mean'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'embedding',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class OutputFaultCriteria(KillCriteria):
    """Kill criteria for output projection faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'out_scale': ['logit_entropy', 'logit_kl_uniform', 'nll'],
            'out_row_drop': ['logit_entropy', 'logit_kl_uniform'],
            'out_noise': ['logit_entropy', 'logit_kl_uniform', 'ece'],
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['logit_entropy'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'output',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class ScoreFaultCriteria(KillCriteria):
    """Kill criteria for score/softmax faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'missing_scaling': ['attention_entropy', 'pre_softmax_score_var', 'pre_softmax_score_mean', 'nll'],
            'wrong_scaling_factor': ['pre_softmax_score_var', 'pre_softmax_score_kurt', 'attention_entropy'],
            'misplaced_dropout': ['pre_softmax_score_var', 'attention_entropy', 'eval_loss'],
            'unsafe_type_cast': ['pre_softmax_score_var', 'pre_softmax_score_skew', 'logit_nan_ratio', 'logit_inf_ratio']
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['attention_entropy', 'pre_softmax_score_var'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'score',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class KernelFaultCriteria(KillCriteria):
    """Kill criteria for kernel/runtime faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'force_unoptimized': ['runtime_step_time', 'kernel_flash_enabled', 'kernel_mem_efficient_enabled'],
            'wrong_layout': ['runtime_step_time', 'kernel_math_enabled', 'runtime_memory_alloc_mb'],
            'inconsistent_dropout': ['runtime_step_time', 'eval_loss', 'eval_perplexity']
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['runtime_step_time'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'kernel',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class VariantFaultCriteria(KillCriteria):
    """Kill criteria for variant faults (architecture-agnostic)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'wrong_variant': ['eval_loss', 'eval_perplexity'],
            'causal_in_noncausal': ['eval_loss', 'eval_perplexity']
        }

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        metric_names = self.metric_map.get(self.fault_name, ['eval_loss'])
        results = {}
        symptoms_detected = []

        for metric in metric_names:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'variant',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


class GenericKillCriteria(KillCriteria):
    """Generic kill criteria for any fault type (architecture-agnostic)."""

    def __init__(self, fault_name: str, primary_metrics: List[str]):
        super().__init__(fault_name)
        self.primary_metrics = primary_metrics

    def evaluate(
        self,
        clean_metrics: Dict[str, List[float]],
        faulty_metrics: Dict[str, List[float]],
        structural_check: bool = True
    ) -> Dict[str, Any]:
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}

        results = {}
        symptoms_detected = []

        for metric in self.primary_metrics:
            result = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = result
            if result.get('killed', False):
                symptoms_detected.append(metric)

        overall_killed = len(symptoms_detected) > 0

        return {
            'fault_name': self.fault_name,
            'fault_type': 'generic',
            'structural_check': structural_check,
            'symptom_checks': results,
            'symptoms_detected': symptoms_detected,
            'killed': overall_killed,
            'reason': 'detected' if overall_killed else 'no_symptoms'
        }


def create_kill_criteria(
    fault_type: str,
    fault_name: str,
    **kwargs
) -> KillCriteria:
    """
    Factory function to create decoder kill criteria.

    Args:
        fault_type: Type of decoder fault
        fault_name: Specific fault name
        **kwargs: Additional parameters

    Returns:
        Decoder kill criteria instance
    """
    # Decoder-specific faults
    if fault_type in ('masking', 'decoder_masking'):
        return MaskingFaultCriteria(fault_name)
    elif fault_type == 'kv_cache':
        return KVCacheFaultCriteria(fault_name)

    # Attention-related faults (work for both encoder and decoder)
    elif fault_type in ('decoder_qkv', 'qkv'):
        return QKVFaultCriteria(fault_name)
    elif fault_type in ('decoder_positional', 'positional'):
        return PositionalFaultCriteria(fault_name)
    elif fault_type == 'score':
        return ScoreFaultCriteria(fault_name)
    elif fault_type == 'kernel':
        return KernelFaultCriteria(fault_name)
    elif fault_type == 'variant':
        return VariantFaultCriteria(fault_name)

    # Architecture-agnostic faults
    elif fault_type == 'ffn':
        return FFNFaultCriteria(fault_name)
    elif fault_type == 'layernorm':
        return LayerNormFaultCriteria(fault_name)
    elif fault_type == 'residual':
        return ResidualFaultCriteria(fault_name)
    elif fault_type == 'embedding':
        return EmbeddingFaultCriteria(fault_name)
    elif fault_type == 'output':
        return OutputFaultCriteria(fault_name)

    # Generic fallback
    elif fault_type == 'generic':
        primary_metrics = kwargs.get('primary_metrics', ['eval_loss', 'eval_perplexity'])
        return GenericKillCriteria(fault_name, primary_metrics)

    else:
        raise ValueError(f"Unknown decoder fault type: {fault_type}")
