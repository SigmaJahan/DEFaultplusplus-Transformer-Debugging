"""Encoder-Specific Kill Criteria.

Per-category kill criteria for encoder fault categories.
K_task uses accuracy drop threshold (not perplexity like decoder).
"""

import logging
from typing import Dict, List, Any

import numpy as np

from src.kill_functions.statistical_tests import exact_permutation_test


class KillCriteria:
    """Base class for encoder kill criteria evaluation."""

    def __init__(self, fault_name: str):
        self.fault_name = fault_name
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _ensure_list(values: Any) -> List[float]:
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
        result: Dict[str, Any] = {'killed': False, 'p_value': 1.0}

        if metric not in clean_metrics or metric not in faulty_metrics:
            result['reason'] = 'metric_not_available'
            return result

        clean_values = clean_metrics[metric]
        faulty_values = faulty_metrics[metric]

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

        if len(clean_values) == 2 and len(faulty_values) == 2:
            try:
                killed, p_value, details = exact_permutation_test(clean_values, faulty_values)
                clean_mean = float(np.mean(clean_values))
                faulty_mean = float(np.mean(faulty_values))
                diff1 = clean_values[0] - faulty_values[0]
                diff2 = clean_values[1] - faulty_values[1]
                directional_agreement = (diff1 * diff2) > 0
                relative_change = abs(clean_mean - faulty_mean) / (abs(clean_mean) + 1e-10)
                min_effect_size = relative_change > 0.01
                final_killed = killed and directional_agreement and min_effect_size
                result.update({
                    'killed': final_killed, 'p_value': p_value,
                    'mean_diff': details['mean_difference'],
                    'clean_mean': clean_mean, 'faulty_mean': faulty_mean,
                    'method': 'exact_permutation_n2',
                    'directional_agreement': directional_agreement,
                    'min_effect_size': min_effect_size,
                    'relative_change': relative_change
                })
            except Exception as exc:
                result['error'] = str(exc)
            return result

        if len(clean_values) < min_samples or len(faulty_values) < min_samples:
            result.update({
                'reason': 'insufficient_samples',
                'clean_samples': len(clean_values),
                'faulty_samples': len(faulty_values),
                'clean_mean': float(np.mean(clean_values)) if clean_values else 0.0,
                'faulty_mean': float(np.mean(faulty_values)) if faulty_values else 0.0
            })
            return result

        try:
            killed, p_value, details = exact_permutation_test(clean_values, faulty_values)
            result.update({
                'killed': killed, 'p_value': p_value,
                'mean_diff': details['mean_difference'],
                'clean_mean': float(np.mean(clean_values)),
                'faulty_mean': float(np.mean(faulty_values)),
                'method': f'exact_permutation_n{len(clean_values)}'
            })
        except Exception as exc:
            result['error'] = str(exc)
        return result

    def _single_seed_kill_test(self, metric: str, clean_val: float, faulty_val: float) -> bool:
        increase_metrics = {
            'mass_pad': 0.1, 'mass_leak': 0.01, 'cross_example_attention': 0.01,
            'loss': 0.001, 'val_loss': 0.001,
            'logit_nan_ratio': 0.001, 'logit_inf_ratio': 0.001,
            'attention_entropy': 0.05,
            'pre_softmax_score_var': 0.1, 'pre_softmax_score_kurt': 0.5,
            'pre_softmax_score_skew': 0.1,
            'runtime_step_time': 0.001, 'runtime_memory_alloc_mb': 10,
            'logit_entropy': 0.01, 'logit_kl_uniform': 0.01, 'ece': 0.001,
            'head_similarity_mean': 0.01,
        }
        decrease_metrics = {
            'accuracy': 0.001, 'val_accuracy': 0.001, 'f1_score': 0.001,
            'val_f1_score': 0.001,
            'grad_norm_total': 0.01, 'update_ratio_total': 0.01,
            'val_positional_invariance': 0.01,
            'kernel_flash_enabled': 0.5, 'kernel_mem_efficient_enabled': 0.5,
        }
        change_metrics = {
            'kernel_fault_force_unoptimized_active': 0.5,
            'kernel_fault_wrong_layout_active': 0.5,
            'kernel_fault_inconsistent_dropout_active': 0.5,
            'positional_accuracy_delta': 0.01, 'positional_margin_delta': 0.01,
            'positional_recv_mid_over_early': 0.05, 'positional_recv_late_over_early': 0.05,
            'embedding_norm_mean': 0.01, 'embedding_subset_norm_mean': 0.01,
            'h1_delta_norm_mean': 0.01,
            'ffn_delta_mean': 0.01, 'ffn_var_ratio_mean': 0.01,
            'ffn_out_skew_mean': 0.01, 'ffn_active_dim_frac_mean': 0.01,
            'residual_cos_mean': 0.01,
            'ln_std_mean': 0.001, 'ln_mean_abs_mean': 0.001,
        }
        if metric in increase_metrics:
            if faulty_val - clean_val > increase_metrics[metric]:
                return True
        if metric in decrease_metrics:
            if clean_val - faulty_val > decrease_metrics[metric]:
                return True
        if metric in change_metrics:
            if abs(faulty_val - clean_val) > change_metrics[metric]:
                return True
        return False


class MaskingFaultCriteria(KillCriteria):
    """K_mask: Kill criteria for attention masking faults."""

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        result = {
            'fault_name': self.fault_name, 'fault_type': 'masking',
            'structural_check': structural_check, 'symptom_checks': {}, 'killed': False,
        }
        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        symptoms = []
        for metric in ('mass_pad', 'cross_example_attention', 'attention_entropy',
                       'val_loss', 'val_accuracy'):
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            result['symptom_checks'][metric] = r
            if r.get('killed', False):
                symptoms.append(metric)

        result['symptoms_detected'] = symptoms
        result['killed'] = structural_check and len(symptoms) > 0
        result['reason'] = 'detected' if result['killed'] else 'no_symptoms'
        return result


class QKVFaultCriteria(KillCriteria):
    """Kill criteria for QKV faults in encoder models."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'zero_query': ['grad_norm_total', 'val_loss', 'val_accuracy'],
            'zero_key': ['grad_norm_total', 'attention_entropy', 'val_loss'],
            'zero_value': ['grad_norm_total', 'val_loss', 'val_accuracy'],
            'swapped_qk': ['attention_entropy', 'val_loss'],
            'tie_heads': ['head_similarity_mean', 'val_loss'],
            'wrong_head_dim': ['grad_norm_total', 'val_loss'],
            'freeze_qkv': ['update_ratio_total', 'grad_norm_total'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        result = {
            'fault_name': self.fault_name, 'fault_type': 'qkv',
            'structural_check': structural_check, 'symptom_checks': {}, 'killed': False,
        }
        if not structural_check:
            result['reason'] = 'structural_check_failed'
            return result

        metric_names = self.metric_map.get(self.fault_name, ['val_loss', 'grad_norm_total'])
        symptoms = []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            result['symptom_checks'][metric] = r
            if r.get('killed', False):
                symptoms.append(metric)

        result['symptoms_detected'] = symptoms
        result['killed'] = structural_check and len(symptoms) > 0
        result['reason'] = 'detected' if result['killed'] else 'no_symptoms'
        return result


class ScoreFaultCriteria(KillCriteria):
    """Kill criteria for score/softmax faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'missing_scaling': ['attention_entropy', 'pre_softmax_score_var', 'pre_softmax_score_mean'],
            'wrong_scaling_factor': ['pre_softmax_score_var', 'pre_softmax_score_kurt', 'attention_entropy'],
            'misplaced_dropout': ['pre_softmax_score_var', 'attention_entropy', 'val_loss'],
            'unsafe_type_cast': ['pre_softmax_score_var', 'pre_softmax_score_skew', 'logit_nan_ratio', 'logit_inf_ratio'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['attention_entropy', 'pre_softmax_score_var'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'score',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class PositionalFaultCriteria(KillCriteria):
    """Kill criteria for positional faults in encoder models."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metrics = [
            'positional_accuracy_delta', 'positional_margin_delta',
            'val_positional_invariance',
            'positional_recv_mid_over_early', 'positional_recv_late_over_early',
        ]

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        results, symptoms = {}, []
        for metric in self.metrics:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'positional',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class KernelFaultCriteria(KillCriteria):
    """Kill criteria for kernel/runtime faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'force_unoptimized': ['runtime_step_time', 'kernel_flash_enabled', 'kernel_mem_efficient_enabled'],
            'wrong_layout': ['runtime_step_time', 'kernel_math_enabled', 'runtime_memory_alloc_mb'],
            'inconsistent_dropout': ['runtime_step_time', 'val_loss'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['runtime_step_time'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'kernel',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class VariantFaultCriteria(KillCriteria):
    """Kill criteria for variant faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'wrong_variant': ['val_loss', 'val_accuracy'],
            'causal_in_noncausal': ['val_loss', 'val_accuracy'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['val_loss'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'variant',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class FFNFaultCriteria(KillCriteria):
    """Kill criteria for FFN faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'ffn_weight_scaling': ['ffn_delta_mean', 'ffn_var_ratio_mean', 'residual_cos_mean'],
            'ffn_neuron_drop': ['ffn_active_dim_frac_mean', 'ffn_var_ratio_mean', 'ffn_delta_mean'],
            'activation_distortion': ['ffn_out_skew_mean', 'ffn_delta_mean', 'logit_entropy'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['ffn_delta_mean', 'residual_cos_mean'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'ffn',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class LayerNormFaultCriteria(KillCriteria):
    """Kill criteria for LayerNorm faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'ln_gamma_fault': ['ln_std_mean', 'residual_cos_mean'],
            'ln_beta_fault': ['ln_mean_abs_mean', 'ln_std_mean'],
            'ln_stats_fault': ['ln_std_mean', 'ffn_out_skew_mean'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['ln_std_mean'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'layernorm',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class ResidualFaultCriteria(KillCriteria):
    """Kill criteria for residual path faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'residual_drop': ['residual_cos_mean', 'ffn_delta_mean'],
            'residual_scale': ['residual_cos_mean', 'ffn_delta_mean'],
            'residual_noise': ['ffn_delta_mean', 'logit_entropy'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['residual_cos_mean'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'residual',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class EmbeddingFaultCriteria(KillCriteria):
    """Kill criteria for embedding faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'embedding_zero': ['embedding_norm_mean', 'embedding_subset_norm_mean', 'h1_delta_norm_mean'],
            'embedding_swap': ['embedding_norm_mean', 'h1_delta_norm_mean'],
            'type_embedding_drop': ['h1_delta_norm_mean', 'embedding_norm_mean'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['embedding_norm_mean'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'embedding',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class OutputFaultCriteria(KillCriteria):
    """Kill criteria for output projection faults."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metric_map = {
            'out_scale': ['logit_entropy', 'logit_kl_uniform', 'val_loss'],
            'out_row_drop': ['logit_entropy', 'logit_kl_uniform'],
            'out_noise': ['logit_entropy', 'logit_kl_uniform', 'ece'],
        }

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        metric_names = self.metric_map.get(self.fault_name, ['logit_entropy'])
        results, symptoms = {}, []
        for metric in metric_names:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'output',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class PoolerFaultCriteria(KillCriteria):
    """Kill criteria for pooler faults (encoder-specific)."""

    def __init__(self, fault_name: str):
        super().__init__(fault_name)
        self.metrics = ['val_accuracy', 'val_loss', 'cls_accuracy', 'cls_f1']

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        results, symptoms = {}, []
        for metric in self.metrics:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'pooler',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


class GenericKillCriteria(KillCriteria):
    """Generic kill criteria for any fault type."""

    def __init__(self, fault_name: str, primary_metrics: List[str]):
        super().__init__(fault_name)
        self.primary_metrics = primary_metrics

    def evaluate(self, clean_metrics, faulty_metrics, structural_check=True, **kwargs):
        if not structural_check:
            return {'killed': False, 'reason': 'structural_check_failed'}
        results, symptoms = {}, []
        for metric in self.primary_metrics:
            r = self._test_metric(metric, clean_metrics, faulty_metrics)
            results[metric] = r
            if r.get('killed', False):
                symptoms.append(metric)
        return {
            'fault_name': self.fault_name, 'fault_type': 'generic',
            'structural_check': structural_check, 'symptom_checks': results,
            'symptoms_detected': symptoms, 'killed': len(symptoms) > 0,
            'reason': 'detected' if symptoms else 'no_symptoms'
        }


def create_kill_criteria(fault_type: str, fault_name: str, **kwargs) -> KillCriteria:
    """Factory function to create encoder kill criteria."""
    if fault_type == 'masking':
        return MaskingFaultCriteria(fault_name)
    elif fault_type == 'qkv':
        return QKVFaultCriteria(fault_name)
    elif fault_type == 'score':
        return ScoreFaultCriteria(fault_name)
    elif fault_type == 'positional':
        return PositionalFaultCriteria(fault_name)
    elif fault_type == 'kernel':
        return KernelFaultCriteria(fault_name)
    elif fault_type == 'variant':
        return VariantFaultCriteria(fault_name)
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
    elif fault_type == 'pooler':
        return PoolerFaultCriteria(fault_name)
    elif fault_type == 'generic':
        primary_metrics = kwargs.get('primary_metrics', ['val_loss', 'val_accuracy'])
        return GenericKillCriteria(fault_name, primary_metrics)
    else:
        raise ValueError(f"Unknown encoder fault type: {fault_type}")
