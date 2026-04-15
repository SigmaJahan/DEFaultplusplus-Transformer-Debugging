"""Metric collector for ABNN Encoder Fault Injection Dataset."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.base_metrics import BaseMetrics
from src.metrics.running_metrics import RunningMetrics
from src.metrics.statistics import EpochAggregator, compute_window_features


class RepresentationDriftTracker:
    def __init__(self, target_epochs: Optional[List[int]] = None, sample_tokens: int = 256):
        self.target_epochs = set(target_epochs or [1, 4, 7, 10])
        self.sample_tokens = sample_tokens
        self.epoch_summaries: Dict[int, Dict[str, Any]] = {}
        self.current_epoch: Optional[int] = None

    def start_epoch(self, epoch_index: int):
        epoch_number = epoch_index + 1
        self.current_epoch = epoch_number
        if epoch_number not in self.target_epochs:
            return
        self.epoch_summaries.setdefault(epoch_number, {
            'sums': defaultdict(lambda: None),
            'counts': defaultdict(int),
            'samples': defaultdict(list),
            'tokens': 0
        })

    def end_epoch(self):
        self.current_epoch = None

    def capture(self, hidden_states: Optional[Any], attention_mask: Optional[torch.Tensor]):
        epoch_number = self.current_epoch
        if epoch_number not in self.target_epochs:
            return
        if hidden_states is None or attention_mask is None:
            return

        data = self.epoch_summaries[epoch_number]
        if data['tokens'] >= self.sample_tokens:
            return

        batch, seq_len = attention_mask.shape
        mask = attention_mask.reshape(-1)
        valid_indices = torch.nonzero(mask > 0, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return

        remaining = self.sample_tokens - data['tokens']
        take = min(int(valid_indices.numel()), remaining)
        valid_indices = valid_indices[:take]

        for layer_idx, layer_hidden in enumerate(hidden_states[1:]):
            flat = layer_hidden.reshape(batch * seq_len, -1)
            selected = flat.index_select(0, valid_indices)
            if selected.numel() == 0:
                continue
            vec_sum = selected.sum(dim=0).detach().cpu()
            existing = data['sums'][layer_idx]
            if existing is None:
                data['sums'][layer_idx] = vec_sum
            else:
                data['sums'][layer_idx] = existing + vec_sum
            data['counts'][layer_idx] += selected.size(0)
            data['samples'][layer_idx].append(selected.detach().cpu())

        data['tokens'] += take

    @staticmethod
    def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        hsic_xy = (X.T @ Y).pow(2).sum()
        hsic_xx = (X.T @ X).pow(2).sum()
        hsic_yy = (Y.T @ Y).pow(2).sum()
        denom = (hsic_xx * hsic_yy).sqrt()
        if denom < 1e-12:
            return 0.0
        return float((hsic_xy / denom).item())

    def get_features(self, num_layers: int) -> Dict[str, float]:
        features: Dict[str, float] = {}
        normalized_vectors: Dict[int, Dict[int, torch.Tensor]] = {}
        sample_matrices: Dict[int, Dict[int, torch.Tensor]] = {}

        for epoch, data in self.epoch_summaries.items():
            epoch_vectors: Dict[int, torch.Tensor] = {}
            epoch_samples: Dict[int, torch.Tensor] = {}
            for layer_idx in range(num_layers):
                sum_vec = data['sums'].get(layer_idx)
                count = data['counts'].get(layer_idx, 0)
                if sum_vec is None or count == 0:
                    continue
                mean_vec = sum_vec / count
                norm = torch.norm(mean_vec)
                if norm > 0:
                    epoch_vectors[layer_idx] = mean_vec / norm
                chunks = data['samples'].get(layer_idx, [])
                if chunks:
                    epoch_samples[layer_idx] = torch.cat(chunks, dim=0).float()
            if epoch_vectors:
                normalized_vectors[epoch] = epoch_vectors
            if epoch_samples:
                sample_matrices[epoch] = epoch_samples

        comparisons = [(1, 4), (4, 7), (7, 10), (1, 10)]
        for start, end in comparisons:
            if start in normalized_vectors and end in normalized_vectors:
                for layer_idx in range(num_layers):
                    vec_a = normalized_vectors[start].get(layer_idx)
                    vec_b = normalized_vectors[end].get(layer_idx)
                    if vec_a is None or vec_b is None:
                        continue
                    cos_sim = float(F.cosine_similarity(
                        vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1
                    ).item())
                    features[f'repr_layer{layer_idx}_cos_{start}_{end}'] = cos_sim
                    features[f'repr_layer{layer_idx}_drift_{start}_{end}'] = 1.0 - cos_sim

            if start in sample_matrices and end in sample_matrices:
                for layer_idx in range(num_layers):
                    X = sample_matrices[start].get(layer_idx)
                    Y = sample_matrices[end].get(layer_idx)
                    if X is None or Y is None:
                        continue
                    n = min(X.size(0), Y.size(0))
                    if n < 2:
                        continue
                    cka = self._linear_cka(X[:n], Y[:n])
                    features[f'repr_layer{layer_idx}_cka_{start}_{end}'] = cka

        for data in self.epoch_summaries.values():
            data['samples'] = defaultdict(list)

        return features


_MC_LOGGER = logging.getLogger("metric_collector")


def _safe_compute(func, *args, label: str = "", **kwargs) -> Dict[str, float]:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        _MC_LOGGER.warning("Metric computation '%s' failed: %s", label, exc)
        return {}


class MetricCollector:
    def __init__(
        self,
        device: torch.device,
        collect_per_batch: bool = False,
        collect_per_epoch: bool = True,
        collect_attention: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.device = device
        self.collect_per_batch = collect_per_batch
        self.collect_per_epoch = collect_per_epoch
        self.collect_attention = collect_attention
        self.config = config or {}

        self.metrics_calculator = BaseMetrics(device, config=self.config)
        self.epoch_aggregator = EpochAggregator()
        self.running_metrics = RunningMetrics(window_size=self.config.get('gradient_window', 20))
        self.representation_tracker = RepresentationDriftTracker(
            target_epochs=self.config.get('representation_epochs', [1, 4, 7, 10]),
            sample_tokens=self.config.get('representation_tokens', 256)
        )

        self.batch_metrics_history: List[Dict[str, float]] = []
        self.epoch_metrics_history: List[Dict[str, float]] = []
        self.validation_history: List[Dict[str, float]] = []
        self.validation_metric_history: Dict[str, List] = defaultdict(list)

        self.batch_counter = 0
        self.active_epoch: Optional[int] = None
        self.position_cutoff = self.config.get('position_cutoff', 64)
        self.num_layers = self.config.get(
            'num_hidden_layers',
            self.config.get('model_config', {}).get('num_hidden_layers', 6)
        )

    def _start_epoch_if_needed(self, epoch: int):
        if self.active_epoch == epoch:
            return
        self.active_epoch = epoch
        self.representation_tracker.start_epoch(epoch)
        self.running_metrics.reset()

    def _attention_layers_to_sample(self, total_layers: int) -> List[int]:
        if total_layers <= 0:
            return []
        if total_layers <= 3:
            return list(range(total_layers))
        return [0, total_layers // 2, total_layers - 1]

    def collect_batch_metrics(
        self,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        outputs: Any,
        labels: torch.Tensor,
        batch_idx: int,
        epoch: int,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        step_time: Optional[float] = None,
        data_time: Optional[float] = None
    ) -> Dict[str, float]:
        self.batch_counter += 1
        self._start_epoch_if_needed(epoch)
        batch = batch or {}

        predictions = outputs.logits if hasattr(outputs, 'logits') else outputs
        attentions = getattr(outputs, 'attentions', None)
        hidden_states = getattr(outputs, 'hidden_states', None)
        attention_mask = batch.get('attention_mask')
        input_ids = batch.get('input_ids')

        metrics: Dict[str, float] = {}

        loss_value = None
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss_value = outputs.loss
        elif loss is not None:
            loss_value = loss

        metrics.update(_safe_compute(
            self.metrics_calculator.compute_training_metrics,
            loss_value, model, optimizer,
            step_time=step_time, data_time=data_time,
            label="training_metrics"
        ))

        metrics.update(_safe_compute(
            self.metrics_calculator.compute_performance_metrics,
            predictions, labels, label="performance_metrics"
        ))

        if self.batch_counter % max(1, self.config.get('activation_interval', 10)) == 0 and hidden_states is not None:
            last_hidden = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
            metrics.update(_safe_compute(
                self.metrics_calculator.compute_statistical_metrics,
                model, last_hidden, label="statistical_metrics"
            ))

        grad_metrics = _safe_compute(
            self.metrics_calculator.compute_gradient_metrics, model, label="gradient_metrics"
        )
        metrics.update(grad_metrics)

        update_ratio_metrics = _safe_compute(
            self.metrics_calculator.compute_update_ratio_metrics, model, label="update_ratio_metrics"
        )
        metrics.update(update_ratio_metrics)

        metrics.update(_safe_compute(
            self.metrics_calculator.compute_positional_performance_metrics,
            model, batch, label="positional_metrics"
        ))

        metrics.update(_safe_compute(
            self.metrics_calculator.compute_structural_metrics,
            hidden_states=hidden_states, model=model,
            attention_mask=attention_mask, input_ids=input_ids,
            logits=predictions, label="structural_metrics"
        ))

        for key, value in grad_metrics.items():
            if not key.startswith('grad_norm_'):
                continue
            self.running_metrics.update(key, value)
            metrics[f'{key}_window_var'] = self.running_metrics.get_variance(key)
            metrics[f'{key}_gns'] = self.running_metrics.get_noise_scale(key)

        metrics['gradient_variance'] = self.running_metrics.get_variance('grad_norm_total')
        metrics['gradient_noise_scale'] = self.running_metrics.get_noise_scale('grad_norm_total')

        if hidden_states is not None and attention_mask is not None:
            self.representation_tracker.capture(hidden_states, attention_mask)

        attention_alias_accumulator = {
            'mass_pad': [],
            'mass_leak': [],
            'cross_example_attention': [],
            'attention_entropy': [],
            'pre_softmax_score_mean': [],
            'pre_softmax_score_var': [],
            'pre_softmax_score_skew': [],
            'pre_softmax_score_kurt': [],
            'head_similarity_mean': [],
            'head_similarity_max': [],
            'attention_rank_mean': [],
        }

        if self.collect_attention and attentions is not None:
            sample_layers = self._attention_layers_to_sample(len(attentions))
            for layer_idx in sample_layers:
                try:
                    layer_input = None
                    if hidden_states is not None and len(hidden_states) > layer_idx:
                        layer_input = hidden_states[layer_idx]
                    attn_metrics = self.metrics_calculator.compute_attention_metrics(
                        attentions[layer_idx],
                        model=model,
                        layer_idx=layer_idx,
                        layer_input=layer_input,
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                        position_cutoff=self.position_cutoff
                    )
                except Exception as exc:
                    _MC_LOGGER.warning("Attention metrics failed for layer %d: %s", layer_idx, exc)
                    attn_metrics = {}
                for key, value in attn_metrics.items():
                    metrics[f'L{layer_idx}_{key}'] = value
                for alias_key in attention_alias_accumulator:
                    metric_key = {
                        'mass_pad': 'attention_mass_pad_mean',
                        'mass_leak': 'attention_mass_leak',
                        'cross_example_attention': 'attention_cross_example_leak',
                    }.get(alias_key, alias_key)
                    if metric_key in attn_metrics:
                        attention_alias_accumulator[alias_key].append(attn_metrics[metric_key])

        metrics.update(self._summarize_attention_aliases(attention_alias_accumulator))

        metrics['batch_idx'] = batch_idx
        metrics['epoch'] = epoch

        self.epoch_aggregator.update(metrics)

        if self.collect_per_batch:
            self.batch_metrics_history.append(metrics.copy())

        return metrics

    def finalize_epoch(self, epoch: int) -> Dict[str, float]:
        epoch_metrics = self.epoch_aggregator.finalize_epoch(epoch)
        epoch_metrics['epoch'] = epoch
        if self.collect_per_epoch:
            self.epoch_metrics_history.append(epoch_metrics.copy())
        self.active_epoch = None
        self.representation_tracker.end_epoch()
        return epoch_metrics

    def record_validation_metrics(self, epoch: int, metrics: Dict[str, float]):
        prefixed = {f'val_{key}': value for key, value in metrics.items()}
        prefixed['epoch'] = epoch
        self.validation_history.append(prefixed)
        for key, value in prefixed.items():
            if key == 'epoch':
                continue
            self.validation_metric_history[key].append((epoch + 1, value))

    def get_epoch_history(self) -> List[Dict[str, float]]:
        return self.epoch_metrics_history

    def get_batch_history(self) -> List[Dict[str, float]]:
        return self.batch_metrics_history

    def get_validation_history(self) -> List[Dict[str, float]]:
        return list(self.validation_history)

    @staticmethod
    def _aggregate_layer_metrics(
        epoch_data: Dict[str, float], num_layers: int
    ) -> Dict[str, float]:
        layer_metrics: Dict[str, Dict[int, float]] = defaultdict(dict)
        patterns = [
            re.compile(r'^L(\d+)_(.+?)_mean$'),
            re.compile(r'^(.+?)_l(\d+)_mean$'),
        ]
        for key, val in epoch_data.items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            for pat in patterns:
                m = pat.match(key)
                if m:
                    groups = m.groups()
                    if groups[0].isdigit():
                        layer_idx, metric_name = int(groups[0]), groups[1]
                    else:
                        metric_name, layer_idx = groups[0], int(groups[1])
                    layer_metrics[metric_name][layer_idx] = val
                    break

        if num_layers <= 0:
            return {}

        base = num_layers // 3
        remainder = num_layers % 3
        sizes = [max(1, base)] * 3
        for i in range(remainder):
            sizes[i] += 1
        early_end = sizes[0]
        mid_end = sizes[0] + sizes[1]

        agg: Dict[str, float] = {}
        for metric_name, layer_vals in layer_metrics.items():
            if not layer_vals:
                continue
            early = [layer_vals[i] for i in range(0, early_end) if i in layer_vals]
            mid = [layer_vals[i] for i in range(early_end, mid_end) if i in layer_vals]
            final_layer = max(layer_vals.keys())
            final_val = layer_vals[final_layer]
            if early:
                arr = np.array(early)
                agg[f'{metric_name}_layer_early_mean'] = float(np.mean(arr))
                agg[f'{metric_name}_layer_early_std'] = float(np.std(arr))
            if mid:
                arr = np.array(mid)
                agg[f'{metric_name}_layer_mid_mean'] = float(np.mean(arr))
                agg[f'{metric_name}_layer_mid_std'] = float(np.std(arr))
            agg[f'{metric_name}_layer_final'] = float(final_val)

        return agg

    def get_final_metrics(self) -> Dict[str, float]:
        if not self.epoch_metrics_history:
            return {}

        final_metrics: Dict[str, float] = {}
        last_epoch = self.epoch_metrics_history[-1]

        final_metrics['final_train_loss'] = last_epoch.get('train_loss_mean', 0.0)
        final_metrics['final_train_accuracy'] = last_epoch.get('accuracy_mean', 0.0)
        final_metrics['final_grad_norm_total'] = last_epoch.get('grad_norm_total_mean', 0.0)

        train_accuracy_series = [epoch.get('accuracy_mean', 0.0) for epoch in self.epoch_metrics_history]
        train_loss_series = [epoch.get('train_loss_mean', math.inf) for epoch in self.epoch_metrics_history]
        final_metrics['best_train_accuracy'] = max(train_accuracy_series) if train_accuracy_series else 0.0
        final_metrics['best_train_loss'] = min(train_loss_series) if train_loss_series else math.inf

        total_epochs = len(self.epoch_metrics_history)
        final_metrics.update(compute_window_features(self.epoch_aggregator.metric_history, total_epochs))
        final_metrics.update(compute_window_features(self.validation_metric_history, total_epochs))

        final_metrics.update(self._aggregate_layer_metrics(last_epoch, self.num_layers))
        final_metrics.update(self.representation_tracker.get_features(self.num_layers))

        positional_aliases = {
            'positional_accuracy_delta_final': 'val_positional_accuracy_delta_final',
            'positional_margin_delta_final': 'val_positional_margin_delta_final',
            'positional_recv_mid_over_early_final': 'val_positional_recv_mid_over_early_final',
            'positional_recv_late_over_early_final': 'val_positional_recv_late_over_early_final',
        }
        for dest_key, src_key in positional_aliases.items():
            if dest_key not in final_metrics and src_key in final_metrics:
                final_metrics[dest_key] = final_metrics[src_key]

        if self.validation_history:
            last_val = self.validation_history[-1]
            if 'val_accuracy' in last_val:
                final_metrics['final_val_accuracy'] = last_val['val_accuracy']
            if 'val_loss' in last_val:
                final_metrics['final_val_loss'] = last_val['val_loss']
            if 'val_perplexity' in last_val:
                final_metrics['final_val_perplexity'] = last_val['val_perplexity']
            if 'val_f1_score' in last_val:
                final_metrics['final_val_f1_score'] = last_val['val_f1_score']

            # Classification metrics
            for key in ('val_cls_accuracy', 'val_cls_f1', 'val_cls_precision',
                        'val_cls_recall', 'val_cls_auc',
                        'val_mlm_accuracy', 'val_mlm_perplexity'):
                if key in last_val:
                    final_metrics[key[4:]] = last_val[key]

            val_accuracies = [entry.get('val_accuracy', 0.0) for entry in self.validation_history]
            val_losses = [entry.get('val_loss', math.inf) for entry in self.validation_history]
            best_val_accuracy = max(val_accuracies) if any(v > 0 for v in val_accuracies) else None
            best_val_loss = min(val_losses) if any(v < math.inf for v in val_losses) else None

            if best_val_accuracy is not None:
                final_metrics['best_val_accuracy'] = best_val_accuracy
            if best_val_loss is not None:
                final_metrics['best_val_loss'] = best_val_loss

        final_metrics['final_loss'] = final_metrics.get('final_val_loss', final_metrics.get('final_train_loss', 0.0))
        final_metrics['final_accuracy'] = final_metrics.get('final_val_accuracy', final_metrics.get('final_train_accuracy', 0.0))
        final_metrics['best_accuracy'] = final_metrics.get('best_val_accuracy', final_metrics.get('best_train_accuracy', 0.0))
        final_metrics['best_loss'] = final_metrics.get('best_val_loss', final_metrics.get('best_train_loss', math.inf))
        last_epoch = self.epoch_metrics_history[-1]
        final_metrics['final_f1_score'] = last_epoch.get('f1_score_mean', 0.0)

        return final_metrics

    def reset(self):
        self.epoch_aggregator.reset()
        self.batch_metrics_history = []
        self.epoch_metrics_history = []
        self.validation_history = []
        self.validation_metric_history = defaultdict(list)
        self.batch_counter = 0
        self.running_metrics.reset()
        self.active_epoch = None

    def get_summary(self) -> str:
        if not self.epoch_metrics_history:
            return "No metrics collected yet."
        final_metrics = self.get_final_metrics()
        summary = [
            "Metric Collection Summary",
            "=" * 50,
            f"Epochs collected: {len(self.epoch_metrics_history)}",
            f"Batches sampled: {len(self.batch_metrics_history)}",
            "",
            "Final Training Metrics:",
            f"  Loss: {final_metrics.get('final_train_loss', 0.0):.4f}",
            f"  Accuracy: {final_metrics.get('final_train_accuracy', 0.0):.4f}",
            "",
            "Best Training Metrics:",
            f"  Best Accuracy: {final_metrics.get('best_train_accuracy', 0.0):.4f}",
            f"  Best Loss: {final_metrics.get('best_train_loss', 0.0):.4f}"
        ]
        if self.validation_history:
            last_val = self.validation_history[-1]
            summary.append("")
            summary.append("Last Validation Epoch:")
            summary.append(f"  Accuracy: {last_val.get('val_accuracy', 0.0):.4f}")
            summary.append(f"  F1 Score: {last_val.get('val_f1_score', 0.0):.4f}")
        return "\n".join(summary)

    @staticmethod
    def _summarize_attention_aliases(alias_values: Dict[str, List[float]]) -> Dict[str, float]:
        def _mean(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        summary = {}
        summary['mass_pad'] = max(alias_values['mass_pad']) if alias_values['mass_pad'] else 0.0
        summary['mass_leak'] = max(alias_values['mass_leak']) if alias_values['mass_leak'] else 0.0
        summary['cross_example_attention'] = max(alias_values['cross_example_attention']) if alias_values['cross_example_attention'] else 0.0
        summary['attention_entropy'] = _mean(alias_values.get('attention_entropy', []))
        summary['pre_softmax_score_mean'] = _mean(alias_values.get('pre_softmax_score_mean', []))
        summary['pre_softmax_score_var'] = _mean(alias_values.get('pre_softmax_score_var', []))
        summary['pre_softmax_score_skew'] = _mean(alias_values.get('pre_softmax_score_skew', []))
        summary['pre_softmax_score_kurt'] = _mean(alias_values.get('pre_softmax_score_kurt', []))
        summary['head_similarity_mean'] = max(alias_values['head_similarity_mean']) if alias_values['head_similarity_mean'] else 0.0
        summary['head_similarity_max'] = max(alias_values['head_similarity_max']) if alias_values['head_similarity_max'] else 0.0
        return summary
