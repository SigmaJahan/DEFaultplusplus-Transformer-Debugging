"""
Base metrics for ABNN Fault Injection Dataset.

Implements step-level metric computations which are later aggregated into
epoch/window statistics by MetricCollector.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np 
import torch
import torch.nn.functional as F
from scipy.stats import entropy, skew, kurtosis, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error


class BaseMetrics:
    """
    Computes training, attention, performance, and structural metrics
    for a single forward/backward pass.
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        self.device = device
        self.config = config or {}

        model_cfg = self.config.get('model_config', {})
        self.model_type = model_cfg.get('model_type', self.config.get('model_type', 'distilbert'))
        self.num_layers = model_cfg.get('num_hidden_layers', self.config.get('num_hidden_layers', 6))

        self.special_token_ids = {
            'pad': self.config.get('pad_token_id', model_cfg.get('pad_token_id')),
            'cls': self.config.get('cls_token_id', model_cfg.get('cls_token_id')),
            'sep': self.config.get('sep_token_id', model_cfg.get('sep_token_id'))
        }
        self.task_info = self.config.get('task_info', {})
        self.task_name = self.task_info.get('task_name')
        self.num_task_labels = self.task_info.get('num_labels', model_cfg.get('num_labels'))
        self.is_regression_task = bool(self.task_info.get('regression')) or (self.num_task_labels == 1 and self.task_name == 'stsb')
        self.grad_activity_threshold = float(self.config.get('grad_activity_threshold', 1e-6))
        self.grad_vanish_threshold = float(self.config.get('grad_vanish_threshold', 1e-4))
        self.grad_explode_threshold = float(self.config.get('grad_explode_threshold', 100.0))
        self.attention_leak_threshold = float(self.config.get('attention_leak_threshold', 1e-6))
        self.ece_num_bins = int(self.config.get('ece_bins', 15))
        self.enable_positional_performance = bool(self.config.get('enable_positional_performance', True))
        self.ffn_probe_tokens = int(self.config.get('ffn_probe_tokens', 256))
        self.ffn_var_activity_threshold = float(self.config.get('ffn_var_activity_threshold', 1e-6))
        # Keep a CPU snapshot of parameters to measure ||Δw|| / ||w||
        self.previous_params: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Training / runtime metrics
    # ------------------------------------------------------------------ #
    def compute_training_metrics(
        self,
        loss: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step_time: Optional[float] = None,
        data_time: Optional[float] = None
    ) -> Dict[str, float]:
        """Collect loss, learning-rate, memory, and runtime stats."""
        metrics: Dict[str, float] = {}

        if isinstance(loss, torch.Tensor):
            metrics['train_loss'] = float(loss.detach().item())
        elif loss is not None:
            metrics['train_loss'] = float(loss)

        if 'train_loss' in metrics:
            metrics['loss'] = metrics['train_loss']

        if optimizer is not None:
            metrics['train_learning_rate'] = optimizer.param_groups[0]['lr']

        if step_time is not None and step_time > 0:
            metrics['runtime_step_time'] = float(step_time)
            metrics['runtime_steps_per_sec'] = 1.0 / step_time

        if data_time is not None and data_time >= 0:
            metrics['runtime_data_time'] = float(data_time)

        # GPU memory (MB)
        if self.device.type == "cuda":
            metrics['runtime_memory_alloc_mb'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            metrics['runtime_memory_reserved_mb'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            metrics['runtime_memory_peak_mb'] = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            free, total = torch.cuda.mem_get_info(self.device)
            metrics['runtime_gpu_total_mb'] = total / 1024 / 1024
            metrics['runtime_gpu_util_pct'] = 100.0 * (1.0 - free / total) if total > 0 else 0.0
        elif self.device.type == "mps":
            metrics['runtime_memory_alloc_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
            metrics['runtime_memory_reserved_mb'] = metrics['runtime_memory_alloc_mb']
            metrics['runtime_memory_peak_mb'] = metrics['runtime_memory_alloc_mb']
        else:
            metrics['runtime_memory_alloc_mb'] = 0.0
            metrics['runtime_memory_reserved_mb'] = 0.0
            metrics['runtime_memory_peak_mb'] = 0.0

        # CPU and RAM usage
        try:
            import psutil
            proc = psutil.Process()
            metrics['runtime_cpu_pct'] = proc.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            metrics['runtime_ram_mb'] = mem_info.rss / 1024 / 1024
            metrics['runtime_ram_pct'] = proc.memory_percent()
        except Exception:
            pass

        # Kernel runtime configuration proxies (best effort)
        kernel_metrics = self._get_kernel_configuration_metrics()
        metrics.update(kernel_metrics)
        state = getattr(model, 'kernel_fault_state', None)
        if isinstance(state, dict):
            for key, active in state.items():
                metrics[f'kernel_fault_{key}_active'] = 1.0 if active else 0.0

        return metrics

    def _get_kernel_configuration_metrics(self) -> Dict[str, float]:
        """Return available runtime kernel flags (CUDA SDPA or fallback)."""
        metrics: Dict[str, float] = {}
        try:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                kernel = torch.backends.cuda.sdp_kernel
                metrics['kernel_flash_enabled'] = float(kernel.is_flash_enabled())
                metrics['kernel_mem_efficient_enabled'] = float(kernel.is_mem_efficient_enabled())
                metrics['kernel_math_enabled'] = float(kernel.is_math_enabled())
        except Exception:
            # Older torch versions or CPU-only environments
            metrics['kernel_flash_enabled'] = 0.0
            metrics['kernel_mem_efficient_enabled'] = 0.0
            metrics['kernel_math_enabled'] = 0.0
        return metrics

    # ------------------------------------------------------------------ #
    # Gradient statistics
    # ------------------------------------------------------------------ #
    def compute_gradient_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute gradient norms per structural block plus total norm.
        """
        metrics: Dict[str, float] = {}
        layer_groups = self._layer_group_patterns()

        total_norm_sq = 0.0
        group_norm_sq = {group: 0.0 for group in layer_groups}
        total_elems = 0
        zero_elems = 0
        grad_abs_min = None
        grad_abs_max = None

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            grad_norm_sq = grad.norm(2).item() ** 2
            total_norm_sq += grad_norm_sq

            # Track absolute min/max and zeros for sparsity/vanishing diagnostics
            grad_abs = grad.abs().detach()
            g_min = grad_abs.min().item() if grad_abs.numel() > 0 else None
            g_max = grad_abs.max().item() if grad_abs.numel() > 0 else None
            if g_min is not None:
                grad_abs_min = g_min if grad_abs_min is None else min(grad_abs_min, g_min)
            if g_max is not None:
                grad_abs_max = g_max if grad_abs_max is None else max(grad_abs_max, g_max)
            zero_elems += int((grad_abs < self.grad_activity_threshold).sum().item())
            total_elems += grad_abs.numel()

            for group, patterns in layer_groups.items():
                if any(pattern in name for pattern in patterns):
                    group_norm_sq[group] += grad_norm_sq

        for group, norm_sq in group_norm_sq.items():
            metrics[f'grad_norm_{group}'] = math.sqrt(norm_sq)
            metrics[f'update_active_{group}'] = 1.0 if norm_sq > self.grad_activity_threshold else 0.0

        metrics['grad_norm_total'] = math.sqrt(total_norm_sq)
        metrics['grad_abs_min'] = float(grad_abs_min) if grad_abs_min is not None else 0.0
        metrics['grad_abs_max'] = float(grad_abs_max) if grad_abs_max is not None else 0.0
        metrics['grad_zero_ratio'] = float(zero_elems / total_elems) if total_elems > 0 else 0.0
        metrics['gradient_vanish'] = 1.0 if metrics['grad_norm_total'] < self.grad_vanish_threshold else 0.0
        metrics['gradient_explode'] = 1.0 if metrics['grad_norm_total'] > self.grad_explode_threshold else 0.0
        return metrics

    def _initialize_previous_params(self, model: torch.nn.Module):
        """Capture a CPU snapshot of parameters for update ratio metrics."""
        self.previous_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.previous_params[name] = param.detach().to(device='cpu', dtype=torch.float32)

    def compute_update_ratio_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Compute ||Δw|| / ||w|| for each structural block since last measurement.
        """
        layer_groups = self._layer_group_patterns()
        default_metrics = {f'update_ratio_{group}': 0.0 for group in layer_groups}
        default_metrics['update_ratio_total'] = 0.0

        if not self.previous_params:
            self._initialize_previous_params(model)
            return default_metrics

        group_delta_sq = {group: 0.0 for group in layer_groups}
        group_weight_sq = {group: 0.0 for group in layer_groups}
        total_delta_sq = 0.0
        total_weight_sq = 0.0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            current = param.detach().to(device='cpu', dtype=torch.float32)
            previous = self.previous_params.get(name)

            if previous is None or previous.shape != current.shape:
                # Parameter added or resized; capture baseline and skip update this round.
                self.previous_params[name] = current
                continue

            delta = current - previous
            delta_norm_sq = float(torch.sum(delta * delta).item())
            weight_norm_sq = float(torch.sum(previous * previous).item())

            total_delta_sq += delta_norm_sq
            total_weight_sq += weight_norm_sq

            for group, patterns in layer_groups.items():
                if any(pattern in name for pattern in patterns):
                    group_delta_sq[group] += delta_norm_sq
                    group_weight_sq[group] += weight_norm_sq

            # Update snapshot for next measurement
            self.previous_params[name] = current

        eps = 1e-12
        metrics = {}
        for group in layer_groups:
            denom = math.sqrt(group_weight_sq[group]) + eps
            ratio = math.sqrt(group_delta_sq[group]) / denom if denom > eps else 0.0
            metrics[f'update_ratio_{group}'] = ratio

        denom_total = math.sqrt(total_weight_sq) + eps
        metrics['update_ratio_total'] = math.sqrt(total_delta_sq) / denom_total if denom_total > eps else 0.0
        return metrics

    def _layer_group_patterns(self) -> Dict[str, Sequence[str]]:
        """Return substrings that identify parameter groups."""
        model_type = (self.model_type or '').lower()
        if model_type in ('bert', 'roberta', 'modernbert'):
            layer_prefix = f'{model_type}.encoder.layer'
            embedding_list = [f'{model_type}.embeddings']
            attn_base = '.attention'
            qkv_patterns = [f'{attn_base}.self.query', f'{attn_base}.self.key', f'{attn_base}.self.value']
            ffn_pattern = '.intermediate'
            ln_patterns = ['attention.output.LayerNorm', 'output.LayerNorm', 'LayerNorm']
        elif model_type in ('gpt2', 'gpt-neo', 'gptj', 'gpt-neox', 'opt'):
            layer_prefix = 'h'
            embedding_list = ['wte', 'wpe']
            attn_base = '.attn'
            qkv_patterns = [f'{attn_base}.c_attn']
            ffn_pattern = '.mlp'
            ln_patterns = ['ln_1', 'ln_2', 'ln_f']
        else:
            layer_prefix = 'distilbert.transformer.layer'
            embedding_list = ['distilbert.embeddings']
            attn_base = '.attention'
            qkv_patterns = [f'{attn_base}.q_lin', f'{attn_base}.k_lin', f'{attn_base}.v_lin']
            ffn_pattern = '.ffn'
            ln_patterns = ['sa_layer_norm', 'output_layer_norm', 'LayerNorm']

        groups: Dict[str, Sequence[str]] = {
            'embedding': embedding_list,
            'classifier': ['pre_classifier', 'classifier', 'score', 'lm_head'],
        }

        for layer_idx in range(self.num_layers):
            prefix = f'{layer_prefix}.{layer_idx}'
            groups[f'layer{layer_idx}_attention'] = [f'{prefix}{attn_base}']
            groups[f'layer{layer_idx}_qkv'] = [f'{prefix}{pat}' for pat in qkv_patterns]
            groups[f'layer{layer_idx}_ffn'] = [f'{prefix}{ffn_pattern}']
            layernorm_patterns = [f'{prefix}.{pat}' for pat in ln_patterns]
            if model_type in ('gpt2', 'gpt-neo', 'gptj', 'gpt-neox', 'opt') and 'ln_f' in ln_patterns:
                layernorm_patterns.append('ln_f')
            groups[f'layer{layer_idx}_layernorm'] = layernorm_patterns

        return groups

    # ------------------------------------------------------------------ #
    # Prediction / classification metrics
    # ------------------------------------------------------------------ #
    def compute_performance_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute accuracy/F1/etc. on a batch."""
        if self.is_regression_task:
            preds_np = predictions.detach().cpu().squeeze(-1).numpy()
            true_labels = labels.detach().cpu().numpy().astype(float)
            if preds_np.ndim == 0:
                preds_np = np.array([preds_np])
            if true_labels.ndim == 0:
                true_labels = np.array([true_labels])
            mse = mean_squared_error(true_labels, preds_np) if len(true_labels) > 0 else 0.0
            mae = mean_absolute_error(true_labels, preds_np) if len(true_labels) > 0 else 0.0
            pearson = pearsonr(true_labels, preds_np)[0] if len(true_labels) > 1 else 0.0
            spearman = spearmanr(true_labels, preds_np).correlation if len(true_labels) > 1 else 0.0
            metrics = {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "regression_mse": float(mse),
                "regression_mae": float(mae),
                "regression_pearson": float(np.nan_to_num(pearson, nan=0.0)),
                "regression_spearman": float(np.nan_to_num(spearman, nan=0.0)),
            }
            total_vals = predictions.numel() if predictions.numel() > 0 else 1
            nan_ratio = float(torch.isnan(predictions).float().sum().item() / total_vals)
            inf_ratio = float(torch.isinf(predictions).float().sum().item() / total_vals)
            metrics["logit_nan_ratio"] = nan_ratio
            metrics["logit_inf_ratio"] = inf_ratio
            metrics["nll"] = 0.0
            metrics["ece"] = 0.0
            metrics["logit_entropy"] = 0.0
            metrics["logit_confidence_mean"] = 0.0
            metrics["logit_kl_uniform"] = 0.0
            return metrics

        pred_classes = predictions.argmax(dim=-1).detach().cpu().numpy()
        true_labels = labels.detach().cpu().numpy()

        # Flatten for decoder models (handles both encoder and decoder cases)
        # Encoder: [batch] -> [batch] (no change)
        # Decoder: [batch, seq_len] -> [batch*seq_len]
        pred_classes_flat = pred_classes.flatten()
        true_labels_flat = true_labels.flatten()

        # Filter out -100 (ignore_index) before computing sklearn metrics
        # -100 is used to mask padding tokens in labels
        valid_mask = true_labels_flat != -100
        pred_classes_filtered = pred_classes_flat[valid_mask]
        true_labels_filtered = true_labels_flat[valid_mask]

        num_unique_labels = len(np.unique(true_labels_filtered))
        is_decoder = predictions.dim() == 3
        if num_unique_labels <= 2 and not is_decoder:
            average = 'binary'
        else:
            average = 'macro'

        metrics = {
            "accuracy": accuracy_score(true_labels_filtered, pred_classes_filtered),
            "f1_score": f1_score(true_labels_filtered, pred_classes_filtered, average=average, zero_division=0),
            "precision": precision_score(true_labels_filtered, pred_classes_filtered, average=average, zero_division=0),
            "recall": recall_score(true_labels_filtered, pred_classes_filtered, average=average, zero_division=0)
        }
        total_vals = predictions.numel() if predictions.numel() > 0 else 1
        nan_ratio = float(torch.isnan(predictions).float().sum().item() / total_vals)
        inf_ratio = float(torch.isinf(predictions).float().sum().item() / total_vals)
        metrics["logit_nan_ratio"] = nan_ratio
        metrics["logit_inf_ratio"] = inf_ratio

        # Negative log-likelihood on the batch (training signal)
        # Handle both encoder [batch, num_classes] and decoder [batch, seq_len, vocab_size] cases
        # IMPORTANT: Use ignore_index=-100 to skip padding tokens (standard PyTorch convention)
        if predictions.dim() == 3:  # Decoder: [batch, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = predictions.shape
            predictions_reshaped = predictions.view(batch_size * seq_len, vocab_size)
            labels_reshaped = labels.view(batch_size * seq_len)
            metrics["nll"] = float(F.cross_entropy(predictions_reshaped, labels_reshaped, ignore_index=-100).detach().item())
        else:  # Encoder: [batch, num_classes]
            metrics["nll"] = float(F.cross_entropy(predictions, labels, ignore_index=-100).detach().item())

        metrics["ece"] = self._compute_ece(predictions, labels, self.ece_num_bins)
        probs = torch.softmax(predictions, dim=-1)
        entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
        metrics["logit_entropy"] = float(entropy.mean().item())
        metrics["logit_confidence_mean"] = float(probs.max(dim=-1).values.mean().item())
        num_classes = predictions.size(-1)
        if num_classes > 0:
            log_k = math.log(max(1, num_classes))
            metrics["logit_kl_uniform"] = float((probs * (probs.log() - log_k)).sum(dim=-1).mean().item())
        else:
            metrics["logit_kl_uniform"] = 0.0

        # Logit margin distribution
        logit_metrics = self._compute_logit_margin_stats(predictions, labels)
        metrics.update(logit_metrics)

        return metrics

    def _compute_logit_margin_stats(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        # Handle both encoder and decoder cases
        if predictions.dim() == 3:  # Decoder: [batch, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = predictions.shape
            # Flatten to [batch*seq_len, vocab_size]
            predictions_flat = predictions.view(batch_size * seq_len, vocab_size)
            labels_flat = labels.view(batch_size * seq_len)
        else:  # Encoder: [batch, num_classes]
            predictions_flat = predictions
            labels_flat = labels

        logits = predictions_flat.detach().cpu().numpy()
        labels_np = labels_flat.detach().cpu().numpy()
        margins = []

        for i in range(len(labels_np)):
            label_idx = int(labels_np[i])
            logit_vec = logits[i]

            # Skip padding tokens (marked with -100) and out-of-bounds indices
            if label_idx == -100 or label_idx >= len(logit_vec) or label_idx < 0:
                continue

            correct_logit = logit_vec[label_idx]
            mask = np.ones_like(logit_vec, dtype=bool)
            mask[label_idx] = False
            second_best = np.max(logit_vec[mask]) if np.any(mask) else correct_logit
            margins.append(correct_logit - second_best)

        if not margins:
            return {}

        margins_arr = np.array(margins, dtype=np.float32)
        return {
            "logit_margin_mean": float(np.mean(margins_arr)),
            "logit_margin_var": float(np.var(margins_arr)),
            "logit_margin_p25": float(np.percentile(margins_arr, 25)),
            "logit_margin_p50": float(np.percentile(margins_arr, 50)),
            "logit_margin_p75": float(np.percentile(margins_arr, 75)),
            "logit_margin_min": float(np.min(margins_arr)),
        }

    @staticmethod
    def _compute_ece(predictions: torch.Tensor, labels: torch.Tensor, num_bins: int) -> float:
        """Expected calibration error using predicted confidence bins."""
        if predictions.numel() == 0:
            return 0.0

        # Handle 3D tensors (decoder) by flattening
        if predictions.dim() == 3:  # [batch, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions.view(batch_size * seq_len, vocab_size)
            labels = labels.view(batch_size * seq_len)

        # Filter out padding tokens (marked with -100)
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return 0.0

        predictions_valid = predictions[valid_mask]
        labels_valid = labels[valid_mask]

        probs = torch.softmax(predictions_valid, dim=-1)
        confidences, pred_classes = probs.max(dim=-1)
        accuracies = pred_classes.eq(labels_valid)
        bins = torch.linspace(0, 1, num_bins + 1, device=predictions.device)

        ece = 0.0
        total = labels_valid.numel()
        if total == 0:
            return 0.0

        for idx in range(num_bins):
            if idx == 0:
                mask = (confidences >= bins[idx]) & (confidences <= bins[idx + 1])
            else:
                mask = (confidences > bins[idx]) & (confidences <= bins[idx + 1])
            count = mask.sum().item()
            if count == 0:
                continue
            acc_bin = accuracies[mask].float().mean().item()
            conf_bin = confidences[mask].mean().item()
            ece += abs(conf_bin - acc_bin) * (count / total)
        return float(ece)

    # ------------------------------------------------------------------ #
    # Attention / positional metrics
    # ------------------------------------------------------------------ #
    def compute_attention_metrics(
        self,
        attention_weights: torch.Tensor,
        model: Optional[torch.nn.Module] = None,
        layer_idx: Optional[int] = None,
        layer_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_cutoff: int = 64
    ) -> Dict[str, float]:
        """
        Compute entropy, sparsity, pad-mass, and positional summaries
        for a single layer's attention tensor.
        """
        attn = attention_weights.detach().float()

        metrics: Dict[str, float] = {}

        probs = torch.clamp(attn, min=1e-12)
        log_probs = probs.log()
        head_entropy = -(probs * log_probs).sum(dim=-1).mean(dim=2)  # [batch, heads]
        head_entropy = head_entropy.mean(dim=0)  # [heads]

        max_weights = attn.max(dim=-1).values  # [batch, heads, seq]
        head_max = max_weights.mean(dim=2).mean(dim=0)

        metrics["attention_entropy_mean"] = float(head_entropy.mean().item())
        metrics["attention_entropy"] = metrics["attention_entropy_mean"]
        metrics["attention_entropy_std"] = float(head_entropy.std(unbiased=False).item())
        metrics["attention_max_mean"] = float(head_max.mean().item())
        metrics["attention_max_std"] = float(head_max.std(unbiased=False).item())
        metrics["attention_sparsity"] = float((attn < 0.01).float().mean().item())
        metrics["attention_weight_magnitude"] = float(attn.abs().mean().item())

        metrics["attention_mass_leak"] = 0.0
        metrics["attention_mass_leak_max"] = 0.0
        metrics["attention_cross_example_leak"] = 0.0
        metrics["attention_mass_future"] = 0.0

        if attention_mask is not None:
            pad_mask = (attention_mask == 0).float().unsqueeze(1).unsqueeze(2).to(attn.device)
            pad_mass = (attn * pad_mask).sum(dim=-1)
            metrics["attention_mass_pad_mean"] = float(pad_mass.mean().item())
            metrics["attention_mass_pad_max"] = float(pad_mass.max().item())

            cross_mask = self._compute_cross_example_mask(attention_mask)
            if cross_mask is not None:
                cross_mask = cross_mask.to(attn.device)
                cross_mass = (attn * cross_mask).sum(dim=-1)
                metrics["attention_mass_leak"] = float(cross_mass.mean().item())
                metrics["attention_mass_leak_max"] = float(cross_mass.max().item())
                leak_indicator = (cross_mass > self.attention_leak_threshold).float()
                metrics["attention_cross_example_leak"] = float(leak_indicator.mean().item())

        future_mass = self._compute_future_attention_mass(attn, attention_mask)
        if future_mass is not None:
            metrics["attention_mass_future"] = float(future_mass)

        if input_ids is not None and self.special_token_ids['cls'] is not None:
            special_mask = self._get_special_token_mask(input_ids)
            special_mask = special_mask.unsqueeze(1).unsqueeze(2).to(attn.device)
            special_mass = (attn * special_mask).sum(dim=-1)
            metrics["attention_mass_special_mean"] = float(special_mass.mean().item())
            metrics["attention_mass_special_std"] = float(special_mass.std(unbiased=False).item())

        # Head redundancy (cosine similarity)
        head_similarity = self.compute_head_similarity(attn)
        metrics.update(head_similarity)

        # SVD-based effective attention rank
        metrics.update(self._compute_attention_rank(attn))

        # Positional profile
        if attention_mask is not None:
            positional_metrics = self.compute_positional_attention_profile(
                attn,
                attention_mask,
                position_cutoff=position_cutoff
            )
            metrics.update(positional_metrics)

        # Score statistics proxy (log of softmax outputs)
        score_proxy = log_probs
        metrics["attention_score_var"] = float(score_proxy.var().item())
        metrics["attention_score_skew"] = float(self._safe_skew(score_proxy.view(-1).cpu().numpy()))

        pre_softmax_stats = self._compute_pre_softmax_stats(
            model=model,
            layer_idx=layer_idx,
            layer_input=layer_input,
            attention_mask=attention_mask
        )
        metrics.update(pre_softmax_stats)

        # QKV alignment: pairwise cosine similarity between Q, K, V weight matrices
        if model is not None and layer_idx is not None:
            qkv_align = self._compute_qkv_alignment(model, layer_idx)
            metrics.update(qkv_align)

        return metrics

    def _compute_qkv_alignment(self, model: torch.nn.Module, layer_idx: int) -> Dict[str, float]:
        """Paper C_int: Q-K, Q-V, K-V pairwise cosine similarities."""
        metrics: Dict[str, float] = {}
        try:
            raw = getattr(model, 'model', model)
            layers = None
            for attr in ['transformer.h', 'gpt_neox.layers', 'model.layers']:
                obj = raw
                for part in attr.split('.'):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None:
                    layers = obj
                    break
            if layers is None or layer_idx >= len(layers):
                return metrics

            layer = layers[layer_idx]
            attn = getattr(layer, 'attn', getattr(layer, 'attention', getattr(layer, 'self_attn', None)))
            if attn is None:
                return metrics

            # GPT-2 style: combined c_attn weight [3*hidden, hidden]
            c_attn = getattr(attn, 'c_attn', None)
            if c_attn is not None and hasattr(c_attn, 'weight'):
                w = c_attn.weight.data.float()
                h = w.size(-1) // 3 if w.size(-1) % 3 == 0 else w.size(-1)
                if w.size(-1) >= 3 * h and h > 0:
                    if w.dim() == 2 and w.size(0) < w.size(1):
                        wq, wk, wv = w[:, :h], w[:, h:2*h], w[:, 2*h:3*h]
                    else:
                        wq, wk, wv = w[:h], w[h:2*h], w[2*h:3*h]
                    qf, kf, vf = wq.reshape(-1), wk.reshape(-1), wv.reshape(-1)
                    metrics['qkv_align_qk'] = float(F.cosine_similarity(qf.unsqueeze(0), kf.unsqueeze(0)).item())
                    metrics['qkv_align_qv'] = float(F.cosine_similarity(qf.unsqueeze(0), vf.unsqueeze(0)).item())
                    metrics['qkv_align_kv'] = float(F.cosine_similarity(kf.unsqueeze(0), vf.unsqueeze(0)).item())
            else:
                # Separate Q, K, V projections
                q_proj = getattr(attn, 'q_proj', getattr(attn, 'query', None))
                k_proj = getattr(attn, 'k_proj', getattr(attn, 'key', None))
                v_proj = getattr(attn, 'v_proj', getattr(attn, 'value', None))
                if q_proj is not None and k_proj is not None and v_proj is not None:
                    qf = q_proj.weight.data.float().reshape(-1)
                    kf = k_proj.weight.data.float().reshape(-1)
                    vf = v_proj.weight.data.float().reshape(-1)
                    if qf.numel() == kf.numel() == vf.numel():
                        metrics['qkv_align_qk'] = float(F.cosine_similarity(qf.unsqueeze(0), kf.unsqueeze(0)).item())
                        metrics['qkv_align_qv'] = float(F.cosine_similarity(qf.unsqueeze(0), vf.unsqueeze(0)).item())
                        metrics['qkv_align_kv'] = float(F.cosine_similarity(kf.unsqueeze(0), vf.unsqueeze(0)).item())
        except Exception:
            pass
        return metrics

    @staticmethod
    def _compute_future_attention_mass(
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Optional[float]:
        """
        Compute average attention mass assigned to future positions.

        For decoder models, this should be near 0 when causal masking is correct.
        """
        if attention_weights is None:
            return None

        attn = attention_weights
        if attn.dim() != 4:
            return None

        batch_size, num_heads, query_len, key_len = attn.shape
        if query_len == 0 or key_len == 0:
            return None

        future_mask = torch.triu(
            torch.ones(query_len, key_len, device=attn.device, dtype=attn.dtype),
            diagonal=1
        )
        future_mass = attn * future_mask

        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask
            if token_mask.dim() >= 3:
                while token_mask.dim() > 2:
                    token_mask = token_mask.max(dim=-2).values
            if token_mask.dim() == 2 and token_mask.size(-1) == key_len:
                if token_mask.max() <= 1.0 and token_mask.min() >= 0.0:
                    token_mask = (token_mask > 0.5).float()
                else:
                    token_mask = (token_mask > -1e4).float()
            else:
                token_mask = None

        if token_mask is not None and token_mask.dim() == 2:
            key_mask = token_mask[:, None, None, :]
            query_mask = token_mask[:, None, :, None]
            future_mass = future_mass * key_mask * query_mask
            valid_queries = query_mask.squeeze(1).squeeze(-1)
            denom = float(valid_queries.sum().item()) * float(num_heads)
            if denom <= 0:
                return 0.0
            return float(future_mass.sum().item() / denom)

        return float(future_mass.sum(dim=-1).mean().item())

    def compute_positional_performance_metrics(
        self,
        model: torch.nn.Module,
        batch: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Measure accuracy/margin when only early or late token windows are visible.
        """
        default_metrics = {
            "positional_accuracy_early": 0.0,
            "positional_accuracy_late": 0.0,
            "positional_accuracy_delta": 0.0,
            "positional_margin_early": 0.0,
            "positional_margin_late": 0.0,
            "positional_margin_delta": 0.0,
            "positional_loss_early": 0.0,
            "positional_loss_late": 0.0
        }

        if not self.enable_positional_performance:
            return default_metrics

        if batch is None:
            return default_metrics

        labels = batch.get('labels')
        attention_mask = batch.get('attention_mask')
        input_ids = batch.get('input_ids')
        if labels is None or attention_mask is None or input_ids is None:
            return default_metrics

        if labels.numel() == 0:
            return default_metrics

        was_training = bool(getattr(model, 'training', False))

        try:
            if was_training and hasattr(model, 'eval'):
                model.eval()

            with torch.no_grad():
                early_batch = self._mask_batch_positions(batch, mode='early')
                late_batch = self._mask_batch_positions(batch, mode='late')

                early_outputs = model(**early_batch)
                late_outputs = model(**late_batch)

                early_logits = self._extract_logits(early_outputs)
                late_logits = self._extract_logits(late_outputs)
                if early_logits.dim() == 3:
                    def _decoder_stats(
                        logits: torch.Tensor,
                        label_tensor: torch.Tensor,
                        attn_mask: Optional[torch.Tensor]
                    ) -> Tuple[float, float, float]:
                        if attn_mask is not None:
                            label_tensor = label_tensor.masked_fill(attn_mask == 0, -100)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = label_tensor[:, 1:].contiguous()
                        valid_mask = shift_labels != -100
                        if valid_mask.sum() == 0:
                            return 0.0, 0.0, 0.0
                        preds = shift_logits.argmax(dim=-1)
                        acc = (preds[valid_mask] == shift_labels[valid_mask]).float().mean().item()
                        margins = self._compute_example_margins(shift_logits, shift_labels)
                        margin_mean = float(margins[valid_mask].mean().item()) if valid_mask.any() else 0.0
                        loss_sum = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction='sum'
                        ).item()
                        loss_mean = float(loss_sum / valid_mask.sum().item())
                        return acc, margin_mean, loss_mean

                    early_acc, early_margin_mean, early_loss = _decoder_stats(
                        early_logits,
                        labels.clone(),
                        early_batch.get('attention_mask')
                    )
                    late_acc, late_margin_mean, late_loss = _decoder_stats(
                        late_logits,
                        labels.clone(),
                        late_batch.get('attention_mask')
                    )
                else:
                    early_preds = early_logits.argmax(dim=-1)
                    late_preds = late_logits.argmax(dim=-1)

                    # Filter out -100 labels (padding tokens) before computing accuracy
                    valid_mask = labels != -100
                    if valid_mask.sum() == 0:
                        return default_metrics

                    early_acc = (early_preds[valid_mask] == labels[valid_mask]).float().mean().item()
                    late_acc = (late_preds[valid_mask] == labels[valid_mask]).float().mean().item()

                    early_margins = self._compute_example_margins(early_logits, labels)
                    late_margins = self._compute_example_margins(late_logits, labels)

                    early_margin_mean = float(early_margins.mean().item())
                    late_margin_mean = float(late_margins.mean().item())

                    early_loss = float(F.cross_entropy(early_logits, labels).item())
                    late_loss = float(F.cross_entropy(late_logits, labels).item())
        except Exception:
            return default_metrics
        finally:
            if was_training and hasattr(model, 'train'):
                model.train()

        metrics = {
            "positional_accuracy_early": float(early_acc),
            "positional_accuracy_late": float(late_acc),
            "positional_accuracy_delta": float(late_acc - early_acc),
            "positional_margin_early": early_margin_mean,
            "positional_margin_late": late_margin_mean,
            "positional_margin_delta": float(late_margin_mean - early_margin_mean),
            "positional_loss_early": early_loss,
            "positional_loss_late": late_loss
        }
        return metrics

    # ------------------------------------------------------------------ #
    # Structural/probing metrics for FFN, LayerNorm, embeddings, outputs
    # ------------------------------------------------------------------ #
    def compute_structural_metrics(
        self,
        hidden_states: Optional[Any],
        model: Optional[torch.nn.Module],
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if hidden_states is None:
            return metrics

        try:
            hs_list = list(hidden_states)
        except Exception:
            return metrics

        if len(hs_list) < 2:
            return metrics

        raw_model = getattr(model, 'model', model) if model is not None else None
        eps = 1e-6
        delta_means = []
        cos_means = []
        var_ratios = []
        ln_std_means = []
        ln_mean_abs_means = []
        active_fracs = []
        skew_vals = []

        max_layers = min(self.num_layers, len(hs_list) - 1)
        probe_tokens = self.ffn_probe_tokens

        for layer_idx in range(max_layers):
            h_in = hs_list[layer_idx]
            h_out = hs_list[layer_idx + 1]
            if h_in is None or h_out is None:
                continue

            flat_in = h_in.reshape(-1, h_in.size(-1))
            flat_out = h_out.reshape(-1, h_out.size(-1))

            if flat_in.size(0) > probe_tokens:
                flat_in = flat_in[:probe_tokens]
                flat_out = flat_out[:probe_tokens]

            if flat_in.numel() == 0 or flat_out.numel() == 0:
                continue

            delta = flat_out - flat_in
            delta_norm = torch.norm(delta, dim=-1)
            mean_delta = delta_norm.mean().item()
            metrics[f'ffn_delta_l{layer_idx}_mean'] = float(mean_delta)
            delta_means.append(mean_delta)

            cos = torch.clamp(F.cosine_similarity(flat_in, flat_out, dim=-1), -1.0, 1.0)
            cos_mean = cos.mean().item()
            metrics[f'residual_cos_l{layer_idx}_mean'] = float(cos_mean)
            cos_means.append(cos_mean)

            var_in = flat_in.var(dim=0, unbiased=False)
            var_out = flat_out.var(dim=0, unbiased=False)
            ratio = (var_out.mean() + eps) / (var_in.mean() + eps)
            metrics[f'ffn_var_ratio_l{layer_idx}'] = float(ratio.item())
            var_ratios.append(ratio.item())

            std_out = torch.sqrt(var_out + eps)
            ln_std_mean = std_out.mean().item()
            metrics[f'ln_std_l{layer_idx}_mean'] = float(ln_std_mean)
            ln_std_means.append(ln_std_mean)

            mean_out = flat_out.mean(dim=0)
            ln_mean_abs = mean_out.abs().mean().item()
            metrics[f'ln_mean_abs_l{layer_idx}_mean'] = float(ln_mean_abs)
            ln_mean_abs_means.append(ln_mean_abs)

            active = (var_out > self.ffn_var_activity_threshold).float()
            active_frac = active.mean().item()
            metrics[f'ffn_active_dim_frac_l{layer_idx}'] = float(active_frac)
            active_fracs.append(active_frac)

            try:
                flat_out_cpu = flat_out.detach().cpu().numpy().reshape(-1)
                skew_val = self._safe_skew(flat_out_cpu)
            except Exception:
                skew_val = 0.0
            metrics[f'ffn_out_skew_l{layer_idx}'] = float(skew_val)
            skew_vals.append(skew_val)

            # FFN output L2 norm (paper C_int: FFN output norm)
            ffn_l2 = torch.norm(flat_out, dim=-1).mean().item()
            metrics[f'ffn_out_norm_l{layer_idx}_mean'] = float(ffn_l2)

            # Inter-layer CKA (paper C_int: repr drift between consecutive layers)
            if flat_in.size(0) >= 4:
                cka_val = self._linear_cka_batch(flat_in, flat_out)
                metrics[f'inter_layer_cka_l{layer_idx}_mean'] = float(cka_val)

        if delta_means:
            metrics['ffn_delta_mean'] = float(np.mean(delta_means))
            metrics['residual_cos_mean'] = float(np.mean(cos_means))
            metrics['ffn_var_ratio_mean'] = float(np.mean(var_ratios))
            metrics['ln_std_mean'] = float(np.mean(ln_std_means))
            metrics['ln_mean_abs_mean'] = float(np.mean(ln_mean_abs_means))
            metrics['ffn_active_dim_frac_mean'] = float(np.mean(active_fracs))
            metrics['ffn_out_skew_mean'] = float(np.mean(skew_vals))

        # Embedding-level probes
        if raw_model is not None and input_ids is not None:
            emb_module = getattr(getattr(raw_model, 'distilbert', None), 'embeddings', None)
            if emb_module is not None and hasattr(emb_module, 'word_embeddings'):
                embeds = emb_module.word_embeddings(input_ids)
                flat_embed = embeds.reshape(-1, embeds.size(-1))
                norms = torch.norm(flat_embed, dim=-1)
                metrics['embedding_norm_mean'] = float(norms.mean().item())
                metrics['embedding_norm_std'] = float(norms.std(unbiased=False).item())

                meta = getattr(raw_model, 'fault_metadata', {})
                subset_ids = []
                if isinstance(meta, dict):
                    if 'embedding_zero' in meta and isinstance(meta['embedding_zero'], dict):
                        subset_ids = meta['embedding_zero'].get('indices', [])
                    elif 'embedding_swap' in meta and isinstance(meta['embedding_swap'], dict):
                        pairs = meta['embedding_swap'].get('pairs', [])
                        subset_ids = list({i for pair in pairs for i in pair})
                if subset_ids:
                    try:
                        idx_tensor = torch.tensor(subset_ids, device=input_ids.device, dtype=input_ids.dtype)
                        subset_mask = torch.isin(input_ids, idx_tensor)
                        if subset_mask.any():
                            selected = flat_embed[subset_mask.reshape(-1)]
                            if selected.numel() > 0:
                                sel_norms = torch.norm(selected, dim=-1)
                                metrics['embedding_subset_norm_mean'] = float(sel_norms.mean().item())
                    except Exception:
                        pass

        # Hidden state drift after first block (captures embedding/type effects)
        if len(hs_list) > 1 and hs_list[0] is not None and hs_list[1] is not None:
            h0 = hs_list[0].reshape(-1, hs_list[0].size(-1))
            h1 = hs_list[1].reshape(-1, hs_list[1].size(-1))
            take = min(h0.size(0), h1.size(0))
            if take > 0:
                h0 = h0[:take]
                h1 = h1[:take]
                delta01 = torch.norm(h1 - h0, dim=-1)
                metrics['h1_delta_norm_mean'] = float(delta01.mean().item())

        # Output projection probes
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
            metrics.setdefault("logit_entropy", float(entropy.mean().item()))
            metrics.setdefault("logit_confidence_mean", float(probs.max(dim=-1).values.mean().item()))
            num_classes = probs.size(-1)
            if num_classes > 0:
                log_k = math.log(max(1, num_classes))
                metrics.setdefault("logit_kl_uniform", float((probs * (probs.log() - log_k)).sum(dim=-1).mean().item()))

        return metrics

    def _get_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return mask for CLS/SEP tokens."""
        special_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        cls_id = self.special_token_ids.get('cls')
        sep_id = self.special_token_ids.get('sep')

        if cls_id is not None:
            special_mask = special_mask + (input_ids == cls_id).float()
        if sep_id is not None:
            special_mask = special_mask + (input_ids == sep_id).float()

        return (special_mask > 0).float()

    def _compute_cross_example_mask(self, attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Mask positions that belong to other batch items (same column index)."""
        if attention_mask is None:
            return None

        mask = attention_mask.float()
        global_active = (mask.sum(dim=0, keepdim=True) > 0).float()
        other_example_mask = (1.0 - mask) * global_active
        if torch.all(other_example_mask <= 0):
            return None

        return other_example_mask.unsqueeze(1).unsqueeze(2)

    def _compute_pre_softmax_stats(
        self,
        model: Optional[torch.nn.Module],
        layer_idx: Optional[int],
        layer_input: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Recompute QK^T scores to capture pre-softmax statistics."""
        if model is None or layer_idx is None or layer_input is None:
            return {}

        attention_module = self._get_attention_module(model, layer_idx)
        if attention_module is None:
            return {}

        with torch.no_grad():
            hidden = layer_input.detach()
            batch_size, seq_len, hidden_size = hidden.shape
            n_heads = getattr(attention_module, 'n_heads', None)
            if n_heads is None:
                n_heads = getattr(model.config, 'num_attention_heads', self.config.get('num_attention_heads', 6))
            dim_per_head = getattr(attention_module, 'dim_per_head', None)
            if dim_per_head is None:
                module_dim = getattr(attention_module, 'dim', hidden_size)
                dim_per_head = module_dim // max(1, n_heads)

            query = attention_module.q_lin(hidden)
            key = attention_module.k_lin(hidden)

            query = query.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)
            key = key.reshape(batch_size, seq_len, n_heads, dim_per_head).transpose(1, 2)

            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_per_head)

            if attention_mask is not None:
                attn_mask = attention_mask.float()
                key_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                query_mask = attn_mask.unsqueeze(1).unsqueeze(-1)
                valid_mask = (key_mask * query_mask).to(dtype=torch.bool)
                if valid_mask.sum() == 0:
                    return {}
                scores = scores.masked_select(valid_mask)
            else:
                scores = scores.reshape(-1)

            if scores.numel() == 0:
                return {}

            scores = scores.detach().cpu().float()
            stats = {
                "pre_softmax_score_mean": float(scores.mean().item()),
                "pre_softmax_score_var": float(scores.var(unbiased=False).item()),
                "pre_softmax_score_skew": float(self._safe_skew(scores.numpy())),
                "pre_softmax_score_kurt": float(self._safe_kurtosis(scores.numpy()))
            }
            return stats

    def _get_attention_module(self, model: torch.nn.Module, layer_idx: int) -> Optional[torch.nn.Module]:
        """Return attention module for DistilBERT layer if available."""
        attention = getattr(getattr(model, 'distilbert', None), 'transformer', None)
        if attention is None or not hasattr(attention, 'layer'):
            return None
        layers = attention.layer
        if layer_idx < 0 or layer_idx >= len(layers):
            return None
        layer = layers[layer_idx]
        return getattr(layer, 'attention', None)

    def _mask_batch_positions(self, batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
        """Return a cloned batch keeping only tokens from the specified segment."""
        clone = self._clone_batch(batch)
        input_ids = clone.get('input_ids')
        attention_mask = clone.get('attention_mask')
        token_type_ids = clone.get('token_type_ids')

        if input_ids is None or attention_mask is None:
            clone.pop('labels', None)
            return clone

        pad_id = self.special_token_ids.get('pad')
        if pad_id is None:
            pad_id = 0

        seq_len = attention_mask.size(1)
        for idx in range(attention_mask.size(0)):
            valid_positions = (attention_mask[idx] != 0).nonzero(as_tuple=False).view(-1)
            if valid_positions.numel() == 0:
                continue
            first = int(valid_positions[0].item())
            last = int(valid_positions[-1].item())
            length = last - first + 1
            if length <= 0:
                continue
            window = max(1, length // 3)
            if mode == 'early':
                start, end = 0, window
            elif mode == 'late':
                start, end = length - window, length
            else:
                start = window
                end = min(length, 2 * window)
            start = max(0, start) + first
            end = min(length, end) + first
            start = max(0, start)
            end = min(seq_len, end)

            mask = torch.zeros_like(attention_mask[idx])
            mask[start:end] = 1
            attention_mask[idx] = mask

            drop_mask = mask == 0
            drop_mask_bool = drop_mask.bool()
            input_ids[idx][drop_mask_bool] = pad_id
            if token_type_ids is not None:
                token_type_ids[idx][drop_mask_bool] = 0

        clone['attention_mask'] = attention_mask
        clone['input_ids'] = input_ids
        if token_type_ids is not None:
            clone['token_type_ids'] = token_type_ids

        clone.pop('labels', None)
        return clone

    @staticmethod
    def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Deep-copy tensor entries to avoid mutating the caller's batch."""
        cloned: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = value
        return cloned

    @staticmethod
    def _safe_skew(values: np.ndarray) -> float:
        """Compute skewness with guard against precision issues."""
        if values.size < 3:
            return 0.0
        try:
            return float(skew(values))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_kurtosis(values: np.ndarray) -> float:
        """Compute kurtosis with guard against precision issues."""
        if values.size < 4:
            return 0.0
        try:
            return float(kurtosis(values, fisher=True, bias=False))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        """Return logits tensor from different HF output variants."""
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    @staticmethod
    def _compute_example_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return per-example logit margins."""
        # Handle both encoder [batch, num_classes] and decoder [batch, seq_len, vocab_size] shapes
        if logits.dim() == 3:  # Decoder case
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            labels_flat = labels.view(batch_size * seq_len)
        else:  # Encoder case
            logits_flat = logits
            labels_flat = labels

        num_classes = logits_flat.size(-1)
        if num_classes < 2:
            return torch.zeros_like(labels, dtype=logits.dtype)

        # Filter out -100 labels (padding tokens)
        valid_mask = labels_flat != -100
        if valid_mask.sum() == 0:
            return torch.zeros_like(labels, dtype=logits.dtype)

        # Compute margins only for valid positions
        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]

        top2 = torch.topk(logits_valid, k=2, dim=-1).values
        target_scores = logits_valid.gather(-1, labels_valid.unsqueeze(-1)).squeeze(-1)
        first = top2[:, 0]
        second = top2[:, 1]
        max_other = torch.where(target_scores == first, second, first)
        margins_valid = target_scores - max_other

        # Create full margins tensor with zeros for invalid positions
        margins_flat = torch.zeros_like(labels_flat, dtype=logits.dtype)
        margins_flat[valid_mask] = margins_valid

        # Reshape back to original label shape
        if logits.dim() == 3:
            return margins_flat.view(batch_size, seq_len)
        else:
            return margins_flat

    @staticmethod
    def compute_head_similarity(attention_weights: torch.Tensor) -> Dict[str, float]:
        """Cosine similarity between attention heads."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        flattened = attention_weights.reshape(batch_size, num_heads, -1)
        mean_patterns = flattened.mean(dim=0)  # [heads, seq_len*seq_len]

        sims = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                cos_sim = F.cosine_similarity(
                    mean_patterns[i].unsqueeze(0),
                    mean_patterns[j].unsqueeze(0),
                    dim=1
                )
                sims.append(float(cos_sim.item()))
        if not sims:
            return {"head_similarity_mean": 0.0, "head_similarity_std": 0.0}

        sims_arr = np.array(sims)
        return {
            "head_similarity_mean": float(np.mean(sims_arr)),
            "head_similarity_std": float(np.std(sims_arr)),
            "head_similarity_max": float(np.max(sims_arr))
        }

    @staticmethod
    def _linear_cka_batch(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Linear CKA between two [n, d] tensors (mean-centered)."""
        X = (X - X.mean(dim=0, keepdim=True)).float()
        Y = (Y - Y.mean(dim=0, keepdim=True)).float()
        hsic_xy = (X.T @ Y).pow(2).sum()
        hsic_xx = (X.T @ X).pow(2).sum()
        hsic_yy = (Y.T @ Y).pow(2).sum()
        denom = (hsic_xx * hsic_yy).sqrt()
        if denom < 1e-12:
            return 0.0
        return float((hsic_xy / denom).clamp(0, 1).item())

    @staticmethod
    def _compute_attention_rank(attention_weights: torch.Tensor) -> Dict[str, float]:
        """SVD-based effective rank of attention weight matrices.

        Effective rank = exp(H(sigma)) where H is the Shannon entropy
        of the normalized singular values. Measures how concentrated
        or uniform the attention distribution is across dimensions.
        """
        batch_size, num_heads, seq_len, key_len = attention_weights.shape
        if seq_len == 0 or key_len == 0:
            return {"attention_rank_mean": 0.0, "attention_rank_std": 0.0}

        mean_attn = attention_weights.mean(dim=0)
        ranks = []
        for h in range(num_heads):
            try:
                s = torch.linalg.svdvals(mean_attn[h].float())
                s = s[s > 1e-12]
                if s.numel() == 0:
                    ranks.append(0.0)
                    continue
                p = s / s.sum()
                entropy = -(p * p.log()).sum().item()
                ranks.append(math.exp(entropy))
            except Exception:
                ranks.append(0.0)

        ranks_arr = np.array(ranks)
        return {
            "attention_rank_mean": float(np.mean(ranks_arr)),
            "attention_rank_std": float(np.std(ranks_arr)),
        }

    @staticmethod
    def compute_positional_attention_profile(
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        position_cutoff: int = 64
    ) -> Dict[str, float]:
        """
        Average attention mass received per absolute position plus region stats.
        """
        seq_len = min(attention_weights.size(-1), position_cutoff)
        attn = attention_weights[..., :seq_len]
        key_mask = attention_mask[:, :seq_len].float()

        recv = attn.mean(dim=2)  # average over query positions -> [batch, heads, seq]
        recv = recv * key_mask.unsqueeze(1)  # zero-out PAD positions

        total_mass = key_mask.sum()
        if total_mass <= 0:
            return {}

        recv_vec = recv.sum(dim=(0, 1)) / total_mass
        recv_np = recv_vec.detach().cpu().numpy()

        region_size = seq_len // 3 if seq_len >= 3 else seq_len
        early = recv_np[:region_size].mean() if region_size > 0 else 0.0
        mid = recv_np[region_size:2 * region_size].mean() if region_size > 0 else 0.0
        late = recv_np[2 * region_size:].mean() if region_size > 0 else 0.0

        return {
            "positional_recv_mean": float(np.mean(recv_np)),
            "positional_recv_var": float(np.var(recv_np)),
            "positional_recv_skew": float(BaseMetrics._safe_skew(recv_np)),
            "positional_recv_early": float(early),
            "positional_recv_mid": float(mid),
            "positional_recv_late": float(late),
            "positional_recv_mid_over_early": float(mid / (early + 1e-8)),
            "positional_recv_late_over_early": float(late / (early + 1e-8)),
        }

    # ------------------------------------------------------------------ #
    # Statistical summaries for weights/activations
    # ------------------------------------------------------------------ #
    def compute_statistical_metrics(
        self,
        model: torch.nn.Module,
        activations: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Simple aggregate statistics of weights/activations."""
        metrics: Dict[str, float] = {}

        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.detach().cpu().numpy().ravel())

        if weights:
            concatenated = np.concatenate(weights)
            metrics["weight_mean"] = float(np.mean(concatenated))
            metrics["weight_std"] = float(np.std(concatenated))
        else:
            metrics["weight_mean"] = 0.0
            metrics["weight_std"] = 0.0

        if activations is not None:
            act = activations.detach().cpu().numpy()
            metrics["activation_mean"] = float(np.mean(act))
            metrics["activation_std"] = float(np.std(act))
        else:
            metrics["activation_mean"] = 0.0
            metrics["activation_std"] = 0.0

        return metrics

    # ------------------------------------------------------------------ #
    def compute_all_metrics(
        self,
        loss: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        activations: Optional[torch.Tensor] = None,
        step_time: Optional[float] = None,
        data_time: Optional[float] = None
    ) -> Dict[str, float]:
        """Convenience wrapper to compute every available metric."""
        all_metrics: Dict[str, float] = {}
        all_metrics.update(self.compute_training_metrics(
            loss,
            model,
            optimizer,
            step_time=step_time,
            data_time=data_time
        ))
        all_metrics.update(self.compute_performance_metrics(predictions, labels))
        all_metrics.update(self.compute_statistical_metrics(model, activations))

        if attention_weights is not None:
            all_metrics.update(self.compute_attention_metrics(attention_weights))

        return all_metrics
