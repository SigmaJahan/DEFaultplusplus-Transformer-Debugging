"""LogitMetrics — classification performance, logit health, ECE, margin stats."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import MetricModule, _extract_logits
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class LogitMetrics(MetricModule):
    """Logit-based performance and health metrics."""

    def __init__(self, inspector: ModelInspector, config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.ece_num_bins = cfg.ece_num_bins

    def collect(
        self,
        *,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        if outputs is None or labels is None:
            return {}

        predictions = _extract_logits(outputs)
        return self._compute_performance(predictions, labels)

    def _compute_performance(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        # Flatten for decoder models
        if predictions.dim() == 3:
            batch_size, seq_len, vocab_size = predictions.shape
            pred_flat = predictions.view(batch_size * seq_len, vocab_size)
            labels_flat = labels.view(batch_size * seq_len)
        else:
            pred_flat = predictions
            labels_flat = labels

        pred_classes = pred_flat.argmax(dim=-1).detach().cpu().numpy().flatten()
        true_labels = labels_flat.detach().cpu().numpy().flatten()

        # Filter ignore_index
        valid_mask = true_labels != -100
        pred_filtered = pred_classes[valid_mask]
        true_filtered = true_labels[valid_mask]

        if len(true_filtered) == 0:
            return {}

        num_unique = len(np.unique(true_filtered))
        average = 'binary' if num_unique <= 2 else 'macro'

        metrics: Dict[str, float] = {
            'accuracy': float(accuracy_score(true_filtered, pred_filtered)),
            'f1_score': float(f1_score(true_filtered, pred_filtered, average=average, zero_division=0)),
            'precision': float(precision_score(true_filtered, pred_filtered, average=average, zero_division=0)),
            'recall': float(recall_score(true_filtered, pred_filtered, average=average, zero_division=0)),
        }

        # Logit health
        total_vals = predictions.numel() if predictions.numel() > 0 else 1
        metrics['logit_nan_ratio'] = float(torch.isnan(predictions).float().sum().item() / total_vals)
        metrics['logit_inf_ratio'] = float(torch.isinf(predictions).float().sum().item() / total_vals)

        # NLL
        if predictions.dim() == 3:
            b, s, v = predictions.shape
            metrics['nll'] = float(F.cross_entropy(
                predictions.view(b * s, v), labels.view(b * s), ignore_index=-100
            ).detach().item())
        else:
            metrics['nll'] = float(F.cross_entropy(predictions, labels, ignore_index=-100).detach().item())

        # ECE
        metrics['ece'] = self._compute_ece(predictions, labels)

        # Entropy and confidence
        probs = torch.softmax(pred_flat, dim=-1)
        ent = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
        metrics['logit_entropy'] = float(ent.mean().item())
        metrics['logit_confidence_mean'] = float(probs.max(dim=-1).values.mean().item())

        num_classes = pred_flat.size(-1)
        if num_classes > 0:
            log_k = math.log(max(1, num_classes))
            metrics['logit_kl_uniform'] = float(
                (probs * (probs.log() - log_k)).sum(dim=-1).mean().item()
            )
        else:
            metrics['logit_kl_uniform'] = 0.0

        # Margin stats
        metrics.update(self._compute_logit_margins(predictions, labels))

        return metrics

    def _compute_ece(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        if predictions.numel() == 0:
            return 0.0

        if predictions.dim() == 3:
            b, s, v = predictions.shape
            predictions = predictions.view(b * s, v)
            labels = labels.view(b * s)

        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return 0.0

        preds_v = predictions[valid_mask]
        labels_v = labels[valid_mask]

        probs = torch.softmax(preds_v, dim=-1)
        confidences, pred_classes = probs.max(dim=-1)
        accuracies = pred_classes.eq(labels_v)
        bins = torch.linspace(0, 1, self.ece_num_bins + 1, device=predictions.device)

        ece = 0.0
        total = labels_v.numel()
        if total == 0:
            return 0.0

        for idx in range(self.ece_num_bins):
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

    @staticmethod
    def _compute_logit_margins(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if predictions.dim() == 3:
            b, s, v = predictions.shape
            pred_flat = predictions.view(b * s, v)
            labels_flat = labels.view(b * s)
        else:
            pred_flat = predictions
            labels_flat = labels

        logits = pred_flat.detach().cpu().numpy()
        labels_np = labels_flat.detach().cpu().numpy()
        margins = []

        for i in range(len(labels_np)):
            label_idx = int(labels_np[i])
            if label_idx == -100 or label_idx < 0:
                continue
            logit_vec = logits[i]
            if label_idx >= len(logit_vec):
                continue
            correct_logit = logit_vec[label_idx]
            mask = np.ones_like(logit_vec, dtype=bool)
            mask[label_idx] = False
            second_best = np.max(logit_vec[mask]) if np.any(mask) else correct_logit
            margins.append(correct_logit - second_best)

        if not margins:
            return {}

        arr = np.array(margins, dtype=np.float32)
        return {
            'logit_margin_mean': float(np.mean(arr)),
            'logit_margin_var': float(np.var(arr)),
            'logit_margin_p25': float(np.percentile(arr, 25)),
            'logit_margin_p50': float(np.percentile(arr, 50)),
            'logit_margin_p75': float(np.percentile(arr, 75)),
            'logit_margin_min': float(np.min(arr)),
        }
