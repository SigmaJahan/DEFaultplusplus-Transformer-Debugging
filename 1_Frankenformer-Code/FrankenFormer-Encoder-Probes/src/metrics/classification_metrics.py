"""Encoder-specific classification metrics (IDs 20-26)."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.constants import (
    METRIC_ID_CLS_ACCURACY,
    METRIC_ID_CLS_F1,
    METRIC_ID_CLS_PRECISION,
    METRIC_ID_CLS_RECALL,
    METRIC_ID_CLS_AUC,
    METRIC_ID_MLM_ACCURACY,
    METRIC_ID_MLM_PERPLEXITY,
)


class ClassificationMetrics:
    def __init__(self, model, tokenizer, device: torch.device, config: Optional[Dict] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}
        self.num_labels = self.config.get('num_labels', getattr(model.config, 'num_labels', 2))

    def compute_classification_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        if predictions is None or labels is None:
            return {}
        if predictions.dim() >= 2:
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions
        preds = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        mask = labels_np >= 0
        preds = preds[mask]
        labels_np = labels_np[mask]
        if len(labels_np) == 0:
            return {'cls_accuracy': 0.0, 'cls_f1': 0.0, 'cls_precision': 0.0, 'cls_recall': 0.0}

        accuracy = float(np.mean(preds == labels_np))
        unique_labels = np.unique(labels_np)
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []

        for cls in unique_labels:
            tp = np.sum((preds == cls) & (labels_np == cls))
            fp = np.sum((preds == cls) & (labels_np != cls))
            fn = np.sum((preds != cls) & (labels_np == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_precision.append(precision)
            per_class_recall.append(recall)
            per_class_f1.append(f1)

        return {
            'cls_accuracy': accuracy,
            'cls_f1': float(np.mean(per_class_f1)),
            'cls_precision': float(np.mean(per_class_precision)),
            'cls_recall': float(np.mean(per_class_recall)),
        }

    def compute_auc_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        if predictions is None or labels is None:
            return {'cls_auc': 0.0}
        labels_np = labels.cpu().numpy().flatten()
        mask = labels_np >= 0
        labels_np = labels_np[mask]
        unique_labels = np.unique(labels_np)
        if len(unique_labels) < 2:
            return {'cls_auc': 0.0}

        if predictions.dim() < 2:
            return {'cls_auc': 0.0}

        probs = F.softmax(predictions, dim=-1).cpu().numpy()
        if mask.sum() < probs.shape[0]:
            probs = probs[mask]

        if self.num_labels == 2 and probs.shape[-1] == 2:
            pos_probs = probs[:, 1]
            auc = self._binary_auc(labels_np, pos_probs)
        else:
            aucs = []
            for cls in unique_labels:
                binary_labels = (labels_np == cls).astype(float)
                if probs.shape[-1] > cls:
                    cls_probs = probs[:, int(cls)]
                else:
                    continue
                auc_cls = self._binary_auc(binary_labels, cls_probs)
                if auc_cls is not None:
                    aucs.append(auc_cls)
            auc = float(np.mean(aucs)) if aucs else 0.0

        return {'cls_auc': auc if auc is not None else 0.0}

    @staticmethod
    def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
        if len(np.unique(labels)) < 2:
            return None
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        n_pos = np.sum(sorted_labels == 1)
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        tp = 0
        auc = 0.0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                auc += tp
        return float(auc / (n_pos * n_neg))

    def compute_mlm_metrics(self, model, dataloader) -> Dict[str, float]:
        model.eval()
        total_correct = 0
        total_masked = 0
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                    labels = batch.get('labels')
                else:
                    continue

                if input_ids is None or labels is None:
                    continue

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                except (RuntimeError, TypeError, AttributeError):
                    continue

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    n_batches += 1

                if outputs.logits is not None:
                    mask = labels != -100
                    if mask.any():
                        preds = outputs.logits[mask].argmax(dim=-1)
                        targets = labels[mask]
                        total_correct += (preds == targets).sum().item()
                        total_masked += mask.sum().item()

        mlm_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        mlm_perplexity = float(np.exp(min(avg_loss, 100.0)))

        return {
            'mlm_accuracy': mlm_accuracy,
            'mlm_perplexity': mlm_perplexity,
        }

    def compute_all_classification_metrics(self, val_dataloader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                    labels = batch.get('labels')
                else:
                    continue

                if input_ids is None or labels is None:
                    continue

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                except (RuntimeError, TypeError, AttributeError):
                    continue

                logits = outputs.logits if hasattr(outputs, 'logits') else None
                if logits is None:
                    continue

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        if not all_logits:
            return {
                'cls_accuracy': 0.0, 'cls_f1': 0.0,
                'cls_precision': 0.0, 'cls_recall': 0.0,
                'cls_auc': 0.0,
            }

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        metrics = {}
        metrics.update(self.compute_classification_metrics(logits, labels))
        metrics.update(self.compute_auc_metrics(logits, labels))
        return metrics
