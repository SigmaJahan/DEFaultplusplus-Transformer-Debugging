"""PositionalMetrics — early/late window accuracy and margins."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import MetricModule, _extract_logits
from ..inspector import ModelInspector
from ...config import ExtractionConfig


class PositionalMetrics(MetricModule):
    """Split sequence into early/late windows and compute per-window performance."""

    def __init__(self, inspector: ModelInspector, config: Optional[ExtractionConfig] = None):
        super().__init__(inspector)
        cfg = config or ExtractionConfig()
        self.pad_token_id = cfg.pad_token_id or 0

    def collect(
        self,
        *,
        model: Optional[torch.nn.Module] = None,
        outputs: Any = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        default = {
            'positional_accuracy_early': 0.0,
            'positional_accuracy_late': 0.0,
            'positional_accuracy_delta': 0.0,
            'positional_margin_early': 0.0,
            'positional_margin_late': 0.0,
            'positional_margin_delta': 0.0,
            'positional_loss_early': 0.0,
            'positional_loss_late': 0.0,
        }

        if model is None or labels is None or attention_mask is None or input_ids is None:
            return default
        if labels.numel() == 0:
            return default

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        was_training = model.training
        try:
            if was_training:
                model.eval()

            with torch.no_grad():
                early_batch = self._mask_batch_positions(batch, mode='early')
                late_batch = self._mask_batch_positions(batch, mode='late')

                early_outputs = model(**early_batch)
                late_outputs = model(**late_batch)

                early_logits = _extract_logits(early_outputs)
                late_logits = _extract_logits(late_outputs)

                if early_logits.dim() == 3:
                    # Decoder case
                    early_acc, early_margin, early_loss = self._decoder_stats(
                        early_logits, labels.clone(), early_batch.get('attention_mask')
                    )
                    late_acc, late_margin, late_loss = self._decoder_stats(
                        late_logits, labels.clone(), late_batch.get('attention_mask')
                    )
                else:
                    # Encoder case
                    valid_mask = labels != -100
                    if valid_mask.sum() == 0:
                        return default

                    early_preds = early_logits.argmax(dim=-1)
                    late_preds = late_logits.argmax(dim=-1)

                    early_acc = (early_preds[valid_mask] == labels[valid_mask]).float().mean().item()
                    late_acc = (late_preds[valid_mask] == labels[valid_mask]).float().mean().item()

                    early_margin = self._compute_margins(early_logits, labels).mean().item()
                    late_margin = self._compute_margins(late_logits, labels).mean().item()

                    early_loss = float(F.cross_entropy(early_logits, labels, ignore_index=-100).item())
                    late_loss = float(F.cross_entropy(late_logits, labels, ignore_index=-100).item())
        except Exception:
            return default
        finally:
            if was_training:
                model.train()

        return {
            'positional_accuracy_early': float(early_acc),
            'positional_accuracy_late': float(late_acc),
            'positional_accuracy_delta': float(late_acc - early_acc),
            'positional_margin_early': float(early_margin),
            'positional_margin_late': float(late_margin),
            'positional_margin_delta': float(late_margin - early_margin),
            'positional_loss_early': float(early_loss),
            'positional_loss_late': float(late_loss),
        }

    def _decoder_stats(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[float, float, float]:
        if attn_mask is not None:
            labels = labels.masked_fill(attn_mask == 0, -100)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels != -100
        if valid_mask.sum() == 0:
            return 0.0, 0.0, 0.0

        preds = shift_logits.argmax(dim=-1)
        acc = (preds[valid_mask] == shift_labels[valid_mask]).float().mean().item()

        margins = self._compute_margins(shift_logits, shift_labels)
        margin_mean = float(margins[valid_mask].mean().item()) if valid_mask.any() else 0.0

        loss_val = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        ).item()
        return acc, margin_mean, float(loss_val)

    @staticmethod
    def _compute_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            b, s, v = logits.shape
            logits_flat = logits.view(b * s, v)
            labels_flat = labels.view(b * s)
        else:
            logits_flat = logits
            labels_flat = labels

        num_classes = logits_flat.size(-1)
        if num_classes < 2:
            return torch.zeros_like(labels, dtype=logits.dtype)

        valid_mask = labels_flat != -100
        if valid_mask.sum() == 0:
            return torch.zeros_like(labels_flat, dtype=logits.dtype)

        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]

        top2 = torch.topk(logits_valid, k=2, dim=-1).values
        target_scores = logits_valid.gather(-1, labels_valid.unsqueeze(-1)).squeeze(-1)
        first, second = top2[:, 0], top2[:, 1]
        max_other = torch.where(target_scores == first, second, first)
        margins_valid = target_scores - max_other

        margins_flat = torch.zeros_like(labels_flat, dtype=logits.dtype)
        margins_flat[valid_mask] = margins_valid

        if logits.dim() == 3:
            return margins_flat.view(logits.size(0), logits.size(1))
        return margins_flat

    def _mask_batch_positions(self, batch: Dict, mode: str) -> Dict:
        clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids = clone.get('input_ids')
        attention_mask = clone.get('attention_mask')

        if input_ids is None or attention_mask is None:
            clone.pop('labels', None)
            return clone

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
                start, end = window, min(length, 2 * window)
            start = max(0, start + first)
            end = min(seq_len, end + first)

            mask = torch.zeros_like(attention_mask[idx])
            mask[start:end] = 1
            attention_mask[idx] = mask
            input_ids[idx][mask == 0] = self.pad_token_id

        clone['attention_mask'] = attention_mask
        clone['input_ids'] = input_ids
        clone.pop('labels', None)
        return clone
