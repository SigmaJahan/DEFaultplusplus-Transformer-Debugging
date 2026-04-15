"""Encoder Trainer for Classification and NER Tasks.

Handles training loop for encoder models with CrossEntropyLoss for
classification and token-level NER. No generation, no MC, no causal LM loss.
"""

import gc
import signal
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.constants import (
    ENCODER_TASK_TYPES,
    DEFAULT_EPOCHS,
    DEFAULT_ENCODER_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LOGGING_STEPS,
)
from src.metrics.classification_metrics import ClassificationMetrics
from src.utils.logger import Logger
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase


class Trainer:
    """Trainer for encoder models on classification and NER tasks."""

    def __init__(
        self,
        model,
        train_dataloader: Optional[DataLoader],
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        logger: Logger,
        config_id: str,
        task_type: str = "cls",
        h5_storage: Optional[HDF5MetricsStorage] = None,
        db_storage: Optional[SQLiteDatabase] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        metric_collector=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.logger = logger
        self.config_id = str(config_id)
        self.task_type = str(task_type)
        self.task_family = self._task_family(task_type)
        self.h5_storage = h5_storage
        self.db_storage = db_storage
        self.run_metadata = run_metadata or {}
        self.metric_collector = metric_collector

        self.epochs = config.get('epochs', DEFAULT_EPOCHS)
        self.learning_rate = config.get('learning_rate', DEFAULT_ENCODER_LEARNING_RATE)
        self.weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        self.warmup_ratio = config.get('warmup_ratio', DEFAULT_WARMUP_RATIO)
        self.max_grad_norm = config.get('max_grad_norm', DEFAULT_MAX_GRAD_NORM)
        self.gradient_accumulation_steps = config.get(
            'gradient_accumulation_steps', DEFAULT_GRADIENT_ACCUMULATION_STEPS
        )
        self.logging_steps = config.get('logging_steps', DEFAULT_LOGGING_STEPS)
        self.num_labels = config.get('num_labels', 2)
        self.is_regression = self.task_type in ('cls_stsb', 'stsb')

        if self.train_dataloader is not None:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

        self.cls_metrics = None
        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
            self.cls_metrics = ClassificationMetrics(
                model=self.model.model,
                tokenizer=self.model.tokenizer,
                device=device,
                config=config,
            )

        self.global_step = 0
        self.current_epoch = 0
        self._terminate_requested = False
        self._terminate_signal = None
        self._partial_saved = False
        self._last_completed_epoch = -1
        self._orig_sigterm = None
        self._orig_sigint = None
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._orig_sigterm = signal.getsignal(signal.SIGTERM)
            self._orig_sigint = signal.getsignal(signal.SIGINT)

            def _handle_signal(signum, _frame):
                if self._terminate_requested:
                    return
                self._terminate_requested = True
                self._terminate_signal = signum
                try:
                    self.logger.warning(
                        f"Received signal {signum}; will save partial metrics at next checkpoint."
                    )
                except Exception:
                    pass

            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)
        except Exception:
            pass

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if self._orig_sigterm is not None:
                signal.signal(signal.SIGTERM, self._orig_sigterm)
            if self._orig_sigint is not None:
                signal.signal(signal.SIGINT, self._orig_sigint)
        except Exception:
            pass

    def _save_partial_metrics(self, reason: str) -> None:
        if self._partial_saved:
            return
        if not self.h5_storage or not self.metric_collector:
            return
        try:
            if self.h5_storage.load_configuration_metrics(self.config_id):
                self._partial_saved = True
                return
        except Exception:
            pass
        try:
            epoch_history = self.metric_collector.epoch_metrics_history
            validation_history = self.metric_collector.validation_history
            final_metrics_dict = self.metric_collector.get_final_metrics()
            metadata = {
                'config_id': self.config_id,
                'task_type': self.task_type,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'run_metadata': self.run_metadata,
                'partial': True,
                'partial_reason': reason,
                'last_completed_epoch': self._last_completed_epoch,
            }
            self.h5_storage.save_configuration_metrics(
                config_id=self.config_id,
                epoch_metrics=epoch_history,
                final_metrics=final_metrics_dict,
                metadata=metadata,
                validation_metrics=validation_history,
            )
            self._partial_saved = True
        except Exception as exc:
            try:
                self.logger.error(f"Failed to save partial metrics: {exc}")
            except Exception:
                pass

    def _check_termination(self) -> None:
        if not self._terminate_requested:
            return
        reason = f"signal_{self._terminate_signal}" if self._terminate_signal else "signal"
        self._save_partial_metrics(reason)
        raise KeyboardInterrupt("Training interrupted by signal")

    @staticmethod
    def _task_family(task_type: str) -> str:
        t = (task_type or "").lower()
        if t in ("cls", "classification") or t.startswith("cls_"):
            return "cls"
        if t in ("ner", "ner_conll2003"):
            return "ner"
        if t == "mlm":
            return "mlm"
        if t in ENCODER_TASK_TYPES:
            return t
        raise ValueError(f"Unknown task type '{task_type}'")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

    def _create_scheduler(self):
        num_training_steps = len(self.train_dataloader) * self.epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train_epoch_cls(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.current_epoch = epoch

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} (CLS)",
            disable=False,
        )

        accumulated_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_samples = 0
        self.optimizer.zero_grad()
        step_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(pbar):
            model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
            batch = {k: v.to(self.device) for k, v in batch.items()
                    if k in model_keys and isinstance(v, torch.Tensor)}

            outputs = self.model.forward_with_attention(**batch)
            loss = outputs.loss

            if loss is None:
                logits = outputs.logits
                labels = batch['labels']
                if self.is_regression:
                    loss = F.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss value: {loss.item()} at batch {batch_idx}")

            epoch_total_loss += loss.item()
            num_batches += 1

            if not self.is_regression and outputs.logits is not None:
                preds = outputs.logits.argmax(dim=-1)
                total_correct += (preds == batch['labels']).sum().item()
                total_samples += batch['labels'].numel()

            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                step_time = time.perf_counter() - step_start_time
                if self.metric_collector and self.global_step % self.logging_steps == 0:
                    self.metric_collector.collect_batch_metrics(
                        loss=accumulated_loss,
                        model=self.model.model,
                        optimizer=self.optimizer,
                        outputs=outputs,
                        labels=batch.get('labels'),
                        batch_idx=batch_idx,
                        epoch=epoch,
                        batch=batch,
                        step_time=step_time,
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

                accumulated_loss = 0.0
                self.global_step += 1
                step_start_time = time.perf_counter()

        if accumulated_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self.global_step += 1

        avg_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {'train_loss': avg_loss, 'train_accuracy': accuracy}

    def train_epoch_ner(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.current_epoch = epoch

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} (NER)",
            disable=False,
        )

        accumulated_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_tokens = 0
        self.optimizer.zero_grad()
        step_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(pbar):
            model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
            batch = {k: v.to(self.device) for k, v in batch.items()
                    if k in model_keys and isinstance(v, torch.Tensor)}

            outputs = self.model.forward_with_attention(**batch)
            loss = outputs.loss

            if loss is None:
                logits = outputs.logits
                labels = batch['labels']
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss value: {loss.item()} at batch {batch_idx}")

            epoch_total_loss += loss.item()
            num_batches += 1

            if outputs.logits is not None:
                preds = outputs.logits.argmax(dim=-1)
                mask = batch['labels'] != -100
                total_correct += (preds[mask] == batch['labels'][mask]).sum().item()
                total_tokens += mask.sum().item()

            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                step_time = time.perf_counter() - step_start_time
                if self.metric_collector and self.global_step % self.logging_steps == 0:
                    self.metric_collector.collect_batch_metrics(
                        loss=accumulated_loss,
                        model=self.model.model,
                        optimizer=self.optimizer,
                        outputs=outputs,
                        labels=batch.get('labels'),
                        batch_idx=batch_idx,
                        epoch=epoch,
                        batch=batch,
                        step_time=step_time,
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

                accumulated_loss = 0.0
                self.global_step += 1
                step_start_time = time.perf_counter()

        if accumulated_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self.global_step += 1

        avg_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        return {'train_loss': avg_loss, 'train_token_accuracy': token_accuracy}

    def evaluate_cls(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating (CLS)"):
                model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
                batch = {k: v.to(self.device) for k, v in batch.items()
                        if k in model_keys and isinstance(v, torch.Tensor)}

                outputs = self.model.forward_with_attention(**batch)
                loss = outputs.loss

                if loss is None and outputs.logits is not None:
                    logits = outputs.logits
                    labels = batch['labels']
                    if self.is_regression:
                        loss = F.mse_loss(logits.squeeze(-1), labels.float())
                    else:
                        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

                if outputs.logits is not None and not self.is_regression:
                    preds = outputs.logits.argmax(dim=-1)
                    all_preds.append(preds.cpu())
                    all_labels.append(batch['labels'].cpu())

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {'val_loss': avg_loss}

        if all_preds and not self.is_regression:
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)

            preds_np = preds.numpy().flatten()
            labels_np = labels.numpy().flatten()
            mask = labels_np >= 0
            preds_np = preds_np[mask]
            labels_np = labels_np[mask]

            if len(labels_np) > 0:
                accuracy = float(np.mean(preds_np == labels_np))
                metrics['val_accuracy'] = accuracy

                unique_labels = np.unique(labels_np)
                per_class_precision, per_class_recall, per_class_f1 = [], [], []
                for cls in unique_labels:
                    tp = np.sum((preds_np == cls) & (labels_np == cls))
                    fp = np.sum((preds_np == cls) & (labels_np != cls))
                    fn = np.sum((preds_np != cls) & (labels_np == cls))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    per_class_precision.append(precision)
                    per_class_recall.append(recall)
                    per_class_f1.append(f1)

                metrics['val_f1'] = float(np.mean(per_class_f1))
                metrics['val_precision'] = float(np.mean(per_class_precision))
                metrics['val_recall'] = float(np.mean(per_class_recall))

        return metrics

    def evaluate_ner(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_tokens = 0
        entity_tp = 0
        entity_fp = 0
        entity_fn = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating (NER)"):
                model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
                batch = {k: v.to(self.device) for k, v in batch.items()
                        if k in model_keys and isinstance(v, torch.Tensor)}

                outputs = self.model.forward_with_attention(**batch)
                loss = outputs.loss
                if loss is None and outputs.logits is not None:
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, self.num_labels),
                        batch['labels'].view(-1),
                        ignore_index=-100,
                    )

                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1

                if outputs.logits is not None:
                    preds = outputs.logits.argmax(dim=-1)
                    mask = batch['labels'] != -100
                    total_correct += (preds[mask] == batch['labels'][mask]).sum().item()
                    total_tokens += mask.sum().item()

                    preds_np = preds.cpu().numpy()
                    labels_np = batch['labels'].cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    for i in range(preds_np.shape[0]):
                        p = preds_np[i][mask_np[i]]
                        l = labels_np[i][mask_np[i]]
                        for cls_id in range(1, self.num_labels):
                            tp = int(np.sum((p == cls_id) & (l == cls_id)))
                            fp = int(np.sum((p == cls_id) & (l != cls_id)))
                            fn = int(np.sum((p != cls_id) & (l == cls_id)))
                            entity_tp += tp
                            entity_fp += fp
                            entity_fn += fn

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0.0
        recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'val_loss': avg_loss,
            'val_token_accuracy': token_accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall,
        }

    def train(self) -> Dict[str, Any]:
        results = {}
        try:
            if self.train_dataloader is not None:
                for epoch in range(self.epochs):
                    self._check_termination()

                    if self.task_family == "cls":
                        train_metrics = self.train_epoch_cls(epoch)
                    elif self.task_family == "ner":
                        train_metrics = self.train_epoch_ner(epoch)
                    else:
                        raise ValueError(f"Unknown task type: {self.task_type}")

                    if self.task_family == "cls":
                        eval_metrics = self.evaluate_cls()
                    elif self.task_family == "ner":
                        eval_metrics = self.evaluate_ner()

                    all_metrics = {**train_metrics, **eval_metrics}
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - " +
                        ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()])
                    )

                    validation_metrics = {}
                    if 'val_loss' in eval_metrics:
                        validation_metrics['loss'] = eval_metrics['val_loss']
                    if 'val_accuracy' in eval_metrics:
                        validation_metrics['accuracy'] = eval_metrics['val_accuracy']
                    if 'val_f1' in eval_metrics:
                        validation_metrics['f1_score'] = eval_metrics['val_f1']
                    if 'val_precision' in eval_metrics:
                        validation_metrics['precision'] = eval_metrics['val_precision']
                    if 'val_recall' in eval_metrics:
                        validation_metrics['recall'] = eval_metrics['val_recall']
                    if 'val_token_accuracy' in eval_metrics:
                        validation_metrics['token_accuracy'] = eval_metrics['val_token_accuracy']

                    if validation_metrics and self.metric_collector:
                        self.metric_collector.record_validation_metrics(epoch, validation_metrics)

                    if self.metric_collector:
                        self.metric_collector.finalize_epoch(epoch)
                        self._last_completed_epoch = epoch

                    results[f'epoch_{epoch}'] = {**train_metrics, **eval_metrics}
                    self._check_termination()

            self._check_termination()

            if self.task_family == "cls":
                final_metrics = self.evaluate_cls()
            elif self.task_family == "ner":
                final_metrics = self.evaluate_ner()
            else:
                final_metrics = {}

            # Classification metrics 20-26
            if self.cls_metrics is not None and self.task_family == "cls":
                try:
                    cls_eval = self.cls_metrics.compute_all_classification_metrics(self.val_dataloader)
                    final_metrics.update(cls_eval)
                except Exception as e:
                    self.logger.warning(f"Failed to compute classification metrics: {e}")

            results['final'] = final_metrics

            if self.h5_storage and self.metric_collector:
                try:
                    epoch_history = self.metric_collector.epoch_metrics_history
                    validation_history = self.metric_collector.validation_history
                    final_metrics_dict = self.metric_collector.get_final_metrics()

                    metadata = {
                        'config_id': self.config_id,
                        'task_type': self.task_type,
                        'epochs': self.epochs,
                        'learning_rate': self.learning_rate,
                        'run_metadata': self.run_metadata,
                    }

                    self.h5_storage.save_configuration_metrics(
                        config_id=self.config_id,
                        epoch_metrics=epoch_history,
                        final_metrics=final_metrics_dict,
                        metadata=metadata,
                        validation_metrics=validation_history,
                    )
                    self.logger.info(f"Saved metrics to HDF5: {self.h5_storage.filepath}")
                except Exception as e:
                    self.logger.error(f"Failed to save metrics to HDF5: {e}")

            return results
        finally:
            self._restore_signal_handlers()
