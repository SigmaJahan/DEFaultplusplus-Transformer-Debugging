"""
Decoder Trainer for Language Modeling and Multiple Choice Tasks.

Handles training loop for decoder models with causal LM loss and MC accuracy.
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
    DECODER_TASK_TYPES,
    DEFAULT_INVARIANCE_PROBE_BATCHES,
    DEFAULT_INVARIANCE_PAD_TOKENS,
    DEFAULT_EPOCHS,
    DEFAULT_DECODER_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LOGGING_STEPS,
)
from src.metrics.metric_collector import MetricCollector
from src.metrics.generation_metrics import GenerationMetrics
from src.utils.logger import Logger
from src.utils.storage import HDF5MetricsStorage, SQLiteDatabase


class Trainer:
    """
    Trainer for decoder models on language modeling and multiple choice tasks.

    Supports:
    - Causal language modeling (WikiText, Lambada)
    - Multiple choice tasks (HellaSwag, PIQA, Winogrande, ARC)
    """

    def __init__(
        self,
        model,
        train_dataloader: Optional[DataLoader],
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        logger: Logger,
        config_id: str,
        task_type: str = "lm",
        h5_storage: Optional[HDF5MetricsStorage] = None,
        db_storage: Optional[SQLiteDatabase] = None,
        run_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize decoder trainer.

        Args:
            model: Decoder model wrapper
            train_dataloader: Training data loader (None for eval-only tasks like Lambada)
            val_dataloader: Validation data loader
            device: Device for training
            config: Training configuration
            logger: Logger instance
            config_id: Unique configuration identifier
            task_type: Task type ("lm", "mc", "lm_completion")
            h5_storage: HDF5 storage for metrics
            db_storage: SQLite database for results
            run_metadata: Additional metadata
        """
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

        # Training parameters
        self.epochs = config.get('epochs', DEFAULT_EPOCHS)
        self.learning_rate = config.get('learning_rate', DEFAULT_DECODER_LEARNING_RATE)
        self.weight_decay = config.get('weight_decay', DEFAULT_WEIGHT_DECAY)
        self.warmup_ratio = config.get('warmup_ratio', DEFAULT_WARMUP_RATIO)
        self.max_grad_norm = config.get('max_grad_norm', DEFAULT_MAX_GRAD_NORM)
        self.gradient_accumulation_steps = config.get(
            'gradient_accumulation_steps',
            DEFAULT_GRADIENT_ACCUMULATION_STEPS
        )
        self.logging_steps = config.get('logging_steps', DEFAULT_LOGGING_STEPS)

        # Initialize optimizer and scheduler if training
        if self.train_dataloader is not None:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

        # Initialize metric collector
        model_config = getattr(self.model.model, 'config', None)
        collector_config = dict(config)
        collector_config['model_type'] = 'decoder'
        collector_config['task_type'] = self.task_type
        collector_config['task_family'] = self.task_family

        if model_config is not None:
            config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else {}
            num_layers = getattr(model_config, 'num_hidden_layers',
                                getattr(model_config, 'n_layer', 6))

            collector_config.update({
                'model_config': config_dict,
                'num_hidden_layers': num_layers,
                'pad_token_id': getattr(model_config, 'pad_token_id', None),
                'eos_token_id': getattr(model_config, 'eos_token_id', None),
            })

        self.metric_collector = MetricCollector(
            device=device,
            collect_per_batch=False,
            collect_per_epoch=True,
            collect_attention=True,
            config=collector_config
        )

        # Initialize decoder-specific generation metrics (metrics 20-26)
        # Only if model has tokenizer (decoder models)
        self.decoder_metrics = None
        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
            self.decoder_metrics = GenerationMetrics(
                model=self.model.model,  # Unwrap to get HF model
                tokenizer=self.model.tokenizer,
                device=device,
                config=collector_config
            )

        # Training state
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
        except Exception as exc:
            try:
                self.logger.warning(f"Could not install signal handlers: {exc}")
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
                validation_metrics=validation_history
            )
            self._partial_saved = True
            try:
                self.logger.warning(f"✓ Saved partial metrics to HDF5: {self.h5_storage.filepath}")
            except Exception:
                pass
        except Exception as exc:
            try:
                self.logger.error(f"Failed to save partial metrics to HDF5: {exc}")
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
        """Map verbose task names to execution family."""
        t = (task_type or "").lower()
        if t in ("lm", "lm_wikitext", "lm_ptb") or t.startswith("lm_"):
            return "lm"
        if t in ("lm_completion",):
            return "lm_completion"
        if t in ("mc", "mc_hellaswag", "mc_piqa", "mc_arc") or t.startswith("mc_"):
            return "mc"
        if t in DECODER_TASK_TYPES:
            return t
        raise ValueError(f"Unknown task type '{task_type}'. Expected one of: {', '.join(DECODER_TASK_TYPES + ['lm', 'mc', 'lm_completion'])}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
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

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        return optimizer

    def _create_scheduler(self):
        """Create linear scheduler with warmup."""
        num_training_steps = len(self.train_dataloader) * self.epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return scheduler

    def _prepend_padding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepend padding tokens and truncate to original length."""
        if pad_tokens <= 0:
            return input_ids, attention_mask

        pad_id = getattr(self.model.model.config, 'pad_token_id', 0)
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            pad_block = torch.full(
                (batch_size, pad_tokens),
                pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            pad_mask = torch.zeros(
                (batch_size, pad_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            input_ids = torch.cat([pad_block, input_ids], dim=1)[:, :seq_len]
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)[:, :seq_len]
            return input_ids, attention_mask

        if input_ids.dim() == 3:
            batch_size, num_choices, seq_len = input_ids.shape
            pad_block = torch.full(
                (batch_size, num_choices, pad_tokens),
                pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device
            )
            pad_mask = torch.zeros(
                (batch_size, num_choices, pad_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            input_ids = torch.cat([pad_block, input_ids], dim=2)[:, :, :seq_len]
            attention_mask = torch.cat([pad_mask, attention_mask], dim=2)[:, :, :seq_len]
            return input_ids, attention_mask

        return input_ids, attention_mask

    def _mask_batch_positions(self, batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
        """Clone a batch and mask to early/mid/late positions, updating labels for LM."""
        clone: Dict[str, Any] = {}
        for key, value in batch.items():
            clone[key] = value.clone() if isinstance(value, torch.Tensor) else value

        input_ids = clone.get('input_ids')
        attention_mask = clone.get('attention_mask')
        labels = clone.get('labels')
        token_type_ids = clone.get('token_type_ids')

        if input_ids is None or attention_mask is None:
            return clone

        pad_id = getattr(self.model.model.config, 'pad_token_id', 0)
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
            start = max(0, start) + first
            end = min(length, end) + first
            start = max(0, start)
            end = min(seq_len, end)

            mask = torch.zeros_like(attention_mask[idx])
            mask[start:end] = 1
            attention_mask[idx] = mask

            drop_mask = mask == 0
            input_ids[idx][drop_mask] = pad_id
            if token_type_ids is not None:
                token_type_ids[idx][drop_mask] = 0
            if labels is not None:
                labels[idx][drop_mask] = -100

        clone['input_ids'] = input_ids
        clone['attention_mask'] = attention_mask
        if token_type_ids is not None:
            clone['token_type_ids'] = token_type_ids
        if labels is not None:
            clone['labels'] = labels

        return clone

    @staticmethod
    def _compute_lm_window_sums(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[int, int, float, float]:
        """Return (correct, count, margin_sum, loss_sum) for masked LM labels."""
        if logits.dim() != 3:
            return 0, 0, 0.0, 0.0

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels != -100
        token_count = int(valid_mask.sum().item())
        if token_count == 0:
            return 0, 0, 0.0, 0.0

        preds = shift_logits.argmax(dim=-1)
        correct = ((preds == shift_labels) & valid_mask).sum().item()

        vocab_size = shift_logits.size(-1)
        margin_sum = 0.0
        if vocab_size >= 2:
            logits_flat = shift_logits.view(-1, vocab_size)
            labels_flat = shift_labels.view(-1)
            valid_flat = labels_flat != -100
            if valid_flat.any():
                logits_valid = logits_flat[valid_flat]
                labels_valid = labels_flat[valid_flat]
                top2 = torch.topk(logits_valid, k=2, dim=-1).values
                target = logits_valid.gather(-1, labels_valid.unsqueeze(-1)).squeeze(-1)
                first = top2[:, 0]
                second = top2[:, 1]
                max_other = torch.where(target == first, second, first)
                margin_sum = float((target - max_other).sum().item())

        loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
        loss_sum = float(loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        ).item())

        return int(correct), token_count, margin_sum, loss_sum

    @staticmethod
    def _compute_invariance_disagreement(
        original_logits: torch.Tensor,
        transformed_logits: torch.Tensor,
        original_mask: Optional[torch.Tensor],
        transformed_mask: Optional[torch.Tensor]
    ) -> Optional[float]:
        """Return mean disagreement between argmax predictions on valid tokens."""
        orig_pred = original_logits.argmax(dim=-1)
        trans_pred = transformed_logits.argmax(dim=-1)

        if orig_pred.shape != trans_pred.shape:
            return None

        if original_mask is None or transformed_mask is None:
            return None

        valid = (original_mask > 0) & (transformed_mask > 0)
        if valid.sum().item() == 0:
            return None

        disagreement = (orig_pred[valid] != trans_pred[valid]).float().mean().item()
        return float(disagreement)

    def train_epoch_lm(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch for language modeling.

        Args:
            epoch: Epoch number

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.current_epoch = epoch

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} (LM)",
            disable=False
        )

        accumulated_loss = 0.0  # For gradient accumulation
        epoch_total_loss = 0.0  # Track total loss for the epoch
        num_batches = 0
        self.optimizer.zero_grad()
        step_start_time = time.perf_counter()
        end = time.perf_counter()
        data_time_accum = 0.0

        for batch_idx, batch in enumerate(pbar):
            data_time = time.perf_counter() - end
            data_time_accum += data_time
            # Move batch to device, filter to model-compatible keys only
            model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
            batch = {k: v.to(self.device) for k, v in batch.items()
                    if k in model_keys and isinstance(v, torch.Tensor)}

            # Validate input_ids and labels are within valid range
            if 'input_ids' in batch:
                max_id = batch['input_ids'].max().item()
                min_id = batch['input_ids'].min().item()
                # Access config through the wrapper's model attribute
                vocab_size = self.model.model.config.vocab_size

                # Debug logging for first batch
                if batch_idx == 0:
                    self.logger.info(f"Batch 0 validation: input_ids range=[{min_id}, {max_id}], vocab_size={vocab_size}")

                if max_id >= vocab_size or min_id < 0:
                    raise ValueError(
                        f"Invalid input_ids: min={min_id}, max={max_id}, vocab_size={vocab_size} "
                        f"at batch {batch_idx}"
                    )

            # Validate labels (excluding -100 which is ignore_index)
            if 'labels' in batch:
                valid_labels = batch['labels'][batch['labels'] != -100]
                if len(valid_labels) > 0:
                    max_label = valid_labels.max().item()
                    min_label = valid_labels.min().item()
                    # Access config through the wrapper's model attribute
                    vocab_size = self.model.model.config.vocab_size

                    # Debug logging for first batch
                    if batch_idx == 0:
                        num_masked = (batch['labels'] == -100).sum().item()
                        total = batch['labels'].numel()
                        self.logger.info(f"Batch 0 validation: labels range=[{min_label}, {max_label}], vocab_size={vocab_size}, masked={num_masked}/{total}")

                    if max_label >= vocab_size or min_label < 0:
                        raise ValueError(
                            f"Invalid labels: min={min_label}, max={max_label}, vocab_size={vocab_size} "
                            f"at batch {batch_idx}"
                        )

            # Forward pass with attention/hidden states for structural probes
            outputs = self.model.forward_with_attention(**batch)
            loss = outputs.loss

            # Check for invalid loss values
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss value: {loss.item()} at batch {batch_idx}")

            # Track epoch loss (unscaled)
            epoch_total_loss += outputs.loss.item()
            num_batches += 1

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # CRITICAL FIX (Bug #23): Collect metrics BEFORE zero_grad()
                # Otherwise all gradient norms will be zero!
                step_time = time.perf_counter() - step_start_time
                if self.global_step % self.logging_steps == 0:
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
                        data_time=data_time_accum
                    )

                # Optimizer step (must be AFTER metric collection)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Free memory (CRITICAL: synchronize before cache clear to prevent bus errors)
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU ops to finish
                    torch.cuda.empty_cache()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'ppl': f'{np.exp(accumulated_loss):.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

                accumulated_loss = 0.0
                self.global_step += 1
                step_start_time = time.perf_counter()
                data_time_accum = 0.0

            end = time.perf_counter()

        # CRITICAL FIX: Handle remainder steps when epoch length isn't divisible by gradient_accumulation_steps
        # Without this, the final partial accumulation window is dropped (no optimizer step)
        if accumulated_loss > 0.0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            # Collect metrics BEFORE zero_grad()
            step_time = time.perf_counter() - step_start_time
            if self.global_step % self.logging_steps == 0:
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
                    data_time=data_time_accum
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Free memory (CRITICAL: synchronize before cache clear to prevent bus errors)
            del outputs
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU ops to finish
                torch.cuda.empty_cache()

            self.global_step += 1

        # Return average loss over all batches
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_epoch_loss}

    def train_epoch_mc(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch for multiple choice.

        Args:
            epoch: Epoch number

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.current_epoch = epoch

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs} (MC)",
            disable=False
        )

        accumulated_loss = 0.0  # For gradient accumulation
        epoch_total_loss = 0.0  # Track total loss for the epoch
        num_batches = 0
        self.optimizer.zero_grad()
        step_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(pbar):
            # Batch has shape: [batch_size, num_choices, seq_len]
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_size, num_choices, seq_len = input_ids.shape

            # Flatten to [batch_size * num_choices, seq_len]
            flat_input_ids = input_ids.view(-1, seq_len)
            flat_attention_mask = attention_mask.view(-1, seq_len)

            # Forward pass with attention/hidden states for structural probes
            outputs = self.model.forward_with_attention(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
            logits = outputs.logits

            # Compute loss for each choice
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = flat_input_ids[..., 1:].contiguous()

            # Compute per-token loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            losses = losses.view(batch_size * num_choices, -1)

            # Average loss per choice
            choice_losses = losses.mean(dim=1)

            # Reshape to [batch_size, num_choices]
            choice_losses = choice_losses.view(batch_size, num_choices)

            # Cross-entropy over choices
            loss = F.cross_entropy(- choice_losses, labels)

            # Track epoch loss (unscaled)
            epoch_total_loss += loss.item()
            num_batches += 1

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Free memory (CRITICAL: synchronize before cache clear to prevent bus errors)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU ops to finish
                    torch.cuda.empty_cache()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

                accumulated_loss = 0.0
                self.global_step += 1

        # CRITICAL FIX: Handle remainder steps when epoch length isn't divisible by gradient_accumulation_steps
        # Without this, the final partial accumulation window is dropped (no optimizer step)
        if accumulated_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Free memory (CRITICAL: synchronize before cache clear to prevent bus errors)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU ops to finish
                torch.cuda.empty_cache()

            self.global_step += 1

        # Return average loss over all batches
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_epoch_loss}

    def evaluate_lm(self) -> Dict[str, float]:
        """
        Evaluate language modeling performance.

        Returns:
            Dictionary with perplexity and loss
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        invariance_disagreements: List[float] = []
        invariance_limit = max(1, self.config.get('invariance_probe_batches', DEFAULT_INVARIANCE_PROBE_BATCHES))
        invariance_seen = 0
        pad_tokens = int(self.config.get('invariance_pad_tokens', DEFAULT_INVARIANCE_PAD_TOKENS))
        enable_positional = bool(self.config.get('enable_positional_performance', True))
        positional_limit = max(
            0,
            int(self.config.get('positional_probe_batches', invariance_limit))
        ) if enable_positional else 0
        positional_seen = 0
        positional_counts = {
            'early_correct': 0,
            'late_correct': 0,
            'early_tokens': 0,
            'late_tokens': 0
        }
        positional_margins = {
            'early_sum': 0.0,
            'late_sum': 0.0
        }
        positional_losses = {
            'early_sum': 0.0,
            'late_sum': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating (LM)"):
                model_keys = {'input_ids', 'attention_mask', 'labels', 'token_type_ids'}
                batch = {k: v.to(self.device) for k, v in batch.items()
                        if k in model_keys and isinstance(v, torch.Tensor)}

                outputs = self.model.forward_with_attention(**batch)
                loss = outputs.loss

                # Count valid label tokens (excluding -100 padding)
                # HuggingFace loss is already averaged over valid tokens
                if 'labels' in batch:
                    num_tokens = (batch['labels'] != -100).sum().item()
                else:
                    num_tokens = batch['input_ids'].numel()

                # Accumulate: loss is mean, so multiply by count to get sum
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                if invariance_seen < invariance_limit and pad_tokens > 0:
                    padded_ids, padded_mask = self._prepend_padding(
                        batch['input_ids'],
                        batch['attention_mask'],
                        pad_tokens
                    )
                    transformed_outputs = self.model.forward_with_attention(
                        input_ids=padded_ids,
                        attention_mask=padded_mask
                    )
                    disagreement = self._compute_invariance_disagreement(
                        outputs.logits,
                        transformed_outputs.logits,
                        batch.get('attention_mask'),
                        padded_mask
                    )
                    if disagreement is not None:
                        invariance_disagreements.append(disagreement)
                    invariance_seen += 1

                if positional_seen < positional_limit and 'labels' in batch:
                    early_batch = self._mask_batch_positions(batch, mode='early')
                    late_batch = self._mask_batch_positions(batch, mode='late')
                    early_inputs = {k: v for k, v in early_batch.items()
                                    if k in model_keys and k != 'labels'}
                    late_inputs = {k: v for k, v in late_batch.items()
                                   if k in model_keys and k != 'labels'}

                    early_outputs = self.model.forward_with_attention(**early_inputs)
                    late_outputs = self.model.forward_with_attention(**late_inputs)

                    early_correct, early_count, early_margin_sum, early_loss_sum = self._compute_lm_window_sums(
                        early_outputs.logits,
                        early_batch['labels']
                    )
                    late_correct, late_count, late_margin_sum, late_loss_sum = self._compute_lm_window_sums(
                        late_outputs.logits,
                        late_batch['labels']
                    )

                    if positional_seen == 0:
                        try:
                            label_count = int((batch['labels'] != -100).sum().item())
                        except Exception:
                            label_count = -1
                        # Extra debug: show label positions + early/late windows for first valid sample
                        try:
                            attn = batch.get('attention_mask')
                            labels_dbg = batch.get('labels')
                            sample_idx = 0
                            if attn is not None and labels_dbg is not None:
                                lengths = attn.sum(dim=1).tolist()
                                # Find first sample with non-zero length
                                for i, length in enumerate(lengths):
                                    if int(length) > 0:
                                        sample_idx = i
                                        break
                                length = int(lengths[sample_idx])
                                window = max(1, length // 3)
                                early_range = (0, min(window, length))
                                late_range = (max(0, length - window), length)
                                label_positions = (labels_dbg[sample_idx] != -100).nonzero(as_tuple=False).view(-1).tolist()
                                valid_positions = (attn[sample_idx] != 0).nonzero(as_tuple=False).view(-1)
                                if valid_positions.numel() > 0:
                                    first_pos = int(valid_positions[0].item())
                                    last_pos = int(valid_positions[-1].item())
                                else:
                                    first_pos = None
                                    last_pos = None
                            else:
                                early_range = None
                                late_range = None
                                label_positions = []
                                first_pos = None
                                last_pos = None
                        except Exception:
                            early_range = None
                            late_range = None
                            label_positions = []
                            first_pos = None
                            last_pos = None
                        self.logger.info(
                            "Positional probe batch 0: label_count=%s early_valid=%s late_valid=%s "
                            "early_range=%s late_range=%s valid_span=(%s,%s) label_positions(sample0)=%s",
                            label_count,
                            int(early_count),
                            int(late_count),
                            early_range,
                            late_range,
                            first_pos,
                            last_pos,
                            label_positions[:10],
                        )

                    positional_counts['early_correct'] += early_correct
                    positional_counts['late_correct'] += late_correct
                    positional_counts['early_tokens'] += early_count
                    positional_counts['late_tokens'] += late_count
                    positional_margins['early_sum'] += early_margin_sum
                    positional_margins['late_sum'] += late_margin_sum
                    positional_losses['early_sum'] += early_loss_sum
                    positional_losses['late_sum'] += late_loss_sum

                    positional_seen += 1

                    del early_outputs
                    del late_outputs

        # CRITICAL: Synchronize GPU before computing metrics to prevent bus errors
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss)

        early_tokens = positional_counts['early_tokens']
        late_tokens = positional_counts['late_tokens']
        early_acc = positional_counts['early_correct'] / early_tokens if early_tokens > 0 else 0.0
        late_acc = positional_counts['late_correct'] / late_tokens if late_tokens > 0 else 0.0
        early_margin = positional_margins['early_sum'] / early_tokens if early_tokens > 0 else 0.0
        late_margin = positional_margins['late_sum'] / late_tokens if late_tokens > 0 else 0.0
        early_loss = positional_losses['early_sum'] / early_tokens if early_tokens > 0 else 0.0
        late_loss = positional_losses['late_sum'] / late_tokens if late_tokens > 0 else 0.0

        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'positional_invariance': float(np.mean(invariance_disagreements)) if invariance_disagreements else 0.0,
            'positional_accuracy_early': float(early_acc),
            'positional_accuracy_late': float(late_acc),
            'positional_accuracy_delta': float(late_acc - early_acc),
            'positional_margin_early': float(early_margin),
            'positional_margin_late': float(late_margin),
            'positional_margin_delta': float(late_margin - early_margin),
            'positional_loss_early': float(early_loss),
            'positional_loss_late': float(late_loss)
        }

        # CRITICAL (Bug #18 Fix): Compute decoder-specific generation metrics (20-26)
        # These measure generation quality, repetition, KV-cache correctness, etc.
        if self.decoder_metrics is not None:
            try:
                generation_metrics = self.decoder_metrics.compute_all_generation_metrics(
                    self.val_dataloader
                )
                metrics.update(generation_metrics)
                self.logger.info(f"✓ Computed generation metrics (20-26): {len(generation_metrics)} metrics")
            except Exception as e:
                self.logger.warning(f"Failed to compute generation metrics: {e}")
                # Continue without generation metrics rather than failing the entire run

        return metrics

    def evaluate_mc(self) -> Dict[str, float]:
        """
        Evaluate multiple choice performance.

        Returns:
            Dictionary with accuracy
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        invariance_disagreements: List[float] = []
        invariance_limit = max(1, self.config.get('invariance_probe_batches', DEFAULT_INVARIANCE_PROBE_BATCHES))
        invariance_seen = 0
        pad_tokens = int(self.config.get('invariance_pad_tokens', DEFAULT_INVARIANCE_PAD_TOKENS))

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating (MC)"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                batch_size, num_choices, seq_len = input_ids.shape

                # Flatten
                flat_input_ids = input_ids.view(-1, seq_len)
                flat_attention_mask = attention_mask.view(-1, seq_len)

                # Forward pass with attention/hidden states for structural probes
                outputs = self.model.forward_with_attention(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
                logits = outputs.logits

                # Compute per-choice loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = flat_input_ids[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                losses = losses.view(batch_size * num_choices, -1)
                choice_losses = losses.mean(dim=1)
                choice_losses = choice_losses.view(batch_size, num_choices)

                # Predict choice with lowest loss
                predictions = torch.argmin(choice_losses, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if invariance_seen < invariance_limit and pad_tokens > 0:
                    padded_ids, padded_mask = self._prepend_padding(
                        input_ids,
                        attention_mask,
                        pad_tokens
                    )
                    flat_padded_ids = padded_ids.view(-1, seq_len)
                    flat_padded_mask = padded_mask.view(-1, seq_len)
                    transformed_outputs = self.model.forward_with_attention(
                        input_ids=flat_padded_ids,
                        attention_mask=flat_padded_mask
                    )
                    disagreement = self._compute_invariance_disagreement(
                        logits,
                        transformed_outputs.logits,
                        flat_attention_mask,
                        flat_padded_mask
                    )
                    if disagreement is not None:
                        invariance_disagreements.append(disagreement)
                    invariance_seen += 1

        # CRITICAL: Synchronize GPU before computing metrics to prevent bus errors
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return {
            'eval_accuracy': accuracy,
            'positional_invariance': float(np.mean(invariance_disagreements)) if invariance_disagreements else 0.0
        }

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary of training results
        """
        results = {}
        try:
            # Training loop (if train_dataloader available)
            if self.train_dataloader is not None:
                for epoch in range(self.epochs):
                    self._check_termination()
                    if self.task_family == "lm" or self.task_family == "lm_completion":
                        train_metrics = self.train_epoch_lm(epoch)
                    elif self.task_family == "mc":
                        train_metrics = self.train_epoch_mc(epoch)
                    else:
                        raise ValueError(f"Unknown task type: {self.task_type}")

                    # Evaluate
                    if self.task_family == "lm" or self.task_family == "lm_completion":
                        eval_metrics = self.evaluate_lm()
                    elif self.task_family == "mc":
                        eval_metrics = self.evaluate_mc()

                    # Log epoch results
                    all_metrics = {**train_metrics, **eval_metrics}
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs} - " +
                                   ", ".join([f"{k}: {v:.4f}" for k, v in all_metrics.items()]))

                    # Record validation metrics with canonical names (no duplicates).
                    # Kill criteria look for val_loss, val_perplexity (added by record_validation_metrics).
                    validation_metrics = {}
                    if 'eval_loss' in eval_metrics:
                        validation_metrics['loss'] = eval_metrics['eval_loss']
                    if 'eval_perplexity' in eval_metrics:
                        validation_metrics['perplexity'] = eval_metrics['eval_perplexity']
                    if 'eval_accuracy' in eval_metrics:
                        validation_metrics['accuracy'] = eval_metrics['eval_accuracy']
                    if 'positional_invariance' in eval_metrics:
                        validation_metrics['positional_invariance'] = eval_metrics['positional_invariance']

                    for key, value in eval_metrics.items():
                        if key.startswith('positional_'):
                            validation_metrics[key] = value

                    # Generation metrics (already properly named from evaluate_lm)
                    for key in ('repetition_max_run', 'repetition_distinct_1', 'repetition_distinct_2',
                                'generation_mean_length', 'generation_eos_ratio',
                                'cache_correctness', 'cache_nll_divergence'):
                        if key in eval_metrics:
                            validation_metrics[key] = eval_metrics[key]

                    if validation_metrics and self.metric_collector:
                        self.metric_collector.record_validation_metrics(epoch, validation_metrics)

                    # CRITICAL FIX (Bug #17): Finalize epoch to populate epoch_metrics_history
                    # Without this, get_final_metrics() returns empty dict causing all zeros
                    if self.metric_collector:
                        self.metric_collector.finalize_epoch(epoch)
                        self._last_completed_epoch = epoch

                    # Store epoch metrics
                    results[f'epoch_{epoch}'] = {**train_metrics, **eval_metrics}

                    self._check_termination()

            else:
                # Eval-only (e.g., Lambada)
                self._check_termination()
                if self.task_family == "lm" or self.task_family == "lm_completion":
                    eval_metrics = self.evaluate_lm()
                elif self.task_family == "mc":
                    eval_metrics = self.evaluate_mc()

                results['eval_only'] = eval_metrics

            self._check_termination()

            # Final evaluation
            if self.task_family == "lm" or self.task_family == "lm_completion":
                final_metrics = self.evaluate_lm()
            elif self.task_family == "mc":
                final_metrics = self.evaluate_mc()

            results['final'] = final_metrics

            # CRITICAL FIX (Bug #20): Save metrics to HDF5
            # Collect all metrics for HDF5 storage
            if self.h5_storage and self.metric_collector:
                try:
                    epoch_history = self.metric_collector.epoch_metrics_history
                    validation_history = self.metric_collector.validation_history
                    final_metrics_dict = self.metric_collector.get_final_metrics()

                    # Prepare metadata
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
                        validation_metrics=validation_history
                    )

                    self.logger.info(f"✓ Saved metrics to HDF5: {self.h5_storage.filepath}")

                except Exception as e:
                    self.logger.error(f"Failed to save metrics to HDF5: {e}")

            return results
        finally:
            self._restore_signal_handlers()
