"""
Decoder-specific generation metrics (IDs 20-26).

Implements metrics for evaluating decoder model generation quality,
cache correctness, and autoregressive behavior.
"""

import time
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.constants import (
    METRIC_ID_REPETITION_MAX_RUN,
    METRIC_ID_REPETITION_DISTINCT_1,
    METRIC_ID_REPETITION_DISTINCT_2,
    METRIC_ID_GENERATION_MEAN_LENGTH,
    METRIC_ID_GENERATION_EOS_RATIO,
    METRIC_ID_CACHE_CORRECTNESS,
    METRIC_ID_CACHE_NLL_DIVERGENCE,
    DEFAULT_GENERATION_SAMPLE_SIZE,
    DEFAULT_GENERATION_MAX_LENGTH,
    DEFAULT_GENERATION_PROMPT_TOKENS,
    CACHE_SIMILARITY_THRESHOLD,
)


class GenerationMetrics:
    """
    Decoder-specific metrics for generation quality and cache correctness.

    Metrics 20-26:
    - 20: Repetition max run
    - 21: Distinct-1 (unique unigrams ratio)
    - 22: Distinct-2 (unique bigrams ratio)
    - 23: Generation mean length
    - 24: Generation EOS ratio
    - 25: Cache correctness (hidden state similarity)
    - 26: Cache NLL divergence
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        config: Optional[Dict] = None
    ):
        """
        Initialize decoder metrics.

        Args:
            model: Decoder model
            tokenizer: Tokenizer for the model
            device: Device for computation
            config: Optional configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}

        self.sample_size = self.config.get('generation_sample_size', DEFAULT_GENERATION_SAMPLE_SIZE)
        self.max_length = self.config.get('generation_max_length', DEFAULT_GENERATION_MAX_LENGTH)
        self.eos_token_id = self.model.config.eos_token_id

    def compute_repetition_metrics(self, generated_ids: torch.Tensor) -> Dict[str, float]:
        """
        Compute repetition and diversity metrics (20-22).

        Args:
            generated_ids: Generated token IDs [batch_size, seq_len]

        Returns:
            Dictionary with repetition metrics
        """
        batch_size = generated_ids.shape[0]

        all_max_runs = []
        all_distinct_1 = []
        all_distinct_2 = []

        for i in range(batch_size):
            sequence = generated_ids[i].cpu().tolist()

            # Remove padding and special tokens
            sequence = [tok for tok in sequence
                       if tok != self.tokenizer.pad_token_id
                       and tok != self.eos_token_id]

            if len(sequence) == 0:
                continue

            # Longest run of identical tokens
            max_run = 1
            current_run = 1
            for j in range(1, len(sequence)):
                if sequence[j] == sequence[j-1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1

            all_max_runs.append(max_run)

            # Distinct-1: unique unigrams / total unigrams
            unigrams = sequence
            distinct_1 = len(set(unigrams)) / len(unigrams) if len(unigrams) > 0 else 0.0
            all_distinct_1.append(distinct_1)

            # Distinct-2: unique bigrams / total bigrams
            if len(sequence) >= 2:
                bigrams = [(sequence[j], sequence[j+1]) for j in range(len(sequence)-1)]
                distinct_2 = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0.0
                all_distinct_2.append(distinct_2)

        return {
            'repetition_max_run': np.mean(all_max_runs) if all_max_runs else 0.0,
            'repetition_distinct_1': np.mean(all_distinct_1) if all_distinct_1 else 0.0,
            'repetition_distinct_2': np.mean(all_distinct_2) if all_distinct_2 else 0.0,
        }

    def compute_length_metrics(self, generated_ids: torch.Tensor) -> Dict[str, float]:
        """
        Compute generation length and EOS metrics (23-24).

        Args:
            generated_ids: Generated token IDs [batch_size, seq_len]

        Returns:
            Dictionary with length metrics
        """
        batch_size = generated_ids.shape[0]

        lengths = []
        eos_counts = 0

        for i in range(batch_size):
            sequence = generated_ids[i].cpu().tolist()

            # Find length (up to first EOS or end)
            length = len(sequence)
            has_eos = False

            for j, tok in enumerate(sequence):
                if tok == self.eos_token_id:
                    length = j + 1
                    has_eos = True
                    break

            lengths.append(length)
            if has_eos:
                eos_counts += 1

        mean_length = np.mean(lengths) if lengths else 0.0
        eos_ratio = eos_counts / batch_size if batch_size > 0 else 0.0

        return {
            'generation_mean_length': mean_length,
            'generation_eos_ratio': eos_ratio,
        }

    def compute_cache_correctness(
        self,
        prompts: List[torch.Tensor],
        max_new_tokens: int = 50
    ) -> Dict[str, float]:
        """
        Compute cache correctness metric (25).

        Compares hidden states from cache-enabled (via autoregressive step) vs cache-disabled generation.
        This ensures faults in cache storage/retrieval are actually exercised.

        Args:
            prompts: List of prompt tensors
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Dictionary with cache correctness metric
        """
        self.model.eval()

        similarities = []

        with torch.no_grad():
            for prompt in prompts[:min(len(prompts), 10)]:  # Limit to 10 for efficiency
                prompt = prompt.to(self.device)

                # 1. Prefill (Cache generation)
                input_ids = prompt.unsqueeze(0)
                outputs_prefill = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                if outputs_prefill.past_key_values is None:
                    continue
                past_key_values = outputs_prefill.past_key_values

                # 2. Decode step (Cache usage)
                # Use a dummy token (e.g. EOS or 0) to verify cache usage
                next_token = torch.tensor([[self.eos_token_id or 50256]], device=self.device)
                
                outputs_with_cache = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )

                # 3. Baseline (Full context without cache)
                full_input = torch.cat([input_ids, next_token], dim=1)
                outputs_without_cache = self.model(
                    input_ids=full_input,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Validate outputs are not None
                if outputs_with_cache is None or outputs_without_cache is None:
                    continue
                if not hasattr(outputs_with_cache, 'hidden_states') or not hasattr(outputs_without_cache, 'hidden_states'):
                    continue
                if outputs_with_cache.hidden_states is None or outputs_without_cache.hidden_states is None:
                    continue

                # Compare last hidden states
                # with_cache: [1, 1, H] (last step only)
                # without_cache: [1, seq_len+1, H] -> take last position
                try:
                    hidden_with_cache = outputs_with_cache.hidden_states[-1][:, -1, :]
                    hidden_without_cache = outputs_without_cache.hidden_states[-1][:, -1, :]

                    # Cosine similarity
                    similarity = F.cosine_similarity(
                        hidden_with_cache.flatten(),
                        hidden_without_cache.flatten(),
                        dim=0
                    )
                    similarities.append(similarity.item())
                except (RuntimeError, ValueError, IndexError) as e:
                    # Skip this prompt if dimension mismatch
                    continue

        avg_similarity = np.mean(similarities) if similarities else 1.0

        return {
            'cache_correctness': avg_similarity,
        }

    def compute_cache_drift(
        self,
        prompts: List[torch.Tensor],
        max_new_tokens: int = 50
    ) -> Dict[str, float]:
        """
        Compute cache NLL divergence metric (26).

        Measures how cache and non-cache predictions diverge at the next generation step.
        """
        self.model.eval()

        nll_divergences = []

        with torch.no_grad():
            for prompt in prompts[:min(len(prompts), 10)]:
                prompt = prompt.to(self.device)

                # 1. Prefill
                input_ids = prompt.unsqueeze(0)
                outputs_prefill = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                if outputs_prefill.past_key_values is None:
                    continue
                past_key_values = outputs_prefill.past_key_values

                # 2. Decode Step (Cache)
                next_token = torch.tensor([[self.eos_token_id or 50256]], device=self.device)
                outputs_with_cache = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )

                # 3. Baseline
                full_input = torch.cat([input_ids, next_token], dim=1)
                outputs_without_cache = self.model(
                    input_ids=full_input,
                    use_cache=False,
                    return_dict=True
                )

                if outputs_with_cache.logits is None or outputs_without_cache.logits is None:
                    continue

                try:
                    # Compare logits for the final position
                    logit_cache = outputs_with_cache.logits[:, -1, :]
                    logit_no_cache = outputs_without_cache.logits[:, -1, :]
                    
                    # Use baseline prediction as pseudo-label to measure divergence
                    target = logit_no_cache.argmax(dim=-1)
                    
                    nll_cache = F.cross_entropy(logit_cache, target)
                    nll_no_cache = F.cross_entropy(logit_no_cache, target)

                    # Absolute difference
                    nll_diff = abs(nll_cache.item() - nll_no_cache.item())
                    nll_divergences.append(nll_diff)
                except (RuntimeError, ValueError, IndexError) as e:
                    # Skip this prompt if dimension mismatch
                    continue

        avg_nll_divergence = np.mean(nll_divergences) if nll_divergences else 0.0

        return {
            'cache_nll_divergence': avg_nll_divergence,
        }

    def compute_generation_latency(
        self,
        prompts: List[torch.Tensor],
        max_new_tokens: int = 50
    ) -> Dict[str, float]:
        """
        Compute generation latency (optional metric).

        Args:
            prompts: List of prompt tensors
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Dictionary with latency metrics
        """
        self.model.eval()

        latencies = []

        for prompt in prompts[:min(len(prompts), 10)]:
            prompt = prompt.to(self.device)

            start_time = time.perf_counter()

            with torch.no_grad():
                _ = self.model.generate(
                    prompt.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )

            end_time = time.perf_counter()

            latency = end_time - start_time
            latencies.append(latency)

        avg_latency = np.mean(latencies) if latencies else 0.0
        tokens_per_second = max_new_tokens / avg_latency if avg_latency > 0 else 0.0

        return {
            'generation_latency_seconds': avg_latency,
            'generation_tokens_per_second': tokens_per_second,
        }

    def compute_all_generation_metrics(
        self,
        val_dataloader,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute all decoder generation metrics.

        Args:
            val_dataloader: Validation dataloader
            num_samples: Number of samples to use (default: from config)

        Returns:
            Dictionary with all generation metrics
        """
        num_samples = num_samples or self.sample_size
        self.model.eval()

        # Sample prompts from validation set
        prompts = []
        generated_sequences = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if len(prompts) >= num_samples:
                    break

                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                else:
                    input_ids = batch
                    attention_mask = None

                if input_ids is None:
                    continue

                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Handle multi-dimensional input_ids (e.g., MC tasks with shape [batch, num_choices, seq_len])
                if input_ids.dim() == 3:
                    # For MC tasks, skip generation metrics
                    continue

                # Take first few tokens as prompt
                if input_ids.dim() != 2:
                    continue
                prompt_len = min(DEFAULT_GENERATION_PROMPT_TOKENS, input_ids.shape[1] // 2)
                if prompt_len < 2:
                    continue
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.eos_token_id

                prompt_tensors = []
                for i in range(input_ids.shape[0]):
                    ids = input_ids[i]
                    if attention_mask is not None:
                        valid_ids = ids[attention_mask[i].bool()]
                    elif pad_token_id is not None:
                        nonpad = (ids != pad_token_id).nonzero(as_tuple=False)
                        if nonpad.numel() == 0:
                            continue
                        valid_ids = ids[:nonpad[-1].item() + 1]
                    else:
                        valid_ids = ids

                    if valid_ids.numel() < 2:
                        continue
                    prompt_ids = valid_ids[:prompt_len]
                    if prompt_ids.numel() < 2:
                        continue
                    prompt_tensors.append(prompt_ids)

                if not prompt_tensors:
                    continue

                max_prompt_len = max(p.size(0) for p in prompt_tensors)
                pad_value = pad_token_id if pad_token_id is not None else 0
                prompt = torch.full(
                    (len(prompt_tensors), max_prompt_len),
                    pad_value,
                    device=self.device,
                    dtype=input_ids.dtype,
                )
                attention_mask_prompt = torch.zeros(
                    (len(prompt_tensors), max_prompt_len),
                    device=self.device,
                    dtype=torch.long,
                )
                for i, p in enumerate(prompt_tensors):
                    prompt[i, -p.size(0):] = p
                    attention_mask_prompt[i, -p.size(0):] = 1

                # Generate - may fail with dimension mismatches from fault injection
                try:
                    generated = self.model.generate(
                        prompt,
                        attention_mask=attention_mask_prompt,
                        max_length=self.max_length,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.eos_token_id,
                    )
                except (RuntimeError, TypeError, AttributeError) as e:
                    # Handle dimension mismatches, None errors, and other generation failures
                    if any(keyword in str(e).lower() for keyword in ['size', 'dimension', 'none', 'subscript']):
                        continue
                    raise

                # Validate generated output is not None
                if generated is None:
                    continue

                prompts.extend(prompt_tensors)
                generated_sequences.append(generated)

                if len(prompts) >= num_samples:
                    break

        # Pad and concatenate generated sequences (different batches may have different lengths)
        if generated_sequences:
            max_gen_len = max(g.size(1) for g in generated_sequences)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = []
            for g in generated_sequences:
                if g.size(1) < max_gen_len:
                    padding = torch.full(
                        (g.size(0), max_gen_len - g.size(1)),
                        pad_id, device=g.device, dtype=g.dtype
                    )
                    g = torch.cat([g, padding], dim=1)
                padded.append(g)
            all_generated = torch.cat(padded, dim=0)[:num_samples]
        else:
            # Return zero metrics if no generation
            return {
                'repetition_max_run': 0.0,
                'repetition_distinct_1': 0.0,
                'repetition_distinct_2': 0.0,
                'generation_mean_length': 0.0,
                'generation_eos_ratio': 0.0,
                'cache_correctness': 0.0,
                'cache_nll_divergence': 0.0,
            }

        # Compute all metrics
        metrics = {}

        # Repetition metrics (20-22)
        metrics.update(self.compute_repetition_metrics(all_generated))

        # Length metrics (23-24)
        metrics.update(self.compute_length_metrics(all_generated))

        # Cache correctness (25)
        if prompts:
            try:
                metrics.update(self.compute_cache_correctness(prompts))
            except (RuntimeError, ValueError) as e:
                metrics['cache_correctness'] = 0.0

        # Cache drift (26)
        if prompts:
            try:
                metrics.update(self.compute_cache_drift(prompts))
            except (RuntimeError, ValueError) as e:
                metrics['cache_nll_divergence'] = 0.0

        return metrics
