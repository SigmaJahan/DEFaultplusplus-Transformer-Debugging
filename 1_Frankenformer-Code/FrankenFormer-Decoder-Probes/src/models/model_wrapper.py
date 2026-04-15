"""
Generic decoder model wrapper for multi-task pipeline.

Uses Hugging Face AutoModelForCausalLM to support
multiple decoder-only checkpoints (GPT-2, GPT-Neo, Pythia, etc.)
while keeping the fault injection interface consistent.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.models.base_model import BaseModelWrapper, FaultInjectorMixin


class ModelWrapper(BaseModelWrapper, FaultInjectorMixin):
    """Wrapper around AutoModelForCausalLM with fault hooks."""

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ):
        # Set device first before calling parent init
        if device is None:
            from src.utils.reproducibility import get_device
            device = get_device()

        # Decoder models don't use num_labels for classification
        # Use 0 as placeholder to satisfy BaseModelWrapper interface
        BaseModelWrapper.__init__(self, model_name, num_labels=0, device=device, cache_dir=cache_dir)
        FaultInjectorMixin.__init__(self)

        self.gradient_checkpointing = gradient_checkpointing
        self.layers: List[nn.Module] = []
        self.tokenizer = None

        # CRITICAL FIX: Call load_model() only once
        # BaseModelWrapper.__init__ no longer calls it, preventing double-loading
        self.load_model()

    def _get_layers(self) -> List[nn.Module]:
        """
        Extract transformer layers from various decoder architectures.

        Raises:
            ValueError: If model architecture is not recognized or unsupported.
        """
        # Try GPT-2 / DistilGPT2 architecture (transformer.h)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)

        # Try GPT-Neo architecture (transformer.h but sometimes different structure)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            return list(self.model.transformer.layers)

        # Try Pythia / GPT-NeoX architecture (gpt_neox.layers)
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return list(self.model.gpt_neox.layers)

        # Try GPT-J architecture (transformer.h)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)

        # Try OPT architecture (model.model.decoder.layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "layers"):
            return list(self.model.model.decoder.layers)

        # Try generic decoder pattern (model.layers)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)

        # Try LLaMA-style architecture (just in case)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)

        # If we get here, the architecture is not supported
        error_msg = (
            f"Cannot locate transformer layers for decoder model '{self.model_name}'.\n"
            f"Model type: {type(self.model).__name__}\n"
            f"Model config type: {getattr(self.model.config, 'model_type', 'unknown')}\n"
            f"Supported architectures: GPT-2, DistilGPT2, GPT-Neo, GPT-J, Pythia (GPT-NeoX).\n"
            f"Model attributes: {dir(self.model)[:10]}...\n"
            f"Please ensure the model has a standard decoder architecture or add support for this model type."
        )
        raise ValueError(error_msg)

    def load_model(self) -> nn.Module:
        """
        Load decoder model from HuggingFace with comprehensive error handling.

        Returns:
            The loaded model.

        Raises:
            ValueError: If model architecture is not supported.
            RuntimeError: If model loading fails.
        """
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                cache_dir=self.cache_dir,
            )

            # Load tokenizer for generation tasks
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            # Set pad token if not present (common for GPT-2)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            # Set padding_side to 'left' for decoder-only models to avoid right-padding warnings
            self.tokenizer.padding_side = 'left'

        except Exception as e:
            raise RuntimeError(
                f"Failed to load decoder model '{self.model_name}' from HuggingFace. "
                f"Check that the model name is correct and you have internet access. "
                f"Error: {e}"
            ) from e

        try:
            if self.gradient_checkpointing:
                self.enable_gradient_checkpointing()
        except Exception as e:
            # Non-fatal - log warning and continue
            import logging
            logging.warning(f"Failed to enable gradient checkpointing: {e}")

        try:
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to move model to device {self.device}. "
                f"Check that CUDA is available if using GPU. "
                f"Error: {e}"
            ) from e

        try:
            self.layers = self._get_layers()
        except ValueError as e:
            # Re-raise with additional context
            raise ValueError(
                f"Decoder model '{self.model_name}' loaded successfully but layer extraction failed. "
                f"This model architecture may not be supported for fault injection. "
                f"Original error: {e}"
            ) from e

        return self.model

    def get_attention_modules(self) -> Dict[str, nn.Module]:
        """Get attention modules from decoder layers."""
        modules: Dict[str, nn.Module] = {}
        for idx, layer in enumerate(self.layers):
            # GPT-2 / GPT-Neo: layer.attn
            if hasattr(layer, "attn"):
                modules[f"layer_{idx}"] = layer.attn
            # Pythia / GPT-NeoX: layer.attention
            elif hasattr(layer, "attention"):
                modules[f"layer_{idx}"] = layer.attention
            # Generic self_attn
            elif hasattr(layer, "self_attn"):
                modules[f"layer_{idx}"] = layer.self_attn
        return modules

    def get_attention_layers(self) -> list:
        """Get list of attention modules."""
        attention_layers = []
        for layer in self.layers:
            if hasattr(layer, "attn"):
                attention_layers.append(layer.attn)
            elif hasattr(layer, "attention"):
                attention_layers.append(layer.attention)
            elif hasattr(layer, "self_attn"):
                attention_layers.append(layer.self_attn)
        return attention_layers

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get specific transformer layer by index."""
        return self.layers[layer_idx]

    def get_embedding_layer(self) -> nn.Module:
        """Get word token embedding layer."""
        # GPT-2 / DistilGPT2: transformer.wte
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte

        # Pythia / GPT-NeoX: gpt_neox.embed_in
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "embed_in"):
            return self.model.gpt_neox.embed_in

        # Generic embedding
        return self.model.get_input_embeddings()

    def get_lm_head(self) -> nn.Module:
        """Get language modeling head (output projection)."""
        # Most models: lm_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head

        # GPT-NeoX: embed_out
        if hasattr(self.model, "embed_out"):
            return self.model.embed_out

        # Generic
        return self.model.get_output_embeddings()

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.layers)

    def get_hidden_size(self) -> int:
        """Get hidden dimension size."""
        # Try n_embd (GPT-2 style)
        if hasattr(self.model.config, "n_embd"):
            return self.model.config.n_embd
        # Try hidden_size (generic)
        return getattr(self.model.config, "hidden_size", 0)

    def get_num_attention_heads(self) -> int:
        """Get number of attention heads."""
        # Try n_head (GPT-2 style)
        if hasattr(self.model.config, "n_head"):
            return self.model.config.n_head
        # Try num_attention_heads (generic)
        return getattr(self.model.config, "num_attention_heads", 0)

    def get_model_config(self):
        """Get model configuration."""
        return self.model.config

    def forward_with_cache(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                          past_key_values: Optional[Tuple] = None, use_cache: bool = True):
        """
        Forward pass with KV-cache support for efficient generation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs from previous forward passes
            use_cache: Whether to return cache for next iteration

        Returns:
            Model outputs including logits and optionally cache
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
        )

    def forward_with_attention(self, **inputs):
        """Forward pass with attention weights and hidden states."""
        return self.model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,  # Disable cache when collecting attention weights
        )

    def get_attention_weights(self, **inputs) -> torch.Tensor:
        """Get attention weights from forward pass."""
        outputs = self.forward_with_attention(**inputs)
        return outputs.attentions

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        min_length: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask for input
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            do_sample: Whether to use sampling or greedy decoding
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            use_cache: Whether to use KV-cache for efficiency
            **kwargs: Additional generation parameters

        Returns:
            Generated token sequences
        """
        # Create attention mask if not provided to avoid warnings
        if attention_mask is None:
            pad_token_id = self.model.config.pad_token_id
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **kwargs
        )

    def get_detailed_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = self.get_model_info()
        config = self.get_model_config()
        details = {
            **base_info,
            "model_type": getattr(config, "model_type", "unknown"),
            "num_layers": self.get_num_layers(),
            "hidden_size": self.get_hidden_size(),
            "num_attention_heads": self.get_num_attention_heads(),
            "max_position_embeddings": getattr(config, "n_positions", getattr(config, "max_position_embeddings", None)),
            "vocab_size": getattr(config, "vocab_size", None),
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        return details


def load_decoder_model(
    model_name: str,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> ModelWrapper:
    """
    Factory function to load a decoder model.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        cache_dir: Directory to cache downloaded models
        gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        ModelWrapper instance
    """
    return ModelWrapper(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        gradient_checkpointing=gradient_checkpointing,
    )
