"""
Base Fault Injection Framework

This module provides the abstract base class for all fault injection implementations.
All fault types must inherit from BaseFault and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import logging
import torch
import torch.nn as nn
import gc
import math


class BaseFault(ABC):
    """
    Abstract base class for all fault injection implementations.

    All fault injectors must:
    1. Store original state before injection
    2. Inject the fault into the model
    3. Restore the model to original state
    4. Support context manager usage
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        fault_name: str,
        description: str
    ):
        """
        Initialize fault injector.

        Args:
            model: The neural network model to inject faults into
            layer_idx: Target layer index (0-based)
            fault_name: Unique identifier for this fault type
            description: Human-readable description of the fault
        """
        self.model = model
        self.layer_idx = layer_idx
        self.fault_name = fault_name
        self.description = description

        # State management
        self.original_state: Dict[str, Any] = {}
        self.is_injected = False
        self.target_layer = None
        self.logger = logging.getLogger(f"faults.{self.__class__.__name__}")
        self._backup_forward: Optional[Any] = None

    def __enter__(self):
        """Context manager entry - inject fault."""
        try:
            self.inject()
        except Exception as e:
            self.logger.error(f"Failed to inject fault {self.fault_name}: {e}")
            # Attempt cleanup even if injection failed
            try:
                self.restore()
            except Exception as restore_error:
                self.logger.error(f"Restoration after failed injection also failed: {restore_error}")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - restore original state.

        Always attempts to restore, even if an exception occurred during the fault.
        Logs restoration errors but doesn't suppress the original exception.
        """
        restoration_error = None
        try:
            self.restore()
        except Exception as e:
            restoration_error = e
            self.logger.error(f"Failed to restore from fault {self.fault_name}: {e}")

        # Always attempt memory cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            self.logger.warning(f"Memory cleanup failed: {cleanup_error}")

        # If restoration failed but there was no original exception, raise the restoration error
        if restoration_error is not None and exc_type is None:
            raise restoration_error

        # Return False to propagate any original exception
        return False

    @abstractmethod
    def inject(self) -> None:
        """
        Inject the fault into the model.

        This method must:
        1. Backup original state to self.original_state
        2. Apply the fault modification
        3. Set self.is_injected = True
        """
        pass

    @abstractmethod
    def restore(self) -> None:
        """
        Restore the model to its original state.

        This method must:
        1. Restore from self.original_state
        2. Clear self.original_state
        3. Set self.is_injected = False
        """
        pass

    # Alias for compatibility with older call sites/tests expecting remove()
    def remove(self) -> None:
        """Remove the injected fault (alias to restore)."""
        self.restore()

    def get_layer(self) -> nn.Module:
        """
        Get the target layer from the model.

        Auto-detects encoder vs decoder and routes to appropriate method.

        Returns:
            The target layer module

        Raises:
            ValueError: If layer index is invalid
        """
        # Detect model type and route to appropriate method
        model_type = self._detect_model_type()

        if model_type == 'decoder':
            return self.get_decoder_layer()

        # Encoder logic (original implementation)
        layers = None
        # Common encoder stacks across supported models
        if hasattr(self.model, 'distilbert'):
            layers = self.model.distilbert.transformer.layer
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layer'):
            layers = self.model.transformer.layer
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            layers = self.model.bert.encoder.layer
        elif hasattr(self.model, 'roberta') and hasattr(self.model.roberta, 'encoder'):
            layers = self.model.roberta.encoder.layer
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'encoder'):
            encoder = self.model.base_model.encoder
            if hasattr(encoder, 'layer'):
                layers = encoder.layer

        if layers is None:
            raise ValueError(f"Cannot find transformer layers in model: {type(self.model)}")

        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise ValueError(
                f"Invalid layer index {self.layer_idx}. "
                f"Model has {len(layers)} layers (0-{len(layers)-1})"
            )

        return layers[self.layer_idx]

    def get_decoder_layer(self) -> nn.Module:
        """
        Get the target layer from decoder architectures.

        Supports GPT-2, GPT-Neo, Pythia (GPT-NeoX), GPT-J, and generic decoder patterns.

        Returns:
            The target layer module

        Raises:
            ValueError: If layer index is invalid or architecture not recognized
        """
        layers = None

        # GPT-2 / DistilGPT2 architecture (transformer.h)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h

        # GPT-Neo architecture (transformer.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers

        # Pythia / GPT-NeoX architecture (gpt_neox.layers)
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            layers = self.model.gpt_neox.layers

        # OPT architecture (model.model.decoder.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            layers = self.model.model.decoder.layers

        # Generic decoder pattern (model.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers

        # LLaMA-style (layers directly)
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers

        if layers is None:
            raise ValueError(
                f"Cannot find decoder transformer layers in model: {type(self.model)}. "
                f"Supported decoder architectures: GPT-2, GPT-Neo, Pythia (GPT-NeoX), GPT-J."
            )

        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise ValueError(
                f"Invalid layer index {self.layer_idx}. "
                f"Decoder model has {len(layers)} layers (0-{len(layers)-1})"
            )

        return layers[self.layer_idx]

    def _detect_model_type(self) -> str:
        """
        Detect whether model is encoder or decoder based on config.

        Returns:
            'encoder' or 'decoder'
        """
        if hasattr(self.model, 'config'):
            model_type = getattr(self.model.config, 'model_type', None)

            # CRITICAL FIX: Expanded decoder type detection to include common models
            # Previously only checked: gpt2, gpt_neo, gpt_neox, gptj
            # Now includes: LLaMA, OPT, BLOOM, Falcon, Mistral, Qwen, Phi, etc.
            decoder_types = [
                # Original supported types
                'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'gpt-j',
                # LLaMA family
                'llama', 'llama2', 'llama3', 'codellama',
                # Meta models
                'opt', 'opt-iml',
                # BigScience
                'bloom', 'bloomz',
                # TII
                'falcon', 'refinedweb',
                # Mistral AI
                'mistral', 'mixtral',
                # Microsoft
                'phi', 'phi-2', 'phi-3',
                # Alibaba
                'qwen', 'qwen2',
                # Google
                'gemma', 'gemma2',
                # EleutherAI
                'pythia',
                # Stability AI
                'stablelm', 'stable-lm',
                # Other common decoders
                'mpt', 'redpajama', 'starcoder', 'santacoder',
                'gpt_bigcode', 'persimmon', 'yi'
            ]

            if model_type in decoder_types:
                return 'decoder'

            # Encoder types
            encoder_types = ['bert', 'roberta', 'distilbert', 'albert', 'modernbert', 'deberta', 'electra']
            if model_type in encoder_types:
                return 'encoder'

        # Fallback: check architecture structure
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return 'decoder'  # GPT-2 style
        elif hasattr(self.model, 'gpt_neox'):
            return 'decoder'  # Pythia style
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return 'decoder'  # LLaMA/OPT/generic decoder style
        elif hasattr(self.model, 'layers'):
            return 'decoder'  # Direct layer access style
        else:
            return 'encoder'  # Default to encoder

    def verify_injection(self) -> bool:
        """
        Verify that the fault was successfully injected.

        Returns:
            True if fault is active, False otherwise
        """
        return self.is_injected

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this fault.

        Returns:
            Dictionary containing fault metadata
        """
        return {
            'fault_name': self.fault_name,
            'description': self.description,
            'layer_idx': self.layer_idx,
            'is_injected': self.is_injected,
            'target_layer': type(self.target_layer).__name__ if self.target_layer else None
        }

    def __repr__(self) -> str:
        """String representation of the fault."""
        return (
            f"{self.__class__.__name__}("
            f"fault_name='{self.fault_name}', "
            f"layer={self.layer_idx}, "
            f"injected={self.is_injected})"
        )

    def _set_kernel_fault_state(self, key: str, active: bool):
        """Helper to expose kernel fault status on the model."""
        state = getattr(self.model, 'kernel_fault_state', {})
        if active:
            state[key] = True
        else:
            state.pop(key, None)
        self.model.kernel_fault_state = state

    def _update_fault_metadata(self, key: str, value: Optional[Any]):
        """
        Helper to stash fault-specific metadata on the model for metrics.

        Args:
            key: Metadata key to set/clear
            value: Value to store; if None the key is removed
        """
        meta = getattr(self.model, 'fault_metadata', {})
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
        self.model.fault_metadata = meta

    def _log_info(self, message: str):
        """Log informational message for fault lifecycle."""
        self.logger.info(message)

    def _log_warning(self, message: str):
        """Log warning message for fault lifecycle."""
        self.logger.warning(message)

    # ------------------------------------------------------------------ #
    # Embedding helpers (multi-model)
    def get_embeddings_module(self) -> nn.Module:
        """
        Return the embeddings module for encoder-style models.

        Supports DistilBERT/BERT/Roberta/ModernBERT-style layouts; falls back
        to model.get_input_embeddings() when needed.
        """
        # Common HF encoder containers
        if hasattr(self.model, "distilbert") and getattr(self.model.distilbert, "embeddings", None) is not None:
            return self.model.distilbert.embeddings
        if hasattr(self.model, "bert") and getattr(self.model.bert, "embeddings", None) is not None:
            return self.model.bert.embeddings
        if hasattr(self.model, "roberta") and getattr(self.model.roberta, "embeddings", None) is not None:
            return self.model.roberta.embeddings
        if hasattr(self.model, "modernbert") and getattr(self.model.modernbert, "embeddings", None) is not None:
            return self.model.modernbert.embeddings
        if hasattr(self.model, "transformer") and getattr(self.model.transformer, "embeddings", None) is not None:
            return self.model.transformer.embeddings
        if hasattr(self.model, "base_model") and getattr(self.model.base_model, "embeddings", None) is not None:
            return self.model.base_model.embeddings

        if hasattr(self.model, "get_input_embeddings"):
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return emb

        raise ValueError(f"Cannot locate embeddings module for model: {type(self.model)}")


class AttentionFault(BaseFault):
    """
    Specialized base class for attention mechanism faults.

    Provides helper methods for accessing attention components.
    """

    def get_attention_module(self) -> nn.Module:
        """
        Get the attention module from the target layer.

        Supports multiple model architectures by trying common attribute names:
        - Encoder models (BERT, DistilBERT, RoBERTa): layer.attention
        - Decoder models (GPT-2, GPT-Neo, DistilGPT2): layer.attn
        - GPT-NeoX models (Pythia): layer.attention
        - Other models: layer.self_attn, layer.self_attention

        Returns:
            The attention module
        """
        layer = self.get_layer()

        # Try common attention attribute names in order of prevalence
        attention_attr_names = [
            'attention',      # BERT, DistilBERT, RoBERTa, GPT-NeoX (Pythia)
            'attn',           # GPT-2, GPT-Neo, DistilGPT2
            'self_attn',      # LLaMA, Mistral, some custom models
            'self_attention', # Some transformer variants
        ]

        for attr_name in attention_attr_names:
            if hasattr(layer, attr_name):
                attention = getattr(layer, attr_name)
                self._ensure_attention_aliases(attention)
                return attention

        # If none found, provide helpful error with available attributes
        available_attrs = [attr for attr in dir(layer) if not attr.startswith('_')]
        raise ValueError(
            f"Cannot find attention module in layer: {type(layer).__name__}\n"
            f"Tried: {', '.join(attention_attr_names)}\n"
            f"Available attributes: {', '.join(available_attrs[:20])}"
        )

    def get_attention_head_size(self) -> int:
        """
        Get the attention head size.

        Supports multiple model architectures:
        - DistilBERT: attention.dim // attention.n_heads
        - BERT: attention.self.attention_head_size
        - GPT-2: attention.head_dim
        - GPT-Neo: attention.head_dim
        - Computed: embed_dim // num_heads

        Returns:
            Size of each attention head
        """
        attention = self.get_attention_module()

        # Try direct head_dim attribute (GPT-2, GPT-Neo)
        if hasattr(attention, 'head_dim'):
            return attention.head_dim

        # Try DistilBERT style (n_heads and dim)
        if hasattr(attention, 'n_heads') and hasattr(attention, 'dim'):
            return attention.dim // attention.n_heads

        # Try BERT style (attention.self.attention_head_size)
        if hasattr(attention, 'self') and hasattr(attention.self, 'attention_head_size'):
            return attention.self.attention_head_size

        # Try to compute from embed_dim / num_heads
        if hasattr(attention, 'embed_dim') and hasattr(attention, 'num_heads'):
            return attention.embed_dim // attention.num_heads
        if hasattr(attention, 'hidden_size') and hasattr(attention, 'num_attention_heads'):
            return attention.hidden_size // attention.num_attention_heads

        raise ValueError(
            f"Cannot determine head size from attention: {type(attention).__name__}\n"
            f"Tried: head_dim, n_heads+dim, self.attention_head_size, embed_dim+num_heads"
        )

    def get_num_heads(self) -> int:
        """
        Get the number of attention heads.

        Supports multiple model architectures:
        - DistilBERT: attention.n_heads
        - BERT: attention.self.num_attention_heads
        - GPT-2/GPT-Neo: attention.num_heads

        Returns:
            Number of attention heads
        """
        attention = self.get_attention_module()

        # Try common attribute names in order
        num_heads_attrs = [
            'num_heads',              # GPT-2, GPT-Neo, many models
            'n_heads',                # DistilBERT
            'num_attention_heads',    # Some models
        ]

        for attr_name in num_heads_attrs:
            if hasattr(attention, attr_name):
                return getattr(attention, attr_name)

        # Try BERT style (attention.self.num_attention_heads)
        if hasattr(attention, 'self') and hasattr(attention.self, 'num_attention_heads'):
            return attention.self.num_attention_heads

        raise ValueError(
            f"Cannot determine number of heads from attention: {type(attention).__name__}\n"
            f"Tried: {', '.join(num_heads_attrs)}, self.num_attention_heads"
        )

    # ------------------------------------------------------------------ #
    # Helpers for multi-encoder compatibility
    def _standardize_attention_inputs(self, args: Any, kwargs: Dict[str, Any]):
        """
        Normalize attention call signatures (DistilBERT qkv vs. BERT hidden_states).
        Returns query, key, value, mask, head_mask, output_attentions.
        """
        query = kwargs.get("query")
        key = kwargs.get("key")
        value = kwargs.get("value")
        mask = kwargs.get("mask", kwargs.get("attention_mask"))
        head_mask = kwargs.get("head_mask")
        output_attentions = kwargs.get("output_attentions", False)

        # Detect call style by inspecting argument characteristics
        # DistilBERT: (query:3D, key:3D, value:3D, mask:2D/4D, ...)
        # BERT: (hidden_states:3D, attention_mask:2D/4D, head_mask:None/tensor, ...)

        # Decoder-style with past_key_value positional args (OPT/LLaMA style)
        if len(args) >= 4:
            possible_past = args[2]
            possible_mask = args[3]
            if (possible_past is None or isinstance(possible_past, (tuple, list))) and (
                possible_mask is None or hasattr(possible_mask, 'dim')
            ):
                query = key = value = args[0]
                if mask is None:
                    mask = possible_mask
                if len(args) >= 5 and head_mask is None:
                    head_mask = args[4]
                if len(args) >= 6 and isinstance(args[5], bool):
                    output_attentions = args[5]
                return query, key, value, mask, head_mask, output_attentions

        # GPT-2 style positional args (hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions)
        if len(args) >= 3:
            possible_past = args[1]
            possible_mask = args[2]
            if isinstance(possible_past, (tuple, list)) and (possible_mask is None or hasattr(possible_mask, 'dim')):
                query = key = value = args[0]
                if mask is None:
                    mask = possible_mask
                if len(args) >= 4 and head_mask is None:
                    head_mask = args[3]
                if len(args) >= 6 and isinstance(args[5], bool):
                    output_attentions = args[5]
                return query, key, value, mask, head_mask, output_attentions

        if len(args) >= 3:
            # Could be DistilBERT (query, key, value, ...) or BERT with many args
            # Check if args[1] and args[2] are 3D tensors (DistilBERT qkv style)
            # or if args[1] is 2D/4D mask and args[2] is None/mask (BERT style)

            is_distilbert_style = False
            if hasattr(args[1], 'dim') and hasattr(args[2], 'dim'):
                # Both args[1] and args[2] are tensors
                if args[1].dim() == 3 and args[2].dim() == 3:
                    # Both are 3D - DistilBERT style (query, key, value)
                    is_distilbert_style = True

            if is_distilbert_style:
                # Distil-style call: query, key, value, [mask], [head_mask]
                query, key, value = args[0], args[1], args[2]
                if len(args) >= 4 and mask is None:
                    mask = args[3]
                if len(args) >= 5 and head_mask is None:
                    head_mask = args[4]
                if len(args) >= 6:
                    output_attentions = args[5]
                return query, key, value, mask, head_mask, output_attentions
            else:
                # BERT-style with 3+ args: hidden_states, attention_mask, head_mask, ...
                query = key = value = args[0]
                if len(args) >= 2 and mask is None:
                    mask = args[1]
                if len(args) >= 3 and head_mask is None:
                    head_mask = args[2]
                if len(args) >= 7 and isinstance(args[6], bool):
                    output_attentions = args[6]
                # args[3:] might be encoder_hidden_states, encoder_attention_mask, etc.
                return query, key, value, mask, head_mask, output_attentions

        if len(args) >= 1:
            # BERT-style call: hidden_states, attention_mask, head_mask, ...
            query = key = value = args[0]
            if len(args) >= 2 and mask is None:
                mask = args[1]
            if len(args) >= 3 and head_mask is None:
                head_mask = args[2]
            if len(args) >= 7 and isinstance(args[6], bool):
                output_attentions = args[6]
            return query, key, value, mask, head_mask, output_attentions

        # Keyword-only invocation (e.g., DistilBERT attention)
        def _get_first(*keys):
            for k in keys:
                v = kwargs.get(k)
                if v is not None:
                    return v
            return None
        if query is None:
            query = _get_first("hidden_states", "input_tensor")
        if key is None:
            key = _get_first("key_value_states", "value_states", "encoder_hidden_states")
        if value is None:
            value = _get_first("value_states", "key_value_states", "encoder_hidden_states")

        if query is None and key is None and value is None:
            raise ValueError("Attention forward called with insufficient arguments for standardization.")

        if key is None:
            key = query
        if value is None:
            value = key

        return query, key, value, mask, head_mask, output_attentions

    def _reshape_attention_mask(self, mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
        """Ensure mask shape matches (batch, heads, seq, seq) additive style for BERT-like attention."""
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # Broadcast to match batch/heads
        if mask.size(-1) != target.size(1):
            mask = mask[..., : target.size(1)]
        return mask

    def _bert_attention_forward(
        self,
        attention: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ):
        """QKV-friendly forward for BertAttention/RobertaAttention style modules."""
        self_attn = attention.self
        attn_mask = self._reshape_attention_mask(mask, query)

        def shape(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (self_attn.num_attention_heads, self_attn.attention_head_size)
            return x.view(*new_x_shape).permute(0, 2, 1, 3)

        q = shape(self_attn.query(query))
        k = shape(self_attn.key(key))
        v = shape(self_attn.value(value))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self_attn.attention_head_size)
        if attn_mask is not None:
            if not torch.is_floating_point(attn_mask):
                attn_mask = attn_mask.float()
            if attn_mask.max() <= 1:
                attn_mask = (1.0 - attn_mask) * -1e4
            attn_scores = attn_scores + attn_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self_attn.dropout(attn_probs)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self_attn.all_head_size,)
        context = context.view(*new_context_shape)

        attention_output = attention.output.dense(context)
        attention_output = attention.output.dropout(attention_output)
        attention_output = attention.output.LayerNorm(attention_output + query)

        outputs = (attention_output,)
        if output_attentions:
            outputs += (attn_probs,)
        return outputs

    def _get_attention_forward(self, attention: nn.Module, forward_fn: Optional[Callable] = None):
        """
        Return a normalized attention forward that accepts (query, key, value, mask, head_mask, output_attentions).
        If forward_fn is provided, it is used instead of attention.forward (useful to avoid recursion).

        Supports:
        - BERT/RoBERTa: attention.self and attention.output
        - DistilBERT: q_lin, k_lin, v_lin
        - GPT-2/GPT-Neo: c_attn (combined QKV projection)
        - Other decoders: Various QKV projection schemes
        """
        target_forward = forward_fn or attention.forward

        # Check for BERT/Roberta style FIRST (before checking q_lin which might be an alias)
        if hasattr(attention, "self") and hasattr(attention, "output"):
            def forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
                return self._bert_attention_forward(attention, query, key, value, mask, head_mask, output_attentions)
            return forward

        # GPT-Neo style: nested attention module (GPTNeoAttention.attention has q_proj, k_proj, v_proj)
        # Must check BEFORE DistilBERT since aliases may have been added
        if hasattr(attention, "attention"):
            inner = attention.attention
            if hasattr(inner, "q_proj") and hasattr(inner, "k_proj") and hasattr(inner, "v_proj"):
                def forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
                    return target_forward(
                        query,
                        attention_mask=mask,
                        head_mask=head_mask,
                        output_attentions=output_attentions
                    )
                return forward

        # Check for native DistilBERT style (has q_lin but NOT attention.self)
        # Excludes GPT-2 (c_attn) and OPT/LLaMA (q_proj) where q_lin is an alias
        if hasattr(attention, "q_lin") and hasattr(attention, "k_lin") and hasattr(attention, "v_lin"):
            if not hasattr(attention, "c_attn") and not hasattr(attention, "q_proj"):
                def forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
                    return target_forward(query, key, value, mask=mask, head_mask=head_mask, output_attentions=output_attentions)
                return forward

        # GPT-2/GPT-Neo style: Uses c_attn (combined QKV) and expects different signature
        # Forward signature: (hidden_states, layer_past, attention_mask, head_mask, ...)
        if hasattr(attention, "c_attn") and hasattr(attention, "c_proj"):
            def forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
                # For GPT-2, query/key/value are the same (hidden_states)
                # The attention module handles splitting into Q, K, V internally
                return target_forward(
                    query,  # hidden_states
                    layer_past=None,
                    attention_mask=mask,
                    head_mask=head_mask,
                    use_cache=False,
                    output_attentions=output_attentions
                )
            return forward

        # Other decoder models with separate q_proj, k_proj, v_proj (OPT, LLaMA, etc.)
        if hasattr(attention, "q_proj") and hasattr(attention, "k_proj") and hasattr(attention, "v_proj"):
            def forward(query, key, value, mask=None, head_mask=None, output_attentions=False):
                return target_forward(
                    query,  # hidden_states
                    attention_mask=mask,
                    layer_head_mask=head_mask,
                    output_attentions=output_attentions
                )
            return forward

        raise ValueError(
            f"Unsupported attention module for fault injection: {type(attention).__name__}\n"
            f"Module must have one of:\n"
            f"  - BERT: self + output attributes\n"
            f"  - DistilBERT: q_lin + k_lin + v_lin\n"
            f"  - GPT-2: c_attn + c_proj\n"
            f"  - Other: q_proj + k_proj + v_proj"
        )

    def _build_attention_wrapper(self, fault_fn):
        """
        Wrap a fault_fn(query, key, value, mask, head_mask, output_attentions)
        so it can be called with either Distil or BERT-style signatures.

        The wrapper accepts multiple signature styles:
        - BERT: (hidden_states, attention_mask, head_mask, ...)
        - DistilBERT: (query, key, value, mask, head_mask, ...)
        - GPT-2: (hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions)
        """
        def wrapped(*args, **kwargs):
            # Check for GPT-2 style use_cache parameter
            use_cache = kwargs.get('use_cache', None)
            if use_cache is None and len(args) >= 5 and isinstance(args[4], bool):
                use_cache = args[4]
            if use_cache is None:
                use_cache = False
            cache_requested = use_cache or (
                'past_key_value' in kwargs or 'past_key_values' in kwargs or 'layer_past' in kwargs
            )
            past_key_value = kwargs.get('past_key_value') or kwargs.get('past_key_values') or kwargs.get('layer_past')
            if past_key_value is None:
                if len(args) >= 3 and isinstance(args[2], (tuple, list)):
                    past_key_value = args[2]
                elif len(args) >= 2 and isinstance(args[1], (tuple, list)):
                    past_key_value = args[1]

            query, key, value, mask, head_mask, output_attentions = self._standardize_attention_inputs(args, kwargs)
            result = fault_fn(query, key, value, mask, head_mask, output_attentions)
            # Ensure result is always a tuple for consistency
            if not isinstance(result, tuple):
                result = (result,)

            backup_forward = getattr(self, "_backup_forward", None)
            if (cache_requested or past_key_value is not None) and backup_forward is not None:
                original = backup_forward(*args, **kwargs)
                if hasattr(original, "to_tuple"):
                    original = original.to_tuple()
                if not isinstance(original, tuple):
                    original = (original,)
                combined = list(original)
                if combined:
                    combined[0] = result[0]
                return tuple(combined)

            # For GPT-2 decoder models with use_cache, ensure we return at least 2 elements
            # Format: (hidden_states, present_key_values) or (hidden_states, present_key_values, attn_weights)
            if use_cache:
                if len(result) == 1:
                    # Add None for present_key_values (cache not computed in faulty forward)
                    result = result + (None,)
                if output_attentions and len(result) == 2:
                    # Add None for attention weights
                    result = result + (None,)
            elif output_attentions and len(result) == 1:
                # Maintain expected tuple length when attentions are requested
                result = result + (None,)

            return result
        return wrapped

    def _get_qkv_projections(self, attention: nn.Module) -> Dict[str, Any]:
        """
        Return projection/out layers and head metadata across encoder and decoder variants.

        For GPT-2 style models with fused c_attn, creates virtual Q/K/V projection wrappers.
        """
        q_proj = getattr(attention, "q_lin", None)
        k_proj = getattr(attention, "k_lin", None)
        v_proj = getattr(attention, "v_lin", None)
        out_proj = getattr(attention, "out_lin", None)
        num_heads = getattr(attention, "n_heads", None)
        dim = getattr(attention, "dim", None)

        # Try BERT-style: attention.self.{query,key,value}
        if q_proj is None and hasattr(attention, "self"):
            self_attn = attention.self
            q_proj = getattr(self_attn, "query", None)
            k_proj = getattr(self_attn, "key", None)
            v_proj = getattr(self_attn, "value", None)
            num_heads = getattr(self_attn, "num_attention_heads", num_heads)
            dim = getattr(self_attn, "all_head_size", dim)
            out_proj = getattr(getattr(attention, "output", None), "dense", out_proj)

        # Handle GPT-2 style fused c_attn (combined QKV projection)
        if (q_proj is None or k_proj is None or v_proj is None) and hasattr(attention, "c_attn"):
            # GPT-2 uses fused c_attn that outputs 3*embed_dim, split into Q, K, V
            # Create virtual projection wrappers for Q, K, V slices
            c_attn = attention.c_attn
            embed_dim = getattr(attention, "embed_dim", None)

            if embed_dim is None:
                # Try to infer from c_attn output dimension
                if hasattr(c_attn, "out_features"):
                    embed_dim = c_attn.out_features // 3
                elif hasattr(c_attn, "nf"):  # GPT-2 Conv1D uses nf
                    embed_dim = c_attn.nf // 3
                else:
                    raise ValueError(f"Cannot determine embed_dim for GPT-2 attention: {type(attention)}")

            # Create virtual projection modules that wrap slices of c_attn
            class _VirtualProjection(nn.Module):
                """Virtual projection that represents a slice of GPT-2's fused c_attn."""
                def __init__(self, c_attn_module, start_idx, end_idx, embed_dim):
                    super().__init__()
                    self.c_attn = c_attn_module
                    self.start_idx = start_idx
                    self.end_idx = end_idx
                    self.embed_dim = embed_dim

                @property
                def weight(self):
                    """Return the slice of c_attn weights corresponding to this projection."""
                    # c_attn.weight is either (3*embed_dim, embed_dim) for Conv1D
                    # or (embed_dim, 3*embed_dim) for Linear
                    if hasattr(self.c_attn, 'nf'):  # Conv1D
                        return self.c_attn.weight[self.start_idx:self.end_idx, :]
                    else:  # Linear
                        return self.c_attn.weight[:, self.start_idx:self.end_idx]

                @property
                def bias(self):
                    """Return the slice of c_attn bias corresponding to this projection."""
                    if self.c_attn.bias is not None:
                        return self.c_attn.bias[self.start_idx:self.end_idx]
                    return None

                def forward(self, x):
                    """Forward pass - apply c_attn and extract the relevant slice."""
                    full_qkv = self.c_attn(x)
                    return full_qkv[..., self.start_idx:self.end_idx]

            # Create virtual projections for Q, K, V
            q_proj = _VirtualProjection(c_attn, 0, embed_dim, embed_dim)
            k_proj = _VirtualProjection(c_attn, embed_dim, 2*embed_dim, embed_dim)
            v_proj = _VirtualProjection(c_attn, 2*embed_dim, 3*embed_dim, embed_dim)

            # Get output projection and other metadata
            out_proj = getattr(attention, "c_proj", out_proj)
            num_heads = getattr(attention, "num_heads", num_heads)
            dim = embed_dim

        # GPT-Neo style: nested attention module (GPTNeoAttention.attention)
        if (q_proj is None or k_proj is None or v_proj is None) and hasattr(attention, "attention"):
            inner = attention.attention
            q_proj = getattr(inner, "q_proj", q_proj)
            k_proj = getattr(inner, "k_proj", k_proj)
            v_proj = getattr(inner, "v_proj", v_proj)
            out_proj = getattr(inner, "out_proj", out_proj)
            num_heads = getattr(inner, "num_heads", num_heads)
            dim = getattr(inner, "embed_dim", dim)

        if q_proj is None or k_proj is None or v_proj is None:
            raise ValueError(f"Cannot resolve QKV projections for attention: {type(attention)}")

        return {
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            "out_proj": out_proj,
            "num_heads": num_heads,
            "dim": dim,
        }

    def _ensure_attention_aliases(self, attention: nn.Module) -> None:
        """
        Add standardized aliases (q_lin, k_lin, v_lin, out_lin, dim, n_heads)
        to various attention modules so fault injectors can operate uniformly.

        Supports:
        - DistilBERT: Already has q_lin, k_lin, v_lin, out_lin
        - BERT/RoBERTa: Has self.query, self.key, self.value
        - GPT-2/GPT-Neo: Has c_attn (combined QKV), c_proj (output)
        - Other decoders: Similar variations
        """
        # Already has aliases - nothing to do
        if hasattr(attention, "q_lin") and hasattr(attention, "k_lin") and hasattr(attention, "v_lin"):
            return

        # BERT/RoBERTa style: attention.self.{query,key,value}
        if hasattr(attention, "self"):
            self_attn = attention.self
            if not hasattr(attention, "q_lin") and hasattr(self_attn, "query"):
                object.__setattr__(attention, "q_lin", self_attn.query)
            if not hasattr(attention, "k_lin") and hasattr(self_attn, "key"):
                object.__setattr__(attention, "k_lin", self_attn.key)
            if not hasattr(attention, "v_lin") and hasattr(self_attn, "value"):
                object.__setattr__(attention, "v_lin", self_attn.value)
            if not hasattr(attention, "out_lin") and hasattr(attention, "output"):
                object.__setattr__(attention, "out_lin", getattr(attention.output, "dense", None))
            if not hasattr(attention, "dropout") and hasattr(self_attn, "dropout"):
                object.__setattr__(attention, "dropout", self_attn.dropout)
            if not hasattr(attention, "dim"):
                object.__setattr__(attention, "dim", getattr(self_attn, "all_head_size", None))
            if not hasattr(attention, "n_heads"):
                object.__setattr__(attention, "n_heads", getattr(self_attn, "num_attention_heads", None))
            return

        # GPT-2/GPT-Neo style: c_attn (combined QKV), c_proj (output)
        # Note: GPT-2 uses a single c_attn projection that outputs 3*embed_dim,
        # which gets split into Q, K, V. We can't create separate q_lin/k_lin/v_lin
        # aliases, but we can mark c_attn as the combined projection.
        if hasattr(attention, "c_attn") and hasattr(attention, "c_proj"):
            # For GPT-2, faults that need separate Q/K/V will need to handle c_attn specially
            # We set q_lin to c_attn to indicate QKV projection exists
            if not hasattr(attention, "q_lin"):
                object.__setattr__(attention, "q_lin", attention.c_attn)  # Combined QKV projection
            if not hasattr(attention, "out_lin"):
                object.__setattr__(attention, "out_lin", attention.c_proj)
            if not hasattr(attention, "dropout") and hasattr(attention, "attn_dropout"):
                object.__setattr__(attention, "dropout", attention.attn_dropout)
            if not hasattr(attention, "dim") and hasattr(attention, "embed_dim"):
                object.__setattr__(attention, "dim", attention.embed_dim)
            if not hasattr(attention, "n_heads") and hasattr(attention, "num_heads"):
                object.__setattr__(attention, "n_heads", attention.num_heads)
            return

        # Other decoder models with separate q_proj, k_proj, v_proj
        if hasattr(attention, "q_proj") and hasattr(attention, "k_proj") and hasattr(attention, "v_proj"):
            if not hasattr(attention, "q_lin"):
                object.__setattr__(attention, "q_lin", attention.q_proj)
            if not hasattr(attention, "k_lin"):
                object.__setattr__(attention, "k_lin", attention.k_proj)
            if not hasattr(attention, "v_lin"):
                object.__setattr__(attention, "v_lin", attention.v_proj)
            if not hasattr(attention, "out_lin") and hasattr(attention, "out_proj"):
                object.__setattr__(attention, "out_lin", attention.out_proj)
            if not hasattr(attention, "dropout") and hasattr(attention, "attn_dropout"):
                object.__setattr__(attention, "dropout", attention.attn_dropout)
            if not hasattr(attention, "dim") and hasattr(attention, "embed_dim"):
                object.__setattr__(attention, "dim", attention.embed_dim)
            if not hasattr(attention, "n_heads"):
                num_heads = getattr(attention, "num_heads", getattr(attention, "num_attention_heads", None))
                if num_heads:
                    object.__setattr__(attention, "n_heads", num_heads)
            return

        # GPT-Neo style: nested attention module (GPTNeoAttention.attention)
        if hasattr(attention, "attention"):
            inner = attention.attention
            if hasattr(inner, "q_proj") and hasattr(inner, "k_proj") and hasattr(inner, "v_proj"):
                if not hasattr(attention, "q_lin"):
                    object.__setattr__(attention, "q_lin", inner.q_proj)
                if not hasattr(attention, "k_lin"):
                    object.__setattr__(attention, "k_lin", inner.k_proj)
                if not hasattr(attention, "v_lin"):
                    object.__setattr__(attention, "v_lin", inner.v_proj)
                if not hasattr(attention, "out_lin") and hasattr(inner, "out_proj"):
                    object.__setattr__(attention, "out_lin", inner.out_proj)
                if not hasattr(attention, "dropout") and hasattr(inner, "attn_dropout"):
                    object.__setattr__(attention, "dropout", inner.attn_dropout)
                if not hasattr(attention, "dim") and hasattr(inner, "embed_dim"):
                    object.__setattr__(attention, "dim", inner.embed_dim)
                if not hasattr(attention, "n_heads") and hasattr(inner, "num_heads"):
                    object.__setattr__(attention, "n_heads", inner.num_heads)
