"""
ModelInspector — Auto-detect HuggingFace model architecture and discover internals.

Uses CATEGORY-BASED auto-discovery (not per-model registry) to support any
BERT-style encoder or GPT-style decoder, including future models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AttentionPattern:
    """Discovered attention module pattern within a layer."""
    module_attr: str  # e.g. 'attention', 'attn', 'self_attn'
    qkv_names: Tuple[str, ...]  # e.g. ('q_lin', 'k_lin', 'v_lin')
    qkv_style: str  # 'separate' or 'fused'


@dataclass
class FFNPattern:
    """Discovered FFN/MLP module pattern within a layer."""
    module_attr: str  # e.g. 'intermediate', 'mlp', 'ffn'


@dataclass
class DiscoveredStructure:
    """Complete discovered model structure."""
    arch_family: str  # 'encoder' or 'decoder'
    backbone: nn.Module
    backbone_path: str
    layers: nn.ModuleList
    layers_path: str
    num_layers: int
    num_heads: int
    hidden_size: int
    attention_pattern: AttentionPattern
    ffn_pattern: FFNPattern
    layernorm_names: List[str]
    embedding: Optional[nn.Module]
    classifier: Optional[nn.Module]


class ModelInspector:
    """
    Auto-discovers the internal structure of any HuggingFace transformer.

    Supports two structural categories:
    - BERT-style encoders (bidirectional attention)
    - GPT-style decoders (causal attention)

    Discovery is based on probing the model's nn.Module tree, NOT
    on maintaining a registry of model names.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = getattr(model, 'config', None)
        if self.config is None:
            raise ValueError(
                f"Model {type(model).__name__} has no 'config' attribute. "
                "ModelInspector requires a HuggingFace model with a config."
            )

        # Step 1: Detect family
        self.arch_family = self._detect_family()

        # Step 2: Extract dimensions from config (reliable)
        self.num_layers = self._get_config_attr('num_hidden_layers', 'n_layer', 'num_layers')
        self.num_heads = self._get_config_attr('num_attention_heads', 'n_head', 'num_heads')
        self.hidden_size = self._get_config_attr('hidden_size', 'n_embd', 'd_model', 'dim')

        # Step 3: Discover structure
        self.backbone, self._backbone_path = self._discover_backbone()
        self.layers, self._layers_path = self._discover_layers()

        if self.num_layers is None:
            self.num_layers = len(self.layers)

        # Step 4: Discover per-layer patterns from first layer
        self._attn_pattern = self._discover_attention(self.layers[0])
        self._ffn_pattern = self._discover_ffn(self.layers[0])
        self._ln_names = self._discover_layernorm(self.layers[0])

        # Step 5: Discover embedding and classifier
        self.embedding = self._discover_embedding()
        self.classifier = self._find_classifier_head()

        logger.info(
            "ModelInspector: family=%s, layers=%d, heads=%s, hidden=%s, "
            "attn=%s, ffn=%s, ln=%s",
            self.arch_family, self.num_layers, self.num_heads, self.hidden_size,
            self._attn_pattern.module_attr if self._attn_pattern else None,
            self._ffn_pattern.module_attr if self._ffn_pattern else None,
            self._ln_names,
        )

    # ------------------------------------------------------------------
    # Family detection
    # ------------------------------------------------------------------
    def _detect_family(self) -> str:
        config = self.config

        # Fast path: config attributes
        if getattr(config, 'is_decoder', False):
            return 'decoder'
        if getattr(config, 'is_encoder_decoder', False):
            raise ValueError(
                f"Encoder-decoder models (e.g., T5, BART) are not supported. "
                f"Your model: {type(self.model).__name__}"
            )

        # Check architectures string
        archs = getattr(config, 'architectures', []) or []
        arch_str = ' '.join(archs).lower()
        model_type = getattr(config, 'model_type', '').lower()

        decoder_keywords = (
            'causallm', 'gpt', 'llama', 'opt', 'bloom', 'falcon', 'mistral',
            'phi', 'qwen', 'gemma', 'mamba', 'rwkv', 'starcoder',
        )
        encoder_keywords = (
            'maskedlm', 'forsequenceclassification', 'fortokenclassification',
            'forquestionanswering', 'bert', 'roberta', 'electra', 'albert',
            'deberta', 'distilbert', 'xlm', 'camembert', 'ernie', 'funnel',
            'modernbert',
        )

        for kw in decoder_keywords:
            if kw in arch_str or kw in model_type:
                return 'decoder'
        for kw in encoder_keywords:
            if kw in arch_str or kw in model_type:
                return 'encoder'

        # Structural probing: check for causal mask modules
        for name, _ in self.model.named_modules():
            name_lower = name.lower()
            if 'causal' in name_lower:
                return 'decoder'

        # Check if model has an 'encoder' submodule (common for BERT-style)
        for child_name, _ in self.model.named_children():
            child_lower = child_name.lower()
            if child_lower in ('encoder', 'bert', 'roberta', 'distilbert', 'albert'):
                return 'encoder'
            if child_lower in ('transformer',):
                # Could be either — check for causal config
                if getattr(config, 'n_embd', None) is not None:
                    return 'decoder'
                return 'encoder'

        raise ValueError(
            f"Could not detect architecture family. Supported: BERT-style encoders, "
            f"GPT-style decoders. Your model: {type(self.model).__name__}. "
            f"config.model_type={getattr(config, 'model_type', 'N/A')}"
        )

    # ------------------------------------------------------------------
    # Config attribute helpers
    # ------------------------------------------------------------------
    def _get_config_attr(self, *names: str) -> Optional[int]:
        for name in names:
            val = getattr(self.config, name, None)
            if val is not None:
                return int(val)
        return None

    # ------------------------------------------------------------------
    # Backbone discovery
    # ------------------------------------------------------------------
    def _discover_backbone(self) -> Tuple[nn.Module, str]:
        """Find the main transformer body."""
        # Try common backbone attribute names
        candidates = [
            'bert', 'roberta', 'distilbert', 'albert', 'electra', 'deberta',
            'transformer', 'gpt_neox', 'gpt', 'model', 'encoder',
        ]
        for attr in candidates:
            backbone = getattr(self.model, attr, None)
            if backbone is not None and isinstance(backbone, nn.Module):
                # Verify it contains something layer-like
                if self._has_modulelist(backbone):
                    return backbone, attr

        # Fallback: walk children looking for one with a ModuleList
        for name, child in self.model.named_children():
            if self._has_modulelist(child):
                return child, name

        raise ValueError(
            f"Could not discover backbone in {type(self.model).__name__}. "
            f"Children: {[n for n, _ in self.model.named_children()]}"
        )

    @staticmethod
    def _has_modulelist(module: nn.Module) -> bool:
        for _, child in module.named_modules():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                return True
        return False

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------
    def _discover_layers(self) -> Tuple[nn.ModuleList, str]:
        """Find the repeating layer stack."""
        # Try common paths from backbone
        layer_paths = [
            'encoder.layer', 'layer', 'h', 'layers',
            'transformer.layer', 'blocks', 'encoder.layers',
        ]
        for path in layer_paths:
            obj = self.backbone
            try:
                for attr in path.split('.'):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                    return obj, f"{self._backbone_path}.{path}"
            except AttributeError:
                continue

        # Fallback: find first ModuleList with attention-like children
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                # Check if children look like transformer layers
                child = module[0]
                for _, sub in child.named_modules():
                    sub_name = type(sub).__name__.lower()
                    if 'attention' in sub_name or 'attn' in sub_name:
                        full_path = f"{self._backbone_path}.{name}" if name else self._backbone_path
                        return module, full_path

        raise ValueError(
            f"Could not discover layer stack in backbone {self._backbone_path}"
        )

    # ------------------------------------------------------------------
    # Attention discovery
    # ------------------------------------------------------------------
    def _discover_attention(self, layer: nn.Module) -> AttentionPattern:
        """Find attention module and Q/K/V projections within a layer."""
        # Try common attention attribute paths
        attn_candidates = ['attention', 'attn', 'self_attn', 'self_attention']

        attn_module = None
        attn_attr = None

        for attr in attn_candidates:
            mod = getattr(layer, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                attn_module = mod
                attn_attr = attr
                break

        if attn_module is None:
            # Search named_modules
            for name, mod in layer.named_modules():
                mod_type = type(mod).__name__.lower()
                if ('attention' in mod_type or 'attn' in mod_type) and name:
                    attn_module = mod
                    attn_attr = name.split('.')[0]
                    break

        if attn_module is None:
            raise ValueError(f"Could not find attention in layer {type(layer).__name__}")

        # Find Q/K/V projections
        qkv_candidates = [
            ('query', 'key', 'value'),
            ('q_proj', 'k_proj', 'v_proj'),
            ('q_lin', 'k_lin', 'v_lin'),
            ('Wq', 'Wk', 'Wv'),
            ('q', 'k', 'v'),
        ]

        # Search within the attention module tree
        for q_name, k_name, v_name in qkv_candidates:
            if self._find_submodule(attn_module, q_name) is not None:
                return AttentionPattern(
                    module_attr=attn_attr,
                    qkv_names=(q_name, k_name, v_name),
                    qkv_style='separate',
                )

        # Check for fused QKV (GPT-2 style c_attn)
        fused_candidates = ['c_attn', 'qkv_proj', 'in_proj']
        for fused_name in fused_candidates:
            if self._find_submodule(attn_module, fused_name) is not None:
                return AttentionPattern(
                    module_attr=attn_attr,
                    qkv_names=(fused_name,),
                    qkv_style='fused',
                )

        # Fallback: return the attention module without specific QKV names
        return AttentionPattern(
            module_attr=attn_attr,
            qkv_names=(),
            qkv_style='unknown',
        )

    @staticmethod
    def _find_submodule(module: nn.Module, name: str) -> Optional[nn.Module]:
        """Find a submodule by name, searching recursively."""
        # Direct attribute
        if hasattr(module, name):
            sub = getattr(module, name)
            if isinstance(sub, nn.Module):
                return sub
        # Search one level of named_children
        for child_name, child in module.named_children():
            if hasattr(child, name):
                sub = getattr(child, name)
                if isinstance(sub, nn.Module):
                    return sub
        return None

    # ------------------------------------------------------------------
    # FFN discovery
    # ------------------------------------------------------------------
    def _discover_ffn(self, layer: nn.Module) -> FFNPattern:
        """Find FFN/MLP module within a layer."""
        ffn_candidates = [
            'intermediate', 'mlp', 'feed_forward', 'ffn',
            'output', 'ff', 'dense',
        ]
        for attr in ffn_candidates:
            mod = getattr(layer, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                return FFNPattern(module_attr=attr)

        # Search named_children
        for name, mod in layer.named_children():
            name_lower = name.lower()
            type_lower = type(mod).__name__.lower()
            if any(kw in name_lower or kw in type_lower for kw in ('mlp', 'ffn', 'feed', 'intermediate')):
                return FFNPattern(module_attr=name)

        return FFNPattern(module_attr='')

    # ------------------------------------------------------------------
    # LayerNorm discovery
    # ------------------------------------------------------------------
    def _discover_layernorm(self, layer: nn.Module) -> List[str]:
        """Find all LayerNorm instances within a layer."""
        ln_names = []
        for name, mod in layer.named_modules():
            if isinstance(mod, (nn.LayerNorm,)):
                ln_names.append(name)
            elif type(mod).__name__ in ('RMSNorm', 'LlamaRMSNorm', 'T5LayerNorm'):
                ln_names.append(name)
        return ln_names

    # ------------------------------------------------------------------
    # Embedding discovery
    # ------------------------------------------------------------------
    def _discover_embedding(self) -> Optional[nn.Module]:
        """Find the word embedding layer."""
        # Try common paths
        embed_paths = [
            ('embeddings', 'word_embeddings'),
            ('embed_tokens',),
            ('wte',),
            ('embed_in',),
            ('embeddings',),
        ]
        for path in embed_paths:
            obj = self.backbone
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Embedding):
                    return obj
            except AttributeError:
                continue

        # Fallback: first nn.Embedding in backbone
        for _, mod in self.backbone.named_modules():
            if isinstance(mod, nn.Embedding):
                return mod

        return None

    # ------------------------------------------------------------------
    # Classifier head discovery
    # ------------------------------------------------------------------
    def _find_classifier_head(self) -> Optional[nn.Module]:
        """Find the output head module."""
        candidates = ['classifier', 'lm_head', 'cls', 'score', 'qa_outputs', 'pre_classifier']
        for attr in candidates:
            mod = getattr(self.model, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                return mod
        return None

    # ------------------------------------------------------------------
    # Parameter groups (replaces hardcoded _layer_group_patterns)
    # ------------------------------------------------------------------
    def get_parameter_groups(self) -> Dict[str, List[str]]:
        """
        Build parameter group mapping using discovered structure.

        Returns dict mapping group_name -> list of parameter name substrings.
        """
        groups: Dict[str, List[str]] = {
            'embedding': [],
            'classifier': [],
        }

        # Find embedding param names
        if self.embedding is not None:
            for pname, _ in self.embedding.named_parameters():
                # Get the full parameter path
                for full_name, param in self.model.named_parameters():
                    if param.data_ptr() == _.data_ptr():
                        groups['embedding'].append(full_name)
                        break

        # Find classifier param names
        if self.classifier is not None:
            for pname, _ in self.classifier.named_parameters():
                for full_name, param in self.model.named_parameters():
                    if param.data_ptr() == _.data_ptr():
                        groups['classifier'].append(full_name)
                        break

        # Per-layer groups
        layers_prefix = self._layers_path
        for i, layer in enumerate(self.layers):
            layer_prefix = f"{layers_prefix}.{i}"
            attn_attr = self._attn_pattern.module_attr if self._attn_pattern else ''
            ffn_attr = self._ffn_pattern.module_attr if self._ffn_pattern else ''

            groups[f'layer{i}_attention'] = [f'{layer_prefix}.{attn_attr}'] if attn_attr else [layer_prefix]

            if self._attn_pattern and self._attn_pattern.qkv_names:
                qkv_patterns = [f'{layer_prefix}.{attn_attr}.{n}'
                                for n in self._attn_pattern.qkv_names]
                # For separate style, also check inside 'self' submodule
                if self._attn_pattern.qkv_style == 'separate':
                    qkv_patterns += [f'{layer_prefix}.{attn_attr}.self.{n}'
                                     for n in self._attn_pattern.qkv_names]
                groups[f'layer{i}_qkv'] = qkv_patterns
            else:
                groups[f'layer{i}_qkv'] = [f'{layer_prefix}.{attn_attr}']

            groups[f'layer{i}_ffn'] = [f'{layer_prefix}.{ffn_attr}'] if ffn_attr else [layer_prefix]

            if self._ln_names:
                groups[f'layer{i}_layernorm'] = [f'{layer_prefix}.{ln}' for ln in self._ln_names]
            else:
                groups[f'layer{i}_layernorm'] = [f'{layer_prefix}.LayerNorm', f'{layer_prefix}.layer_norm']

        return groups

    # ------------------------------------------------------------------
    # Attention hooks
    # ------------------------------------------------------------------
    def register_attention_hooks(self, callback) -> List[Any]:
        """Enable attention output and register hooks as fallback."""
        # Newer transformers requires eager attention for output_attentions
        if hasattr(self.config, '_attn_implementation'):
            self.config._attn_implementation = 'eager'
        self.config.output_attentions = True
        return []

    # ------------------------------------------------------------------
    # Layer sampling
    # ------------------------------------------------------------------
    def get_sampled_layer_indices(self, strategy: str = 'early_mid_late') -> List[int]:
        """Return layer indices for sampling (early, mid, late)."""
        n = self.num_layers
        if n <= 0:
            return []
        if n <= 3:
            return list(range(n))
        return [0, n // 2, n - 1]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_attention_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Return the attention module for a specific layer."""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return None
        layer = self.layers[layer_idx]
        if self._attn_pattern is None:
            return None
        return getattr(layer, self._attn_pattern.module_attr, None)

    def get_qkv_modules(self, layer_idx: int) -> Optional[Tuple[nn.Module, ...]]:
        """Return Q, K, V projection modules for a layer."""
        attn = self.get_attention_module(layer_idx)
        if attn is None or not self._attn_pattern or not self._attn_pattern.qkv_names:
            return None

        modules = []
        for name in self._attn_pattern.qkv_names:
            mod = self._find_submodule(attn, name)
            if mod is not None:
                modules.append(mod)
        return tuple(modules) if modules else None

    def get_ffn_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Return the FFN/MLP module for a specific layer."""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return None
        layer = self.layers[layer_idx]
        if self._ffn_pattern is None or not self._ffn_pattern.module_attr:
            return None
        return getattr(layer, self._ffn_pattern.module_attr, None)
