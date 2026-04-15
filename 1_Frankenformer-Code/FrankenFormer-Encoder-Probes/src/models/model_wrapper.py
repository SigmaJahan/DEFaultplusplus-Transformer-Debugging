"""
Generic encoder model wrapper for multi-task pipeline.

Uses Hugging Face AutoModelForSequenceClassification / AutoModelForMaskedLM
to support multiple encoder checkpoints (BERT, RoBERTa, DistilBERT, ELECTRA)
while keeping the fault injection interface consistent.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.models.base_model import BaseModelWrapper, FaultInjectorMixin


class ModelWrapper(BaseModelWrapper, FaultInjectorMixin):
    """Wrapper around AutoModelForSequenceClassification with fault hooks."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ):
        if device is None:
            from src.utils.reproducibility import get_device
            device = get_device()

        BaseModelWrapper.__init__(self, model_name, num_labels=num_labels, device=device, cache_dir=cache_dir)
        FaultInjectorMixin.__init__(self)

        self.gradient_checkpointing = gradient_checkpointing
        self.layers: List[nn.Module] = []
        self.tokenizer = None

        self.load_model()

    def _get_layers(self) -> List[nn.Module]:
        """Extract transformer layers from various encoder architectures."""
        # BERT / RoBERTa: bert.encoder.layer or roberta.encoder.layer
        for attr in ['bert', 'roberta', 'electra']:
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                encoder = getattr(backbone, 'encoder', None)
                if encoder is not None and hasattr(encoder, 'layer'):
                    return list(encoder.layer)

        # DistilBERT: distilbert.transformer.layer
        distilbert = getattr(self.model, 'distilbert', None)
        if distilbert is not None:
            transformer = getattr(distilbert, 'transformer', None)
            if transformer is not None and hasattr(transformer, 'layer'):
                return list(transformer.layer)

        # ModernBERT: model.encoder.layers
        model_attr = getattr(self.model, 'model', None)
        if model_attr is not None:
            encoder = getattr(model_attr, 'encoder', None)
            if encoder is not None and hasattr(encoder, 'layers'):
                return list(encoder.layers)

        # Generic fallback
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return list(self.model.encoder.layer)

        error_msg = (
            f"Cannot locate transformer layers for encoder model '{self.model_name}'.\n"
            f"Model type: {type(self.model).__name__}\n"
            f"Supported architectures: BERT, DistilBERT, RoBERTa, ELECTRA.\n"
        )
        raise ValueError(error_msg)

    def load_model(self) -> nn.Module:
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir,
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                cache_dir=self.cache_dir,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load encoder model '{self.model_name}' from HuggingFace. "
                f"Error: {e}"
            ) from e

        try:
            if self.gradient_checkpointing:
                self.enable_gradient_checkpointing()
        except Exception as e:
            import logging
            logging.warning(f"Failed to enable gradient checkpointing: {e}")

        try:
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to move model to device {self.device}. Error: {e}"
            ) from e

        try:
            self.layers = self._get_layers()
        except ValueError as e:
            raise ValueError(
                f"Encoder model '{self.model_name}' loaded but layer extraction failed. "
                f"Original error: {e}"
            ) from e

        return self.model

    def get_attention_modules(self) -> Dict[str, nn.Module]:
        modules: Dict[str, nn.Module] = {}
        for idx, layer in enumerate(self.layers):
            # BERT / RoBERTa: layer.attention
            if hasattr(layer, "attention"):
                modules[f"layer_{idx}"] = layer.attention
            # DistilBERT: layer.attention (same attribute name)
            elif hasattr(layer, "self_attn"):
                modules[f"layer_{idx}"] = layer.self_attn
        return modules

    def get_attention_layers(self) -> list:
        attention_layers = []
        for layer in self.layers:
            if hasattr(layer, "attention"):
                attention_layers.append(layer.attention)
            elif hasattr(layer, "self_attn"):
                attention_layers.append(layer.self_attn)
        return attention_layers

    def get_layer(self, layer_idx: int) -> nn.Module:
        return self.layers[layer_idx]

    def get_embedding_layer(self) -> nn.Module:
        for attr in ['bert', 'roberta', 'electra', 'distilbert']:
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                embeddings = getattr(backbone, 'embeddings', None)
                if embeddings is not None:
                    word_emb = getattr(embeddings, 'word_embeddings', None)
                    if word_emb is not None:
                        return word_emb
        return self.model.get_input_embeddings()

    def get_classifier_head(self) -> nn.Module:
        if hasattr(self.model, "classifier"):
            return self.model.classifier
        if hasattr(self.model, "pre_classifier"):
            return self.model.pre_classifier
        raise AttributeError(f"No classifier head found for {self.model_name}")

    def get_pooler(self) -> Optional[nn.Module]:
        for attr in ['bert', 'roberta']:
            backbone = getattr(self.model, attr, None)
            if backbone is not None:
                pooler = getattr(backbone, 'pooler', None)
                if pooler is not None:
                    return pooler
        return None

    def get_num_layers(self) -> int:
        return len(self.layers)

    def get_hidden_size(self) -> int:
        if hasattr(self.model.config, "dim"):
            return self.model.config.dim
        return getattr(self.model.config, "hidden_size", 0)

    def get_num_attention_heads(self) -> int:
        if hasattr(self.model.config, "n_heads"):
            return self.model.config.n_heads
        return getattr(self.model.config, "num_attention_heads", 0)

    def get_model_config(self):
        return self.model.config

    def forward_with_attention(self, **inputs):
        return self.model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

    def get_attention_weights(self, **inputs) -> torch.Tensor:
        outputs = self.forward_with_attention(**inputs)
        return outputs.attentions

    def get_detailed_model_info(self) -> Dict[str, Any]:
        base_info = self.get_model_info()
        config = self.get_model_config()
        details = {
            **base_info,
            "model_type": getattr(config, "model_type", "unknown"),
            "num_layers": self.get_num_layers(),
            "hidden_size": self.get_hidden_size(),
            "num_attention_heads": self.get_num_attention_heads(),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        return details


def load_encoder_model(
    model_name: str,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> ModelWrapper:
    return ModelWrapper(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
        cache_dir=cache_dir,
        gradient_checkpointing=gradient_checkpointing,
    )
