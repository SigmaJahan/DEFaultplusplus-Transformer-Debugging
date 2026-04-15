from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy
import torch.nn as nn


class BaseFault(ABC):
    def __init__(self):
        self._backups: Dict[str, Any] = {}
        self._active = False

    @abstractmethod
    def inject(self, model: nn.Module, layer_idx: int = 0, **params) -> None:
        ...

    @abstractmethod
    def restore(self, model: nn.Module) -> None:
        ...

    def remove(self, model: nn.Module) -> None:
        self.restore(model)
        self._backups.clear()
        self._active = False

    def _backup_param(self, key: str, value: Any):
        if key not in self._backups:
            self._backups[key] = copy.deepcopy(value)

    def _get_backup(self, key: str) -> Any:
        return self._backups.get(key)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class AttentionFault(BaseFault):
    def _get_encoder_layers(self, model: nn.Module):
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return model.encoder.layer
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            return model.transformer.layer
        if hasattr(model, 'roberta'):
            return model.roberta.encoder.layer
        if hasattr(model, 'bert'):
            return model.bert.encoder.layer
        if hasattr(model, 'electra'):
            return model.electra.encoder.layer
        if hasattr(model, 'distilbert'):
            return model.distilbert.transformer.layer
        raise ValueError("Cannot find encoder layers in model")

    def _get_attention(self, model: nn.Module, layer_idx: int):
        layer = self._get_encoder_layers(model)[layer_idx]
        return get_attention_module_from_layer(layer)


def get_attention_module_from_layer(layer):
    if hasattr(layer, 'attention'):
        if hasattr(layer.attention, 'self'):
            return layer.attention.self
        return layer.attention
    raise ValueError("Cannot find attention module in layer")
