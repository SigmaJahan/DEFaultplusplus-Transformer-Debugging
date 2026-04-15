import math
import torch.nn as nn
from .base_fault import AttentionFault


class MissingScalingFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        head_dim = getattr(attn, 'attention_head_size',
                           getattr(attn, 'dim', 64) // getattr(attn, 'n_heads', 1))
        self._backup_param('scale', 1.0 / math.sqrt(head_dim))
        if hasattr(attn, 'attention_head_size'):
            attn.attention_head_size = 1
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        if hasattr(attn, 'attention_head_size'):
            scale = self._backups['scale']
            attn.attention_head_size = round(1.0 / (scale ** 2))
        self._active = False


class WrongScalingFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        factor = params.get('factor', 10.0)
        attn = self._get_attention(model, layer_idx)
        if hasattr(attn, 'attention_head_size'):
            self._backup_param('attention_head_size', attn.attention_head_size)
            attn.attention_head_size = int(attn.attention_head_size * factor)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        attn = self._get_attention(model, layer_idx)
        if 'attention_head_size' in self._backups:
            attn.attention_head_size = self._backups['attention_head_size']
        self._active = False


SCORE_FAULTS = {
    'missing_scaling': MissingScalingFault,
    'wrong_scaling': WrongScalingFault,
}


def create_score_fault(name: str, **kwargs) -> AttentionFault:
    return SCORE_FAULTS[name](**kwargs)
