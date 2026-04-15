import torch
import torch.nn as nn
from .base_fault import AttentionFault


class ForceUnoptimizedFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        self._backup_param('cudnn_enabled', torch.backends.cudnn.enabled)
        torch.backends.cudnn.enabled = False
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        torch.backends.cudnn.enabled = self._backups.get('cudnn_enabled', True)
        self._active = False


class WrongLayoutFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        attn = self._get_attention(model, layer_idx)
        self._orig_forward = attn.forward
        self._active = True

        def patched_forward(*args, **kwargs):
            result = self._orig_forward(*args, **kwargs)
            if isinstance(result, tuple) and result[0] is not None:
                out = result[0].transpose(-2, -1).contiguous().transpose(-2, -1)
                return (out,) + result[1:]
            return result

        self._backup_param('forward', self._orig_forward)
        attn.forward = patched_forward
        self._attn_ref = attn

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._attn_ref.forward = self._backups['forward']
        self._active = False


class InconsistentDropoutFault(AttentionFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        rate = params.get('rate', 0.5)
        attn = self._get_attention(model, layer_idx)
        dropout_attr = None
        for attr in ('dropout', 'attention_dropout', 'attn_dropout'):
            if hasattr(attn, attr):
                dropout_attr = attr
                break
        if dropout_attr:
            dropout_module = getattr(attn, dropout_attr)
            self._backup_param('dropout_p', dropout_module.p)
            dropout_module.p = rate
            self._dropout_ref = dropout_module
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        if hasattr(self, '_dropout_ref') and 'dropout_p' in self._backups:
            self._dropout_ref.p = self._backups['dropout_p']
        self._active = False


KERNEL_FAULTS = {
    'force_unoptimized': ForceUnoptimizedFault,
    'wrong_layout': WrongLayoutFault,
    'inconsistent_dropout': InconsistentDropoutFault,
}


def create_kernel_fault(name: str, **kwargs) -> AttentionFault:
    return KERNEL_FAULTS[name](**kwargs)
