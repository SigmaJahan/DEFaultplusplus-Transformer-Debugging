import torch
import torch.nn as nn
from .base_fault import BaseFault


class ZeroMaskFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        self._active = True
        self._orig_forward = model.forward

        def patched_forward(*args, **kwargs):
            kwargs['attention_mask'] = None
            return self._orig_forward(*args, **kwargs)

        self._backup_param('forward', self._orig_forward)
        model.forward = patched_forward

    def restore(self, model: nn.Module):
        if 'forward' in self._backups:
            model.forward = self._backups['forward']
        self._active = False


class InvertedMaskFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        self._active = True
        self._orig_forward = model.forward

        def patched_forward(*args, **kwargs):
            mask = kwargs.get('attention_mask', None)
            if mask is not None:
                kwargs['attention_mask'] = 1 - mask
            return self._orig_forward(*args, **kwargs)

        self._backup_param('forward', self._orig_forward)
        model.forward = patched_forward

    def restore(self, model: nn.Module):
        if 'forward' in self._backups:
            model.forward = self._backups['forward']
        self._active = False


class WrongMaskBroadcastFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        self._active = True
        self._orig_forward = model.forward

        def patched_forward(*args, **kwargs):
            mask = kwargs.get('attention_mask', None)
            if mask is not None and mask.dim() == 2:
                kwargs['attention_mask'] = mask.unsqueeze(1)
            return self._orig_forward(*args, **kwargs)

        self._backup_param('forward', self._orig_forward)
        model.forward = patched_forward

    def restore(self, model: nn.Module):
        if 'forward' in self._backups:
            model.forward = self._backups['forward']
        self._active = False


MASKING_FAULTS = {
    'zero_mask': ZeroMaskFault,
    'inverted_mask': InvertedMaskFault,
    'wrong_broadcast': WrongMaskBroadcastFault,
}


def create_masking_fault(name: str, **kwargs) -> BaseFault:
    return MASKING_FAULTS[name](**kwargs)
