import torch
import torch.nn as nn
from .base_fault import BaseFault


def _get_output_module(model, layer_idx):
    for attr in ('encoder.layer', 'bert.encoder.layer', 'roberta.encoder.layer',
                 'electra.encoder.layer', 'transformer.layer', 'distilbert.transformer.layer'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            layer = obj[layer_idx]
            break
        except AttributeError:
            continue
    else:
        raise ValueError("Cannot find encoder layers")
    if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
        return layer.output
    if hasattr(layer, 'sa_layer_norm'):
        return layer
    raise ValueError("Cannot find output module in layer")


class ResidualDropFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        out_mod = _get_output_module(model, layer_idx)
        self._orig_forward = out_mod.forward

        def patched_forward(hidden_states, input_tensor=None):
            dense_out = out_mod.dense(hidden_states)
            dense_out = out_mod.dropout(dense_out)
            return out_mod.LayerNorm(dense_out)

        self._backup_param('forward', self._orig_forward)
        out_mod.forward = patched_forward
        self._mod_ref = out_mod
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._mod_ref.forward = self._backups['forward']
        self._active = False


class ResidualScaleFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        scale = params.get('scale', 0.0)
        out_mod = _get_output_module(model, layer_idx)
        self._orig_forward = out_mod.forward

        def patched_forward(hidden_states, input_tensor=None):
            dense_out = out_mod.dense(hidden_states)
            dense_out = out_mod.dropout(dense_out)
            if input_tensor is not None:
                return out_mod.LayerNorm(dense_out + input_tensor * scale)
            return out_mod.LayerNorm(dense_out)

        self._backup_param('forward', self._orig_forward)
        out_mod.forward = patched_forward
        self._mod_ref = out_mod
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._mod_ref.forward = self._backups['forward']
        self._active = False


class ResidualNoiseFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        std = params.get('std', 0.1)
        out_mod = _get_output_module(model, layer_idx)
        self._orig_forward = out_mod.forward

        def patched_forward(hidden_states, input_tensor=None):
            dense_out = out_mod.dense(hidden_states)
            dense_out = out_mod.dropout(dense_out)
            if input_tensor is not None:
                noisy_residual = input_tensor + torch.randn_like(input_tensor) * std
                return out_mod.LayerNorm(dense_out + noisy_residual)
            return out_mod.LayerNorm(dense_out)

        self._backup_param('forward', self._orig_forward)
        out_mod.forward = patched_forward
        self._mod_ref = out_mod
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        self._mod_ref.forward = self._backups['forward']
        self._active = False


RESIDUAL_FAULTS = {
    'residual_drop': ResidualDropFault,
    'residual_scale': ResidualScaleFault,
    'residual_noise': ResidualNoiseFault,
}


def create_residual_fault(name: str, **kwargs) -> BaseFault:
    return RESIDUAL_FAULTS[name](**kwargs)
