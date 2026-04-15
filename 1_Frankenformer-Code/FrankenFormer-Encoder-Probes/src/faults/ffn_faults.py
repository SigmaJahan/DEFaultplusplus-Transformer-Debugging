import torch
import torch.nn as nn
from .base_fault import BaseFault


def _get_ffn_layers(model, layer_idx):
    layers = None
    for attr in ('encoder.layer', 'bert.encoder.layer', 'roberta.encoder.layer',
                 'electra.encoder.layer', 'transformer.layer', 'distilbert.transformer.layer'):
        obj = model
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError("Cannot find encoder layers")
    layer = layers[layer_idx]
    if hasattr(layer, 'intermediate'):
        return layer.intermediate.dense, getattr(layer, 'output', None)
    if hasattr(layer, 'ffn'):
        return layer.ffn.lin1, layer.ffn.lin2
    raise ValueError("Cannot find FFN in layer")


class FFNWeightScalingFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        scale = params.get('scale', 10.0)
        dense, _ = _get_ffn_layers(model, layer_idx)
        self._backup_param('weight', dense.weight.data.clone())
        dense.weight.data.mul_(scale)
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        dense, _ = _get_ffn_layers(model, layer_idx)
        dense.weight.data.copy_(self._backups['weight'])
        self._active = False


class FFNNeuronDropFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        fraction = params.get('fraction', 0.1)
        dense, _ = _get_ffn_layers(model, layer_idx)
        self._backup_param('weight', dense.weight.data.clone())
        self._backup_param('bias', dense.bias.data.clone() if dense.bias is not None else None)
        num_drop = int(dense.weight.shape[0] * fraction)
        indices = torch.randperm(dense.weight.shape[0])[:num_drop]
        dense.weight.data[indices] = 0
        if dense.bias is not None:
            dense.bias.data[indices] = 0
        self._active = True
        self._model_ref = (model, layer_idx)

    def restore(self, model: nn.Module):
        if not self._active:
            return
        _, layer_idx = self._model_ref
        dense, _ = _get_ffn_layers(model, layer_idx)
        dense.weight.data.copy_(self._backups['weight'])
        if self._backups['bias'] is not None:
            dense.bias.data.copy_(self._backups['bias'])
        self._active = False


class ActivationDistortionFault(BaseFault):
    def inject(self, model: nn.Module, layer_idx: int = 0, **params):
        target_act = params.get('activation', 'relu')
        layers = None
        for attr in ('encoder.layer', 'bert.encoder.layer', 'roberta.encoder.layer',
                     'electra.encoder.layer', 'transformer.layer', 'distilbert.transformer.layer'):
            obj = model
            try:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue
        layer = layers[layer_idx]
        if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'intermediate_act_fn'):
            self._backup_param('act_fn', layer.intermediate.intermediate_act_fn)
            layer.intermediate.intermediate_act_fn = getattr(nn.functional, target_act)
            self._layer_ref = layer
        elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'activation'):
            self._backup_param('act_fn', layer.ffn.activation)
            layer.ffn.activation = getattr(nn.functional, target_act)
            self._layer_ref = layer
        self._active = True

    def restore(self, model: nn.Module):
        if not self._active:
            return
        layer = self._layer_ref
        if hasattr(layer, 'intermediate') and 'act_fn' in self._backups:
            layer.intermediate.intermediate_act_fn = self._backups['act_fn']
        elif hasattr(layer, 'ffn') and 'act_fn' in self._backups:
            layer.ffn.activation = self._backups['act_fn']
        self._active = False


FFN_FAULTS = {
    'ffn_weight_scaling': FFNWeightScalingFault,
    'ffn_neuron_drop': FFNNeuronDropFault,
    'activation_distortion': ActivationDistortionFault,
}


def create_ffn_fault(name: str, **kwargs) -> BaseFault:
    return FFN_FAULTS[name](**kwargs)
