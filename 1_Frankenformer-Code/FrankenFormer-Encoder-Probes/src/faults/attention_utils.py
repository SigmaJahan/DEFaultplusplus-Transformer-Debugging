import torch.nn as nn


def is_encoder_architecture(model: nn.Module) -> bool:
    model_type = getattr(getattr(model, 'config', None), 'model_type', '')
    return model_type in ('bert', 'roberta', 'distilbert', 'electra')


def get_attention_module_from_layer(layer):
    if hasattr(layer, 'attention'):
        if hasattr(layer.attention, 'self'):
            return layer.attention.self
        return layer.attention
    raise ValueError("Cannot find attention module in layer")


def get_qkv_projections(attn_module):
    if hasattr(attn_module, 'query'):
        return attn_module.query, attn_module.key, attn_module.value
    if hasattr(attn_module, 'q_lin'):
        return attn_module.q_lin, attn_module.k_lin, attn_module.v_lin
    raise ValueError("Cannot find QKV projections in attention module")
