from .base_fault import BaseFault, AttentionFault, get_attention_module_from_layer
from .attention_utils import (
    is_encoder_architecture, get_qkv_projections,
    get_attention_module_from_layer as get_attn_module,
)
from .masking_faults import (
    ZeroMaskFault, InvertedMaskFault, WrongMaskBroadcastFault,
    MASKING_FAULTS, create_masking_fault,
)
from .qkv_faults import (
    ZeroQueryFault, ZeroKeyFault, ZeroValueFault,
    SwappedQKFault, TieHeadsFault, WrongHeadDimFault, FreezeQKVFault,
    QKV_FAULTS, create_qkv_fault,
)
from .score_faults import (
    MissingScalingFault, WrongScalingFault,
    SCORE_FAULTS, create_score_fault,
)
from .positional_faults import (
    MissingPositionalFault, OffByOneFault, TruncatePositionsFault, DoublePositionFault,
    POSITIONAL_FAULTS, create_positional_fault,
)
from .kernel_faults import (
    ForceUnoptimizedFault, WrongLayoutFault, InconsistentDropoutFault,
    KERNEL_FAULTS, create_kernel_fault,
)
from .variant_faults import (
    WrongVariantFault, CausalInBidirectionalFault,
    VARIANT_FAULTS, create_variant_fault,
)
from .embedding_faults import (
    EmbeddingZeroFault, EmbeddingSwapFault, TypeEmbeddingDropFault,
    EMBEDDING_FAULTS, create_embedding_fault,
)
from .ffn_faults import (
    FFNWeightScalingFault, FFNNeuronDropFault, ActivationDistortionFault,
    FFN_FAULTS, create_ffn_fault,
)
from .layernorm_faults import (
    LNGammaFault, LNBetaFault, LNStatsFault,
    LAYERNORM_FAULTS, create_layernorm_fault,
)
from .residual_faults import (
    ResidualDropFault, ResidualScaleFault, ResidualNoiseFault,
    RESIDUAL_FAULTS, create_residual_fault,
)
from .output_faults import (
    OutScaleFault, OutRowDropFault, OutNoiseFault,
    OUTPUT_FAULTS, create_output_fault,
)
from .pooler_faults import (
    PoolerScaleFault, PoolerZeroFault, PoolerNoiseFault,
    POOLER_FAULTS, create_pooler_fault,
)

ALL_FAULTS = {
    **MASKING_FAULTS,
    **QKV_FAULTS,
    **SCORE_FAULTS,
    **POSITIONAL_FAULTS,
    **KERNEL_FAULTS,
    **VARIANT_FAULTS,
    **EMBEDDING_FAULTS,
    **FFN_FAULTS,
    **LAYERNORM_FAULTS,
    **RESIDUAL_FAULTS,
    **OUTPUT_FAULTS,
    **POOLER_FAULTS,
}
