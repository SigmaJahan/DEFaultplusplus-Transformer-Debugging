from .issue_23349_jax_seq_lengths import CASE as ROW01
from .issue_103082_sdpa_causal_lneqs import CASE as ROW02
from .issue_19045_bert_relative_cache import CASE as ROW09
from .issue_17886_t5_prune_relative_bias import CASE as ROW10
from .issue_6_qkv_projection_loader import CASE as ROW35
from .issue_20_sparse_cache_logits import CASE as ROW18
from .issue_37574_swa_cache_roll import CASE as ROW19
from .issue_36096_flex_attention_weights import CASE as ROW24
from .issue_116333_kernel_stride_check import CASE as ROW39
from .issue_35896_qwen2_window_layers import CASE as ROW44
from .issue_11903_diffusers_qkv_fusion import CASE as ROW11903

CASES = [
    ROW01,
    ROW02,
    ROW09,
    ROW10,
    ROW35,
    ROW18,
    ROW19,
    ROW24,
    ROW39,
    ROW44,
    ROW11903,
]
