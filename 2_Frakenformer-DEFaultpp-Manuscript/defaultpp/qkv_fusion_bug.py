class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def fuse_qkv_projections(self):
        """Fuse Q/K/V into a single linear layer for efficiency."""
        w = torch.cat([self.q_proj.weight,
                       self.k_proj.weight,
                       self.v_proj.weight])
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_qkv.weight.data.copy_(w)
        # BUG: original q_proj, k_proj, v_proj NOT deleted

    def forward(self, x, context=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # uses fused layer
        # ... attention computation with q, k, v ...
        return self.out_proj(attn_output)

# --- Downstream: LoRA targets the stale projections ---
lora_config = LoraConfig(
    target_modules=["q_proj", "v_proj"],  # dead parameters!
)
model = get_peft_model(pipeline.unet, lora_config)
model.train()  # LoRA adapts stale layers; no effect on forward pass
