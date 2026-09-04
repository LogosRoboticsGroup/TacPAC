# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted from the official Wan2.2 repository: https://github.com/Wan-Video/Wan2.2
# Multi-view extension of WanModel.
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import ModelMixin

from starVLA.model.modules.wan_video.wan_submodules import (
    WanRMSNorm, WanLayerNorm, Head, WanAttentionBlock,
    rope_params, sinusoidal_embedding_1d,
)
from starVLA.model.modules.wan_video.wan_dit import WanModel

__all__ = ['WanMultiViewModel', 'WanMultiViewAttentionBlock']


class WanCrossViewAttention(nn.Module):
    """
    Simple SDPA-based cross-view attention.
    """
    
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
    def forward(self, x, attn_bias=None):
        """
        Args:
            x: [B, V*L, C]
            attn_bias: [B, 1, V*L, V*L] float32 additive mask, or None
        Returns:
            [B, V*L, C]
        """
        B, S, C = x.shape
        n, d = self.num_heads, self.head_dim
        
        q = self.norm_q(self.q(x)).reshape(B, S, n, d).permute(0, 2, 1, 3)  # [B, n, S, d]
        k = self.norm_k(self.k(x)).reshape(B, S, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, S, n, d).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.permute(0, 2, 1, 3).reshape(B, S, C)
        return self.o(out)
    

def _build_cross_view_attn_bias(view_mask, n_view, L, device):
    """
    Build additive attention bias for cross-view attention.

    Args:
        view_mask: [B, V] bool — which views are valid
        n_view: int V
        L: int — tokens per view (padded seq_len)
        device, dtype

    Returns:
        attn_bias [B, 1, V*L, V*L] — 0 where attention is allowed, -1e6 where masked
    """
    B = view_mask.shape[0]
    vm = view_mask.to(device=device, dtype=torch.bool)  # [B, V]

    # valid[b, v1, v2] = True if v1 can attend v2
    valid = vm.unsqueeze(2) & vm.unsqueeze(1)  # [B, V, V]
    # Each view always attends itself (diagonal)
    eye = torch.eye(n_view, dtype=torch.bool, device=device).unsqueeze(0)  # [1, V, V]
    valid = valid | eye  # [B, V, V]

    # Expand to token level: [B, V*L, V*L]
    # valid[b, v1, v2] → all tokens of v1 can attend all tokens of v2
    valid_tok = (
        valid
        .unsqueeze(2).unsqueeze(4)
        .expand(B, n_view, L, n_view, L)
        .reshape(B, n_view * L, n_view * L)
    )

    attn_bias = torch.zeros(B, 1, n_view * L, n_view * L, device=device, dtype=torch.float32)
    attn_bias.masked_fill_(~valid_tok.unsqueeze(1), -1e6)
    return attn_bias


class WanMultiViewAttentionBlock(WanAttentionBlock):
    """
    Extends WanAttentionBlock with a cross-view attention layer inserted between
    self-attention and cross-attention (text).
    
    When n_view == 1, cross-view attention is completely skipped, making this block
    numerically identical to WanAttentionBlock.
    """
    
    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.norm_cv = WanLayerNorm(dim, eps)
        self.cross_view_attn = WanCrossViewAttention(dim, num_heads, qk_norm, eps)

    @torch.no_grad()
    def init_cross_view_from_self_attn(self):
        self.cross_view_attn.q.load_state_dict(self.self_attn.q.state_dict())
        self.cross_view_attn.k.load_state_dict(self.self_attn.k.state_dict())
        self.cross_view_attn.v.load_state_dict(self.self_attn.v.state_dict())
        self.cross_view_attn.o.load_state_dict(self.self_attn.o.state_dict())
        self.cross_view_attn.norm_q.load_state_dict(self.self_attn.norm_q.state_dict())
        self.cross_view_attn.norm_k.load_state_dict(self.self_attn.norm_k.state_dict())
        
    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                e_cv=None, n_view=1, view_mask=None):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        if n_view > 1:
            BV, L, C = x.shape
            x_cv = x.reshape(BV // n_view, n_view * L, C)
            attn_bias = _build_cross_view_attn_bias(
                view_mask, n_view, L, x.device) if view_mask is not None else None
            cv = self.cross_view_attn(self.norm_cv(x_cv), attn_bias).reshape(BV, L, C)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + cv * e_cv

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x
    
class WanMultiViewModel(WanModel):
    """
    Extends WanModel with:
      - Per-block cross-view attention (copy-initialized from self-attention)
      - Zero-initialized timestep-conditioned cross-view gate
      - View embedding (zero-init)
      
    Loading pretrain WanModel weights via strict=False preserves the original
    distribution because the cross-view residual gate is zero-initialized.
    """
    
    _no_split_modules = ['WanMultiViewAttentionBlock']
    
    @register_to_config
    def __init__(self,
                 model_type='ti2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=48,
                 dim=3072,
                 ffn_dim=14336,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=48,
                 num_heads=24,
                 num_layers=30,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_view=None):
        ModelMixin.__init__(self)
        
        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type
        
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if num_view is None:
            raise ValueError("WanMultiViewModel requires `num_view` (the data mix's video-key count).")
        self.num_view = int(num_view)
        
        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim)
        )
        
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )
        self.cross_view_time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim)
        )
        
        # blocks
        self.blocks = nn.ModuleList([
            WanMultiViewAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                                       cross_attn_norm, eps) for _ in range(num_layers)
        ])
        
        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # RoPE frequencies (single concatenated tensor, matching official)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        
        # view embedding - zero-init, so adding it has no effect at initialization
        self.view_embed = nn.Parameter(torch.zeros(self.num_view, dim))
        
        # initialize weights
        self.init_weights()
        self.init_cross_view_from_self_attn()
        nn.init.zeros_(self.cross_view_time_projection[-1].weight)
        nn.init.zeros_(self.cross_view_time_projection[-1].bias)
        self.gradient_checkpointing = False
        
    @staticmethod
    def state_dict_has_cross_view_weights(state_dict):
        required = (
            "cross_view_attn.q.weight",
            "cross_view_attn.k.weight",
            "cross_view_attn.v.weight",
            "cross_view_attn.o.weight",
        )
        return all(any(name in key for key in state_dict) for name in required)

    @torch.no_grad()
    def init_cross_view_from_self_attn(self):
        for block in self.blocks:
            block.init_cross_view_from_self_attn()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        n_view=1,
        view_mask=None,
        return_hidden_states=False,
    ):
        """
        Forward pass through the diffusion model.

        Args:
            x (Tensor): Input video latents [B*V, C_in, T, H, W]
            t (Tensor): Timesteps [B*V] (scalar per sample) or [B*V, seq_len] (per-token)
            context (List[Tensor]): List of text embeddings, each [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            y (Tensor, optional): Conditional video latents [B*V, C_in, T, H, W] for i2v mode
            n_view (int): Number of views; batch dimension is B*n_view
            view_mask (Tensor, optional): [B, V] bool mask for valid views
            return_hidden_states (bool): If True, return dict with noise_pred and
                video_states_buffer (list of per-block hidden states [B*V, seq_len, dim]).
                If False (default), return noise_pred Tensor only (backward compatible).

        Returns:
            Tensor or dict: Denoised video latents [B*V, C_out, T, H, W], or dict with
                "noise_pred" and "video_states_buffer" when return_hidden_states=True.
        """
        if self.model_type == 'i2v':
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        # patch embedding: batched conv3d over the full batch
        BV = x.shape[0]
        x = self.patch_embedding(x)                          # [BV, dim, T', H', W']
        patch_grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)                    # [BV, T'H'W', dim]
        seq_lens = torch.full((BV,), x.size(1), dtype=torch.long, device=device)
        assert x.size(1) <= seq_len

        # inject view embedding (zero at init → no effect when loading pretrain)
        if n_view > 1:
            B = BV // n_view
            view_indices = torch.arange(n_view, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.view_embed[view_indices.reshape(BV)].unsqueeze(1)

        # time embeddings — support per-token timesteps [BV, seq_len]
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            e_cv = self.cross_view_time_projection(e)
            assert (
                e.dtype == torch.float32
                and e0.dtype == torch.float32
                and e_cv.dtype == torch.float32
            )

        # context (text embedding)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=patch_grid,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            e_cv=e_cv,
            n_view=n_view,
            view_mask=view_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        hidden_states_buffer = [] if return_hidden_states else None

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
            if return_hidden_states:
                hidden_states_buffer.append(x)

        # head
        x = self.head(x, e)

        # unpatchify
        noise_pred = self.unpatchify(x, patch_grid)

        if return_hidden_states:
            return {"noise_pred": noise_pred, "video_states_buffer": hidden_states_buffer}
        return noise_pred


if __name__ == "__main__":
    """
    Equivalence test: WanMultiViewModel(n_view=1) must match WanModel output.
    """
    import sys

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 for testing to avoid the autocast requirement that WanAttentionBlock has
    # (it uses .float() internally which requires autocast for bfloat16 weights)
    dtype = torch.float32

    # Minimal config for fast test
    kwargs = dict(
        model_type='ti2v',
        patch_size=(1, 2, 2),
        text_len=16,
        in_dim=4,
        dim=64,
        ffn_dim=128,
        freq_dim=32,
        text_dim=32,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )

    print("Building WanModel and WanMultiViewModel...")
    model_orig = WanModel(**kwargs).to(dtype).to(device)
    model_mv = WanMultiViewModel(**kwargs, num_view=3).to(dtype).to(device)

    # Load WanModel weights into WanMultiViewModel (strict=False)
    missing, unexpected = model_mv.load_state_dict(model_orig.state_dict(), strict=False)
    print(f"  Missing keys (new params): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Dummy input
    T, H, W = 5, 8, 8  # small spatial dims
    x_in = torch.randn(1, kwargs['in_dim'], T, H, W, dtype=dtype, device=device)
    T_patch = T
    H_patch = H // kwargs['patch_size'][1]
    W_patch = W // kwargs['patch_size'][2]
    seq_len = T_patch * H_patch * W_patch

    t = torch.tensor([500.0], device=device, dtype=torch.float32)
    t_expanded = t.unsqueeze(1).expand(1, seq_len)
    ctx = torch.randn(1, kwargs['text_len'], kwargs['text_dim'], dtype=dtype, device=device)
    ctx_list = [ctx[0]]

    with torch.no_grad():
        out_orig = model_orig(x_in, t=t_expanded, context=ctx_list, seq_len=seq_len)
        out_mv_1v = model_mv(x_in, t=t_expanded, context=ctx_list,
                             seq_len=seq_len, n_view=1)

    diff = (out_orig.float() - out_mv_1v.float()).abs().max()
    print(f"\n[TEST 1] n_view=1 equivalence: max_diff = {diff:.2e}", end="")
    if diff < 1e-3:
        print(" ✓ PASS")
    else:
        print(" ✗ FAIL (expected < 1e-3)")
        sys.exit(1)

    # Test n_view=3 forward pass (3 copies of same input)
    print("[TEST 2] n_view=3 forward pass...", end=" ")
    x3 = x_in.expand(3, -1, -1, -1, -1).contiguous()   # [3, C, T, H, W]
    ctx3 = [ctx[0]] * 3
    t3 = t_expanded.expand(3, -1)
    view_mask_3 = torch.ones(1, 3, dtype=torch.bool, device=device)

    with torch.no_grad():
        out_3v = model_mv(x3, t=t3, context=ctx3, seq_len=seq_len,
                          n_view=3, view_mask=view_mask_3)

    assert out_3v.shape == x3.shape
    print(f"output shape: {list(out_3v.shape)} ✓ PASS")

    # Test return_hidden_states=False (default) backward compatibility
    print("[TEST 3] return_hidden_states=False backward compat...", end=" ")
    with torch.no_grad():
        out_default = model_mv(x_in, t=t_expanded, context=ctx_list, seq_len=seq_len, n_view=1)
    assert isinstance(out_default, torch.Tensor), "Expected Tensor when return_hidden_states=False"
    assert out_default.shape == x_in.shape
    print(f"type=Tensor, shape={list(out_default.shape)} ✓ PASS")

    # Test return_hidden_states=True returns dict with correct buffer
    print("[TEST 5] return_hidden_states=True dict format...", end=" ")
    with torch.no_grad():
        out_hs = model_mv(x_in, t=t_expanded, context=ctx_list, seq_len=seq_len,
                          n_view=1, return_hidden_states=True)
    assert isinstance(out_hs, dict), "Expected dict when return_hidden_states=True"
    assert "noise_pred" in out_hs and "video_states_buffer" in out_hs
    buf = out_hs["video_states_buffer"]
    assert len(buf) == kwargs['num_layers'], f"Expected {kwargs['num_layers']} layers, got {len(buf)}"
    assert buf[0].shape == (1, seq_len, kwargs['dim']), f"Unexpected shape: {buf[0].shape}"
    assert out_hs["noise_pred"].shape == x_in.shape
    print(f"num_layers={len(buf)}, hidden shape={list(buf[0].shape)} ✓ PASS")

    print("\nAll tests passed!")
