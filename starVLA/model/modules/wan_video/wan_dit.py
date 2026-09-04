# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted from the official Wan2.2 repository: https://github.com/Wan-Video/Wan2.2
import json
import os

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from starVLA.model.modules.wan_video.wan_submodules import (
    WanRMSNorm, WanLayerNorm, Head, WanAttentionBlock,
    rope_params, sinusoidal_embedding_1d,
)

__all__ = ['WanModel', 'WanModelStateDictConverter']

class WanModel(ModelMixin, ConfigMixin):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

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
                 eps=1e-6):
        super().__init__()

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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
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

        # initialize weights
        self.init_weights()
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        """
        Forward pass through the diffusion model.

        Args:
            x (Tensor): Input video latents [B, C_in, T, H, W]
            t (Tensor): Timesteps [B] (scalar per sample) or [B, seq_len] (per-token)
            context (List[Tensor]): List of text embeddings, each [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            y (Tensor, optional): Conditional video latents [B, C_in, T, H, W] for i2v mode

        Returns:
            Tensor: Denoised video latents [B, C_out, T, H, W]
        """
        if self.model_type == 'i2v':
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        # patch embedding: batched conv3d over the full batch
        B = x.shape[0]
        x = self.patch_embedding(x)                          # [B, dim, T', H', W']
        patch_grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)                    # [B, T'H'W', dim]
        seq_lens = torch.full((B,), x.size(1), dtype=torch.long, device=device)
        assert x.size(1) <= seq_len

        # time embeddings — support per-token timesteps [B, seq_len]
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

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
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        return self.unpatchify(x, patch_grid)

    def unpatchify(self, x, patch_grid):
        c = self.out_dim
        B = x.shape[0]
        t, h, w = patch_grid
        pt, ph, pw = self.patch_size
        x = x[:, :t * h * w].view(B, t, h, w, pt, ph, pw, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, t * pt, h * ph, w * pw)
        return x.float()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)


class WanModelStateDictConverter:

    @staticmethod
    def load_state_dict(path):
        state_dict = {}

        if os.path.isdir(path):
            safetensors_index = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
            safetensors_single = os.path.join(path, "diffusion_pytorch_model.safetensors")

            if os.path.exists(safetensors_index):
                from safetensors.torch import load_file
                with open(safetensors_index, 'r') as f:
                    index = json.load(f)
                for shard_file in set(index["weight_map"].values()):
                    shard_path = os.path.join(path, shard_file)
                    state_dict.update(load_file(shard_path))
            elif os.path.exists(safetensors_single):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_single)
            else:
                raise FileNotFoundError(
                    f"No safetensors files found in {path}")
        elif path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        elif path.endswith('.pth') or path.endswith('.pt'):
            try:
                state_dict = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            except (TypeError, RuntimeError):
                state_dict = torch.load(path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported file format: {path}")

        return state_dict

    @staticmethod
    def load_pretrained(model, path, strict=False):
        state_dict = WanModelStateDictConverter.load_state_dict(path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        return model
