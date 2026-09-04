import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from starVLA.model.modules.wan_video.wan_submodules import (
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d as wan_sinusoidal_embedding_1d,
)


def masked_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
    compatibility_mode: bool = True,
) -> torch.Tensor:
    if not compatibility_mode:
        raise NotImplementedError(
            "Only compatibility mode is implemented for masked attention. "
            "Please set compatibility_mode=True."
        )
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    embedding = wan_sinusoidal_embedding_1d(dim, position)
    if position.dtype.is_floating_point:
        return embedding.to(dtype=position.dtype)
    return embedding.to(dtype=torch.float32)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    return rope_params(end, dim, theta)


def precompute_freqs_cis_3d(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_dim = 2 * (dim // 6)
    w_dim = 2 * (dim // 6)
    f_dim = dim - h_dim - w_dim
    return (
        rope_params(end, f_dim, theta),
        rope_params(end, h_dim, theta),
        rope_params(end, w_dim, theta),
    )


def rope_apply_1d_packed(
    x: torch.Tensor,
    freqs: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int,
    num_query_per_group: int,
    num_key_per_group: int,
    mode: str = "causal",
) -> torch.Tensor:
    if mode not in {"causal", "group_diagonal"}:
        raise ValueError(f"Mode {mode} must be 'causal' or 'group_diagonal'")

    total_num_query_tokens = num_temporal_groups * num_query_per_group
    total_num_key_tokens = num_temporal_groups * num_key_per_group
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)

    query_time_indices = query_time_indices.unsqueeze(1)
    key_time_indices = key_time_indices.unsqueeze(0)
    if mode == "causal":
        attn_mask = query_time_indices >= key_time_indices
    else:
        attn_mask = query_time_indices == key_time_indices

    if attn_mask.shape != (total_num_query_tokens, total_num_key_tokens):
        raise RuntimeError("Attention mask shape mismatch")
    return attn_mask


class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return masked_flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)


class MoTCompatibleSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply_1d_packed(q, freqs, self.num_heads)
        k = rope_apply_1d_packed(k, freqs, self.num_heads)
        x = masked_flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class MoTCompatibleCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = masked_flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class MoTCompatibleDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = MoTCompatibleSelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = MoTCompatibleCrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(input_x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, has_pos_emb: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))

        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


RMSNorm = WanRMSNorm
SelfAttention = MoTCompatibleSelfAttention
CrossAttention = MoTCompatibleCrossAttention
DiTBlock = MoTCompatibleDiTBlock
flash_attention = masked_flash_attention
rope_apply = rope_apply_1d_packed


__all__ = [
    "AttentionModule",
    "CrossAttention",
    "DiTBlock",
    "GateModule",
    "Head",
    "MLP",
    "MoTCompatibleCrossAttention",
    "MoTCompatibleDiTBlock",
    "MoTCompatibleSelfAttention",
    "RMSNorm",
    "SelfAttention",
    "WanLayerNorm",
    "WanRMSNorm",
    "create_group_causal_attn_mask",
    "flash_attention",
    "masked_flash_attention",
    "modulate",
    "precompute_freqs_cis",
    "precompute_freqs_cis_3d",
    "rope_apply",
    "rope_apply_1d_packed",
    "sinusoidal_embedding_1d",
]
