import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Dict, List, Tuple

from starVLA.training.trainer_utils import initialize_overwatch

from .wan_dit_adapters import (
    GateModule,
    SelfAttention,
    WanLayerNorm,
    flash_attention,
    modulate,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    rope_apply,
    sinusoidal_embedding_1d,
)
from .expert_utils import validate_attention_config

logger = initialize_overwatch(__name__)


class TacDiTBlock(nn.Module):
    """DiT block without cross-attention: adaLN-modulated self-attention + FFN.

    Attribute names mirror `MoTCompatibleDiTBlock` so weights can be copied from a
    trained ActionExpert block via `load_state_dict(strict=False)`.
    """

    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
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
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        q = self.self_attn.norm_q(self.self_attn.q(attn_input))
        k = self.self_attn.norm_k(self.self_attn.k(attn_input))
        v = self.self_attn.v(attn_input)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        attn_out = flash_attention(
            q=q,
            k=torch.cat([k_cache, k], dim=1),
            v=torch.cat([v_cache, v], dim=1),
            num_heads=self.num_heads,
        )
        x = self.gate(x, gate_msa, self.self_attn.o(attn_out))
        x = self.gate(x, gate_mlp, self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class TactileExpert(nn.Module):
    """Single-pass corrector: real-time tactile + action chunk -> delta actions.

    Queries = [fresh tactile tokens, action tokens]; keys/values = cached tactile/action
    K/V from the joint denoise prefill plus the expert's own tokens. No cross-attention,
    no diffusion timestep — the adaLN condition encodes the execution offset instead.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        patch_dim: int,
        ffn_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        validate_attention_config(num_heads=num_heads, attn_head_dim=attn_head_dim)

        # 触觉 token 由 video expert 的触觉 patchifier 产出(stage 2 冻结),这里只把它的
        # 宽度投到本 expert 的 hidden_dim;两者相等时不加参数。
        self.patch_proj = nn.Identity() if patch_dim == hidden_dim else nn.Linear(patch_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.executed_embedding = nn.Embedding(2, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [TacDiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, action_dim)

        # Same deterministic rope tables as the video (3D) / action (1D) experts, so the
        # fresh tokens live in the same positional space as the cached K/V they attend to.
        self.freqs_3d = precompute_freqs_cis_3d(attn_head_dim)
        self.freqs_1d = precompute_freqs_cis(attn_head_dim, end=1024)

        nn.init.zeros_(self.executed_embedding.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def init_from_action_expert(self, action_expert: nn.Module) -> None:
        """Warm-start from the trained action expert; it already knows how to attend the
        cached video/action K/V inside MoT. New parts (patch proj/executed/head) keep their init."""
        self.action_encoder.load_state_dict(action_expert.action_encoder.state_dict())
        self.time_embedding.load_state_dict(action_expert.time_embedding.state_dict())
        self.time_projection.load_state_dict(action_expert.time_projection.state_dict())
        for tac_block, action_block in zip(self.blocks, action_expert.blocks):
            missing, _unexpected_cross_attn = tac_block.load_state_dict(action_block.state_dict(), strict=False)
            if missing:
                raise ValueError(f"TacDiTBlock init_from_action_expert missing keys: {missing}")
        logger.info("TactileExpert initialized %d blocks from action expert.", len(self.blocks))

    def build_freqs(
        self,
        frame_ids: torch.Tensor,
        tac_grid: Tuple[int, int, int],
        action_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample rope for [tactile tokens, action tokens].

        Fresh tactile tokens reuse the video 3D rope at (frame_ids, h, w_start+w) so each
        taxel patch sits at rope-distance zero from its counterpart in the temporally
        matched cached frame; action tokens reuse the action expert's 1D rope.

        Args:
            frame_ids: [B] matched latent frame index per sample.
            tac_grid: (grid_h, w_start, w_tac) token-grid geometry of the tactile slice.
            action_len: action token count T.

        Returns:
            Complex freqs [B, S_tac + T, 1, attn_head_dim // 2].
        """
        grid_h, w_start, w_tac = tac_grid
        batch_size = frame_ids.shape[0]
        f_band, h_band, w_band = (band.to(device) for band in self.freqs_3d)
        spatial = torch.cat(
            [
                h_band[:grid_h].view(grid_h, 1, -1).expand(grid_h, w_tac, -1),
                w_band[w_start : w_start + w_tac].view(1, w_tac, -1).expand(grid_h, w_tac, -1),
            ],
            dim=-1,
        ).reshape(1, grid_h * w_tac, -1)
        frame = f_band[frame_ids.to(device)].unsqueeze(1).expand(-1, grid_h * w_tac, -1)
        tac_freqs = torch.cat([frame, spatial.expand(batch_size, -1, -1)], dim=-1)
        action_freqs = self.freqs_1d[:action_len].to(device).view(1, action_len, -1).expand(batch_size, -1, -1)
        return torch.cat([tac_freqs, action_freqs], dim=1).unsqueeze(2)

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        actions: torch.Tensor,
        executed_mask: torch.Tensor,
        offset: torch.Tensor,
        frame_ids: torch.Tensor,
        tac_grid: Tuple[int, int, int],
        kv_cache: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """One corrector pass.

        Args:
            tactile_tokens: [B, S_tac, patch_dim] current tactile canvas through the frozen
                tactile patchifier of the video expert.
            actions: [B, T, action_dim] executed prefix + planned suffix (normalized).
            executed_mask: [B, T] bool, True for already-executed steps.
            offset: [B] executed step count k (adaLN condition).
            frame_ids: [B] temporally matched latent frame per sample.
            tac_grid: (grid_h, w_start, w_tac) tactile token-grid geometry.
            kv_cache: per-layer {"k","v"} [B, S_kv, num_heads*attn_head_dim] from the prefill.

        Returns:
            Delta actions [B, T, action_dim].
        """
        if len(kv_cache) != len(self.blocks):
            raise ValueError(f"`kv_cache` must contain {len(self.blocks)} layers, got {len(kv_cache)}.")
        x_tac = self.patch_proj(tactile_tokens)
        x_action = self.action_encoder(actions) + self.executed_embedding(executed_mask.long())
        x = torch.cat([x_tac, x_action], dim=1)

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, offset.float()).to(dtype=x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        num_tac_tokens = x_tac.shape[1]
        grid_h, _, w_tac = tac_grid
        if num_tac_tokens != grid_h * w_tac:
            raise ValueError(f"Tactile token count {num_tac_tokens} does not match tac_grid {tac_grid}.")
        freqs = self.build_freqs(frame_ids, tac_grid, action_len=actions.shape[1], device=x.device)

        for block, layer_cache in zip(self.blocks, kv_cache):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t_mod, freqs, layer_cache["k"], layer_cache["v"], use_reentrant=False
                )
            else:
                x = block(x, t_mod, freqs, layer_cache["k"], layer_cache["v"])
        return self.head(x[:, num_tac_tokens:])
