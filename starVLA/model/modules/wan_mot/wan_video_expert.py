import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, Optional
from einops import rearrange

from starVLA.training.trainer_utils import initialize_overwatch
from .wan_dit_adapters import (
    AttentionModule,
    CrossAttention,
    DiTBlock,
    GateModule,
    Head,
    MLP,
    RMSNorm,
    SelfAttention,
    WanLayerNorm,
    create_group_causal_attn_mask,
    flash_attention,
    modulate,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    rope_apply,
    sinusoidal_embedding_1d,
)
from .expert_utils import run_dit_blocks, validate_attention_config

logger = initialize_overwatch(__name__)


class CrossViewAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float):
        super().__init__()
        attn_dim = num_heads * attn_head_dim
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, attn_dim)
        self.k = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(hidden_dim, attn_dim)
        self.o = nn.Linear(attn_dim, hidden_dim)
        self.norm_q = RMSNorm(attn_dim, eps=eps)
        self.norm_k = RMSNorm(attn_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        return self.o(flash_attention(q=q, k=k, v=v, num_heads=self.num_heads))


def _warm_start_tactile_patch(module: "WanVideoExpert", incompatible_keys) -> None:
    """Fill the tactile patchifier from the visual one whenever a checkpoint predates it.

    Registered as a load-state-dict post hook rather than called from the framework: weights
    arrive through several paths (pretrained Wan DiT, `pretrained_checkpoint`, stage/resume
    checkpoints, inference `from_pretrained`) and only a hook covers all of them. Without it a
    single-patchifier checkpoint would silently keep whatever the tactile conv was initialized
    with instead of reproducing its old tokens.
    """
    if not any(key.endswith("patch_embedding_tactile.weight") for key in incompatible_keys.missing_keys):
        return
    module.init_tactile_patch_from_visual()
    incompatible_keys.missing_keys[:] = [
        key for key in incompatible_keys.missing_keys if "patch_embedding_tactile." not in key
    ]
    logger.info("Checkpoint has no tactile patchifier; warm-started it from the visual patchifier.")


class WanVideoExpert(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        validate_attention_config(num_heads=num_heads, attn_head_dim=attn_head_dim)
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # 触觉列走独立 patchifier;纯视觉数据下 `modality_split` 保持 None,这一路不参与前向。
        self.patch_embedding_tactile = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.modality_split: Optional[Tuple[int, int]] = None
        self.register_load_state_dict_post_hook(_warm_start_tactile_patch)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")
            

    def set_modality_split(self, vision_width: int, canvas_width: int) -> None:
        """Route canvas columns `[0, vision_width)` to the visual patchifier and the rest to the
        tactile one. Widths are the pixel-space canvas geometry; the latent seam scales by the
        same ratio the split VAE encode uses, so both cut at the same place."""
        vision_width, canvas_width = int(vision_width), int(canvas_width)
        if not 0 < vision_width < canvas_width:
            raise ValueError(f"Invalid modality split: vision {vision_width} of canvas {canvas_width}")
        self.modality_split = (vision_width, canvas_width)

    @torch.no_grad()
    def init_tactile_patch_from_visual(self) -> None:
        """Warm-start the tactile patchifier from the visual one, so a checkpoint trained with a
        single patchifier keeps producing exactly its old tokens."""
        self.patch_embedding_tactile.load_state_dict(self.patch_embedding.state_dict())

    def tactile_split_width(self, width: int) -> int:
        """Latent column where the vision segment ends, aligned to the patch grid."""
        vision_width, canvas_width = self.modality_split
        split = width * vision_width // canvas_width
        patch_w = int(self.patch_size[2])
        if split % patch_w or (width - split) % patch_w:
            raise ValueError(
                f"Modality seam at latent column {split} of {width} must align with patch width {patch_w}."
            )
        return split

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        if self.modality_split is None:
            x = self.patch_embedding(x)
        else:
            split = self.tactile_split_width(x.shape[-1])
            x = torch.cat(
                [self.patch_embedding(x[..., :split]), self.patch_embedding_tactile(x[..., split:])],
                dim=-1,
            )
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")
            
            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, 0, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim, 
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w # query length
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        kwargs = dict(
            context=context_emb,
            t_mod=t_mod,
            freqs=freqs,
            context_mask=context_attn_mask,
            self_attn_mask=self_attn_mask,
        )

        x_tokens = run_dit_blocks(
            blocks=self.blocks,
            x=x_tokens,
            kwargs=kwargs,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )

        return self.post_dit(x_tokens, pre_state)


class WanMultiViewVideoExpert(WanVideoExpert):
    def __init__(self, *args, num_view: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_view = int(num_view)
        self.view_embed = nn.Parameter(torch.zeros(self.num_view, self.hidden_dim))
        self.cross_view_norms = nn.ModuleList(
            [WanLayerNorm(self.hidden_dim, eps=kwargs.get("eps", 1e-6)) for _ in range(len(self.blocks))]
        )
        self.cross_view_attns = nn.ModuleList(
            [
                CrossViewAttention(self.hidden_dim, self.attn_head_dim, self.num_heads, kwargs.get("eps", 1e-6))
                for _ in range(len(self.blocks))
            ]
        )
        self.cross_view_time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.init_cross_view_from_self_attn()
        nn.init.zeros_(self.cross_view_time_projection[-1].weight)
        nn.init.zeros_(self.cross_view_time_projection[-1].bias)

    @staticmethod
    def state_dict_has_cross_view_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
        required = (
            "cross_view_attns.0.q.weight",
            "cross_view_attns.0.k.weight",
            "cross_view_attns.0.v.weight",
            "cross_view_attns.0.o.weight",
        )
        return all(any(name in key for key in state_dict) for name in required)

    @torch.no_grad()
    def init_cross_view_from_self_attn(self) -> None:
        for block, cross_view_attn in zip(self.blocks, self.cross_view_attns):
            cross_view_attn.q.load_state_dict(block.self_attn.q.state_dict())
            cross_view_attn.k.load_state_dict(block.self_attn.k.state_dict())
            cross_view_attn.v.load_state_dict(block.self_attn.v.state_dict())
            cross_view_attn.o.load_state_dict(block.self_attn.o.state_dict())
            cross_view_attn.norm_q.load_state_dict(block.self_attn.norm_q.state_dict())
            cross_view_attn.norm_k.load_state_dict(block.self_attn.norm_k.state_dict())

    def pre_dit_multiview(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ) -> Dict[str, Any]:
        batch_size, n_view = x.shape[:2]
        flat_x = x.flatten(0, 1)
        flat_timestep = timestep.repeat_interleave(n_view)
        flat_context = context.repeat_interleave(n_view, dim=0)
        flat_context_mask = None if context_mask is None else context_mask.repeat_interleave(n_view, dim=0)
        pre_state = super().pre_dit(
            x=flat_x,
            timestep=flat_timestep,
            context=flat_context,
            context_mask=flat_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

        tokens_per_view = pre_state["tokens"].shape[1]
        if n_view != self.num_view:
            raise ValueError(f"WanMultiViewVideoExpert expects {self.num_view} views (video keys), got {n_view}.")
        view_embed = self.view_embed.to(device=pre_state["tokens"].device, dtype=pre_state["tokens"].dtype)
        tokens = pre_state["tokens"].reshape(batch_size, n_view, tokens_per_view, self.hidden_dim)
        tokens = tokens + view_embed.view(1, n_view, 1, self.hidden_dim)

        pre_state["tokens"] = tokens.reshape(batch_size, n_view * tokens_per_view, self.hidden_dim)
        pre_state["t"] = pre_state["t"].reshape(batch_size, n_view * tokens_per_view, self.hidden_dim)
        pre_state["t_mod"] = pre_state["t_mod"].reshape(batch_size, n_view * tokens_per_view, 6, self.hidden_dim)
        pre_state["e_cv"] = self.cross_view_time_projection(
            pre_state["t"].reshape(batch_size, n_view * tokens_per_view, self.hidden_dim)
        )
        pre_state["freqs"] = pre_state["freqs"].repeat(n_view, 1, 1)
        pre_state["context"] = pre_state["context"].reshape(batch_size, n_view, *pre_state["context"].shape[1:])[:, 0]
        pre_state["context_mask"] = pre_state["context_mask"].reshape(
            batch_size, n_view * tokens_per_view, pre_state["context_mask"].shape[-1]
        )
        pre_state["meta"].update({
            "batch_size": batch_size,
            "n_view": n_view,
            "tokens_per_view": tokens_per_view,
        })
        return pre_state

    def apply_cross_view(
        self,
        layer_idx: int,
        x: torch.Tensor,
        e_cv: torch.Tensor,
        n_view: int,
        tokens_per_view: int,
    ) -> torch.Tensor:
        if n_view <= 1:
            return x
        cv = self.cross_view_attns[layer_idx](self.cross_view_norms[layer_idx](x))
        return x + cv * e_cv[:, : n_view * tokens_per_view]

    def post_dit_multiview(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        batch_size = int(pre_state["meta"]["batch_size"])
        n_view = int(pre_state["meta"]["n_view"])
        tokens_per_view = int(pre_state["meta"]["tokens_per_view"])
        f, h, w = pre_state["meta"]["grid_size"]
        flat_tokens = x_tokens.reshape(batch_size * n_view, tokens_per_view, self.hidden_dim)
        flat_t = pre_state["t"].reshape(batch_size * n_view, tokens_per_view, self.hidden_dim)
        x = self.head(flat_tokens, flat_t)
        x = self.unpatchify(x, (f, h, w))
        return x.reshape(batch_size, n_view, *x.shape[1:])
