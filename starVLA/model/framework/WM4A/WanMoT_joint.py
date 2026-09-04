from typing import Any, Dict, List, Optional, Tuple

import torch

from starVLA.model.modules.wan_mot.time_alignment import (
    TemporalAlignmentEmbedding,
    build_action_tau_ids,
    build_video_tau_ids,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY

from .WanMoT import WanMoT, _to_plain_dict


@FRAMEWORK_REGISTRY.register("WanMoTJoint")
class WanMoTJoint(WanMoT):
    """WanMoT variant that jointly denoises video and action latents."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if bool(getattr(self.video_expert, "action_conditioned", False)):
            raise ValueError("WanMoTJoint requires `video_expert.action_conditioned=false`.")
        self.time_alignment_cfg = _to_plain_dict(getattr(self.framework_config, "time_alignment", {}))
        self.mot.time_alignment = self._build_time_alignment()

    def _build_time_alignment(self) -> Optional[TemporalAlignmentEmbedding]:
        cfg = self.time_alignment_cfg
        if not bool(cfg.get("enabled", False)):
            return None
        attn_dim = int(self.video_expert.num_heads) * int(self.video_expert.attn_head_dim)
        max_tau = int(getattr(self.config.datasets.vla_data, "num_future_frames", 8)) // int(
            getattr(self.vae, "temporal_downsample_factor", 4)
        )
        return TemporalAlignmentEmbedding(
            max_tau=max_tau,
            attn_dim=attn_dim,
        ).to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    def _build_mot_tau_ids(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
        *,
        n_view: int = 1,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if getattr(self.mot, "time_alignment", None) is None:
            return None
        latent_frames = int(video_pre["meta"]["grid_size"][0])
        device = video_pre["tokens"].device
        return {
            "video": build_video_tau_ids(
                latent_frames=latent_frames,
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                n_view=n_view,
                device=device,
            ),
            "action": build_action_tau_ids(
                action_horizon=int(action_pre["tokens"].shape[1]),
                latent_frames=latent_frames,
                future_frame_stride=int(getattr(self.config.datasets.vla_data, "future_frame_stride", 1)),
                vae_temporal_stride=int(getattr(self.vae, "temporal_downsample_factor", 4)),
                device=device,
            ),
        }

    def _run_mot(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        return self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            tau_ids_all=self._build_mot_tau_ids(video_pre, action_pre),
        )

    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._autocast_context(dtype=latents_video.dtype):
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            tokens_out = self._run_mot(video_pre, action_pre)
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _decode_joint_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self._decode_video_latents(latents)

    def _prepare_joint_conditioning(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        state: Optional[Any],
        context: Optional[Any],
        context_mask: Optional[Any],
        image_is_tactile: Optional[Any],
        caller: str,
        *,
        require_proprio: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.device, torch.dtype]:
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images(batch_images)[:, :, :, :1, :, :]
        images = self._resize_views_if_needed(images)
        batch_size, n_view, _, _, _, _ = images.shape
        view_mask_tensor = self._prepare_view_mask(view_mask, batch_size, n_view, images.device)
        image_is_tactile_tensor = self._prepare_image_is_tactile(image_is_tactile, batch_size, n_view, images.device)
        images = self._normalize_views_for_vae(images, image_is_tactile_tensor)
        video = self._concat_views(images, view_mask_tensor)
        video = self._resize_video_for_vae(video).to(device=device, dtype=dtype, non_blocking=True)

        context, context_mask = self._prepare_runtime_text_context_with_cache(
            context,
            context_mask,
            instructions,
            batch_size,
            device,
            dtype,
            caller,
        )

        state_tensor = self._batch_to_tensor(state, device=device, dtype=dtype)
        if require_proprio or state_tensor is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, state_tensor)
        return video, context, context_mask, device, dtype

    def _resolve_joint_seed(self, seed: Optional[Any]) -> Optional[int]:
        if seed is not None:
            return int(seed)
        predict_cfg = getattr(self.action_config, "predict", None)
        if predict_cfg is None:
            return None
        if hasattr(predict_cfg, "seed"):
            return int(predict_cfg.seed)
        try:
            if "seed" in predict_cfg:
                return int(predict_cfg["seed"])
        except TypeError:
            pass
        return None

    def _build_joint_inference_schedule(
        self,
        *,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_steps = int(self.infer_video_scheduler.num_train_timesteps)
        action_steps = int(self.infer_action_scheduler.num_train_timesteps)
        if video_steps != action_steps:
            raise ValueError(
                "WanMoTJoint inference requires video/action schedulers to share "
                f"`num_train_timesteps`, got {video_steps} and {action_steps}."
            )
        return self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=shift_override,
        )

    @torch.inference_mode()
    def _infer_joint_latents(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
    ) -> dict:
        if num_video_frames <= 1 or num_video_frames % 4 != 1:
            raise ValueError(f"WanMoTJoint video inference requires T > 1 and T % 4 == 1, got T={num_video_frames}.")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")

        batch_size = video.shape[0]
        device = video.device
        dtype = video.dtype

        first_frame_latents = self._encode_first_frame_latents(video)
        temporal_factor = int(getattr(self.vae, "temporal_downsample_factor", 4))
        upsampling_factor = int(getattr(self.vae, "upsampling_factor", 16))
        latent_t = (num_video_frames - 1) // temporal_factor + 1
        latent_h = video.shape[-2] // upsampling_factor
        latent_w = video.shape[-1] // upsampling_factor
        z_dim = int(getattr(self.vae, "z_dim", getattr(self.video_config, "z_dim", 48)))

        seed = self._resolve_joint_seed(seed)
        video_generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, z_dim, latent_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=video_generator,
        )
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=action_generator,
        )
        latents_video[:, :, :1] = first_frame_latents

        infer_timesteps, infer_deltas = self._build_joint_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.expand(batch_size).to(device=device, dtype=dtype)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)
            latents_video[:, :, :1] = first_frame_latents

        return {"video_latents": latents_video, "action_latents": latents_action}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        fps: Optional[List[float]] = None,
        state: Optional[Any] = None,
        actions: Optional[Any] = None,
        data_id: Optional[Any] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        del fps, data_id
        if actions is not None:
            raise ValueError("WanMoTJoint.predict_action does not support action prefixes.")
        self.eval()

        video, context, context_mask, _, _ = self._prepare_joint_conditioning(
            batch_images=batch_images,
            view_mask=view_mask,
            instructions=instructions,
            state=state,
            context=context,
            context_mask=context_mask,
            image_is_tactile=kwargs.get("image_is_tactile", None),
            caller="predict_action",
            require_proprio=True,
        )
        default_num_video_frames = int(getattr(self.config.datasets.vla_data, "num_future_frames", 8)) + 1
        joint_out = self._infer_joint_latents(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(kwargs.get("num_video_frames", default_num_video_frames)),
            action_horizon=int(kwargs.get("action_horizon", self.action_horizon)),
            num_inference_steps=int(kwargs.get("num_inference_steps", getattr(self.action_config, "num_inference_steps", 10))),
            sigma_shift=kwargs.get("sigma_shift", None),
            seed=kwargs.get("seed", None),
        )
        actions_out = joint_out["action_latents"].detach().to(device="cpu", dtype=torch.float32).numpy()
        output = {"normalized_actions": actions_out}
        if bool(kwargs.get("return_video", False)):
            video_out = self._decode_joint_video_latents(joint_out["video_latents"])
            output["video"] = (video_out * 0.5 + 0.5).clamp_(0, 1).detach().to(device="cpu", dtype=torch.float32)
        return output

    @torch.inference_mode()
    def predict_video(
        self,
        batch_images: Any,
        view_mask: Optional[Any],
        instructions: List[str],
        fps: Optional[List[float]] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        del fps
        self.eval()

        state = kwargs.get("state", kwargs.get("proprio", None))
        video, context, context_mask, _, _ = self._prepare_joint_conditioning(
            batch_images=batch_images,
            view_mask=view_mask,
            instructions=instructions,
            state=state,
            context=context,
            context_mask=context_mask,
            image_is_tactile=kwargs.get("image_is_tactile", None),
            caller="predict_video",
            require_proprio=False,
        )
        default_num_video_frames = int(getattr(self.config.datasets.vla_data, "num_future_frames", 8)) + 1
        joint_out = self._infer_joint_latents(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(kwargs.get("num_video_frames", default_num_video_frames)),
            action_horizon=int(kwargs.get("action_horizon", self.action_horizon)),
            num_inference_steps=int(
                kwargs.get(
                    "num_inference_steps",
                    getattr(self.video_config, "num_inference_steps", getattr(self.action_config, "num_inference_steps", 10)),
                )
            ),
            sigma_shift=kwargs.get("sigma_shift", None),
            seed=kwargs.get("seed", None),
        )
        video_out = self._decode_video_latents(joint_out["video_latents"])
        video_out = (video_out * 0.5 + 0.5).clamp_(0, 1)
        return {"video": video_out}
