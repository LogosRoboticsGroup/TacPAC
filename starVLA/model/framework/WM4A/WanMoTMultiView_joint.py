from typing import Any, Dict, List, Optional, Tuple

import torch

from starVLA.model.modules.wan_mot.wan_video_expert import WanMultiViewVideoExpert
from starVLA.model.tools import FRAMEWORK_REGISTRY

from .WanMoT_joint import WanMoTJoint


@FRAMEWORK_REGISTRY.register("WanMoTMultiViewJoint")
class WanMoTMultiViewJoint(WanMoTJoint):
    """WanMoT joint variant that keeps camera views separate."""

    def _build_video_expert(self, video_model_config: dict) -> WanMultiViewVideoExpert:
        from starVLA.dataloader.vla.mixtures import num_video_keys

        num_view = num_video_keys(self.config.datasets.vla_data.data_mix)
        return WanMultiViewVideoExpert(**dict(video_model_config), num_view=num_view)

    def _resolve_image_size(self) -> Tuple[int, int]:
        # Views stay separate; each goes through the VAE at the per-view size.
        return tuple(map(int, self.view_image_size))

    def _normalize_and_resize_multiview_video(
        self,
        video: torch.Tensor,
        image_is_tactile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        video = self._normalize_views_for_vae(video, image_is_tactile)
        batch_size, n_view = video.shape[:2]
        flat = video.reshape(batch_size * n_view, *video.shape[2:])
        flat = self._resize_video_for_vae(flat)
        return flat.reshape(batch_size, n_view, *flat.shape[1:])

    @torch.no_grad()
    def _encode_multiview_latents(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, n_view = video.shape[:2]
        flat = video.reshape(batch_size * n_view, *video.shape[2:])
        # Views are already separate here — no modality split of the per-view frame.
        latents = self._run_vae_encode(flat)
        return latents.reshape(batch_size, n_view, *latents.shape[1:])

    @torch.no_grad()
    def _encode_first_frame_latents_multiview(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 6 or video.shape[3] != 1:
            raise ValueError(f"`video` must be [B,V,3,1,H,W], got {tuple(video.shape)}")
        return self._encode_multiview_latents(video)

    @torch.no_grad()
    def _decode_multiview_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, n_view = latents.shape[:2]
        flat = latents.reshape(batch_size * n_view, *latents.shape[2:])
        video = self._run_vae_decode(flat)
        return video.reshape(batch_size, n_view, *video.shape[1:])

    @torch.no_grad()
    def _decode_joint_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self._decode_multiview_latents(latents)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        *,
        n_view: int = 1,
        tokens_per_view: Optional[int] = None,
    ) -> torch.Tensor:
        tokens_per_view = video_seq_len if tokens_per_view is None else int(tokens_per_view)
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        view_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=tokens_per_view,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        for view_idx in range(int(n_view)):
            start = view_idx * tokens_per_view
            end = start + tokens_per_view
            mask[start:end, start:end] = view_mask
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    def _run_mot(
        self,
        video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        video_meta = video_pre["meta"]
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_meta["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            n_view=int(video_meta["n_view"]),
            tokens_per_view=int(video_meta["tokens_per_view"]),
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
            cross_view_all={
                "video": {
                    "e_cv": video_pre["e_cv"],
                    "n_view": int(video_meta["n_view"]),
                    "tokens_per_view": int(video_meta["tokens_per_view"]),
                }
            },
            tau_ids_all=self._build_mot_tau_ids(
                video_pre,
                action_pre,
                n_view=int(video_meta["n_view"]),
            ),
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
            video_pre = self.video_expert.pre_dit_multiview(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            tokens_out = self._run_mot(video_pre, action_pre)
            pred_video = self.video_expert.post_dit_multiview(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

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
        del view_mask
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images(batch_images)[:, :, :, :1, :, :]
        images = self._resize_views_if_needed(images)
        batch_size = images.shape[0]
        n_view = images.shape[1]
        image_is_tactile_tensor = self._prepare_image_is_tactile(image_is_tactile, batch_size, n_view, images.device)
        video = self._normalize_and_resize_multiview_video(
            images,
            image_is_tactile=image_is_tactile_tensor,
        ).to(device=device, dtype=dtype, non_blocking=True)

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

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        del kwargs
        if examples is None:
            raise ValueError("WanMoTMultiViewJoint.forward requires `examples`.")
        if not bool(getattr(self.config.datasets.vla_data, "use_future_frames", False)):
            raise ValueError("WanMoTMultiViewJoint training requires datasets.vla_data.use_future_frames=true.")
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images([example["image"] for example in examples])
        images = self._resize_views_if_needed(images)
        batch_size, n_view, _, num_frames, _, _ = images.shape
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(f"WanMoTMultiViewJoint video frames must satisfy T > 1 and T % 4 == 1, got T={num_frames}.")
        image_is_tactile = self._prepare_examples_image_is_tactile(examples, batch_size, n_view, images.device)
        video = self._normalize_and_resize_multiview_video(
            images,
            image_is_tactile=image_is_tactile,
        ).to(device=device, dtype=dtype, non_blocking=True)

        context, context_mask = self._prepare_text_context(examples, device, dtype)
        state = None
        if "state" in examples[0]:
            state = torch.stack([self._batch_to_tensor(example["state"]) for example in examples])
            state = state.to(device=device, dtype=dtype, non_blocking=True)
        context, context_mask = self._append_proprio_to_context(context, context_mask, state)

        actions = torch.stack([self._batch_to_tensor(example["actions"]) for example in examples])
        actions = actions[:, -self.action_horizon :, :].to(device=device, dtype=dtype, non_blocking=True)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.shape[-1]}.")
        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = torch.stack([self._batch_to_tensor(example["action_mask"]) for example in examples])
            action_mask = action_mask[:, -self.action_horizon :, :].to(device=device, dtype=torch.bool)

        image_is_pad = None
        if "image_is_pad" in examples[0]:
            image_is_pad = torch.stack([self._batch_to_tensor(example["image_is_pad"]) for example in examples])
            image_is_pad = image_is_pad[:, :num_frames].to(device=device, dtype=torch.bool)

        with torch.no_grad():
            input_latents = self._encode_multiview_latents(video)
        input_latents = input_latents.to(device=device, dtype=dtype)
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        first_frame_latents = input_latents[:, :, :, :1]
        noisy_video[:, :, :, :1] = first_frame_latents

        noise_action = torch.randn_like(actions)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_action = self.train_action_scheduler.add_noise(actions, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(actions, noise_action, timestep_action)

        pred_video, pred_action_velocity = self._predict_joint_noise(
            latents_video=noisy_video,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
        )

        pred_video = pred_video[:, :, :, 1:]
        target_video = target_video[:, :, :, 1:]
        flat_pred_video = pred_video.flatten(0, 1)
        flat_target_video = target_video.flatten(0, 1)
        flat_image_is_pad = None if image_is_pad is None else image_is_pad.repeat_interleave(n_view, dim=0)
        loss_video_per_view = self._compute_video_loss_per_sample(
            pred_video=flat_pred_video,
            target_video=flat_target_video,
            image_is_pad=flat_image_is_pad,
            include_initial_video_step=False,
        ).reshape(batch_size, n_view)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=loss_video_per_view.device,
            dtype=loss_video_per_view.dtype,
        )
        loss_video = (loss_video_per_view.mean(dim=1) * video_weight).mean()

        if self.action_supervision_target == "action_space":
            pred_action_clean = self.train_action_scheduler.recover_sample_from_velocity(
                noisy_samples=noisy_action,
                pred_velocity=pred_action_velocity,
                timestep=timestep_action,
            )
            loss_action_per_sample = self._compute_action_loss_per_sample(pred_action_clean, actions, action_mask)
        else:
            loss_action_per_sample = self._compute_action_loss_per_sample(
                pred_action_velocity,
                target_action,
                action_mask,
            )
        if self.action_loss_weight_mode == "none":
            action_weight = torch.ones_like(loss_action_per_sample)
        else:
            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                device=loss_action_per_sample.device,
                dtype=loss_action_per_sample.dtype,
            )
        loss_action = (loss_action_per_sample * action_weight).mean()
        total_loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return {
            "total_loss": total_loss,
            "action_loss": float(loss_action.detach().item()),
            "video_loss": float(loss_video.detach().item()),
        }

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
            raise ValueError(f"WanMoTMultiViewJoint video inference requires T > 1 and T % 4 == 1, got T={num_video_frames}.")
        batch_size, n_view = video.shape[:2]
        device = video.device
        dtype = video.dtype

        first_frame_latents = self._encode_first_frame_latents_multiview(video)
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
            (batch_size, n_view, z_dim, latent_t, latent_h, latent_w),
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
        latents_video[:, :, :, :1] = first_frame_latents

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
            latents_video[:, :, :, :1] = first_frame_latents

        return {"video_latents": latents_video, "action_latents": latents_action}

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
        video_out = self._decode_multiview_latents(joint_out["video_latents"])
        video_out = (video_out * 0.5 + 0.5).clamp_(0, 1)
        return {"video": video_out}
