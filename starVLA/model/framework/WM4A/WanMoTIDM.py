from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from starVLA.model.modules.wan_mot.time_alignment import (
    build_action_tau_ids,
    build_video_tau_ids,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY

from .WanMoT import _to_plain_dict
from .WanMoT_joint import WanMoTJoint


@FRAMEWORK_REGISTRY.register("WanMoTIDM")
class WanMoTIDM(WanMoTJoint):
    """IDM variant with teacher-forcing video conditioning for action denoising.

    This mirrors FastWAMIDM:
      - training uses [noisy_video, cond_video, noisy_action] in one MoT pass;
      - inference first denoises video, then denoises action against cached video K/V.
    """

    video_cond_noise_prob = 0.5

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        idm_cfg = _to_plain_dict(getattr(self.framework_config, "idm", {}))
        self.video_cond_noise_prob = float(
            idm_cfg.get("video_cond_noise_prob", type(self).video_cond_noise_prob)
        )

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        *,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        noisy_end = noisy_video_seq_len
        cond_end = noisy_end + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:noisy_end, :noisy_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_video_seq_len,
            video_tokens_per_frame=noisy_video_tokens_per_frame,
            device=device,
        )
        mask[noisy_end:cond_end, noisy_end:cond_end] = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_video_seq_len,
            video_tokens_per_frame=cond_video_tokens_per_frame,
            device=device,
        )
        mask[cond_end:, cond_end:] = True
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def _build_video_tau_ids_from_pre(self, video_pre: Dict[str, Any]) -> Optional[torch.Tensor]:
        if getattr(self.mot, "time_alignment", None) is None:
            return None
        return build_video_tau_ids(
            latent_frames=int(video_pre["meta"]["grid_size"][0]),
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            n_view=int(video_pre["meta"].get("n_view", 1)),
            device=video_pre["tokens"].device,
        )

    def _build_action_tau_ids_from_pre(
        self,
        video_pre: Dict[str, Any],
        action_seq_len: int,
    ) -> Optional[torch.Tensor]:
        if getattr(self.mot, "time_alignment", None) is None:
            return None
        return build_action_tau_ids(
            action_horizon=int(action_seq_len),
            latent_frames=int(video_pre["meta"]["grid_size"][0]),
            future_frame_stride=int(getattr(self.config.datasets.vla_data, "future_frame_stride", 1)),
            vae_temporal_stride=int(getattr(self.vae, "temporal_downsample_factor", 4)),
            device=video_pre["tokens"].device,
        )

    def _build_teacher_forcing_tau_ids(
        self,
        *,
        noisy_video_pre: Dict[str, Any],
        cond_video_pre: Dict[str, Any],
        action_pre: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if getattr(self.mot, "time_alignment", None) is None:
            return None
        noisy_video_tau = self._build_video_tau_ids_from_pre(noisy_video_pre)
        cond_video_tau = self._build_video_tau_ids_from_pre(cond_video_pre)
        action_tau = self._build_action_tau_ids_from_pre(cond_video_pre, action_pre["tokens"].shape[1])
        return {
            "video": torch.cat([noisy_video_tau, cond_video_tau], dim=0),
            "action": action_tau,
        }

    def _prepare_training_batch(self, examples: List[dict]) -> Dict[str, Any]:
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images([example["image"] for example in examples])
        images = self._resize_views_if_needed(images)
        batch_size, n_view, _, num_frames, _, _ = images.shape
        if num_frames <= 1 or num_frames % 4 != 1:
            raise ValueError(f"WanMoTIDM video frames must satisfy T > 1 and T % 4 == 1, got T={num_frames}.")

        image_is_tactile = self._prepare_examples_image_is_tactile(examples, batch_size, n_view, images.device)
        view_mask = self._prepare_view_mask(
            [example["view_mask"] for example in examples] if "view_mask" in examples[0] else None,
            batch_size,
            n_view,
            images.device,
        )
        images = self._normalize_views_for_vae(images, image_is_tactile)
        video = self._concat_views(images, view_mask)
        video = self._resize_video_for_vae(video).to(device=device, dtype=dtype, non_blocking=True)

        context, context_mask = self._prepare_text_context(examples, device, dtype)
        state = None
        if "state" in examples[0]:
            state = torch.stack([self._batch_to_tensor(example["state"]) for example in examples])
            state = state.to(device=device, dtype=dtype, non_blocking=True)
        context, context_mask = self._append_proprio_to_context(context, context_mask, state)

        actions = torch.stack([self._batch_to_tensor(example["actions"]) for example in examples])
        actions = actions[:, -self.action_horizon :, :].to(device=device, dtype=dtype, non_blocking=True)

        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = torch.stack([self._batch_to_tensor(example["action_mask"]) for example in examples])
            action_mask = action_mask[:, -self.action_horizon :, :].to(device=device, dtype=torch.bool)

        image_is_pad = None
        if "image_is_pad" in examples[0]:
            image_is_pad = torch.stack([self._batch_to_tensor(example["image_is_pad"]) for example in examples])
            image_is_pad = image_is_pad[:, :num_frames].to(device=device, dtype=torch.bool)

        return {
            "video": video,
            "context": context,
            "context_mask": context_mask,
            "actions": actions,
            "action_mask": action_mask,
            "image_is_pad": image_is_pad,
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
        }

    def _compute_action_loss(
        self,
        *,
        pred_action_velocity: torch.Tensor,
        noisy_action: torch.Tensor,
        target_action: torch.Tensor,
        actions: torch.Tensor,
        timestep_action: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.action_supervision_target == "action_space":
            pred_action = self.train_action_scheduler.recover_sample_from_velocity(
                noisy_samples=noisy_action,
                pred_velocity=pred_action_velocity,
                timestep=timestep_action,
            )
            loss_per_sample = self._compute_action_loss_per_sample(pred_action, actions, action_mask)
        else:
            loss_per_sample = self._compute_action_loss_per_sample(
                pred_action_velocity,
                target_action,
                action_mask,
            )

        if self.action_loss_weight_mode == "none":
            return loss_per_sample.mean()
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=loss_per_sample.device,
            dtype=loss_per_sample.dtype,
        )
        return (loss_per_sample * action_weight).mean()

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        del kwargs
        batch = self._prepare_training_batch(examples)
        video = batch["video"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        actions = batch["actions"]
        action_mask = batch["action_mask"]
        image_is_pad = batch["image_is_pad"]
        batch_size = batch["batch_size"]
        device = batch["device"]
        dtype = batch["dtype"]

        with torch.no_grad():
            input_latents = self._encode_video_latents(video)
        input_latents = input_latents.to(device=device, dtype=dtype)
        first_frame_latents = input_latents[:, :, :1]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_video = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        noisy_video[:, :, :1] = first_frame_latents

        noise_action = torch.randn_like(actions)
        timestep_action = self.train_action_scheduler.sample_training_t(batch_size, device=device, dtype=dtype)
        noisy_action = self.train_action_scheduler.add_noise(actions, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(actions, noise_action, timestep_action)

        cond_noise_mask = torch.rand((batch_size,), device=device) < self.video_cond_noise_prob
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=dtype, device=device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any().item()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents,
                noise_video_cond,
                timestep_video_cond_sampled,
            )
            cond_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_selector, latents_cond_noisy, input_latents)
        latents_cond = latents_cond.clone()
        latents_cond[:, :, :1] = first_frame_latents

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        with self._autocast_context(dtype=dtype):
            video_pre_noisy = self.video_expert.pre_dit(
                x=noisy_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_pre_cond = self.video_expert.pre_dit(
                x=latents_cond,
                timestep=timestep_video_cond,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )

            noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
            cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
            noisy_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])
            cond_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])

            merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
            merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
            merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
            merged_video_context_mask = torch.cat(
                [video_pre_noisy["context_mask"], video_pre_cond["context_mask"]],
                dim=1,
            )

            attention_mask = self._build_teacher_forcing_attention_mask(
                noisy_video_seq_len=noisy_video_seq_len,
                cond_video_seq_len=cond_video_seq_len,
                action_seq_len=action_pre["tokens"].shape[1],
                noisy_video_tokens_per_frame=noisy_tokens_per_frame,
                cond_video_tokens_per_frame=cond_tokens_per_frame,
                device=merged_video_tokens.device,
            )
            tokens_out = self.mot(
                embeds_all={
                    "video": merged_video_tokens,
                    "action": action_pre["tokens"],
                },
                attention_mask=attention_mask,
                freqs_all={
                    "video": merged_video_freqs,
                    "action": action_pre["freqs"],
                },
                context_all={
                    "video": {
                        "context": video_pre_noisy["context"],
                        "mask": merged_video_context_mask,
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                },
                t_mod_all={
                    "video": merged_video_t_mod,
                    "action": action_pre["t_mod"],
                },
                tau_ids_all=self._build_teacher_forcing_tau_ids(
                    noisy_video_pre=video_pre_noisy,
                    cond_video_pre=video_pre_cond,
                    action_pre=action_pre,
                ),
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"][:, :noisy_video_seq_len], video_pre_noisy)
            pred_action_velocity = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_video = pred_video[:, :, 1:]
        target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            device=loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_action = self._compute_action_loss(
            pred_action_velocity=pred_action_velocity,
            noisy_action=noisy_action,
            target_action=target_action,
            actions=actions,
            timestep_action=timestep_action,
            action_mask=action_mask,
        )
        total_loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return {
            "total_loss": total_loss,
            "action_loss": float(loss_action.detach().item()),
            "video_loss": float(loss_video.detach().item()),
        }

    @torch.inference_mode()
    def _infer_video_latents_idm(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        num_video_frames: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
    ) -> torch.Tensor:
        if num_video_frames <= 1 or num_video_frames % 4 != 1:
            raise ValueError(f"WanMoTIDM video inference requires T > 1 and T % 4 == 1, got T={num_video_frames}.")

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
        generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, z_dim, latent_t, latent_h, latent_w),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        latents_video[:, :, :1] = first_frame_latents

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.expand(batch_size).to(device=device, dtype=dtype)
            with self._autocast_context(dtype=dtype):
                pred_video = self.video_expert(
                    x=latents_video,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)
            latents_video[:, :, :1] = first_frame_latents
        return latents_video

    @torch.inference_mode()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: List[dict],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_tau_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with self._autocast_context(dtype=latents_action.dtype):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_tau_ids=action_tau_ids,
            )
            return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.inference_mode()
    def _infer_idm_latents(
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
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")

        batch_size = video.shape[0]
        device = video.device
        dtype = video.dtype
        seed = self._resolve_joint_seed(seed)

        latents_video = self._infer_video_latents_idm(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=num_video_frames,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
        )

        action_generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=action_generator,
        )

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video_cond = torch.zeros((batch_size,), device=device, dtype=dtype)
        with self._autocast_context(dtype=dtype):
            video_pre_cond = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video_cond,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre_cond["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_horizon,
                video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
                device=video_pre_cond["tokens"].device,
            )
            video_tau_ids = self._build_video_tau_ids_from_pre(video_pre_cond)
            action_tau_ids = self._build_action_tau_ids_from_pre(video_pre_cond, action_horizon)
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre_cond["tokens"],
                video_freqs=video_pre_cond["freqs"],
                video_t_mod=video_pre_cond["t_mod"],
                video_context_payload={"context": video_pre_cond["context"], "mask": video_pre_cond["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                video_tau_ids=video_tau_ids,
            )

        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t_action.expand(batch_size).to(device=device, dtype=dtype)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_tau_ids=action_tau_ids,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

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
            raise ValueError("WanMoTIDM.predict_action does not support action prefixes.")
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
        idm_out = self._infer_idm_latents(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(kwargs.get("num_video_frames", default_num_video_frames)),
            action_horizon=int(kwargs.get("action_horizon", self.action_horizon)),
            num_inference_steps=int(kwargs.get("num_inference_steps", getattr(self.action_config, "num_inference_steps", 10))),
            sigma_shift=kwargs.get("sigma_shift", None),
            seed=kwargs.get("seed", None),
        )
        actions_out = idm_out["action_latents"].detach().to(device="cpu", dtype=torch.float32).numpy()
        output = {"normalized_actions": actions_out}
        if bool(kwargs.get("return_video", False)):
            video_out = self._decode_joint_video_latents(idm_out["video_latents"])
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
        video_latents = self._infer_video_latents_idm(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(kwargs.get("num_video_frames", default_num_video_frames)),
            num_inference_steps=int(
                kwargs.get(
                    "num_inference_steps",
                    getattr(self.video_config, "num_inference_steps", getattr(self.action_config, "num_inference_steps", 10)),
                )
            ),
            sigma_shift=kwargs.get("sigma_shift", None),
            seed=kwargs.get("seed", None),
        )
        video_out = self._decode_video_latents(video_latents)
        video_out = (video_out * 0.5 + 0.5).clamp_(0, 1)
        return {"video": video_out}
