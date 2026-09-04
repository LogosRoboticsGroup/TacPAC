from typing import Any, Dict, List, Optional, Tuple

import torch

from starVLA.model.modules.wan_mot.tactile_expert import TactileExpert
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

from .WanMoT_joint import WanMoTJoint
from .WanMoT import _to_plain_dict

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("WanMoTJointTacExpert")
class WanMoTJointTacExpert(WanMoTJoint):
    """WanMoTJoint + asynchronous tactile expert.

    Phase-2 training (frozen base): run the base 10-step joint denoise to get the plan and
    a t=0 prefill KV cache (tactile video tokens + action tokens), then supervise the
    tactile expert with delta = GT - plan on the not-yet-executed suffix.

    Inference runs in three steps, in this order: `predict_action` denoises and returns the
    chunk right away, `prefill_tactile_cache` then pays the extra joint forward once the chunk
    is out, and each `correct_action` is one tactile tick patching the not-yet-executed suffix.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        if self.n_tactile_views == 0:
            raise ValueError("WanMoTJointTacExpert requires tactile views in the data mix.")
        self.tactile_config = config.framework.tactile_model
        tac_model_config = _to_plain_dict(getattr(self.tactile_config, "config"))
        tac_model_config.setdefault("patch_dim", int(self.video_expert.hidden_dim))
        self.tactile_expert = TactileExpert(**tac_model_config).to(dtype=self.torch_dtype)
        if int(self.tactile_expert.num_heads) != int(self.video_expert.num_heads):
            raise ValueError("TactileExpert `num_heads` must match video expert to consume cached K/V.")
        if int(self.tactile_expert.attn_head_dim) != int(self.video_expert.attn_head_dim):
            raise ValueError("TactileExpert `attn_head_dim` must match video expert to consume cached K/V.")
        if len(self.tactile_expert.blocks) != len(self.video_expert.blocks):
            raise ValueError("TactileExpert `num_layers` must match the base experts for 1:1 K/V consumption.")

        self._tac_init_from_action_expert = bool(getattr(self.tactile_config, "init_from_action_expert", True))
        if self._tac_init_from_action_expert:
            self.tactile_expert.init_from_action_expert(self.action_expert)
        self.steps_per_latent_frame = int(getattr(self.vae, "temporal_downsample_factor", 4)) * int(
            getattr(self.config.datasets.vla_data, "future_frame_stride", 1)
        )
        self._tac_cache: Optional[dict] = None
        self._pending_prefill: Optional[dict] = None

    def reset(self):
        super().reset()
        self._tac_cache = None
        self._pending_prefill = None

    def compile(self):
        super().compile()
        self.tactile_expert = torch.compile(self.tactile_expert)

    def _before_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        converted = super()._before_load_state_dict(state_dict)
        if "mot" in state_dict and "tactile_expert" in state_dict:
            converted.update({f"tactile_expert.{key}": value for key, value in state_dict["tactile_expert"].items()})
        return converted

    def _after_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super()._after_load_state_dict(state_dict)
        if self._tac_init_from_action_expert and not any(key.startswith("tactile_expert.") for key in state_dict):
            logger.info("Checkpoint has no tactile_expert weights; warm-starting from the loaded action expert.")
            self.tactile_expert.init_from_action_expert(self.action_expert)

    def _tac_token_geometry(self, grid_size: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """(latent_frames, grid_h, w_start, w_tac) of the tactile slice in the token grid.
        Split by actual canvas widths (same formula as the VAE split), so it stays correct
        when `tactile_image_size` shrinks the tactile columns."""
        latent_frames, grid_h, grid_w = (int(v) for v in grid_size)
        w_start = self._modality_split_width(grid_w)
        return latent_frames, grid_h, w_start, grid_w - w_start

    def _tactile_now_canvas(self, tactile_now: torch.Tensor) -> torch.Tensor:
        """`tactile_now` [B,M,V_tac,C,H,W] -> VAE input [B*M,C,1,H,W_tac], laid out exactly like
        the tactile region of the main canvas so the fresh tokens match the cached slice."""
        views = tactile_now.flatten(0, 1).unsqueeze(3)  # [B*M, V_tac, C, 1, H, W]
        return self._tactile_canvas(views).clamp(-1.0, 1.0)

    @staticmethod
    def _tac_token_index(
        latent_frames: int,
        grid_h: int,
        grid_w: int,
        w_start: int,
        w_tac: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Flat indices of tactile tokens inside the [f*h*w] video token sequence."""
        per_frame = torch.arange(grid_h * grid_w, device=device).reshape(grid_h, grid_w)[:, w_start : w_start + w_tac]
        frame_base = torch.arange(latent_frames, device=device).view(-1, 1) * (grid_h * grid_w)
        return (frame_base + per_frame.reshape(1, -1)).reshape(-1)

    @torch.no_grad()
    def _prefill_tac_cache(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict:
        """One clean t=0 joint pass over [denoised video + plan]; cache per-layer K/V of the
        tactile token slice and the action tokens. Shared verbatim by training and inference."""
        batch_size = latents_video.shape[0]
        device = latents_video.device
        dtype = latents_video.dtype
        timestep = torch.zeros((batch_size,), device=device, dtype=dtype)
        with self._autocast_context(dtype=dtype):
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=int(latents_action.shape[1]),
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            tau_ids = self._build_mot_tau_ids(video_pre, action_pre)
            video_kv = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                video_tau_ids=None if tau_ids is None else tau_ids.get("video"),
            )
            action_kv: List[Dict[str, torch.Tensor]] = []
            self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=video_kv,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_tau_ids=None if tau_ids is None else tau_ids.get("action"),
                kv_out=action_kv,
            )
        latent_frames, grid_h, grid_w = (int(v) for v in video_pre["meta"]["grid_size"])
        latent_frames, grid_h, w_start, w_tac = self._tac_token_geometry((latent_frames, grid_h, grid_w))
        tac_idx = self._tac_token_index(latent_frames, grid_h, grid_w, w_start, w_tac, device)
        kv_cache = [
            {
                "k": torch.cat([layer_video["k"][:, tac_idx], layer_action["k"]], dim=1),
                "v": torch.cat([layer_video["v"][:, tac_idx], layer_action["v"]], dim=1),
            }
            for layer_video, layer_action in zip(video_kv, action_kv)
        ]
        return {
            "kv": kv_cache,
            "latent_frames": latent_frames,
            "tac_grid": (grid_h, w_start, w_tac),
        }

    @torch.no_grad()
    def _patchify_tactile(self, tactile_latents: torch.Tensor) -> torch.Tensor:
        """Fresh tactile latent -> tokens [B, S_tac, video hidden], through the video expert's
        tactile patchifier. It stays frozen in phase 2: the cached tactile K/V came out of these
        exact weights, so the fresh tokens have to enter the same embedding space."""
        return self.video_expert.patch_embedding_tactile(tactile_latents).flatten(2).transpose(1, 2)

    def _run_tactile_expert(
        self,
        cache: dict,
        tactile_latents: torch.Tensor,
        actions_input: torch.Tensor,
        executed_mask: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        frame_ids = (offsets // self.steps_per_latent_frame + 1).clamp(max=cache["latent_frames"] - 1)
        with self._autocast_context(dtype=tactile_latents.dtype):
            return self.tactile_expert(
                tactile_tokens=self._patchify_tactile(tactile_latents),
                actions=actions_input,
                executed_mask=executed_mask,
                offset=offsets,
                frame_ids=frame_ids,
                tac_grid=cache["tac_grid"],
                kv_cache=cache["kv"],
            )

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        del kwargs
        if examples is None:
            raise ValueError("WanMoTJointTacExpert.forward requires `examples`.")
        if "tactile_now" not in examples[0]:
            raise ValueError(
                "WanMoTJointTacExpert training requires `tactile_now`/`tactile_offset` from the dataloader; "
                "set datasets.vla_data.tactile_offsets_per_sample > 0."
            )
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        images = self._prepare_batch_images([example["image"] for example in examples])[:, :, :, :1]
        images = self._resize_views_if_needed(images)
        batch_size, n_view = images.shape[:2]
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

        actions_gt = torch.stack([self._batch_to_tensor(example["actions"]) for example in examples])
        actions_gt = actions_gt[:, -self.action_horizon :, :].to(device=device, dtype=dtype, non_blocking=True)
        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = torch.stack([self._batch_to_tensor(example["action_mask"]) for example in examples])
            action_mask = action_mask[:, -self.action_horizon :, :].to(device=device, dtype=torch.bool)

        # Frozen-base rollout: exactly the inference path, so the KV/plan distribution the
        # tactile expert trains on is the one it will see at deployment.
        joint_out = self._infer_joint_latents(
            video=video,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(getattr(self.config.datasets.vla_data, "num_future_frames", 8)) + 1,
            action_horizon=self.action_horizon,
            num_inference_steps=int(getattr(self.action_config, "num_inference_steps", 10)),
            sigma_shift=None,
            seed=None,
        )
        plan = joint_out["action_latents"].clone()
        tac_cache = self._prefill_tac_cache(joint_out["video_latents"].clone(), plan, context, context_mask)

        tactile_now = torch.stack([self._batch_to_tensor(example["tactile_now"]) for example in examples])
        offsets = torch.stack([self._batch_to_tensor(example["tactile_offset"]) for example in examples])
        num_offsets, n_tac_view = int(tactile_now.shape[1]), int(tactile_now.shape[2])
        if n_tac_view != self.n_tactile_views:
            raise ValueError(f"Expected {self.n_tactile_views} tactile views in `tactile_now`, got {n_tac_view}.")
        canvas = self._tactile_now_canvas(tactile_now.to(device=device, dtype=dtype, non_blocking=True))
        tactile_latents = self._run_vae_encode(canvas)

        horizon = self.action_horizon
        offsets_flat = offsets.reshape(-1).to(device=device, dtype=torch.long)
        plan_expand = plan.unsqueeze(1).expand(-1, num_offsets, -1, -1).reshape(-1, horizon, plan.shape[-1])
        gt_expand = actions_gt.unsqueeze(1).expand(-1, num_offsets, -1, -1).reshape(-1, horizon, plan.shape[-1])
        executed = torch.arange(horizon, device=device).view(1, -1) < offsets_flat.view(-1, 1)
        actions_input = torch.where(executed.unsqueeze(-1), gt_expand, plan_expand)

        cache_expanded = {
            "kv": [
                {
                    "k": layer["k"].repeat_interleave(num_offsets, dim=0),
                    "v": layer["v"].repeat_interleave(num_offsets, dim=0),
                }
                for layer in tac_cache["kv"]
            ],
            "latent_frames": tac_cache["latent_frames"],
            "tac_grid": tac_cache["tac_grid"],
        }
        delta = self._run_tactile_expert(cache_expanded, tactile_latents, actions_input, executed, offsets_flat)

        target = gt_expand - actions_input
        suffix = ~executed
        if action_mask is not None:
            mask = action_mask.unsqueeze(1).expand(-1, num_offsets, -1, -1).reshape(-1, horizon, action_mask.shape[-1])
            mask = mask & suffix.unsqueeze(-1)
        else:
            mask = suffix
        loss = self._compute_action_loss_per_sample(delta, target, mask).mean()
        with torch.no_grad():
            valid = mask if mask.ndim == 3 else mask.unsqueeze(-1).expand_as(target)
            target_abs = target.detach().abs()[valid].mean() if valid.any() else target.new_zeros(())
            delta_abs = delta.detach().abs()[valid].mean() if valid.any() else delta.new_zeros(())
        return {
            "total_loss": loss,
            "tac_loss": float(loss.detach().item()),
            "tac_target_abs": float(target_abs.item()),
            "tac_delta_abs": float(delta_abs.item()),
        }

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
            raise ValueError("WanMoTJointTacExpert.predict_action does not support action prefixes.")
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
        plan = joint_out["action_latents"]
        # Prefill deliberately does NOT run here: the chunk goes out on the wire first, the
        # caller then pays the extra joint forward via `prefill_tactile_cache()`.
        self._tac_cache = None
        self._pending_prefill = {
            "video_latents": joint_out["video_latents"],
            "plan": plan,
            "context": context,
            "context_mask": context_mask,
        }

        output = {"normalized_actions": plan.detach().to(device="cpu", dtype=torch.float32).numpy()}
        if bool(kwargs.get("return_video", False)):
            video_out = self._decode_joint_video_latents(joint_out["video_latents"])
            output["video"] = (video_out * 0.5 + 0.5).clamp_(0, 1).detach().to(device="cpu", dtype=torch.float32)
        return output

    @property
    def tactile_prefill_pending(self) -> bool:
        """True when `predict_action` left a plan whose tactile cache is not built yet."""
        return self._pending_prefill is not None

    @torch.inference_mode()
    def prefill_tactile_cache(self) -> None:
        """Build the tactile K/V cache for the plan `predict_action` just returned.

        Costs one clean t=0 joint forward, so call it *after* the action chunk is on the wire:
        the robot starts executing a forward pass earlier, and the cache is ready long before
        the first tactile tick. Every `correct_action` needs it.
        """
        pending = self._pending_prefill
        if pending is None:
            raise ValueError("prefill_tactile_cache requires a fresh plan; call predict_action first.")
        plan = pending["plan"]
        cache = self._prefill_tac_cache(
            pending["video_latents"], plan, pending["context"], pending["context_mask"]
        )
        cache["plan"] = plan
        cache["current"] = plan.clone()
        self._tac_cache = cache
        self._pending_prefill = None

    @torch.inference_mode()
    def correct_action(
        self,
        batch_tactile_images: Any,
        offset: int,
        image_is_tactile: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        """One tactile tick against the active chunk.

        Args:
            batch_tactile_images: raw tactile views only, [B, V_tac, C, H, W].
            offset: executed step count k of the active chunk; the corrected suffix covers [k:].

        Returns:
            `normalized_actions`: corrected suffix [B, H-k, action_dim] replacing the remainder
            of the chunk; `delta`: the applied correction.
        """
        del kwargs
        if self._tac_cache is None:
            # Deliberately not a lazy prefill: a tick that silently costs a full joint forward
            # would blow the real-time budget it is supposed to fit in.
            raise ValueError(
                "correct_action requires a prefilled plan cache; call predict_action then "
                "prefill_tactile_cache() first."
            )
        self.eval()
        cache = self._tac_cache
        device = next(self.video_expert.parameters()).device
        dtype = next(self.video_expert.parameters()).dtype

        horizon = int(cache["plan"].shape[1])
        offset = int(offset)
        if not 0 <= offset < horizon:
            raise ValueError(f"`offset` must be in [0, {horizon}), got {offset}.")

        images = self._prepare_batch_images(batch_tactile_images)[:, :, :, :1]
        preprocessor = getattr(self, "_tactile_preprocessor", None)
        if preprocessor is not None:
            images, auto_tactile_mask = preprocessor.apply_tactile_tensor(images)
            if auto_tactile_mask is not None:
                image_is_tactile = auto_tactile_mask
        images = self._resize_views_if_needed(images)
        batch_size, n_view = images.shape[:2]
        if n_view != self.n_tactile_views:
            raise ValueError(f"correct_action expects {self.n_tactile_views} tactile views, got {n_view}.")
        if batch_size != cache["plan"].shape[0]:
            raise ValueError(f"Batch mismatch with active plan: {batch_size} vs {cache['plan'].shape[0]}.")
        image_is_tactile_tensor = self._prepare_image_is_tactile(image_is_tactile, batch_size, n_view, images.device)
        images = self._normalize_views_for_vae(images, image_is_tactile_tensor)
        # `_concat_views` 拿的是整张画布(视觉在前),这里只有触觉视图,必须走 `_tactile_canvas`,
        # 与训练侧 `_tactile_now_canvas` 同一布局。
        canvas = self._tactile_canvas(images).clamp(-1.0, 1.0).to(device=device, dtype=dtype)
        tactile_latents = self._run_vae_encode(canvas)

        actions_input = torch.cat([cache["current"][:, :offset], cache["plan"][:, offset:]], dim=1)
        executed = (torch.arange(horizon, device=device) < offset).view(1, -1).expand(batch_size, -1)
        offsets = torch.full((batch_size,), offset, device=device, dtype=torch.long)
        delta = self._run_tactile_expert(cache, tactile_latents, actions_input, executed, offsets)

        corrected_suffix = cache["plan"][:, offset:] + delta[:, offset:]
        cache["current"] = torch.cat([cache["current"][:, :offset], corrected_suffix], dim=1)
        return {
            "normalized_actions": corrected_suffix.detach().to(device="cpu", dtype=torch.float32).numpy(),
            # Full chunk (executed prefix + corrected suffix): the wire protocol replaces the
            # whole plan, and callers un-normalize against the plan's original observation.
            "normalized_chunk": cache["current"].detach().to(device="cpu", dtype=torch.float32).numpy(),
            "delta": delta[:, offset:].detach().to(device="cpu", dtype=torch.float32).numpy(),
            "offset": offset,
        }
