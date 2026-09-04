from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from einops import rearrange
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.WM4A.LTXQformerFM import LTXQformerFM, _dtype, _to_dict
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model,
)
from starVLA.model.modules.ltx_video import get_diffusion_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class LTXFMDefaultConfig:
    name: str = "LTXFM"
    torch_dtype: str = "bfloat16"
    skip_dit_load_from_pretrain: bool = False
    enable_prompt_cache: bool = True
    enable_cudnn_benchmark: bool = True
    detach_condition_features: bool = False
    video_model: dict = field(default_factory=dict)
    ltx_layers: dict = field(default_factory=lambda: {"start": 0, "end": 28})
    action_model: dict = field(
        default_factory=lambda: {
            "model_path": None,
            "skip_load_from_pretrain": False,
            "action_model_type": "LayerwiseFM",
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 100,
            "state_dim": 100,
            "action_horizon": 32,
            "repeated_diffusion_steps": 1,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "input_embedding_dim": 1024,
                "attention_head_dim": 64,
                "num_attention_heads": 16,
                "num_layers": 28,
                "cross_attention_dim": 2048,
                "output_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": False,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
        }
    )
    loss: dict = field(default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0})


@FRAMEWORK_REGISTRY.register("LTXFM")
class LTXFM(LTXQformerFM):
    """LTX layer-wise video features directly cross-attended by a LayerwiseFM action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        del kwargs
        self.config = merge_framework_config(LTXFMDefaultConfig, config)
        self.video_config = self.config.framework.video_model
        self.action_config = self.config.framework.action_model
        self.torch_dtype = _dtype(getattr(self.config.framework, "torch_dtype", "bfloat16"))
        self.skip_dit_load_from_pretrain = bool(
            getattr(self.config.framework, "skip_dit_load_from_pretrain", False)
        )
        self.enable_prompt_cache = bool(getattr(self.config.framework, "enable_prompt_cache", True))
        self._cached_prompt_key = None
        self._cached_text_cond = None

        pretrained = self.video_config.pretrained_model_name_or_path
        self.video_tokenizer, self.video_text_encoder = self._build_text_components(pretrained)
        self.video_vae = AutoencoderKLLTXVideo.from_pretrained(
            pretrained,
            subfolder="vae",
            torch_dtype=self.torch_dtype,
        ).eval()
        for module in (self.video_text_encoder, self.video_vae):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

        self.text_len = int(getattr(self.video_config, "text_context_len", 128))
        self.diffusion_model = get_diffusion_model(
            config=self.video_config,
            skip_load_from_pretrain=self.skip_dit_load_from_pretrain,
        ).to(self.torch_dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler(**_to_dict(self.video_config.scheduler))
        self.register_buffer("scheduler_sigmas", self.scheduler.sigmas.clone().to(self.torch_dtype), persistent=False)
        self._init_uncond_text()

        self.vae_spatial_compression_ratio = int(self.video_vae.spatial_compression_ratio)
        self.vae_temporal_compression_ratio = int(self.video_vae.temporal_compression_ratio)
        self.image_size = tuple(getattr(self.config.datasets.vla_data, "image_size", [256, 256]))
        self.use_future_frames = bool(getattr(self.config.datasets.vla_data, "use_future_frames", False))

        ltx_hidden = (
            int(self.video_config.diffusion_model.config.num_attention_heads)
            * int(self.video_config.diffusion_model.config.attention_head_dim)
        )
        layer_cfg = self.config.framework.ltx_layers
        self.ltx_start_layer = int(getattr(layer_cfg, "start", 0))
        self.ltx_end_layer = int(getattr(layer_cfg, "end", self.video_config.diffusion_model.config.num_layers))
        self.num_action_layers = self.ltx_end_layer - self.ltx_start_layer

        dit_cfg = self.config.framework.action_model.diffusion_model_cfg
        head_dim = int(dit_cfg.get("attention_head_dim", 64))
        action_hidden = int(dit_cfg.get("input_embedding_dim", getattr(self.action_config, "hidden_size", 1024)))
        dit_cfg.attention_head_dim = head_dim
        dit_cfg.input_embedding_dim = action_hidden
        dit_cfg.cross_attention_dim = ltx_hidden
        dit_cfg.num_attention_heads = action_hidden // head_dim
        dit_cfg.num_layers = self.num_action_layers
        dit_cfg.interleave_self_attention = False

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self._load_action_model_if_needed()

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        loss_cfg = getattr(self.config.framework, "loss", {})
        self.loss_lambda_video = float(getattr(loss_cfg, "lambda_video", 1.0))
        self.loss_lambda_action = float(getattr(loss_cfg, "lambda_action", 1.0))

        if bool(getattr(self.config.framework, "enable_cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True

    def _load_action_model_if_needed(self) -> None:
        if self.skip_dit_load_from_pretrain or bool(getattr(self.action_config, "skip_load_from_pretrain", False)):
            return
        path = getattr(self.action_config, "model_path", None)
        if not path:
            return
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu")
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
        if any(key.startswith("action_model.") for key in state_dict):
            state_dict = {
                key.removeprefix("action_model."): value
                for key, value in state_dict.items()
                if key.startswith("action_model.")
            }
        self.action_model.load_state_dict(state_dict, strict=False)

    def compile(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version.")
        mode, fullgraph = "reduce-overhead", False
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.video_vae = torch.compile(self.video_vae, mode=mode, fullgraph=fullgraph)
        self.diffusion_model = torch.compile(self.diffusion_model, mode=mode, fullgraph=fullgraph)
        self.action_model.action_encoder = torch.compile(self.action_model.action_encoder, mode=mode, fullgraph=fullgraph)
        self.action_model.model = torch.compile(self.action_model.model, mode=mode, fullgraph=fullgraph)
        self.action_model.action_decoder = torch.compile(self.action_model.action_decoder, mode=mode, fullgraph=fullgraph)

    def _action_conditions(self, video_states: List[torch.Tensor], n_view: int, detach: bool = False) -> List[torch.Tensor]:
        states = video_states[self.ltx_start_layer : self.ltx_end_layer]
        if detach:
            states = [state.detach() for state in states]
        action_dtype = next(self.action_model.parameters()).dtype
        return [rearrange(state, "(b v) l c -> b (v l) c", v=n_view).to(dtype=action_dtype) for state in states]

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        if examples is None:
            raise ValueError("LTXFM.forward requires examples.")

        images = self._prepare_images(
            [example["image"] for example in examples],
            [example["image_is_tactile"] for example in examples] if "image_is_tactile" in examples[0] else None,
        )
        batch_size, n_view = images.shape[:2]
        device = next(self.diffusion_model.parameters()).device
        view_mask = self._view_mask(
            [example["view_mask"] for example in examples] if "view_mask" in examples[0] else None,
            batch_size,
            n_view,
            images.device,
        )
        fps = self._fps(
            [example.get("fps", getattr(self.video_config, "fps", 24)) for example in examples],
            batch_size,
            n_view,
            device,
            self.diffusion_model.dtype,
        )
        prompt_embeds, prompt_mask = self._text_from_examples(examples)

        if float(getattr(self.video_config, "caption_dropout_p", 0.0)) > 0:
            drop = (torch.rand(batch_size, device=device) < float(self.video_config.caption_dropout_p))[:, None, None]
            prompt_embeds = torch.where(drop, self.uncond_prompt_embeds.to(device, prompt_embeds.dtype), prompt_embeds)
            prompt_mask = torch.where(drop.squeeze(-1), self.uncond_prompt_attention_mask.to(device), prompt_mask)

        video_out, n_view, video_target, video_mask = self._run_ltx(images, view_mask, prompt_embeds, prompt_mask, fps)
        video_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.use_future_frames:
            video_loss = ((video_out["noise_pred"].float() - video_target.float()).pow(2) * video_mask).sum((1, 2))
            video_loss = (video_loss / video_mask.sum((1, 2)).clamp_min(1)).mean()

        detach = bool(getattr(self.config.trainer, "knowledge_isolation", False)) or bool(
            getattr(self.config.framework, "detach_condition_features", False)
        )
        action_conds = self._action_conditions(video_out["video_states_buffer"], n_view, detach=detach)
        action_dtype = next(self.action_model.parameters()).dtype
        actions = self._tensor([example["actions"] for example in examples], device=action_conds[0].device, dtype=action_dtype)
        actions = actions[:, -self.action_horizon :, :]
        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = self._tensor(
                [example["action_mask"] for example in examples],
                device=action_conds[0].device,
                dtype=torch.bool,
            )
            action_mask = action_mask[:, -self.action_horizon :]
        state = self._state(
            [example["state"] for example in examples] if "state" in examples[0] else None,
            action_conds[0].device,
            action_dtype,
        )

        repeat = int(getattr(self.config.framework.action_model, "repeated_diffusion_steps", 1))
        action_conds = [cond.repeat(repeat, 1, 1) for cond in action_conds]
        actions = actions.repeat(repeat, 1, 1)
        state = state.repeat(repeat, 1, 1) if state is not None else None
        action_mask = (
            action_mask.repeat(repeat, *([1] * (action_mask.ndim - 1))) if action_mask is not None else None
        )

        noise = torch.randn_like(actions)
        t = self.action_model.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_actions = (1 - t) * noise + t * actions
        t_discretized = (t[:, 0, 0] * self.action_model.num_timestep_buckets).long()
        pred_action_velocity = self.action_model.predict_velocity(
            action_conds,
            noisy_actions,
            t_discretized,
            state=state,
        )
        pred_actions = noisy_actions + (1 - t) * pred_action_velocity
        action_loss = self.action_model._compute_action_loss(pred_actions, actions, action_mask)
        total_loss = self.loss_lambda_action * action_loss + self.loss_lambda_video * video_loss
        return {"total_loss": total_loss, "action_loss": action_loss.detach(), "video_loss": video_loss.detach()}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: Optional[List[dict]] = None,
        batch_images: Optional[Any] = None,
        view_mask: Optional[Any] = None,
        instructions: Optional[List[str]] = None,
        fps: Optional[List[float]] = None,
        state: Optional[Any] = None,
        context: Optional[Any] = None,
        context_mask: Optional[Any] = None,
        image_is_tactile: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        del kwargs
        if examples is not None:
            examples = examples if isinstance(examples, list) else [examples]
            batch_images = [example["image"] for example in examples]
            view_mask = [example["view_mask"] for example in examples] if "view_mask" in examples[0] else None
            instructions = [example["lang"] for example in examples]
            fps = [example.get("fps", getattr(self.video_config, "fps", 24)) for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            image_is_tactile = [example["image_is_tactile"] for example in examples] if "image_is_tactile" in examples[0] else None
            if "context" in examples[0]:
                context, context_mask = [example["context"] for example in examples], [example["context_mask"] for example in examples]

        self.eval()
        images = self._prepare_images(batch_images, image_is_tactile, first_frame_only=True)
        batch_size, n_view = images.shape[:2]
        device = next(self.diffusion_model.parameters()).device
        view_mask = self._view_mask(view_mask, batch_size, n_view, images.device)
        fps = self._fps(fps, batch_size, n_view, device, self.diffusion_model.dtype)
        prompt_embeds, prompt_mask = self._runtime_text(instructions, context, context_mask)

        video_out, n_view, _, _ = self._run_ltx(images, view_mask, prompt_embeds, prompt_mask, fps, with_targets=False)
        action_conds = self._action_conditions(video_out["video_states_buffer"], n_view)
        action_dtype = next(self.action_model.parameters()).dtype
        state = self._state(state, action_conds[0].device, action_dtype)
        actions = self.action_model.predict_action(action_conds, state)
        return {"normalized_actions": actions.detach().cpu().float().numpy()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/vla/starvla_ltx_fm.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    print(LTXFM(cfg))
