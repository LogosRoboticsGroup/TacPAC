from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import functional as _F
from transformers import T5EncoderModel, T5Tokenizer

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.ltx_video import get_diffusion_model
from starVLA.model.modules.ltx_video.utils import (
    _pack_latents,
    gen_noise_from_condition_frame_latent,
    get_text_conditions,
)
from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.utils.text_embedding_cache import (
    DEFAULT_LTX_ENCODER_ID,
    format_text_prompt,
    maybe_load_text_embedding_cache,
)


class _TextCacheCfg(NamedTuple):
    cache_dir: Optional[str]
    context_len: int
    encoder_id: str
    required: bool
    prompt_template: Optional[str]


def _image_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.array(value, copy=True))
    else:
        tensor = _F.pil_to_tensor(value)
    if tensor.ndim >= 3 and tensor.shape[-1] in (1, 3, 4) and tensor.shape[-3] not in (1, 3, 4):
        order = (2, 0, 1) if tensor.ndim == 3 else (3, 0, 1, 2)
        tensor = tensor.permute(*order)
    return tensor


def _to_dict(value: Any) -> dict:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(str(value).lower(), torch.bfloat16)


@dataclass
class LTXQformerFMDefaultConfig:
    name: str = "LTXQformerFM"
    torch_dtype: str = "bfloat16"
    skip_dit_load_from_pretrain: bool = False
    enable_prompt_cache: bool = True
    enable_cudnn_benchmark: bool = True
    video_model: dict = field(default_factory=dict)
    layer_qformer: dict = field(
        default_factory=lambda: {
            "qformer_start_layer": 0,
            "qformer_end_layer": 28,
            "num_query_tokens": 64,
            "input_dim": 2048,
            "ouptput_dim": 512,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "action_dim": 100,
            "state_dim": 100,
            "action_horizon": 16,
            "repeated_diffusion_steps": 1,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 512,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )
    loss: dict = field(default_factory=lambda: {"lambda_video": 1.0, "lambda_action": 1.0})


@FRAMEWORK_REGISTRY.register("LTXQformerFM")
class LTXQformerFM(baseframework):
    """LTX video features, LayerwiseQFormer pooling, and the existing GR00T action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.config = merge_framework_config(LTXQformerFMDefaultConfig, config)
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

        diffusion_hidden = (
            int(self.video_config.diffusion_model.config.num_attention_heads)
            * int(self.video_config.diffusion_model.config.attention_head_dim)
        )
        self.config.framework.layer_qformer.input_dim = diffusion_hidden
        self.layer_qformer = get_layerwise_qformer(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.config.framework.layer_qformer.ouptput_dim
        )
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self._load_action_model_if_needed()

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        loss_cfg = getattr(self.config.framework, "loss", {})
        self.loss_lambda_video = float(getattr(loss_cfg, "lambda_video", 1.0))
        self.loss_lambda_action = float(getattr(loss_cfg, "lambda_action", 1.0))

        if bool(getattr(self.config.framework, "enable_cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True

    def _build_text_components(self, pretrained: str):
        if not bool(getattr(self.video_config, "load_text_encoder", True)):
            return None, None
        tokenizer = T5Tokenizer.from_pretrained(pretrained, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            pretrained,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
        ).eval()
        return tokenizer, text_encoder

    def _init_uncond_text(self) -> None:
        device = next(self.diffusion_model.parameters()).device
        dtype = self.diffusion_model.dtype
        text_uncond = self._load_text_context_from_cache((), device=device, dtype=dtype, prompt="")
        if text_uncond is None:
            text_uncond = self._encode_text([""], cache=False)
        self.register_buffer("uncond_prompt_embeds", text_uncond[0], persistent=False)
        self.register_buffer("uncond_prompt_attention_mask", text_uncond[1], persistent=False)

    def _before_load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        current = self.state_dict()
        return {key: value for key, value in state_dict.items() if key not in current or value.shape == current[key].shape}

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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.video_text_encoder is not None:
            self.video_text_encoder.eval()
        self.video_vae.eval()
        return self

    def compile(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version.")
        mode, fullgraph = "reduce-overhead", False
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.video_vae = torch.compile(self.video_vae, mode=mode, fullgraph=fullgraph)
        self.diffusion_model = torch.compile(self.diffusion_model, mode=mode, fullgraph=fullgraph)
        self.layer_qformer = torch.compile(self.layer_qformer, mode=mode, fullgraph=fullgraph)
        self.action_model.action_encoder = torch.compile(self.action_model.action_encoder, mode=mode, fullgraph=fullgraph)
        self.action_model.model = torch.compile(self.action_model.model, mode=mode, fullgraph=fullgraph)
        self.action_model.action_decoder = torch.compile(self.action_model.action_decoder, mode=mode, fullgraph=fullgraph)

    def _autocast(self):
        device = next(self.diffusion_model.parameters()).device
        if device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=self.diffusion_model.dtype)

    def _tensor(self, value: Any, device=None, dtype=None) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, list):
            value = torch.stack([item if torch.is_tensor(item) else torch.as_tensor(item) for item in value])
        elif not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype, non_blocking=True) if device is not None or dtype is not None else value

    def _prepare_images(self, batch_images: Any, image_is_tactile=None, first_frame_only: bool = False) -> torch.Tensor:
        if isinstance(batch_images, torch.Tensor):
            images = batch_images.detach().clone()
        elif isinstance(batch_images, np.ndarray):
            images = torch.as_tensor(np.array(batch_images, copy=True))
        else:
            images = torch.stack(
                [torch.stack([_image_to_tensor(image) for image in sample], dim=0) for sample in batch_images],
                dim=0,
            )

        if images.ndim == 5 and images.shape[-1] in (1, 3, 4) and images.shape[2] not in (1, 3, 4):
            images = images.permute(0, 1, 4, 2, 3)
        if images.ndim == 6 and images.shape[-1] in (1, 3, 4) and images.shape[2] not in (1, 3, 4):
            images = images.permute(0, 1, 5, 2, 3, 4)
        if images.ndim == 5:
            images = images.unsqueeze(3)
        if first_frame_only:
            images = images[:, :, :, :1]

        target_h, target_w = int(self.image_size[0]), int(self.image_size[1])
        if images.shape[-2:] != (target_h, target_w):
            b, v, c, t, _, _ = images.shape
            flat = images.reshape(b * v, c, t, images.shape[-2], images.shape[-1]).float()
            images = F.interpolate(flat, size=(t, target_h, target_w), mode="trilinear", align_corners=False)
            images = images.reshape(b, v, c, t, target_h, target_w)

        rgb = images.float()
        if rgb.max() > 2.0:
            rgb = rgb / 255.0
        rgb = rgb * 2.0 - 1.0
        if image_is_tactile is None:
            return rgb

        tactile_mask = self._tensor(image_is_tactile, device=images.device, dtype=torch.bool).reshape(images.shape[0], images.shape[1])
        tactile = images.float().clamp(-1.0, 1.0)
        return torch.where(tactile_mask[:, :, None, None, None, None], tactile, rgb)

    def _view_mask(self, view_mask: Any, batch_size: int, n_view: int, device) -> torch.Tensor:
        if view_mask is None:
            return torch.ones(batch_size, n_view, dtype=torch.bool, device=device)
        return self._tensor(view_mask, device=device, dtype=torch.bool).reshape(batch_size, n_view)

    def _fps(self, fps: Optional[Any], batch_size: int, n_view: int, device, dtype) -> torch.Tensor:
        if fps is None:
            fps = [float(getattr(self.video_config, "fps", 24))] * batch_size
        fps = self._tensor(fps, device=device, dtype=dtype).reshape(batch_size, 1).repeat(1, n_view)
        return rearrange(fps, "b v -> (b v)")

    def _format_prompt(self, instruction: str) -> str:
        return format_text_prompt(instruction, getattr(self.video_config, "prompt_template", None))

    def _encode_text(self, prompts: Sequence[str], cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.video_tokenizer is None or self.video_text_encoder is None:
            raise RuntimeError("LTX text encoding requires `framework.video_model.load_text_encoder=true` or cached context.")
        prompt_key = tuple(prompts)
        if cache and self.enable_prompt_cache and self._cached_prompt_key == prompt_key and self._cached_text_cond is not None:
            prompt_embeds, prompt_mask = self._cached_text_cond
        else:
            text = get_text_conditions(
                self.video_tokenizer,
                self.video_text_encoder,
                prompt=list(prompts),
                max_sequence_length=self.text_len,
            )
            prompt_embeds, prompt_mask = text["prompt_embeds"], text["prompt_attention_mask"]
            if cache and self.enable_prompt_cache:
                self._cached_prompt_key = prompt_key
                self._cached_text_cond = (prompt_embeds.detach().cpu(), prompt_mask.detach().cpu())
        device = next(self.diffusion_model.parameters()).device if hasattr(self, "diffusion_model") else prompt_embeds.device
        dtype = getattr(self, "diffusion_model", self.video_text_encoder).dtype
        return prompt_embeds.to(device=device, dtype=dtype), prompt_mask.to(device=device)

    def _text_cache_cfg(self) -> _TextCacheCfg:
        data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        video_template = getattr(self.video_config, "prompt_template", None)
        if data_cfg is None:
            return _TextCacheCfg(None, self.text_len, DEFAULT_LTX_ENCODER_ID, False, video_template)
        return _TextCacheCfg(
            cache_dir=getattr(data_cfg, "text_embedding_cache_dir", None),
            context_len=int(getattr(data_cfg, "text_context_len", self.text_len)),
            encoder_id=str(getattr(data_cfg, "text_cache_encoder_id", DEFAULT_LTX_ENCODER_ID)),
            required=bool(getattr(data_cfg, "require_text_embedding_cache", False)),
            prompt_template=getattr(data_cfg, "text_context_prompt_template", video_template),
        )

    def _load_text_context_from_cache(
        self,
        instructions: Sequence[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        prompt: Optional[str] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Stack cached embeddings for ``instructions`` (templated) or a verbatim ``prompt``.

        Pass ``prompt`` (e.g. the unconditional ``""``) to read a single entry without
        applying the prompt template. Returns ``None`` if any entry is missing.
        """
        cfg = self._text_cache_cfg()
        if not cfg.cache_dir:
            return None
        tasks = [None] if prompt is not None else [str(i) for i in instructions]
        contexts = []
        masks = []
        for task in tasks:
            cached_text = maybe_load_text_embedding_cache(
                cfg.cache_dir,
                task,
                context_len=cfg.context_len,
                encoder_id=cfg.encoder_id,
                required=cfg.required,
                prompt_template=cfg.prompt_template,
                prompt=prompt,
            )
            if cached_text is None:
                return None
            context, mask = cached_text
            contexts.append(context)
            masks.append(mask)
        return (
            torch.stack(contexts, dim=0).to(device=device, dtype=dtype),
            torch.stack(masks, dim=0).to(device=device, dtype=torch.bool),
        )

    def _text_from_examples(self, examples: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.diffusion_model.parameters()).device
        if "context" in examples[0]:
            context = self._tensor([example["context"] for example in examples], device=device, dtype=self.diffusion_model.dtype)
            mask = self._tensor([example["context_mask"] for example in examples], device=context.device, dtype=torch.bool)
            return context, mask
        cached_text = self._load_text_context_from_cache(
            [example["lang"] for example in examples],
            device=device,
            dtype=self.diffusion_model.dtype,
        )
        if cached_text is not None:
            return cached_text
        return self._encode_text([self._format_prompt(example["lang"]) for example in examples])

    def _runtime_text(self, instructions, context=None, context_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.diffusion_model.parameters()).device
        if context is not None and context_mask is not None:
            return (
                self._tensor(context, device=device, dtype=self.diffusion_model.dtype),
                self._tensor(context_mask, device=device, dtype=torch.bool),
            )
        cached_text = self._load_text_context_from_cache(instructions, device=device, dtype=self.diffusion_model.dtype)
        if cached_text is not None:
            return cached_text
        return self._encode_text([self._format_prompt(instruction) for instruction in instructions])

    def _state(self, state: Any, device, dtype) -> Optional[torch.Tensor]:
        state = self._tensor(state, device=device, dtype=dtype)
        return state.unsqueeze(1) if state is not None and state.ndim == 2 else state

    @torch.no_grad()
    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        dist = self.video_vae.encode(images).latent_dist
        latents = dist.mode() if getattr(self.video_config, "noise_seed", None) else dist.sample()
        mean = self.video_vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = self.video_vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        return (latents - mean) / std

    def _run_ltx(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        fps: torch.Tensor,
        with_targets: bool = True,
    ) -> Tuple[dict, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = next(self.diffusion_model.parameters()).device
        dtype = self.diffusion_model.dtype
        batch_size, n_view, _, _, height, width = images.shape
        flat_images = rearrange(images, "b v c t h w -> (b v) c t h w").to(device=device, dtype=dtype)

        latent_frames = int(self.video_config.num_frames) // self.vae_temporal_compression_ratio + 1
        latent_h = height // self.vae_spatial_compression_ratio
        latent_w = width // self.vae_spatial_compression_ratio

        latents = self._encode_latents(flat_images).to(dtype=dtype)
        latents = rearrange(
            _pack_latents(latents),
            "(b v) (f h w) c -> (b v) c f h w",
            b=batch_size,
            h=latent_h,
            w=latent_w,
        )
        condition_latents = latents[:, :, : int(self.video_config.n_prev)]
        target_latents = rearrange(latents, "bv c f h w -> bv (f h w) c")

        view_index = torch.zeros(batch_size, n_view, dtype=torch.long, device=device).flatten()
        timesteps = (self.scheduler_sigmas[view_index] * 1000.0).long()
        noisy_latents, conditioning_mask, _ = gen_noise_from_condition_frame_latent(
            condition_latents,
            latent_frames,
            latent_h,
            latent_w,
            noise_to_condition_frames=0,
        )
        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)

        with self._autocast():
            result = self.diffusion_model(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                encoder_attention_mask=prompt_mask,
                num_frames=latent_frames,
                height=latent_h,
                width=latent_w,
                n_view=n_view,
                rope_interpolation_scale=(
                    self.vae_temporal_compression_ratio / fps,
                    self.vae_spatial_compression_ratio,
                    self.vae_spatial_compression_ratio,
                ),
                video_attention_mask=view_mask.flatten().to(device=device, dtype=dtype),
            )
        if not with_targets:
            return result, n_view, None, None
        target = noisy_latents - target_latents
        mask = 1 - conditioning_mask.unsqueeze(-1).expand_as(target)
        return result, n_view, target, mask

    def _action_condition(self, video_states: List[torch.Tensor], n_view: int, detach: bool = False) -> torch.Tensor:
        start = int(self.config.framework.layer_qformer.qformer_start_layer)
        end = int(self.config.framework.layer_qformer.qformer_end_layer)
        states = video_states[start:end]
        if detach:
            states = [state.detach() for state in states]
        qformer_dtype = next(self.layer_qformer.parameters()).dtype
        states = [rearrange(state, "(b v) l c -> b (v l) c", v=n_view).to(dtype=qformer_dtype) for state in states]
        return self.layer_qformer(states)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        del kwargs
        if examples is None:
            raise ValueError("LTXQformerFM.forward requires examples.")

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
        fps = self._fps([example.get("fps", getattr(self.video_config, "fps", 24)) for example in examples], batch_size, n_view, device, self.diffusion_model.dtype)
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

        action_dtype = next(self.action_model.parameters()).dtype
        action_cond = self._action_condition(
            video_out["video_states_buffer"],
            n_view,
            detach=bool(getattr(self.config.trainer, "knowledge_isolation", False)),
        ).to(dtype=action_dtype)
        actions = self._tensor([example["actions"] for example in examples], device=action_cond.device, dtype=action_dtype)
        actions = actions[:, -self.action_horizon :, :]
        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = self._tensor(
                [example["action_mask"] for example in examples],
                device=action_cond.device,
                dtype=torch.bool,
            )
            action_mask = action_mask[:, -self.action_horizon :, :]
        state = self._state(
            [example["state"] for example in examples] if "state" in examples[0] else None,
            action_cond.device,
            action_dtype,
        )

        repeat = int(getattr(self.config.framework.action_model, "repeated_diffusion_steps", 1))
        action_loss = self.action_model(
            action_cond.repeat(repeat, 1, 1),
            actions.repeat(repeat, 1, 1),
            state.repeat(repeat, 1, 1) if state is not None else None,
            action_mask=action_mask.repeat(repeat, 1, 1) if action_mask is not None else None,
        )
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
        action_dtype = next(self.action_model.parameters()).dtype
        action_cond = self._action_condition(video_out["video_states_buffer"], n_view).to(dtype=action_dtype)
        state = self._state(state, action_cond.device, action_dtype)
        actions = self.action_model.predict_action(action_cond, state)
        return {"normalized_actions": actions.detach().cpu().float().numpy()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/vla/starvla_ltx_qformer_fm.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    print(LTXQformerFM(cfg))
