from contextlib import nullcontext
from typing import List

import os
import torch
from tqdm import tqdm

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.wan_video.wan_dit import WanModelStateDictConverter
from starVLA.model.modules.wan_video.wan_dit_multiview import WanMultiViewModel
from starVLA.model.modules.wan_video.wan_vae import Wan2_2_VAE
from starVLA.model.modules.wan_video.wan_text_encoder import WanTextEncoder
from starVLA.model.modules.wan_video.flow_match_scheduler import FlowMatchScheduler
from starVLA.model.modules.wan_video.flow_unipc_scheduler import FlowUniPCMultistepScheduler
from transformers import AutoTokenizer

logger = initialize_overwatch(__name__)


def _looks_like_local_path(path: str) -> bool:
    return os.path.isabs(path) or os.path.sep in path or (os.path.altsep and os.path.altsep in path)


@FRAMEWORK_REGISTRY.register("MultiViewWanVideo")
class MultiViewWanVideo(baseframework):

    def __init__(
        self,
        config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.wan_config = config.framework.wan_video


        # VAE
        self.vae = Wan2_2_VAE(
            z_dim=getattr(self.wan_config, "z_dim", 48),
            dim=getattr(self.wan_config, "vae_dim", 160),
        )
        vae_path = getattr(self.wan_config, "vae_path", None)
        if vae_path and os.path.exists(vae_path):
            vae_sd = WanModelStateDictConverter.load_state_dict(vae_path)
            if not any(k.startswith("model.") for k in vae_sd.keys()):
                vae_sd = {"model." + k: v for k, v in vae_sd.items()}
            if "model_state" in vae_sd:
                vae_sd = {"model." + k: v for k, v in vae_sd["model_state"].items()}
            self.vae.load_state_dict(vae_sd, strict=False)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        # Text encoder
        self.text_encoder = WanTextEncoder(
            **getattr(self.wan_config, "text_encoder_kwargs", {}),
        )
        te_path = getattr(self.wan_config, "text_encoder_path", None)
        if te_path and os.path.exists(te_path):
            te_sd = WanModelStateDictConverter.load_state_dict(te_path)
            self.text_encoder.load_state_dict(te_sd, strict=False)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        from starVLA.dataloader.video.mixtures import num_video_keys

        num_view = num_video_keys(self.config.datasets.video_data.data_mix)
        dit_kwargs = dict(getattr(self.wan_config, "dit_kwargs", {}))
        for key in ["model_type", "in_dim", "dim", "ffn_dim", "out_dim", "num_heads", "num_layers"]:
            if hasattr(self.wan_config, key) and key not in dit_kwargs:
                dit_kwargs[key] = getattr(self.wan_config, key)
        self.dit = WanMultiViewModel(**dit_kwargs, num_view=num_view)
        dit_path = getattr(self.wan_config, "dit_path", None)
        if dit_path and os.path.exists(dit_path):
            try:
                dit_sd = WanModelStateDictConverter.load_state_dict(dit_path)
                has_cross_view = self.dit.state_dict_has_cross_view_weights(dit_sd)
                missing_keys, unexpected_keys = self.dit.load_state_dict(
                    dit_sd, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                force_copy = getattr(
                    self.wan_config, "force_copy_cross_view_from_self_attn", False)
                copy_cross_view = getattr(
                    self.wan_config, "copy_cross_view_from_self_attn", True)
                if copy_cross_view and (force_copy or not has_cross_view):
                    self.dit.init_cross_view_from_self_attn()
            except FileNotFoundError:
                logger.warning(f"DiT weights not fully downloaded at {dit_path}, using random init")

        # Tokenizer
        tokenizer_path = getattr(self.wan_config, "tokenizer_path", "google/umt5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Schedulers
        self.scheduler = FlowMatchScheduler(
            **getattr(self.wan_config, "scheduler_kwargs", {}),
        )
        self.inference_scheduler = FlowUniPCMultistepScheduler(
            **getattr(self.wan_config, "inference_scheduler_kwargs", {}),
        )

        # Inference params
        self.num_inference_steps = getattr(self.wan_config, "num_inference_steps", 50)
        self.guidance_scale = getattr(self.wan_config, "guidance_scale", 5.0)
        self.caption_dropout_p = getattr(self.wan_config, "caption_dropout_p", 0.1)
        
        # VAE stride for computing latent dimensions
        self.vae_stride = (4, 16, 16)
        
        # Pre-compute empty-string text embedding (used for CFG null context and caption dropout)
        self.empty_text_embed = self._encode_text([""], "cpu", torch.float32)[0]
    
    @staticmethod
    def _unwrap_compiled_module(module):
        return getattr(module, "_orig_mod", module) 
    
    def _encode_text(self, prompts, device, dtype):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=getattr(self.wan_config, "text_len", 512),
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        with torch.inference_mode():
            text_embeds = self.text_encoder(input_ids, mask=attention_mask)
        return [text_embeds[i][attention_mask[i].bool()] for i in range(len(prompts))]
        
    def _compute_seq_len(self, latent_t, latent_h, latent_w):
        patch_t, patch_h, patch_w = self.dit.patch_size
        return (latent_t // patch_t) * (latent_h // patch_h) * (latent_w // patch_w)

    def _build_per_token_ts(self, latent_t, latent_h, latent_w, batch_size, device, timesteps):
        ph, pw = self.dit.patch_size[1], self.dit.patch_size[2]
        spatial_mask = torch.ones(
            latent_t,
            latent_h // ph,
            latent_w // pw,
            device=device,
            dtype=torch.float32,
        )
        spatial_mask[0] = 0.0
        return spatial_mask.flatten().unsqueeze(0) * timesteps.float().reshape(batch_size, 1)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        device = next(self.dit.parameters()).device
        dtype = self.dit.dtype
        
        prompts = [ex["prompt"] for ex in examples]
        
        images = torch.stack([ex["image"] for ex in examples]).to(device=device, dtype=dtype)
        batch_size = images.shape[0]
        view_mask = None

        n_view = images.shape[2] # image shape: [B, C, n_view, T, H, W]
        if "view_mask" in examples[0]:
            view_mask = torch.stack([ex["view_mask"] for ex in examples]).to(device=device, dtype=torch.bool)
        else:
            view_mask = torch.ones(batch_size, n_view, dtype=torch.bool, device=device)
        images = images.permute(0, 2, 1, 3, 4, 5).flatten(0, 1) # [B, C, n_view, T, H, W] -> [B, n_view, C, T, H, W] -> [B * n_view, C, T, H, W]
        
        # Normalize pixels to [-1, 1]
        images = images / 255.0 * 2.0 - 1.0
        
        # VAE encode full video → [B, C, T_lat, H_lat, W_lat]
        # single_encode accepts a batched [B, C, T, H, W] tensor directly,
        # bypassing the per-sample for-loop in Wan2_2_VAE.encode.
        with torch.inference_mode():
            latents = self.vae.single_encode(images)
        latents = latents.to(dtype)
        total_views, ch, latent_t, latent_h, latent_w = latents.shape
        
        # Extract first-frame latent for I2V conditioning
        z = latents[:, :, :1, :, :]
        
        # Text encoding with caption dropout
        context = self._encode_text(prompts, device, dtype)
        if n_view > 1:
            context = [c for c in context for _ in range(n_view)]
        dropout_mask = (torch.rand(batch_size, device=device) < self.caption_dropout_p).repeat_interleave(n_view)
        if dropout_mask.any():
            empty_embed = self.empty_text_embed.to(device=device, dtype=dtype)
            for i in range(total_views):
                if dropout_mask[i]:
                    context[i] = empty_embed
                    
        # Timestamp sampling
        self.scheduler.set_timesteps(
            self.scheduler.num_train_timesteps, training=True,
        )
        beta_dist = torch.distributions.Beta(1.5, 1.0)
        t_sample = beta_dist.sample((batch_size,)).to(device)
        timestep_indices = (t_sample * len(self.scheduler.timesteps)).long().clamp(
            0, len(self.scheduler.timesteps) - 1)
        timesteps = self.scheduler.timesteps[timestep_indices.cpu()].to(device).repeat_interleave(n_view)
        
        # Flow matching: add noise to full video latent
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps).to(dtype)
        target = self.scheduler.training_target(latents, noise, timesteps).to(dtype)

        # I2V latent masking: first frame stays clean, rest is noisy
        frame_mask = torch.ones(1, 1, latent_t, 1, 1, device=device, dtype=dtype)
        frame_mask[:, :, 0] = 0.0
        noisy_latents = (1.0 - frame_mask) * z + frame_mask * noisy_latents

        # Per-token timesteps at patch resolution: first-frame tokens get t=0
        seq_len = self._compute_seq_len(latent_t, latent_h, latent_w)
        per_token_ts = self._build_per_token_ts(
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            batch_size=total_views,
            device=device,
            timesteps=timesteps,
        )
        
        # DiT forward — pass batched tensor directly, returns [B, C, T, H, W]
        noise_pred = self.dit(
            x=noisy_latents,
            t=per_token_ts,
            context=context,
            seq_len=seq_len,
            n_view=n_view,
            view_mask=view_mask,
        )
        
        # Loss: weighted MSE on velocity, exclude first frame
        loss_weights = self.scheduler.training_weight(timesteps).to(device)
        while len(loss_weights.shape) < len(noise_pred.shape):
            loss_weights = loss_weights.unsqueeze(-1)
        
        # Reuse frame_mask: first latent frame = 0 (conditioning, no loss), rest = 1
        per_elem_loss = loss_weights * (noise_pred.float() - target.float()).pow(2) * frame_mask
        valid_view_count = total_views
        if view_mask is not None:
            view_weights = view_mask.reshape(-1).to(device=device, dtype=per_elem_loss.dtype).view(total_views, 1, 1, 1, 1)
            per_elem_loss = per_elem_loss * view_weights
            valid_view_count = int(view_mask.sum().item())

        loss = per_elem_loss.sum() / (frame_mask.sum() * max(valid_view_count, 1)).clamp_min(1) / (ch * latent_h * latent_w)
        return {"video_loss": loss}
    
    @torch.no_grad()
    def predict_video(
        self,
        image: torch.Tensor = None,
        prompt: List[str] = None,
        negative_prompt: str = "",
        num_frames: int = None,
        height: int = None,
        width: int = None,
        seed: int = 42,
        view_mask: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Generate video(s) using the official Wan2.2 TI2V approach.

        image shapes accepted:
          [C, H, W] or [C, 1, H, W]  → single sample (B=1)
          [B, C, 1, H, W]             → batched (B samples)
          [B, C, V, T, H, W]          → batched multiview

        Returns:
          {"video": [B, C, T, H, W]} in 0-1
        """
        dit = self._unwrap_compiled_module(self.dit)
        vae = self._unwrap_compiled_module(self.vae)
        device = next(dit.parameters()).device
        dtype = dit.dtype
        
        if num_frames is None:
            num_frames = self.config.datasets.video_data.num_frames
        if height is None:
            height = getattr(self.wan_config, "target_height", self.config.datasets.video_data.image_size[0])
        if width is None:
            width = getattr(self.wan_config, "target_width", self.config.datasets.video_data.image_size[1])
            
        latent_t = (num_frames - 1) // self.vae_stride[0] + 1
        latent_h = height // self.vae_stride[1]
        latent_w = width // self.vae_stride[2]
        ch = dit.in_dim
        
        patch_t, patch_h, patch_w = dit.patch_size
        seq_len = (latent_t // patch_t) * (latent_h // patch_h) * (latent_w // patch_w)
        
        z = None
        if image is not None:
            img = image

            if img.dim() == 3:      # [C, H, W]
                img = img.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            elif img.dim() == 4:    # [C, T, H, W]
                img = img.unsqueeze(0).unsqueeze(2)
            elif img.dim() == 5:    # [B, C, T, H, W]
                img = img.unsqueeze(2)
            elif img.dim() != 6:    # [B, C, V, T, H, W]
                raise ValueError(f"Unsupported image ndim={img.dim()}")

            batch_size, n_view = img.shape[0], img.shape[2]
            img = img[:, :, :, :1, :, :].permute(0, 2, 1, 3, 4, 5).flatten(0, 1) # [B, C, V, T, H, W] -> [B, V, C, T, H, W] -> [B * V, C, T, H, W]

            img_norm = (img.float() / 255.0 * 2.0 - 1.0).to(device=device, dtype=dtype)
            with torch.inference_mode():
                z = self.vae.single_encode(img_norm)
            z = z.to(dtype)
        else:
            batch_size, n_view = len(prompt), 1
            
        total_views = batch_size * n_view
        
        if len(prompt) == batch_size:
            context = self._encode_text(prompt, device, dtype)
            if n_view > 1:
                context = [c for c in context for _ in range(n_view)]
        else:
            raise ValueError(f"Expected {batch_size} or {total_views} prompts, got {len(prompt)}")
        
        vm = None
        if view_mask is not None:
            vm = torch.as_tensor(view_mask, device=device, dtype=torch.bool)
            if vm.dim() == 1:
                vm = vm.unsqueeze(0).expand(batch_size, -1)
            if vm.shape != (batch_size, n_view):
                raise ValueError(f"Expected view_mask shape {(batch_size, n_view)}, got {tuple(vm.shape)}")
            
        if negative_prompt:
            negative_context = self._encode_text([negative_prompt] * total_views, device, dtype)
        else:
            empty_embed = self.empty_text_embed.to(device=device, dtype=dtype)
            negative_context = [empty_embed] * total_views
            
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(
            total_views,
            ch,
            latent_t,
            latent_h,
            latent_w,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).to(dtype)
        
        i2v_mask = torch.ones(1, 1, latent_t, 1, 1, device=device, dtype=dtype)
        i2v_mask[:, :, 0] = 0.0
        latent = noise if z is None else (1.0 - i2v_mask) * z + i2v_mask * noise
        
        self.inference_scheduler.set_timesteps(
            self.num_inference_steps,
            device=device,
            shift=getattr(self.wan_config, "noise_shift", 5.0),
        )
        timesteps = self.inference_scheduler.timesteps
        
        arg_c = {
            "context": context,
            "seq_len": seq_len,
            "n_view": n_view,
            "view_mask": vm,
        }
        arg_null = {
            "context": negative_context,
            "seq_len": seq_len,
            "n_view": n_view,
            "view_mask": vm,
        }
        
        use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
            timestep = t.to(device).unsqueeze(0)
            timestep_input = self._build_per_token_ts(
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                batch_size=total_views,
                device=device,
                timesteps=timestep.expand(total_views),
            )
            
            with torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else nullcontext():
                noise_pred_cond = self.dit(latent, t=timestep_input, **arg_c)
                noise_pred_uncond = self.dit(latent, t=timestep_input, **arg_null)
                
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = self.inference_scheduler.step(
                noise_pred,
                t,
                latent,
                step_index=step_idx,
                return_dict=False,
            )[0]
            
            if z is not None:
                latent = (1.0 - i2v_mask) * z + i2v_mask * latent
                
        with torch.inference_mode():
            videos = self.vae.decode(latent.to(dtype))

        videos = videos.clamp(-1, 1) * 0.5 + 0.5
        return {"video": videos}
    
    
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/video/starvla_video_multiview.yaml",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MultiViewWanVideo(cfg)
    model = model.to(torch.bfloat16)
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("Full model OOM, moving only DiT to GPU")
        model.dit = model.dit.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  DiT: {sum(p.numel() for p in model.dit.parameters()) / 1e6:.1f}M")
    print(f"  VAE: {sum(p.numel() for p in model.vae.parameters()) / 1e6:.1f}M")
    print(f"  TextEncoder: {sum(p.numel() for p in model.text_encoder.parameters()) / 1e6:.1f}M")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
