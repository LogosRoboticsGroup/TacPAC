from typing import List, Optional, Tuple

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.wan_video.wan_dit import WanModel, WanModelStateDictConverter
from starVLA.model.modules.wan_video.wan_vae import Wan2_2_VAE
from starVLA.model.modules.wan_video.wan_text_encoder import WanTextEncoder
from starVLA.model.modules.wan_video.flow_match_scheduler import FlowMatchScheduler
from starVLA.model.modules.wan_video.flow_unipc_scheduler import FlowUniPCMultistepScheduler
from transformers import AutoTokenizer

logger = initialize_overwatch(__name__)

@FRAMEWORK_REGISTRY.register("WanVideo")
class WanVideo(baseframework):
    
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
            vae_sd = torch.load(vae_path, map_location="cpu")
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
            te_sd = torch.load(te_path, map_location="cpu")
            self.text_encoder.load_state_dict(te_sd, strict=False)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        # DiT (no CLIP image encoder needed — I2V uses latent masking)
        dit_kwargs = dict(getattr(self.wan_config, "dit_kwargs", {}))
        for k in ['model_type', 'in_dim', 'dim', 'ffn_dim', 'out_dim', 'num_heads', 'num_layers']:
            if hasattr(self.wan_config, k) and k not in dit_kwargs:
                dit_kwargs[k] = getattr(self.wan_config, k)
        self.dit = WanModel(**dit_kwargs)
        dit_path = getattr(self.wan_config, "dit_path", None)
        if dit_path and os.path.exists(dit_path):
            try:
                WanModelStateDictConverter.load_pretrained(
                    self.dit, dit_path, strict=False,
                )
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
        
    def compile(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version.")
        mode = "reduce-overhead"
        fullgraph = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        self.vae = torch.compile(self.vae, mode=mode, fullgraph=fullgraph)
        self.dit = torch.compile(self.dit, mode=mode, fullgraph=fullgraph)

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
        context = [text_embeds[i][attention_mask[i].bool()] for i in range(len(prompts))]
        return context
    
    def _compute_seq_len(self, latent_t, latent_h, latent_w):
        patch_t, patch_h, patch_w = self.dit.patch_size
        return (latent_t // patch_t) * (latent_h // patch_h) * (latent_w // patch_w)
    
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        device = next(self.dit.parameters()).device
        dtype = self.dit.dtype
        
        prompts = [ex["prompt"] for ex in examples]
        
        images = torch.stack([ex["image"] for ex in examples]).to(dtype)
        batch_size = images.shape[0]
        # Dataloader returns [B, C, n_view, T, H, W] - take first view
        if images.dim() == 6:
            images = images[:, :, 0, :, :, :]  # [B, C, T, H, W]
        
        # Normalize pixels to [-1, 1]
        images = images / 255.0 * 2.0 - 1.0
        
        # VAE encode full video → [B, C, T_lat, H_lat, W_lat]
        # single_encode accepts a batched [B, C, T, H, W] tensor directly,
        # bypassing the per-sample for-loop in Wan2_2_VAE.encode.
        with torch.inference_mode():
            latents = self.vae.single_encode(images)
        latents = latents.to(dtype)
        B, ch, latent_t, latent_h, latent_w = latents.shape
        
        # Extract first-frame latent for I2V conditioning
        z = latents[:, :, :1, :, :]  # [B, C, 1, H_lat, W_lat]
        
        # Text encoding with caption dropout
        context = self._encode_text(prompts, device, dtype)
        dropout_mask = torch.rand(batch_size, device=device) < self.caption_dropout_p
        if dropout_mask.any():
            empty_embed = self.empty_text_embed.to(device=device, dtype=dtype)
            for i in range(batch_size):
                if dropout_mask[i]:
                    context[i] = empty_embed
                    
        # Timestamp sampling
        self.scheduler.set_timesteps(
            self.scheduler.num_train_timesteps, training=True,
        )
        beta_dist = torch.distributions.Beta(1.5, 1.0)
        t_sample = beta_dist.sample((B,)).to(device)
        timestep_indices = (t_sample * len(self.scheduler.timesteps)).long().clamp(
            0, len(self.scheduler.timesteps) - 1)
        timesteps = self.scheduler.timesteps[timestep_indices.cpu()].to(device)
        
        # Flow matching: add noise to full video latent
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps).to(dtype)
        target = self.scheduler.training_target(latents, noise, timesteps).to(dtype)

        # I2V latent masking: first frame stays clean, rest is noisy
        mask = torch.ones(1, 1, latent_t, 1, 1, device=device, dtype=dtype)
        mask[:, :, 0, :, :] = 0.0  # first frame = 0 (clean), rest = 1 (noisy)
        noisy_latents = (1.0 - mask) * z + mask * noisy_latents

        # Per-token timesteps at patch resolution: first-frame tokens get t=0
        seq_len = self._compute_seq_len(latent_t, latent_h, latent_w)
        ph, pw = self.dit.patch_size[1], self.dit.patch_size[2]
        spatial_mask = torch.ones(latent_t, latent_h // ph, latent_w // pw, device=device, dtype=torch.float32)
        spatial_mask[0] = 0.0  # first temporal frame is clean
        per_token_ts = spatial_mask.flatten().unsqueeze(0) * timesteps.float().unsqueeze(1)  # [B, seq_len]

        # DiT forward — pass batched tensor directly, returns [B, C, T, H, W]
        noise_pred = self.dit(
            x=noisy_latents,
            t=per_token_ts,
            context=context,
            seq_len=seq_len,
        )
        
        # Loss: weighted MSE on velocity, exclude first frame
        loss_weights = self.scheduler.training_weight(timesteps).to(device)
        while len(loss_weights.shape) < len(noise_pred.shape):
            loss_weights = loss_weights.unsqueeze(-1)

        # Reuse mask: first latent frame = 0 (conditioning, no loss), rest = 1
        per_elem_loss = loss_weights * (noise_pred.float() - target.float()).pow(2) * mask
        loss = per_elem_loss.sum() / (mask.sum() * B).clamp_min(1) / (ch * latent_h * latent_w)

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
        **kwargs,
    ) -> dict:
        """
        Generate video(s) using the official Wan2.2 TI2V approach.

        image shapes accepted:
          [C, H, W] or [C, 1, H, W]  → single sample (B=1)
          [B, C, 1, H, W]             → batched (B samples)

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
        
        # Normalize image to [B, C, 1, H, W]
        z = None
        i2v_mask = None
        if image is not None:
            img = image
            if img.dim() == 6:      # [B, C, n_view, T, H, W] — take first view/frame
                img = img[:, :, 0, 0, :, :].unsqueeze(2)  # [B, C, 1, H, W]
            elif img.dim() == 3:    # [C, H, W]
                img = img.unsqueeze(1).unsqueeze(0)        # [1, C, 1, H, W]
            elif img.dim() == 4:    # [C, 1, H, W]
                img = img.unsqueeze(0)                     # [1, C, 1, H, W]
            # img is now [B, C, 1, H, W]
            B = img.shape[0]
            img_norm = (img.float() / 255.0 * 2.0 - 1.0).to(device=device, dtype=dtype)
            with torch.inference_mode():
                z = vae.single_encode(img_norm)  # [B, C, 1, H_lat, W_lat]
            z = z.to(dtype)
        else:
            assert prompt is not None
            B = len(prompt)
            
        if len(prompt) == 1 and B > 1:
            prompt = prompt * B
            
        # Text encoding
        context = self._encode_text(prompt, device, dtype)
        if negative_prompt:
            empty_context = self._encode_text([negative_prompt] * B, device, dtype)
        else:
            empty_embed = self.empty_text_embed.to(device=device, dtype=dtype)
            empty_context = [empty_embed] * B
            
        # Noise [B, C, T_lat, H_lat, W_lat]
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            B, ch, latent_t, latent_h, latent_w,
            dtype=torch.float32, device=device, generator=seed_g,
        ).to(dtype)
        
        # I2V masking: first frame clean, rest noise
        i2v_mask = torch.ones(1, 1, latent_t, 1, 1, device=device, dtype=dtype)
        i2v_mask[:, :, 0] = 0.0
        
        if z is not None:
            latent = (1.0 - i2v_mask) * z + i2v_mask * noise  # [B, C, T_lat, H_lat, W_lat]
        else:
            latent = noise
            
        # Scheduler setup
        self.inference_scheduler.set_timesteps(
            self.num_inference_steps, device=device,
            shift=getattr(self.wan_config, "noise_shift", 5.0),
        )
        timesteps = self.inference_scheduler.timesteps

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': empty_context, 'seq_len': seq_len}
        
        # Per-token timestep mask at patch resolution [latent_t, latent_h/ph, latent_w/pw]
        ph, pw = dit.patch_size[1], dit.patch_size[2]
        ts_mask = torch.ones(latent_t, latent_h // ph, latent_w // pw, device=device, dtype=torch.float32)
        ts_mask[0] = 0.0  # first temporal frame is clean (t=0)
        
        # Denoising loop
        for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising")):
            timestep = t.to(device).unsqueeze(0)

            # Per-token timesteps: first-frame tokens t=0, rest = current t
            temp_ts = (ts_mask * timestep.float()).flatten()      # [seq_len]
            timestep_input = temp_ts.unsqueeze(0).expand(B, -1)  # [B, seq_len]

            with torch.autocast("cuda", dtype=dtype):
                # Pass batched tensor directly — returns [B, C, T, H, W]
                noise_pred_cond = dit(latent, t=timestep_input, **arg_c)
                noise_pred_uncond = dit(latent, t=timestep_input, **arg_null)

            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond)

            # Scheduler step: call once with full batch (stateful multi-step scheduler)
            latent = self.inference_scheduler.step(
                noise_pred, t, latent, step_index=step_idx, return_dict=False)[0]

            # Re-apply clean first frame (critical for I2V)
            if z is not None:
                latent = (1.0 - i2v_mask) * z + i2v_mask * latent

        # VAE decode [B, C, T_lat, H_lat, W_lat] → [B, C, T, H, W]
        with torch.inference_mode():
            videos = vae.decode(latent.to(dtype))

        videos = videos.clamp(-1, 1) * 0.5 + 0.5
        return {"video": videos}
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="starVLA/config/training/video/starvla_video.yaml")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = WanVideo(cfg)
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
