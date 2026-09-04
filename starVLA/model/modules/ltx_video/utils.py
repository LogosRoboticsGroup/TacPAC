from typing import Dict, List, Optional, Union
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

import torchvision
import torchvision.transforms as transforms



def unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents

def gen_noise_from_condition_frame_latent(
    condition_frame_latent, latent_num_frames,
    latent_height=12, latent_width=16,
    generator=None, noise_to_condition_frames=0.05, pack_latent=True
):

    """
    To train the model for memory-frames conditioning,
    we occasionally set the timestep of the tokens
    belonging to the condition video frames to a small
    random value and noise these tokens to the corresponding
    level. The model quickly learns to utilize
    this new information (when provided) as a conditioning signal
    
    condition_frame_latent: (b v) c m h w

    modified from 

    """

    mem_size = condition_frame_latent.shape[2]
    num_channels_latents = condition_frame_latent.shape[1] # 128
    batch_size = condition_frame_latent.size(0)   # bv
    # latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

    shape = (batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width)
    mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)

    init_latents = condition_frame_latent[:,:,:1].repeat(1, 1, latent_num_frames, 1, 1)
    init_latents[:,:,:mem_size] = condition_frame_latent
    conditioning_mask = torch.zeros(mask_shape, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    conditioning_mask[:, :, :mem_size] = 1.0

    # similar to conditioning mask but useful to timesteps
    cond_indicator = torch.zeros((1, 1, latent_num_frames, 1, 1), device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    cond_indicator[:, :, :mem_size] = 1.0

    rand_noise_ff = random.random() * noise_to_condition_frames

    first_frame_mask = conditioning_mask.clone()
    first_frame_mask[:, :, :mem_size] = 1.0 - rand_noise_ff # diffusion forcing

    noise = randn_tensor(shape, generator=generator, device=condition_frame_latent.device, dtype=condition_frame_latent.dtype)
    latents = init_latents * first_frame_mask + noise * (1 - first_frame_mask)

    if pack_latent:
        conditioning_mask = _pack_latents(conditioning_mask).squeeze(-1)
        cond_indicator = _pack_latents(cond_indicator).squeeze(-1)

        latents = _pack_latents(latents)

    # pack_latents: b c f h w -> b (f h w) c
    # unpack_latents: b (f h w) c -> b c f h w

    return latents, conditioning_mask, cond_indicator

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def _encode_prompt(
    tokenizer,
    text_encoder,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length,
) -> torch.Tensor:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_attention_mask = text_inputs.attention_mask
    prompt_attention_mask = prompt_attention_mask.bool().to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}

@torch.no_grad()
def get_text_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 128,
    **kwargs,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    if isinstance(prompt, str):
        prompt = [prompt]

    return _encode_prompt(tokenizer, text_encoder, prompt, device, dtype, max_sequence_length)
