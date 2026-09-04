from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint


def validate_attention_config(num_heads: int, attn_head_dim: int) -> None:
    if num_heads <= 0:
        raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
    if attn_head_dim <= 0:
        raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
    if attn_head_dim % 2 != 0:
        raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")


def run_dit_blocks(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    kwargs: dict[str, Any],
    use_gradient_checkpointing: bool,
) -> torch.Tensor:
    def create_custom_forward(module: nn.Module):
        def custom_forward(*inputs, **kwargs):
            return module(*inputs, **kwargs)

        return custom_forward

    for block in blocks:
        if torch.is_grad_enabled() and use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **kwargs)
    return x


__all__ = ["run_dit_blocks", "validate_attention_config"]
