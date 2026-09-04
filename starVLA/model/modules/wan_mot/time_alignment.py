from __future__ import annotations

import torch
import torch.nn as nn


def build_video_tau_ids(
    *,
    latent_frames: int,
    tokens_per_frame: int,
    n_view: int = 1,
    device: torch.device,
) -> torch.Tensor:
    per_view = torch.arange(latent_frames, device=device, dtype=torch.long).repeat_interleave(tokens_per_frame)
    return per_view.repeat(int(n_view))


def build_action_tau_ids(
    *,
    action_horizon: int,
    latent_frames: int,
    future_frame_stride: int,
    vae_temporal_stride: int = 4,
    device: torch.device,
) -> torch.Tensor:
    tokens_per_tau = int(vae_temporal_stride) * int(future_frame_stride)
    expected_horizon = max(int(latent_frames) - 1, 0) * tokens_per_tau
    if int(action_horizon) != expected_horizon:
        raise ValueError(
            "Action/video temporal layout mismatch: "
            f"action_horizon={action_horizon}, expected={expected_horizon} "
            f"(latent_frames={latent_frames}, vae_temporal_stride={vae_temporal_stride}, "
            f"future_frame_stride={future_frame_stride})."
        )
    tau = torch.arange(action_horizon, device=device, dtype=torch.long).div(tokens_per_tau, rounding_mode="floor") + 1
    return tau.clamp(max=max(int(latent_frames) - 1, 0))


class TemporalAlignmentEmbedding(nn.Module):
    def __init__(
        self,
        max_tau: int,
        attn_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(max_tau) + 1, int(attn_dim))
        nn.init.zeros_(self.embedding.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tau_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tau_ids is None:
            return q, k, v
        tau_emb = self.embedding(tau_ids.to(device=q.device)).to(dtype=q.dtype).unsqueeze(0)
        return q + tau_emb, k + tau_emb, v
