"""Denoised tactile "stress" view, shared by the training dataloader and inference clients.

Pipeline (per tactile view): gray -> |frame - median_baseline| -> subtract per-pixel
noise floor -> blur -> gain -> stress in [0, 1]. Training feeds ``stress * 2 - 1``
through the ``image_is_tactile`` branch; inference applies the same transform in the
shared policy preprocessing path.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

DEFAULT_GAIN = 10.0
DEFAULT_MARGIN = 2.0
DEFAULT_BLUR_SIGMA = 2.0
DEFAULT_BASELINE_FRAMES = 15


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB channel-last (..., H, W, 3) -> gray (..., H, W), float32."""
    return rgb.astype(np.float32) @ _GRAY_WEIGHTS


def compute_stress_baseline(gray_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gray frames (N, H, W) -> (median baseline, per-pixel 99th-percentile noise floor)."""
    stack = gray_frames.astype(np.float32)
    baseline = np.median(stack, axis=0)
    noise_floor = np.percentile(np.abs(stack - baseline), 99, axis=0).astype(np.float32)
    return baseline.astype(np.float32), noise_floor


def stress_map(
    gray: np.ndarray,
    baseline: np.ndarray,
    noise_floor: np.ndarray,
    gain: float = DEFAULT_GAIN,
    margin: float = DEFAULT_MARGIN,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> np.ndarray:
    """Gray frame(s) (H, W) or (N, H, W) -> denoised stress in [0, 1], same shape."""
    if gray.ndim == 3:
        return np.stack(
            [stress_map(frame, baseline, noise_floor, gain, margin, blur_sigma) for frame in gray]
        )
    cleaned = np.maximum(np.abs(gray.astype(np.float32) - baseline) - noise_floor - margin, 0.0)
    if blur_sigma > 0:
        cleaned = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(cleaned * gain / 255.0, 0.0, 1.0)


def stress_view(
    frames: torch.Tensor,
    baseline: np.ndarray,
    noise_floor: np.ndarray,
    gain: float = DEFAULT_GAIN,
) -> torch.Tensor:
    """Training-side view: frames (N, C, H, W) uint8 -> (N, 3, H, W) float in [-1, 1]."""
    gray = to_gray(frames.numpy().transpose(0, 2, 3, 1))
    stress = stress_map(gray, baseline, noise_floor, gain)
    view = torch.from_numpy(stress * 2.0 - 1.0)
    return view[:, None].expand(-1, 3, -1, -1).contiguous()


def video_stress_baseline(
    decode_fn,
    video_path: str,
    count: int,
    cache: dict,
    fps: float,
    cache_cap: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline/noise floor from a video's first ``count`` frames, memoized in ``cache``.

    ``decode_fn(video_path, indexes, indexes_s)`` must return (N, C, H, W) uint8 frames.
    """
    if video_path not in cache:
        if len(cache) >= cache_cap:
            cache.pop(next(iter(cache)))
        indexes = np.arange(count, dtype=np.int64)
        frames = decode_fn(video_path, indexes, (indexes / fps).tolist())
        gray = to_gray(frames.numpy().transpose(0, 2, 3, 1))
        cache[video_path] = compute_stress_baseline(gray)
    return cache[video_path]


class TactileInferencePreprocessor:
    """Stateful online preprocessing for any tactile stream."""

    def __init__(
        self,
        mode: str = "residual",
        gain: float = DEFAULT_GAIN,
        baseline_frames: int = DEFAULT_BASELINE_FRAMES,
        view_indices=(),
    ) -> None:
        self.mode = mode
        self.gain = float(gain)
        self.baseline_frames = int(baseline_frames)
        self.view_indices = tuple(view_indices)
        self.reset()

    @classmethod
    def from_config(cls, config):
        vla_data = _config_value(_config_value(config, "datasets"), "vla_data")
        data_mix = _config_value(vla_data, "data_mix")
        if data_mix is None:
            return None

        from starVLA.dataloader.vla.mixtures import video_keys

        substring = str(_config_value(vla_data, "tactile_key_substring", "tactile"))
        indices = [index for index, key in enumerate(video_keys(data_mix)) if substring in str(key)]
        if not indices:
            return None

        enabled = bool(_config_value(vla_data, "enable_tactile_residual", True))
        training_mode = _config_value(vla_data, "tactile_residual_mode", "signed")
        mode = "raw" if not enabled else ("stress" if training_mode == "stress" else "residual")
        return cls(
            mode=mode,
            gain=_config_value(vla_data, "tactile_stress_gain", DEFAULT_GAIN),
            baseline_frames=_config_value(vla_data, "tactile_stress_baseline_frames", DEFAULT_BASELINE_FRAMES),
            view_indices=indices,
        )

    def reset(self) -> None:
        self._references = {}
        self._gray_buffers = {}
        self._stress_baselines = {}

    def process(self, frame: np.ndarray, stream_id) -> np.ndarray:
        if self.mode == "raw":
            return np.asarray(frame)

        frame = np.asarray(frame, dtype=np.float32)
        if self.mode != "stress":
            if stream_id not in self._references:
                self._references[stream_id] = frame.copy()
            return np.clip((frame - self._references[stream_id]) / 255.0, -1.0, 1.0)

        gray = to_gray(frame)
        if stream_id not in self._stress_baselines:
            buffer = self._gray_buffers.setdefault(stream_id, [])
            if len(buffer) < self.baseline_frames:
                buffer.append(gray)
            baseline = compute_stress_baseline(np.stack(buffer))
            if len(buffer) == self.baseline_frames:
                self._stress_baselines[stream_id] = baseline
        else:
            baseline = self._stress_baselines[stream_id]
        stress = stress_map(gray, *baseline, gain=self.gain)
        return np.repeat((stress * 2.0 - 1.0)[..., None], 3, axis=2).astype(np.float32)

    def apply(self, data: dict) -> None:
        if self.mode == "raw" or "batch_images" not in data:
            return

        batch_images = [list(images) for images in data["batch_images"]]
        tactile_masks = []
        for batch_index, images in enumerate(batch_images):
            mask = [False] * len(images)
            for view_index in self.view_indices:
                image = images[view_index]
                if isinstance(image, torch.Tensor):
                    image = np.moveaxis(image.detach().cpu().numpy(), 0, -1)
                images[view_index] = self.process(image, (batch_index, view_index))
                mask[view_index] = True
            tactile_masks.append(mask)
        data["batch_images"] = batch_images
        data["image_is_tactile"] = tactile_masks

    def apply_tactile_tensor(self, images: torch.Tensor):
        if self.mode == "raw":
            return images, None

        images = images.float()
        for batch_index in range(images.shape[0]):
            for local_index, view_index in enumerate(self.view_indices):
                frame = np.moveaxis(images[batch_index, local_index, :, 0].detach().cpu().numpy(), 0, -1)
                processed = self.process(frame, (batch_index, view_index))
                images[batch_index, local_index, :, 0] = torch.from_numpy(processed).permute(2, 0, 1).to(images.device)
        mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
        return images, mask
