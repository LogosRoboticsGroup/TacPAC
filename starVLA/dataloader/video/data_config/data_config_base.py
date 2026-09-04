from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms


class VideoBaseDataConfig:
    video_keys: Sequence[str] | None = None
    image_size = (256, 256)
    resize_method = "cover_crop"
    _pixel_transforms_resize = None

    def set_image_size(self, image_size):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self._pixel_transforms_resize = None

    @property
    def pixel_transforms_resize(self):
        if self._pixel_transforms_resize is None:
            if self.resize_method in {"cover_crop", "center_crop_resize"}:
                self._pixel_transforms_resize = transforms.CenterCrop(self.image_size)
            elif self.resize_method == "resize":
                self._pixel_transforms_resize = transforms.Resize(self.image_size)
            else:
                raise NotImplementedError(f"Unknown resize method: {self.resize_method}")
        return self._pixel_transforms_resize

    def resolve_video_keys(self, discovered_keys: Iterable[str] | None, override_keys: Sequence[str] | None = None) -> List[str]:
        discovered = [key for key in (discovered_keys or []) if key is not None]
        override = [key for key in (override_keys or []) if key is not None]
        defaults = [key for key in (self.video_keys or []) if key is not None]

        if discovered:
            if override:
                resolved = [key for key in override if key in discovered]
                if resolved:
                    return resolved
            if defaults:
                preferred = [key for key in defaults if key in discovered]
                if preferred:
                    remainder = [key for key in discovered if key not in preferred]
                    return preferred + remainder
            return discovered

        if override:
            return override
        return defaults

    def get_decode_resize_shape(self, height: int, width: int) -> tuple[int, int]:
        target_h, target_w = self.image_size
        if self.resize_method in {"cover_crop", "center_crop_resize"}:
            scale = max(target_h / float(height), target_w / float(width))
            return max(1, int(round(height * scale))), max(1, int(round(width * scale)))
        if self.resize_method == "resize":
            return target_h, target_w
        raise NotImplementedError(f"Unknown resize method: {self.resize_method}")

    def format_video(self, video) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            if hasattr(video, "asnumpy"):
                video = torch.from_numpy(video.asnumpy())
            else:
                video = torch.from_numpy(np.asarray(video))

        if video.ndim == 4 and video.shape[-1] in {1, 3, 4}:
            video = video.permute(3, 0, 1, 2)
        elif video.ndim == 4 and video.shape[1] in {1, 3, 4}:
            video = video.permute(1, 0, 2, 3)
        elif video.ndim != 4:
            raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")

        return self.pixel_transforms_resize(video).float()
