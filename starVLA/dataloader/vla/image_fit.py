"""How a camera frame is fit to the model's square input.

Two modes, and which one a camera gets is a property of the modality:

  stretch — squash the whole frame into the target. Tactile frames are a full sensor
            readout rather than a composition, so cropping would cut away part of the
            contact surface.
  crop    — aspect-preserving resize + center crop, matching training's
            `BaseDataConfig.pixel_transforms_resize` (`Resize(min)` then `CenterCrop`).

Dataset conversion (`scripts/data/convert_flexiv_to_lerobot`) and the deployment server
(`deployment/model_server/infersystem_protocol`) both fit frames, at different points in
the pipeline and with different image libraries. They share the geometry from here so a
model cannot end up trained on one framing and served the other.

Deliberately dependency-free (plain arithmetic) so both sides can import it.
"""

from __future__ import annotations

RESIZE_MODES = ("stretch", "crop")


def resize_crop_geometry(
    src_hw: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return (resize_h, resize_w, top, left) for an aspect-preserving resize + center crop.

    The short side is scaled to the target and the excess cropped away. For a square
    `image_size` the scale is exactly `min(image_size) / min(src_hw)`, so it lands within
    1px of torchvision's `Resize(min(image_size))`; for a non-square one it covers the
    target instead of zero-padding it.
    """
    src_h, src_w = src_hw
    target_h, target_w = image_size
    scale = max(target_h / src_h, target_w / src_w)
    resize_h = max(target_h, round(src_h * scale))
    resize_w = max(target_w, round(src_w * scale))
    top = int(round((resize_h - target_h) / 2.0))
    left = int(round((resize_w - target_w) / 2.0))
    return resize_h, resize_w, top, left
