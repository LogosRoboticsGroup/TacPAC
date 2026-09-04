"""Inference-side image fitting must match how the dataset videos were baked.

`convert_flexiv_to_lerobot` stretches tactile frames (a full sensor readout) and
aspect-preserving-resizes + center crops RGB frames; the server has to do the same to the
live camera frames or the model sees a different framing than it trained on.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from deployment.model_server.infersystem_protocol import (
    InferSystemAdapterConfig,
    decode_image_bytes,
    infersystem_to_starvla_payload,
    infersystem_to_tactile_payload,
)
from starVLA.dataloader.vla.data_config.data_config import FlexivTacDataConfig


CAMERAS = ("third_view", "left_wrist_view", "left_tactile", "right_tactile")
SRC_H, SRC_W = 480, 640  # the flexiv cameras' native resolution


def _png(array: np.ndarray) -> bytes:
    return cv2.imencode(".png", array[..., ::-1])[1].tobytes()


def _gradient_frame() -> np.ndarray:
    """Column ramp: the crop is visible as a shifted value range."""
    ramp = np.linspace(0, 255, SRC_W, dtype=np.float32)
    return np.repeat(np.tile(ramp, (SRC_H, 1))[..., None], 3, axis=2).astype(np.uint8)


def _config(**kwargs) -> InferSystemAdapterConfig:
    base = dict(camera_order=CAMERAS, image_resize_size=(256, 256), n_tactile_views=2)
    base.update(kwargs)
    return InferSystemAdapterConfig(**base)


class TrainingParityTest(unittest.TestCase):
    """The served frame must be the frame training's transform would have produced. The
    conversion script bakes the same geometry from `starVLA.dataloader.vla.image_fit`."""

    def _training_view(self, frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        config = FlexivTacDataConfig()
        config.set_image_size(image_size)
        return config.resize_image(frame).permute(1, 2, 0).numpy()

    def test_crop_mode_matches_pixel_transforms_resize(self):
        for image_size in [(224, 224), (256, 256)]:
            frame = _gradient_frame()
            served = decode_image_bytes(_png(frame), resize_size=image_size, resize_mode="crop")
            trained = self._training_view(frame, image_size)
            self.assertEqual(served.shape, trained.shape, image_size)
            # PIL vs torchvision bilinear differ slightly; the framing must not.
            self.assertLess(
                float(np.abs(served.astype(np.float32) - trained.astype(np.float32)).mean()),
                2.0,
                image_size,
            )

    def test_stretch_mode_does_not_match_it(self):
        frame = _gradient_frame()
        served = decode_image_bytes(_png(frame), resize_size=(256, 256), resize_mode="stretch")
        trained = self._training_view(frame, (256, 256))
        self.assertGreater(
            float(np.abs(served.astype(np.float32) - trained.astype(np.float32)).mean()), 10.0
        )


class DecodeModeTest(unittest.TestCase):
    def test_stretch_keeps_the_whole_frame(self):
        out = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="stretch")
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertLess(int(out[0, 0, 0]), 5)  # left edge survives
        self.assertGreater(int(out[0, -1, 0]), 250)  # right edge survives

    def test_crop_drops_the_sides_and_keeps_the_aspect_ratio(self):
        out = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="crop")
        self.assertEqual(out.shape, (256, 256, 3))
        # 640x480 -> short side to 256 (341x256) -> center crop 256 wide: ~12.5% off each side.
        self.assertGreater(int(out[0, 0, 0]), 20)
        self.assertLess(int(out[0, -1, 0]), 235)

    def test_unknown_mode_raises(self):
        with self.assertRaises(Exception):
            decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="fit")


class ModalityRoutingTest(unittest.TestCase):
    def _request(self) -> dict:
        msg = {"cmd": "predict", "state": [0.0] * 8}
        for name in CAMERAS:
            msg[name] = _png(_gradient_frame())
        return msg

    def test_rgb_is_cropped_and_tactile_is_stretched(self):
        images = infersystem_to_starvla_payload(self._request(), _config())["batch_images"][0]
        stretched = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="stretch")
        cropped = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="crop")

        for index in (0, 1):  # vision
            np.testing.assert_array_equal(images[index], cropped)
        for index in (2, 3):  # tactile
            np.testing.assert_array_equal(images[index], stretched)

    def test_tactile_tick_payload_stretches_too(self):
        payload = infersystem_to_tactile_payload(self._request(), _config())
        stretched = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="stretch")
        images = payload["batch_tactile_images"][0]
        self.assertEqual(len(images), 2)
        for image in images:
            np.testing.assert_array_equal(image, stretched)

    def test_legacy_stretch_mode_restores_the_old_behaviour(self):
        """Checkpoints trained on stretch-baked datasets keep their framing."""
        images = infersystem_to_starvla_payload(self._request(), _config(rgb_resize_mode="stretch"))["batch_images"][0]
        stretched = decode_image_bytes(_png(_gradient_frame()), resize_size=(256, 256), resize_mode="stretch")
        for image in images:
            np.testing.assert_array_equal(image, stretched)

    def test_no_tactile_views_leaves_every_camera_on_the_rgb_path(self):
        config = _config(camera_order=("third_view", "left_wrist_view"), n_tactile_views=0)
        self.assertEqual(config.tactile_cameras, frozenset())
        self.assertEqual(config.resize_mode_for("third_view"), "crop")

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            _config(rgb_resize_mode="fit")
        with self.assertRaises(ValueError):
            _config(n_tactile_views=9)


if __name__ == "__main__":
    unittest.main()
