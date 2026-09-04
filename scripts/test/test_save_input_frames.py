from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from deployment.model_server.infersystem_protocol import InferSystemAdapterConfig
from deployment.model_server.server_infersystem import InferSystemPolicyServer


def _png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image[..., ::-1])[1].tobytes()  # cv2 wants BGR


class SaveInputFramesTest(unittest.TestCase):
    def test_input_model_png_shows_transformed_payload_images(self):
        server = InferSystemPolicyServer.__new__(InferSystemPolicyServer)
        server._adapter_config = InferSystemAdapterConfig(
            camera_order=("cam_view", "cam_tactile"),
            image_resize_size=(32, 32),
        )
        server._save_generated_video_every = 1

        raw_view = np.full((32, 32, 3), 60, dtype=np.uint8)
        raw_tactile = np.full((32, 32, 3), 200, dtype=np.uint8)
        msg = {"cmd": "predict", "cam_view": _png(raw_view), "cam_tactile": _png(raw_tactile)}
        # what the model actually receives: tactile already transformed (e.g. residual-encoded)
        model_images = [raw_view, np.full((32, 32, 3), 128, dtype=np.uint8)]

        with tempfile.TemporaryDirectory() as tmp:
            server._save_generated_video_dir = Path(tmp)
            paths = server._maybe_save_input_frames(msg, "stem", 1, model_images)
            saved = imageio.imread(paths["model"])

        self.assertTrue(np.all(saved[:, :32] == 60))  # vision half untouched
        self.assertTrue(np.all(saved[:, 32:] == 128))  # tactile half is the model input, not raw


if __name__ == "__main__":
    unittest.main()
