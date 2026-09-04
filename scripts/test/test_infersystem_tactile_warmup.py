from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from deployment.model_server.infersystem_protocol import InferSystemAdapterConfig
from deployment.model_server.server_infersystem import InferSystemPolicyServer
from starVLA.dataloader.vla.tactile_stress import TactileInferencePreprocessor


def _png(gray_level, rng, noise=2.0, size=32):
    gray = gray_level + rng.uniform(-noise, noise, (size, size, 1)).astype(np.float32)
    image = np.clip(np.repeat(gray, 3, axis=2), 0, 255).astype(np.uint8)
    return cv2.imencode(".png", image[..., ::-1])[1].tobytes()


class SeedTactileWarmupTest(unittest.TestCase):
    def _server(self, preprocessor):
        server = InferSystemPolicyServer.__new__(InferSystemPolicyServer)
        server._adapter_config = InferSystemAdapterConfig(
            camera_order=("cam_view", "cam_tactile"),
            image_resize_size=(32, 32),
        )
        server._policy = SimpleNamespace(_tactile_preprocessor=preprocessor)
        return server

    def test_warmup_freezes_stress_baseline_at_view_index(self):
        rng = np.random.default_rng(0)
        preprocessor = TactileInferencePreprocessor(mode="stress", baseline_frames=8, view_indices=(1,))
        server = self._server(preprocessor)

        msg = {"cmd": "reset", "tactile_warmup": {"cam_tactile": [_png(100, rng) for _ in range(8)]}}
        counts = server._seed_tactile_warmup(msg)

        self.assertEqual(counts, {"cam_tactile": 8})
        self.assertIn((0, 1), preprocessor._stress_baselines)  # frozen for apply()'s stream id

    def test_warmup_seeds_residual_reference(self):
        rng = np.random.default_rng(1)
        preprocessor = TactileInferencePreprocessor(mode="residual", view_indices=(1,))
        server = self._server(preprocessor)

        server._seed_tactile_warmup({"tactile_warmup": {"cam_tactile": [_png(100, rng)]}})
        self.assertIn((0, 1), preprocessor._references)

    def test_no_preprocessor_or_no_warmup_is_noop(self):
        server = self._server(None)
        self.assertEqual(server._seed_tactile_warmup({"tactile_warmup": {}}), {})
        self.assertEqual(server._seed_tactile_warmup({}), {})


if __name__ == "__main__":
    unittest.main()
