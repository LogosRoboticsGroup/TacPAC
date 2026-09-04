from __future__ import annotations

import unittest

import numpy as np
import torch

from starVLA.dataloader.vla.tactile_stress import (
    TactileInferencePreprocessor,
    compute_stress_baseline,
    stress_map,
    stress_view,
    to_gray,
    video_stress_baseline,
)


def _noisy_frames(rng, count, height=32, width=32, level=100.0, noise=3.0):
    return level + rng.uniform(-noise, noise, size=(count, height, width)).astype(np.float32)


class StressBaselineTest(unittest.TestCase):
    def test_baseline_is_median_and_floor_covers_noise(self):
        rng = np.random.default_rng(0)
        gray = _noisy_frames(rng, 16)
        baseline, noise_floor = compute_stress_baseline(gray)
        self.assertEqual(baseline.shape, (32, 32))
        self.assertEqual(noise_floor.shape, (32, 32))
        # uniform +-3 noise: per-pixel median stays within the support, mean error is small
        self.assertLessEqual(np.abs(baseline - 100.0).max(), 3.0)
        self.assertLess(np.abs(baseline - 100.0).mean(), 1.0)
        self.assertTrue(np.all(noise_floor >= 0.0))
        self.assertTrue(np.all(noise_floor <= 6.5))


class StressMapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.rng = rng
        self.baseline, self.noise_floor = compute_stress_baseline(_noisy_frames(rng, 16))

    def test_noise_only_frame_is_all_zero(self):
        frame = _noisy_frames(self.rng, 1)[0]
        stress = stress_map(frame, self.baseline, self.noise_floor)
        self.assertEqual(stress.shape, frame.shape)
        self.assertTrue(np.all(stress == 0.0))

    def test_contact_blob_survives_and_background_stays_zero(self):
        frame = self.baseline.copy()
        frame[12:20, 12:20] += 40.0
        stress = stress_map(frame, self.baseline, self.noise_floor)
        self.assertGreater(stress[16, 16], 0.5)
        self.assertTrue(np.all(stress[:4, :4] == 0.0))
        self.assertLessEqual(stress.max(), 1.0)


class StressViewTest(unittest.TestCase):
    def test_output_range_channels_and_no_contact_is_black(self):
        rng = np.random.default_rng(2)
        # baseline and live frames go through the same uint8 quantization, as in the real pipeline
        gray = _noisy_frames(rng, 16, noise=2.0).astype(np.uint8).astype(np.float32)
        baseline, noise_floor = compute_stress_baseline(gray)
        frames = torch.from_numpy(
            np.repeat(_noisy_frames(rng, 4, noise=2.0)[:, None], 3, axis=1)
        ).to(torch.uint8)
        view = stress_view(frames, baseline, noise_floor)
        self.assertEqual(tuple(view.shape), (4, 3, 32, 32))
        self.assertTrue(torch.equal(view[:, 0], view[:, 1]))
        self.assertTrue(torch.all(view >= -1.0))
        self.assertTrue(torch.all(view <= 1.0))
        self.assertTrue(torch.all(view == -1.0))  # noise-only input -> black


class VideoStressBaselineTest(unittest.TestCase):
    def test_decodes_first_frames_once_per_video(self):
        rng = np.random.default_rng(4)
        calls = []

        def decode_fn(video_path, indexes, indexes_s):
            calls.append((video_path, list(indexes), list(indexes_s)))
            gray = _noisy_frames(rng, len(indexes))
            return torch.from_numpy(np.repeat(gray[:, None], 3, axis=1)).to(torch.uint8)

        cache = {}
        baseline, noise_floor = video_stress_baseline(decode_fn, "a.mp4", 8, cache, fps=30.0)
        self.assertEqual(baseline.shape, (32, 32))
        self.assertEqual(noise_floor.shape, (32, 32))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], list(range(8)))
        self.assertAlmostEqual(calls[0][2][3], 3 / 30.0)

        again = video_stress_baseline(decode_fn, "a.mp4", 8, cache, fps=30.0)
        self.assertEqual(len(calls), 1)  # cached, no re-decode
        self.assertTrue(np.array_equal(again[0], baseline))

    def test_cache_is_capped(self):
        rng = np.random.default_rng(5)

        def decode_fn(video_path, indexes, indexes_s):
            gray = _noisy_frames(rng, len(indexes))
            return torch.from_numpy(np.repeat(gray[:, None], 3, axis=1)).to(torch.uint8)

        cache = {}
        for idx in range(200):
            video_stress_baseline(decode_fn, f"{idx}.mp4", 4, cache, fps=30.0)
        self.assertLessEqual(len(cache), 128)


class TactileInferencePreprocessorTest(unittest.TestCase):
    @staticmethod
    def _raw(rng, height=32, width=32, noise=2.0):
        gray = 100.0 + rng.uniform(-noise, noise, size=(height, width, 1)).astype(np.float32)
        return np.repeat(gray, 3, axis=2).astype(np.uint8)

    def test_raw_mode_is_passthrough(self):
        processor = TactileInferencePreprocessor(mode="raw", view_indices=(1,))
        images = [[np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 42, dtype=np.uint8)]]
        processor.apply({"batch_images": images})
        self.assertTrue(np.all(images[0][1] == 42))

    def test_residual_streams_are_independent_and_resettable(self):
        processor = TactileInferencePreprocessor(mode="residual")
        first_a = np.full((8, 8, 3), 80, dtype=np.uint8)
        first_b = np.full((8, 8, 3), 160, dtype=np.uint8)
        self.assertTrue(np.all(processor.process(first_a, "finger_a") == 0.0))
        self.assertTrue(np.all(processor.process(first_b, "finger_b") == 0.0))

        moved_a = first_a + 20
        moved_b = first_b - 20
        self.assertTrue(np.allclose(processor.process(moved_a, "finger_a"), 20.0 / 255.0))
        self.assertTrue(np.allclose(processor.process(moved_b, "finger_b"), -20.0 / 255.0))

        processor.reset()
        self.assertTrue(np.all(processor.process(moved_a, "finger_a") == 0.0))

    def test_stress_streams_are_independent(self):
        rng = np.random.default_rng(6)
        processor = TactileInferencePreprocessor(mode="stress", gain=10.0, baseline_frames=8)
        for _ in range(10):
            left = processor.process(self._raw(rng), "finger_a")
            right = processor.process(self._raw(rng), "finger_b")
            self.assertEqual(left.shape, (32, 32, 3))
            self.assertEqual(left.dtype, np.float32)
            self.assertTrue(np.all(left == -1.0))
            self.assertTrue(np.all(right == -1.0))

        contact = self._raw(rng).astype(np.float32)
        contact[12:20, 12:20] += 40.0
        left = processor.process(contact.astype(np.uint8), "finger_a")
        right = processor.process(self._raw(rng), "finger_b")
        self.assertGreater(float(left[16, 16, 0]), 0.5)
        self.assertEqual(float(left[0, 0, 0]), -1.0)
        self.assertTrue(np.all(right == -1.0))


class TactileModeConfigTest(unittest.TestCase):
    @staticmethod
    def _server_config(data_mix, **overrides):
        vla_data = {"data_mix": data_mix, **overrides}
        return TactileInferencePreprocessor.from_config({"datasets": {"vla_data": vla_data}})

    def test_training_config_drives_all_tactile_views(self):
        cases = (
            ({}, "residual"),
            ({"tactile_residual_mode": "stress"}, "stress"),
            ({"enable_tactile_residual": False}, "raw"),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                config = self._server_config("flexiv_plug_4views", **overrides)
                self.assertEqual(config.mode, expected)
                self.assertEqual(config.view_indices, (2, 3))

        aloha = self._server_config("aloha_wipe_board")
        self.assertEqual(aloha.view_indices, (3, 4, 5, 6))
        flexiv = self._server_config("flexiv_plug_4views")
        self.assertEqual(flexiv.view_indices, (2, 3))
        tuned = self._server_config(
            "flexiv_plug_4views",
            tactile_residual_mode="stress",
            tactile_stress_gain=4.0,
            tactile_stress_baseline_frames=6,
        )
        self.assertEqual(tuned.gain, 4.0)
        self.assertEqual(tuned.baseline_frames, 6)
        self.assertIsNone(self._server_config("flexiv_plug_2views"))

    def test_base_preprocess_applies_tactile_transform(self):
        import torch.nn as nn

        from starVLA.model.framework.base_framework import baseframework

        class IdentityDataConfig:
            input_transform = staticmethod(lambda data: data)
            normalize_data = staticmethod(lambda data, stats: data)
            pad_data = staticmethod(lambda data: data)

        class StubFramework(baseframework):
            def __init__(self):
                nn.Module.__init__(self)
                self.config = {
                    "datasets": {
                        "vla_data": {
                            "data_mix": "flexiv_plug_4views",
                            "tactile_residual_mode": "signed",
                        }
                    }
                }
                self.data_config = IdentityDataConfig()
                self.norm_stats = {"flexiv_tac": {}}
                self.stat_key = "flexiv_tac"
                self._configure_tactile_preprocessor()

        model = StubFramework()
        metadata = model.get_metadata()
        self.assertEqual(metadata["tactile_mode"], "residual")
        self.assertEqual(metadata["tactile_view_indices"], [2, 3])
        vision = np.full((8, 8, 3), 20, dtype=np.uint8)
        tactile = np.full((8, 8, 3), 100, dtype=np.uint8)
        first = model.preprocess({"batch_images": [[vision.copy(), vision.copy(), tactile.copy(), tactile.copy()]]})
        self.assertEqual(first["image_is_tactile"], [[False, False, True, True]])
        self.assertTrue(np.all(first["batch_images"][0][0] == 20))
        self.assertTrue(np.all(first["batch_images"][0][2] == 0.0))

        moved = tactile + 25
        second = model.preprocess({"batch_images": [[vision, vision, moved, moved]]})
        self.assertTrue(np.allclose(second["batch_images"][0][2], 25.0 / 255.0))

        tactile_only = torch.from_numpy(np.stack([moved, moved])).permute(0, 3, 1, 2)[None, :, :, None]
        tactile_only, tactile_mask = model._tactile_preprocessor.apply_tactile_tensor(tactile_only)
        self.assertTrue(torch.allclose(tactile_only, torch.full_like(tactile_only, 25.0 / 255.0)))
        self.assertTrue(torch.all(tactile_mask))

        model.reset()
        reset = model.preprocess({"batch_images": [[vision, vision, moved, moved]]})
        self.assertTrue(np.all(reset["batch_images"][0][2] == 0.0))

    def test_common_client_keeps_tactile_transport_lossless(self):
        from deployment.model_server.inferencer import M1Inference

        client = M1Inference.__new__(M1Inference)
        client.tactile_view_indices = {2, 3}
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        encoded = client._encode_images([image, image, image, image])
        self.assertTrue(encoded[0].startswith(b"\xff\xd8"))
        self.assertTrue(encoded[1].startswith(b"\xff\xd8"))
        self.assertTrue(encoded[2].startswith(b"\x89PNG"))
        self.assertTrue(encoded[3].startswith(b"\x89PNG"))


class GrayTest(unittest.TestCase):
    def test_bt601_weights(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[..., 0] = 255.0
        gray = to_gray(rgb)
        self.assertEqual(gray.shape, (2, 2))
        self.assertTrue(np.allclose(gray, 255.0 * 0.299, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
