from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.WM4A.WanMoT import WanMoT, modality_view_counts


class FakeVAE(nn.Module):
    """Records per-call input widths; downsamples spatially by 8 like the real VAE."""

    def __init__(self):
        super().__init__()
        self.encode_widths = []
        self.decode_widths = []

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        self.encode_widths.append(video.shape[-1])
        return F.avg_pool3d(video, kernel_size=(1, 8, 8))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_widths.append(latents.shape[-1])
        return F.interpolate(latents, scale_factor=(1, 8, 8), mode="nearest")


def _make_model(n_vision: int, n_tactile: int) -> WanMoT:
    model = WanMoT.__new__(WanMoT)
    nn.Module.__init__(model)
    model.vae = FakeVAE()
    model.n_vision_views = n_vision
    model.n_tactile_views = n_tactile
    model.tactile_image_size = None
    return model


class ModalityViewCountsTest(unittest.TestCase):
    def test_flexiv_tac_keys(self):
        keys = [
            "observation.images.third_view",
            "observation.images.left_wrist_view",
            "observation.images.left_wrist_left_tactile",
            "observation.images.left_wrist_right_tactile",
        ]
        self.assertEqual(modality_view_counts(keys), (2, 2))

    def test_notac_keys(self):
        keys = ["observation.images.third_view", "observation.images.left_wrist_view"]
        self.assertEqual(modality_view_counts(keys), (2, 0))

    def test_aloha_uneven_split(self):
        keys = [
            "observation.image.third_view",
            "observation.image.left_wrist_view",
            "observation.image.right_wrist_view",
            "observation.image.left_wrist_left_tactile",
            "observation.image.left_wrist_right_tactile",
            "observation.image.right_wrist_left_tactile",
            "observation.image.right_wrist_right_tactile",
        ]
        self.assertEqual(modality_view_counts(keys), (3, 4))

    def test_tactile_not_suffix_raises(self):
        keys = [
            "observation.images.left_tactile",
            "observation.images.third_view",
        ]
        with self.assertRaises(ValueError):
            modality_view_counts(keys)


class SplitEncodeTest(unittest.TestCase):
    def test_encode_splits_canvas_into_two_vae_passes(self):
        model = _make_model(n_vision=2, n_tactile=2)
        video = torch.randn(1, 3, 5, 32, 128)  # 4 views, each 32 wide
        latents = model._encode_video_latents(video)
        self.assertEqual(model.vae.encode_widths, [64, 64])
        self.assertEqual(tuple(latents.shape), (1, 3, 5, 4, 16))

    def test_encode_uneven_split(self):
        model = _make_model(n_vision=3, n_tactile=4)
        video = torch.randn(1, 3, 5, 32, 224)  # 7 views, each 32 wide
        latents = model._encode_video_latents(video)
        self.assertEqual(model.vae.encode_widths, [96, 128])
        self.assertEqual(tuple(latents.shape), (1, 3, 5, 4, 28))

    def test_encode_no_tactile_single_pass(self):
        model = _make_model(n_vision=2, n_tactile=0)
        video = torch.randn(1, 3, 5, 32, 64)
        latents = model._encode_video_latents(video)
        self.assertEqual(model.vae.encode_widths, [64])
        self.assertEqual(tuple(latents.shape), (1, 3, 5, 4, 8))

    def test_encode_halves_concat_in_order(self):
        model = _make_model(n_vision=2, n_tactile=2)
        video = torch.zeros(1, 3, 1, 8, 32)
        video[..., :16] = 1.0  # vision half all ones, tactile half zeros
        latents = model._encode_video_latents(video)
        self.assertTrue(torch.all(latents[..., :2] == 1.0))
        self.assertTrue(torch.all(latents[..., 2:] == 0.0))


class SplitDecodeTest(unittest.TestCase):
    def test_decode_splits_latents_into_two_vae_passes(self):
        model = _make_model(n_vision=2, n_tactile=2)
        latents = torch.randn(1, 3, 5, 4, 16)
        video = model._decode_video_latents(latents)
        self.assertEqual(model.vae.decode_widths, [8, 8])
        self.assertEqual(tuple(video.shape), (1, 3, 5, 32, 128))

    def test_decode_no_tactile_single_pass(self):
        model = _make_model(n_vision=2, n_tactile=0)
        latents = torch.randn(1, 3, 5, 4, 8)
        video = model._decode_video_latents(latents)
        self.assertEqual(model.vae.decode_widths, [8])
        self.assertEqual(tuple(video.shape), (1, 3, 5, 32, 64))


if __name__ == "__main__":
    unittest.main()
