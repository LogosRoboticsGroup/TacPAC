from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.WM4A.WanMoT import WanMoT


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


def _make_model(n_vision=2, n_tactile=2, view=(256, 256), tactile=(128, 128)) -> WanMoT:
    model = WanMoT.__new__(WanMoT)
    nn.Module.__init__(model)
    model.vae = FakeVAE()
    model.concat_multi_camera = "horizontal"
    model.view_image_size = view
    model.tactile_image_size = tactile
    model.n_vision_views = n_vision
    model.n_tactile_views = n_tactile
    model.image_size = model._horizontal_canvas_size()
    return model


def _view_images(values, height=256, width=256, frames=1):
    views = [torch.full((1, 3, frames, height, width), float(v)) for v in values]
    return torch.stack(views, dim=1)


class CanvasSizeTest(unittest.TestCase):
    def test_mixed_canvas_size(self):
        model = _make_model()
        self.assertEqual(model.image_size, (256, 2 * 256 + 128))

    def test_legacy_canvas_size(self):
        model = _make_model(tactile=None)
        self.assertEqual(model.image_size, (256, 4 * 256))

    def test_equal_height_gives_one_view_per_column(self):
        model = _make_model(tactile=(256, 256))
        self.assertEqual(model.image_size, (256, 4 * 256))

    def test_quarter_height_stacks_four_per_column(self):
        model = _make_model(n_tactile=4, tactile=(64, 128))
        self.assertEqual(model.image_size, (256, 2 * 256 + 128))

    def test_tactile_count_must_fill_columns(self):
        with self.assertRaises(ValueError):
            _make_model(n_vision=3, n_tactile=1)

    def test_tactile_height_must_divide_view_height(self):
        with self.assertRaises(ValueError):
            _make_model(tactile=(96, 128))


class ConcatLayoutTest(unittest.TestCase):
    def test_tactile_pair_stacks_vertically_after_vision(self):
        model = _make_model()
        canvas = model._concat_views(_view_images([0.0, 1.0, 2.0, 3.0]), None)
        self.assertEqual(tuple(canvas.shape), (1, 3, 1, 256, 640))
        self.assertTrue(torch.all(canvas[..., :, :256] == 0.0))
        self.assertTrue(torch.all(canvas[..., :, 256:512] == 1.0))
        self.assertTrue(torch.all(canvas[..., :128, 512:] == 2.0))  # left tactile on top
        self.assertTrue(torch.all(canvas[..., 128:, 512:] == 3.0))  # right tactile below

    def test_equal_height_concat_matches_legacy(self):
        images = _view_images([0.0, 1.0, 2.0, 3.0])
        mixed = _make_model(tactile=(256, 256))._concat_views(images, None)
        legacy = _make_model(tactile=None)._concat_views(images, None)
        self.assertTrue(torch.equal(mixed, legacy))

    def test_legacy_concat_unchanged(self):
        model = _make_model(tactile=None)
        canvas = model._concat_views(_view_images([0.0, 1.0, 2.0, 3.0]), None)
        self.assertEqual(tuple(canvas.shape), (1, 3, 1, 256, 1024))
        self.assertTrue(torch.all(canvas[..., 512:768] == 2.0))


class TactileExpertCanvasTest(unittest.TestCase):
    """The tactile expert encodes `tactile_now` on its own canvas, then reads cached K/V for
    the tactile slice of the main canvas. The two canvases must agree pixel-for-pixel or the
    token grids disagree (TactileExpert.forward rejects the mismatch)."""

    def _assert_matches_main_canvas(self, model, values):
        images = _view_images(values)
        main = model._concat_views(images, None)
        split_w = model._modality_split_width(main.shape[-1])
        tactile = model._tactile_canvas(images[:, model.n_vision_views :])
        self.assertTrue(torch.equal(tactile, main[..., split_w:]))

    def test_mixed_size(self):
        self._assert_matches_main_canvas(_make_model(), [0.0, 1.0, 2.0, 3.0])

    def test_equal_size(self):
        self._assert_matches_main_canvas(_make_model(tactile=(256, 256)), [0.0, 1.0, 2.0, 3.0])

    def test_four_per_column(self):
        model = _make_model(n_tactile=4, tactile=(64, 128))
        self._assert_matches_main_canvas(model, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_legacy(self):
        self._assert_matches_main_canvas(_make_model(tactile=None), [0.0, 1.0, 2.0, 3.0])


class SplitEncodeDecodeTest(unittest.TestCase):
    def test_encode_splits_at_actual_vision_width(self):
        model = _make_model()
        canvas = model._concat_views(_view_images([0.0, 1.0, 2.0, 3.0]), None)
        latents = model._encode_video_latents(canvas)
        self.assertEqual(model.vae.encode_widths, [512, 128])
        self.assertEqual(latents.shape[-1], 80)

    def test_decode_splits_latents_proportionally(self):
        model = _make_model()
        model._decode_video_latents(torch.zeros(1, 4, 1, 32, 80))
        self.assertEqual(model.vae.decode_widths, [64, 16])

    def test_legacy_split_by_view_count(self):
        model = _make_model(tactile=None)
        canvas = model._concat_views(_view_images([0.0, 1.0, 2.0, 3.0]), None)
        model._encode_video_latents(canvas)
        self.assertEqual(model.vae.encode_widths, [512, 512])


if __name__ == "__main__":
    unittest.main()
