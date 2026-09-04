from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from starVLA.model.framework.WM4A.WanMoT import WanMoT
from starVLA.model.modules.wan_mot.wan_video_expert import WanVideoExpert


HIDDEN, HEADS, HEAD_DIM, LAYERS = 64, 4, 16, 2
TEXT_DIM, FREQ_DIM, Z_DIM = 32, 32, 4


def _make_video_expert() -> WanVideoExpert:
    return WanVideoExpert(
        hidden_dim=HIDDEN,
        in_dim=Z_DIM,
        ffn_dim=128,
        out_dim=Z_DIM,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
    )


def _make_framework(n_vision: int, n_tactile: int, tactile_image_size=None) -> WanMoT:
    model = WanMoT.__new__(WanMoT)
    nn.Module.__init__(model)
    model.n_vision_views, model.n_tactile_views = n_vision, n_tactile
    model.view_image_size = (256, 256)
    model.tactile_image_size = tactile_image_size
    model.image_size = model._horizontal_canvas_size()
    model.video_expert = _make_video_expert()
    return model


class ModalitySplitGeometryTest(unittest.TestCase):
    def test_equal_size_views_split_by_view_count(self):
        model = _make_framework(n_vision=2, n_tactile=2)
        self.assertEqual(model._modality_split_geometry(), (2, 4))
        self.assertEqual(model._modality_split_width(20), 10)

    def test_mixed_size_views_split_by_pixel_width(self):
        # visual 256 x2 + tactile 128 pair column -> canvas 640, vision owns 512.
        model = _make_framework(n_vision=2, n_tactile=2, tactile_image_size=(128, 128))
        self.assertEqual(model._modality_split_geometry(), (512, 640))
        self.assertEqual(model._modality_split_width(20), 16)

    def test_expert_seam_matches_vae_seam(self):
        model = _make_framework(n_vision=2, n_tactile=2, tactile_image_size=(128, 128))
        model.video_expert.set_modality_split(*model._modality_split_geometry())
        latent_w = model.image_size[1] // 16  # VAE spatial stride
        self.assertEqual(
            model.video_expert.tactile_split_width(latent_w),
            model._modality_split_width(latent_w),
        )


class PatchifyRoutingTest(unittest.TestCase):
    def _latents(self, width: int) -> torch.Tensor:
        return torch.randn(2, Z_DIM, 3, 8, width)

    def test_no_split_uses_visual_patchifier_only(self):
        expert = _make_video_expert()
        latents = self._latents(20)
        tokens = expert.patchify(latents)
        self.assertTrue(torch.equal(tokens, expert.patch_embedding(latents)))
        self.assertEqual(tuple(tokens.shape), (2, HIDDEN, 3, 4, 10))

    def test_split_routes_each_half_to_its_own_patchifier(self):
        expert = _make_video_expert()
        expert.set_modality_split(512, 640)  # 4/5 of the canvas is vision
        latents = self._latents(20)
        tokens = expert.patchify(latents)
        self.assertEqual(tuple(tokens.shape), (2, HIDDEN, 3, 4, 10))
        self.assertTrue(torch.equal(tokens[..., :8], expert.patch_embedding(latents[..., :16])))
        self.assertTrue(torch.equal(tokens[..., 8:], expert.patch_embedding_tactile(latents[..., 16:])))

    def test_warm_start_reproduces_single_patchifier_output(self):
        expert = _make_video_expert()
        latents = self._latents(20)
        single = expert.patchify(latents)
        expert.set_modality_split(512, 640)
        expert.init_tactile_patch_from_visual()
        self.assertTrue(torch.allclose(expert.patchify(latents), single))

    def test_misaligned_seam_raises(self):
        expert = _make_video_expert()
        expert.set_modality_split(1, 4)  # latent seam at column 5, patch width 2
        with self.assertRaises(ValueError):
            expert.patchify(self._latents(20))

    def test_gradients_reach_both_patchifiers(self):
        expert = _make_video_expert()
        expert.set_modality_split(512, 640)
        expert.patchify(self._latents(20)).square().mean().backward()
        for patch in (expert.patch_embedding, expert.patch_embedding_tactile):
            self.assertIsNotNone(patch.weight.grad)
            self.assertTrue(torch.any(patch.weight.grad != 0))


class LegacyCheckpointLoadTest(unittest.TestCase):
    """A checkpoint trained with one patchifier must warm-start on every load path — the
    training entry (`pretrained_checkpoint`) calls plain `load_state_dict`, with none of the
    framework's `_after_load_state_dict` hooks."""

    def _legacy_state_dict(self, expert: WanVideoExpert) -> dict:
        return {key: value for key, value in expert.state_dict().items() if "patch_embedding_tactile" not in key}

    def test_bare_load_state_dict_fills_the_tactile_patchifier(self):
        trained = _make_video_expert()
        nn.init.normal_(trained.patch_embedding.weight, std=0.5)
        fresh = _make_video_expert()

        missing, _unexpected = fresh.load_state_dict(self._legacy_state_dict(trained), strict=False)

        self.assertTrue(torch.equal(fresh.patch_embedding_tactile.weight, trained.patch_embedding.weight))
        self.assertFalse([key for key in missing if "patch_embedding_tactile" in key])

    def test_nested_load_through_the_parent_module_also_warm_starts(self):
        trained = _make_video_expert()
        nn.init.normal_(trained.patch_embedding.weight, std=0.5)
        parent = nn.Module()
        parent.video = _make_video_expert()

        legacy = {f"video.{key}": value for key, value in self._legacy_state_dict(trained).items()}
        parent.load_state_dict(legacy, strict=False)

        self.assertTrue(torch.equal(parent.video.patch_embedding_tactile.weight, trained.patch_embedding.weight))

    def test_checkpoint_with_a_tactile_patchifier_is_left_alone(self):
        trained = _make_video_expert()
        nn.init.normal_(trained.patch_embedding_tactile.weight, std=0.5)
        fresh = _make_video_expert()

        fresh.load_state_dict(trained.state_dict(), strict=True)

        self.assertTrue(torch.equal(fresh.patch_embedding_tactile.weight, trained.patch_embedding_tactile.weight))
        self.assertFalse(torch.equal(fresh.patch_embedding_tactile.weight, fresh.patch_embedding.weight))


if __name__ == "__main__":
    unittest.main()
