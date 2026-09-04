from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.WM4A.WanMoTJointTacExpert import WanMoTJointTacExpert
from starVLA.model.modules.wan_mot.action_expert import ActionExpert
from starVLA.model.modules.wan_mot.mot import MoT
from starVLA.model.modules.wan_mot.tactile_expert import TactileExpert
from starVLA.model.modules.wan_mot.wan_video_expert import WanVideoExpert


HIDDEN, HEADS, HEAD_DIM, LAYERS = 64, 4, 16, 2
ACTION_DIM, TEXT_DIM, FREQ_DIM, Z_DIM = 6, 32, 32, 4
HORIZON = 5


def _make_tactile_expert(patch_dim: int = HIDDEN) -> TactileExpert:
    return TactileExpert(
        hidden_dim=HIDDEN,
        action_dim=ACTION_DIM,
        patch_dim=patch_dim,
        ffn_dim=128,
        freq_dim=FREQ_DIM,
        eps=1e-6,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
    )


def _make_action_expert() -> ActionExpert:
    return ActionExpert(
        hidden_dim=HIDDEN,
        action_dim=ACTION_DIM,
        ffn_dim=128,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        eps=1e-6,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
    )


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


def _make_framework() -> WanMoTJointTacExpert:
    """Bare instance with just the pieces `_prefill_tac_cache`/`_run_tactile_expert` touch."""
    model = WanMoTJointTacExpert.__new__(WanMoTJointTacExpert)
    nn.Module.__init__(model)
    model.video_expert = _make_video_expert()
    model.action_expert = _make_action_expert()
    model.mot = MoT(
        mixtures={"video": model.video_expert, "action": model.action_expert},
        enable_gradient_checkpointing=False,
    )
    model.tactile_expert = _make_tactile_expert()
    model.n_vision_views = 1
    model.n_tactile_views = 1
    model.tactile_image_size = None
    model.video_expert.set_modality_split(*model._modality_split_geometry())
    model.steps_per_latent_frame = 4
    model.vae = nn.Identity()  # WanMoT.train() touches vae/text_encoder
    model.text_encoder = None
    model.eval()
    return model


def _fake_kv_cache(batch_size: int, kv_len: int) -> list[dict[str, torch.Tensor]]:
    attn_dim = HEADS * HEAD_DIM
    return [
        {"k": torch.randn(batch_size, kv_len, attn_dim), "v": torch.randn(batch_size, kv_len, attn_dim)}
        for _ in range(LAYERS)
    ]


class TactileExpertModuleTest(unittest.TestCase):
    def _forward(self, expert, batch_size=2, patch_dim=HIDDEN):
        tactile_tokens = torch.randn(batch_size, 4, patch_dim)  # 2x2 tactile token grid
        actions = torch.randn(batch_size, HORIZON, ACTION_DIM)
        offset = torch.tensor([0, 3][:batch_size])
        executed = torch.arange(HORIZON).view(1, -1) < offset.view(-1, 1)
        frame_ids = torch.tensor([1, 1][:batch_size])
        return expert(
            tactile_tokens=tactile_tokens,
            actions=actions,
            executed_mask=executed,
            offset=offset,
            frame_ids=frame_ids,
            tac_grid=(2, 2, 2),
            kv_cache=_fake_kv_cache(batch_size, kv_len=13),
        )

    def test_zero_init_head_outputs_zero_delta(self):
        delta = self._forward(_make_tactile_expert())
        self.assertEqual(tuple(delta.shape), (2, HORIZON, ACTION_DIM))
        self.assertTrue(torch.all(delta == 0))

    def test_gradients_reach_all_trainable_parts(self):
        expert = _make_tactile_expert(patch_dim=2 * HIDDEN)
        nn.init.normal_(expert.head.weight, std=0.02)
        delta = self._forward(expert, patch_dim=2 * HIDDEN)
        delta.square().mean().backward()
        for name in ["patch_proj.weight", "action_encoder.weight", "head.weight", "blocks.0.self_attn.q.weight"]:
            param = dict(expert.named_parameters())[name]
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.any(param.grad != 0), name)

    def test_patch_proj_is_identity_when_dims_match(self):
        self.assertIsInstance(_make_tactile_expert().patch_proj, nn.Identity)
        self.assertIsInstance(_make_tactile_expert(patch_dim=2 * HIDDEN).patch_proj, nn.Linear)

    def test_build_freqs_frame_band_varies_spatial_band_shared(self):
        expert = _make_tactile_expert()
        freqs = expert.build_freqs(torch.tensor([0, 2]), tac_grid=(2, 2, 2), action_len=HORIZON, device="cpu")
        self.assertEqual(tuple(freqs.shape), (2, 4 + HORIZON, 1, HEAD_DIM // 2))
        self.assertTrue(freqs.is_complex())
        f_half = HEAD_DIM // 2 - 2 * (2 * (HEAD_DIM // 6) // 2)  # f-band complex width
        tac = freqs[:, :4, 0]
        self.assertFalse(torch.allclose(tac[0, :, :f_half], tac[1, :, :f_half]))  # frame band differs
        self.assertTrue(torch.allclose(tac[0, :, f_half:], tac[1, :, f_half:]))  # spatial band shared
        self.assertTrue(torch.allclose(freqs[0, 4:], freqs[1, 4:]))  # action band shared


class InitFromActionExpertTest(unittest.TestCase):
    def test_copies_blocks_and_encoders_keeps_zero_head(self):
        action_expert = _make_action_expert()
        tac = _make_tactile_expert()
        tac.init_from_action_expert(action_expert)
        for layer_idx in range(LAYERS):
            self.assertTrue(
                torch.equal(
                    tac.blocks[layer_idx].self_attn.q.weight,
                    action_expert.blocks[layer_idx].self_attn.q.weight,
                )
            )
            self.assertTrue(
                torch.equal(tac.blocks[layer_idx].ffn[0].weight, action_expert.blocks[layer_idx].ffn[0].weight)
            )
            self.assertTrue(
                torch.equal(tac.blocks[layer_idx].modulation, action_expert.blocks[layer_idx].modulation)
            )
        self.assertTrue(torch.equal(tac.action_encoder.weight, action_expert.action_encoder.weight))
        self.assertTrue(torch.all(tac.head.weight == 0))
        self.assertTrue(torch.all(tac.executed_embedding.weight == 0))


class TacTokenGeometryTest(unittest.TestCase):
    def test_flexiv_tac_layout(self):
        model = WanMoTJointTacExpert.__new__(WanMoTJointTacExpert)
        model.n_vision_views, model.n_tactile_views = 2, 2
        model.tactile_image_size = None
        latent_frames, grid_h, w_start, w_tac = model._tac_token_geometry((3, 7, 28))
        self.assertEqual((latent_frames, grid_h, w_start, w_tac), (3, 7, 14, 14))
        idx = WanMoTJointTacExpert._tac_token_index(3, 7, 28, 14, 14, torch.device("cpu"))
        self.assertEqual(idx.numel(), 3 * 7 * 14)
        self.assertTrue(torch.equal(idx[:14], torch.arange(14, 28)))  # frame 0, row 0
        self.assertTrue(torch.equal(idx[14:28], torch.arange(42, 56)))  # frame 0, row 1
        self.assertEqual(int(idx[7 * 14]), 196 + 14)  # frame 1 starts one full frame later

    def test_mixed_view_sizes_split_by_actual_width(self):
        # visual 256x256 x2 + tactile 128x128 pair column -> canvas 640, token grid 20:
        # vision occupies 16 columns, tactile 4 (NOT the per-view-count split of 10/10).
        model = WanMoTJointTacExpert.__new__(WanMoTJointTacExpert)
        model.n_vision_views, model.n_tactile_views = 2, 2
        model.view_image_size = (256, 256)
        model.tactile_image_size = (128, 128)
        model.image_size = (256, 640)
        latent_frames, grid_h, w_start, w_tac = model._tac_token_geometry((3, 8, 20))
        self.assertEqual((latent_frames, grid_h, w_start, w_tac), (3, 8, 16, 4))


class TactileNowCanvasTest(unittest.TestCase):
    """The fresh `tactile_now` tokens and the cached tactile slice must count the same, or
    TactileExpert.forward rejects them. `tactile_now` arrives at `view_image_size`."""

    DOWNSAMPLE, PATCH = 16, 2  # VAE spatial stride, DiT patch size

    def _token_counts(self, view, tactile, n_tactile=2):
        model = WanMoTJointTacExpert.__new__(WanMoTJointTacExpert)
        model.n_vision_views, model.n_tactile_views = 2, n_tactile
        model.view_image_size, model.tactile_image_size = view, tactile
        model.image_size = model._horizontal_canvas_size()
        grid = tuple(size // self.DOWNSAMPLE // self.PATCH for size in model.image_size)
        _, grid_h, _, w_tac = model._tac_token_geometry((3,) + grid)

        canvas = model._tactile_now_canvas(torch.zeros(1, 2, n_tactile, 3, *view))
        fresh = [size // self.DOWNSAMPLE // self.PATCH for size in canvas.shape[-2:]]
        return grid_h * w_tac, fresh[0] * fresh[1]

    def test_stacked_pair(self):
        self.assertEqual(*self._token_counts((256, 256), (128, 128)))

    def test_equal_size_one_per_column(self):
        self.assertEqual(*self._token_counts((256, 256), (256, 256)))

    def test_four_per_column(self):
        self.assertEqual(*self._token_counts((256, 256), (64, 128), n_tactile=4))

    def test_legacy_no_tactile_size(self):
        self.assertEqual(*self._token_counts((224, 224), None))


class PrefillAndCorrectFlowTest(unittest.TestCase):
    def _prefill(self, model, batch_size=2):
        latents_video = torch.randn(batch_size, Z_DIM, 2, 4, 8)  # grid f=2, h=2, w=4
        latents_action = torch.randn(batch_size, HORIZON, ACTION_DIM)
        context = torch.randn(batch_size, 3, TEXT_DIM)
        context_mask = torch.ones(batch_size, 3, dtype=torch.bool)
        return model._prefill_tac_cache(latents_video, latents_action, context, context_mask)

    def test_prefill_cache_geometry(self):
        model = _make_framework()
        cache = self._prefill(model)
        self.assertEqual(len(cache["kv"]), LAYERS)
        self.assertEqual(cache["latent_frames"], 2)
        self.assertEqual(cache["tac_grid"], (2, 2, 2))
        # 2 frames x (2x2) tactile tokens + HORIZON action tokens
        self.assertEqual(tuple(cache["kv"][0]["k"].shape), (2, 2 * 4 + HORIZON, HEADS * HEAD_DIM))
        self.assertEqual(tuple(cache["kv"][0]["v"].shape), (2, 2 * 4 + HORIZON, HEADS * HEAD_DIM))

    def test_tick_zero_init_delta_and_frame_clamp(self):
        model = _make_framework()
        cache = self._prefill(model)
        tactile_latents = torch.randn(2, Z_DIM, 1, 4, 4)
        actions_input = torch.randn(2, HORIZON, ACTION_DIM)
        offsets = torch.tensor([0, 4])  # frame ids 1 and 2 -> clamped to latent_frames-1 = 1
        executed = torch.arange(HORIZON).view(1, -1) < offsets.view(-1, 1)
        delta = model._run_tactile_expert(cache, tactile_latents, actions_input, executed, offsets)
        self.assertEqual(tuple(delta.shape), (2, HORIZON, ACTION_DIM))
        self.assertTrue(torch.all(delta == 0))

    def test_tokens_come_from_the_frozen_tactile_patchifier(self):
        model = _make_framework()
        tactile_latents = torch.randn(2, Z_DIM, 1, 4, 4)
        tokens = model._patchify_tactile(tactile_latents)
        expected = model.video_expert.patch_embedding_tactile(tactile_latents).flatten(2).transpose(1, 2)
        self.assertEqual(tuple(tokens.shape), (2, 4, HIDDEN))
        self.assertTrue(torch.equal(tokens, expected))
        self.assertFalse(tokens.requires_grad)  # phase-2 keeps the patchifier frozen

    def test_gradients_stop_at_the_tactile_patchifier(self):
        model = _make_framework()
        nn.init.normal_(model.tactile_expert.head.weight, std=0.02)
        cache = self._prefill(model)
        offsets = torch.tensor([0, 4])
        executed = torch.arange(HORIZON).view(1, -1) < offsets.view(-1, 1)
        delta = model._run_tactile_expert(
            cache, torch.randn(2, Z_DIM, 1, 4, 4), torch.randn(2, HORIZON, ACTION_DIM), executed, offsets
        )
        delta.square().mean().backward()
        self.assertIsNone(model.video_expert.patch_embedding_tactile.weight.grad)
        self.assertIsNotNone(model.tactile_expert.head.weight.grad)


class MoTActionKVCaptureTest(unittest.TestCase):
    def test_capture_matches_plain_forward(self):
        model = _make_framework()
        batch_size = 2
        latents_video = torch.randn(batch_size, Z_DIM, 2, 4, 8)
        latents_action = torch.randn(batch_size, HORIZON, ACTION_DIM)
        context = torch.randn(batch_size, 3, TEXT_DIM)
        context_mask = torch.ones(batch_size, 3, dtype=torch.bool)
        timestep = torch.zeros(batch_size)

        video_pre = model.video_expert.pre_dit(
            x=latents_video, timestep=timestep, context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=True,
        )
        action_pre = model.action_expert.pre_dit(
            action_tokens=latents_action, timestep=timestep, context=context, context_mask=context_mask,
        )
        video_seq_len = video_pre["tokens"].shape[1]
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=HORIZON,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=torch.device("cpu"),
        )
        video_kv = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        common = dict(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            video_kv_cache=video_kv,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        with torch.no_grad():
            plain = model.mot.forward_action_with_video_cache(**common)
            captured_kv = []
            with_capture = model.mot.forward_action_with_video_cache(**common, kv_out=captured_kv)
        self.assertTrue(torch.allclose(plain, with_capture))
        self.assertEqual(len(captured_kv), LAYERS)
        for layer in captured_kv:
            self.assertEqual(tuple(layer["k"].shape), (batch_size, HORIZON, HEADS * HEAD_DIM))


class SampleTactileOffsetsTest(unittest.TestCase):
    def test_valid_range_clamped_to_episode_end(self):
        import sys
        import types

        for missing in ("pinocchio", "decord"):  # only touched at import time / inside unused methods
            sys.modules.setdefault(missing, types.ModuleType(missing))
        from starVLA.dataloader.vla.dataset.lerobot_dataset import LeRobotV2Dataset

        dataset = LeRobotV2Dataset.__new__(LeRobotV2Dataset)
        dataset.episode_step_indices = None
        dataset.tactile_offsets_per_sample = 64
        dataset.config = type("Cfg", (), {"action_horizon": 32})()
        offsets, raw = dataset._sample_tactile_offsets(episode_idx=0, step_idx=8, total_frames=10)
        self.assertEqual(offsets.shape, (64,))
        self.assertTrue((offsets >= 0).all() and (offsets < 2).all())  # only 2 valid steps remain
        self.assertTrue((raw == 8 + offsets).all())

        offsets, raw = dataset._sample_tactile_offsets(episode_idx=0, step_idx=0, total_frames=1000)
        self.assertTrue((offsets < 32).all())
        self.assertTrue((raw == offsets).all())


if __name__ == "__main__":
    unittest.main()
