"""GPU smoke test for WanMoTJointTacExpert: random-weight experts (skip pretrain), real VAE,
real flexiv tactile canvas geometry (4 views 224x224, 2 tactile). Covers the phase-2 training forward
(frozen denoise -> prefill -> tactile loss -> backward) and predict_action/correct_action.

Run from repo root: PYTHONPATH=. python scripts/test/smoke_tac_expert_gpu.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.WanMoTJointTacExpert import WanMoTJointTacExpert

CONFIG_YAML = "starVLA/config/training/vla/starvla_wam.yaml"
SMALL = {"num_layers": 4, "num_heads": 6, "attn_head_dim": 64}


def build_model():
    cfg = OmegaConf.load(CONFIG_YAML)
    overrides = {
        "framework.name": "WanMoTJointTacExpert",
        "framework.skip_dit_load_from_pretrain": True,
        "framework.video_model.load_text_encoder": False,
        "framework.action_model.action_dim": 8,
        "datasets.vla_data.data_mix": "flexiv_plug_4views",
        "datasets.vla_data.tactile_offsets_per_sample": 2,
        "trainer.enable_gradient_checkpointing": False,
        "trainer.enable_compile": False,
        **{f"framework.video_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 384, "ffn_dim": 1024}.items()},
        **{f"framework.action_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 256, "ffn_dim": 512}.items()},
        **{f"framework.tactile_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 256, "ffn_dim": 512}.items()},
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    # Direct class import (build_framework auto-imports every framework; some need deps
    # absent on dev boxes).
    model = WanMoTJointTacExpert(cfg)
    return model.cuda()


def fake_examples(batch_size=2, num_offsets=2, horizon=32):
    examples = []
    for _ in range(batch_size):
        vision = [torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8).float() for _ in range(2)]
        tactile = [torch.randn(3, 224, 224).clamp(-1, 1) * 0.2 for _ in range(2)]
        examples.append(
            {
                "image": vision + tactile,
                "image_is_tactile": torch.tensor([False, False, True, True]),
                "view_mask": torch.ones(4, dtype=torch.bool),
                "lang": "lift the can",
                "context": torch.randn(128, 4096),
                "context_mask": torch.ones(128, dtype=torch.bool),
                "state": torch.randn(1, 8),
                "actions": torch.randn(horizon, 8).clamp(-1, 1),
                "action_mask": torch.ones(horizon, 8, dtype=torch.bool),
                "tactile_now": (torch.randn(num_offsets, 2, 3, 224, 224) * 0.2).clamp(-1, 1),
                "tactile_offset": torch.tensor(np.random.randint(0, horizon, size=num_offsets)),
            }
        )
    return examples


def main():
    torch.manual_seed(0)
    model = build_model()
    print("model built; tactile expert params:", sum(p.numel() for p in model.tactile_expert.parameters()) / 1e6, "M")

    # --- phase-2 training forward ---
    model.train()
    out = model.forward(fake_examples())
    print("train forward:", {k: (float(v) if not torch.is_tensor(v) else "tensor") for k, v in out.items()})
    out["total_loss"].backward()
    tac_grads = [p.grad for p in model.tactile_expert.parameters() if p.grad is not None]
    base_grads = [p.grad for p in model.video_expert.parameters() if p.grad is not None]
    assert tac_grads and any(torch.any(g != 0) for g in tac_grads), "tactile expert got no gradients"
    assert not base_grads, "frozen base unexpectedly received gradients"
    print(f"backward ok: {len(tac_grads)} tactile grads, base grads: {len(base_grads)}")
    model.zero_grad(set_to_none=True)

    # --- inference: plan -> prefill -> ticks (deployment order: chunk goes out first) ---
    batch_images = np.random.randint(0, 256, (1, 4, 3, 224, 224), dtype=np.uint8)
    predict_start = time.perf_counter()
    plan_out = model.predict_action(
        batch_images=batch_images,
        view_mask=[[True] * 4],
        instructions=["lift the can"],
        state=np.random.randn(1, 1, 8).astype(np.float32),
        context=torch.randn(1, 128, 4096),
        context_mask=torch.ones(1, 128, dtype=torch.bool),
    )
    torch.cuda.synchronize()
    predict_ms = 1e3 * (time.perf_counter() - predict_start)
    plan = plan_out["normalized_actions"]
    assert plan.shape == (1, 32, 8)

    tac_images = np.random.randint(0, 256, (1, 2, 3, 224, 224), dtype=np.uint8)
    try:
        model.correct_action(batch_tactile_images=tac_images, offset=0)
        raise AssertionError("correct_action must fail before prefill_tactile_cache()")
    except ValueError:
        pass

    prefill_start = time.perf_counter()
    model.prefill_tactile_cache()
    torch.cuda.synchronize()
    prefill_ms = 1e3 * (time.perf_counter() - prefill_start)
    print(f"plan: {plan.shape}, predict {predict_ms:.1f} ms (chunk out), prefill {prefill_ms:.1f} ms (after)")

    for offset in (0, 7, 31):
        tick_start = time.perf_counter()
        tick = model.correct_action(batch_tactile_images=tac_images, offset=offset)
        torch.cuda.synchronize()
        suffix = tick["normalized_actions"]
        assert suffix.shape == (1, 32 - offset, 8), suffix.shape
        # The wire protocol replaces the whole plan, so the tick also reports the full chunk.
        chunk = tick["normalized_chunk"]
        assert chunk.shape == (1, 32, 8), chunk.shape
        assert np.allclose(chunk[:, offset:], suffix), "chunk suffix must be the corrected suffix"
        # zero-init head -> delta must be exactly zero, suffix == plan suffix
        assert np.allclose(tick["delta"], 0.0), "zero-init tactile head should output zero delta"
        assert np.allclose(suffix, plan[:, offset:]), "corrected suffix must equal plan under zero delta"
        print(f"tick k={offset:2d}: suffix {suffix.shape}, {1e3 * (time.perf_counter() - tick_start):.1f} ms")

    model.reset()
    try:
        model.correct_action(batch_tactile_images=tac_images, offset=0)
        raise AssertionError("correct_action must fail after reset()")
    except ValueError:
        print("reset() clears the plan cache as expected")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
