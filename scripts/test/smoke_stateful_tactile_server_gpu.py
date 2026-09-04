"""End-to-end loopback for the InferSystem stateful tactile protocol (GPU).

Runs the real `InferSystemPolicyServer` over a real ZMQ socket against a random-weight
WanMoTJointTacExpert (small DiT, real VAE) that carries the deployed checkpoint's norm
stats — so PNG decode, msgpack, the FlexivTac state-relative transform, prepare/refine
and the retry replay all run for real. Only the weights are fake.

Requests are byte-identical to what the robot-side `TactilePlanWorker` sends
(`Inference/tactile_plan_worker.py` in the InferSystem repo).

Run from repo root: PYTHONPATH=. python scripts/test/smoke_stateful_tactile_server_gpu.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import msgpack
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf

from deployment.model_server.infersystem_protocol import InferSystemAdapterConfig
from deployment.model_server.server_infersystem import InferSystemPolicyServer
from starVLA.model.framework.WM4A.WanMoTJointTacExpert import WanMoTJointTacExpert

CONFIG_YAML = "starVLA/config/training/vla/starvla_wam.yaml"
RUN_DIR = Path("results/Checkpoints/vla/0725_WanMoTJointTacExpert_flexiv_insert_board_4views")
STAT_KEY = "flexiv_tac"
CAMERAS = (
    "observation.images.third_view",
    "observation.images.left_wrist_view",
    "observation.images.left_wrist_left_tactile",
    "observation.images.left_wrist_right_tactile",
)
ENDPOINT = "tcp://127.0.0.1:5599"
SMALL = {"num_layers": 4, "num_heads": 6, "attn_head_dim": 64}


def build_policy():
    cfg = OmegaConf.load(CONFIG_YAML)
    overrides = {
        "framework.name": "WanMoTJointTacExpert",
        "framework.skip_dit_load_from_pretrain": True,
        "framework.video_model.load_text_encoder": False,
        "framework.action_model.action_dim": 8,
        "framework.action_model.state_dim": 8,
        "datasets.vla_data.data_mix": "flexiv_insert_board_4views",
        "datasets.vla_data.tactile_offsets_per_sample": 0,
        "trainer.enable_gradient_checkpointing": False,
        "trainer.enable_compile": False,
        **{f"framework.video_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 384, "ffn_dim": 1024}.items()},
        **{f"framework.action_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 256, "ffn_dim": 512}.items()},
        **{f"framework.tactile_model.config.{k}": v for k, v in {**SMALL, "hidden_dim": 256, "ffn_dim": 512}.items()},
    }
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    policy = WanMoTJointTacExpert(cfg).cuda().eval()

    def fake_encode_text(prompts, device, dtype):
        """Stand in for the 20GB T5 (not on this box); everything above it stays real."""
        generator = torch.Generator().manual_seed(0)
        context = torch.randn(len(prompts), policy.text_len, policy.text_dim, generator=generator)
        return context.to(device=device, dtype=dtype), torch.ones(
            len(prompts), policy.text_len, dtype=torch.bool, device=device
        )

    policy._encode_text = fake_encode_text
    # Real deployment stats/data-config, fake weights.
    policy.norm_stats = json.loads((RUN_DIR / "dataset_statistics.json").read_text())
    policy.set_dataconfig(STAT_KEY)
    return policy


def png(value: int, size: int = 224) -> bytes:
    image = np.full((size, size, 3), value, dtype=np.uint8)
    return cv2.imencode(".png", image)[1].tobytes()


def observation(state_offset: float = 0.0) -> dict:
    msg = {"state": (np.arange(8, dtype=np.float32) * 0.01 + state_offset).tolist(),
           "prompt": "plug out the pc board and insert into the side slot"}
    for index, name in enumerate(CAMERAS):
        msg[name] = png(40 + 30 * index)
    return msg


class Client:
    """Same wire calls as InferSystem's InferenceClient."""

    def __init__(self, endpoint: str):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 600_000)
        self._socket.connect(endpoint)

    def request(self, payload: dict) -> dict:
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        resp = msgpack.unpackb(self._socket.recv(), raw=False)
        if resp.get("status") == "error":
            raise RuntimeError(f"server error: {resp.get('message')}")
        return resp

    def stateful(self, control: dict, *, state_offset: float = 0.0, actions_prefix=None, request_id=None) -> dict:
        payload = {"cmd": "predict", **observation(state_offset)}
        payload["stateful_tactile"] = control
        payload["request_id"] = request_id or uuid4().hex
        if actions_prefix is not None:
            payload["actions"] = actions_prefix
        return self.request(payload)

    def close(self):
        self._socket.close()
        self._context.term()


def check_plan_meta(resp: dict, control: dict) -> np.ndarray:
    meta = resp["plan"]
    assert meta["plan_id"] == control["plan_id"], meta
    assert meta["op"] == control["op"], meta
    assert int(meta["action_offset"]) == int(control["action_offset"]), meta
    assert int(meta["tactile_seq"]) == int(control["tactile_seq"]), meta
    actions = np.asarray(resp["actions"], dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    assert actions.shape[0] == int(meta["action_horizon"]), (actions.shape, meta)
    return actions


def main() -> None:
    torch.manual_seed(0)
    policy = build_policy()
    server = InferSystemPolicyServer(
        policy,
        bind=ENDPOINT,
        adapter_config=InferSystemAdapterConfig(
            camera_order=CAMERAS,
            stat_key=STAT_KEY,
            image_resize_size=tuple(policy.data_config.image_size),
            n_tactile_views=policy.n_tactile_views,
            rgb_resize_mode="crop",  # datasets re-baked with aspect-preserving RGB
        ),
        log_timing_every=0,
    )
    threading.Thread(target=server.run, name="infersystem", daemon=True).start()
    time.sleep(1.0)
    client = Client(ENDPOINT)

    metadata = client.request({"cmd": "metadata"})["metadata"]
    assert metadata.get("supports_stateful_tactile"), metadata
    print("metadata ok: supports_stateful_tactile=True, chunk =", metadata.get("action_chunk_size"))

    client.request({"cmd": "reset"})

    plan_id = uuid4().hex
    prepare = {"op": "prepare", "plan_id": plan_id, "action_offset": 0, "tactile_seq": 0,
               "previous_plan_id": None, "previous_action_offset": None, "force_replan": False}
    started = time.perf_counter()
    plan_actions = check_plan_meta(client.stateful(prepare), prepare)
    print(f"prepare ok: actions {plan_actions.shape}, {1e3 * (time.perf_counter() - started):.0f} ms")

    horizon = plan_actions.shape[0]
    merged = plan_actions.copy()
    for seq, offset in enumerate((2, 5, 11), start=1):
        control = {"op": "refine", "plan_id": plan_id, "action_offset": offset, "tactile_seq": seq}
        started = time.perf_counter()
        # The robot moved since the plan was made, and echoes the executed prefix back.
        refined = check_plan_meta(
            client.stateful(control, state_offset=0.05 * seq, actions_prefix=merged[:offset].tolist()),
            control,
        )
        assert refined.shape == plan_actions.shape, refined.shape
        # Prefix is anchored: already-executed steps must come back unchanged, or the robot
        # would see its own past rewritten (this is what the state-relative reuse protects).
        np.testing.assert_allclose(refined[:offset], merged[:offset], rtol=1e-4, atol=1e-4)
        merged[offset:] = refined[offset:]
        print(f"refine k={offset:2d} ok: {1e3 * (time.perf_counter() - started):.0f} ms")

    # Retry replay: same request_id must not run the model again.
    control = {"op": "refine", "plan_id": plan_id, "action_offset": 3, "tactile_seq": 99}
    rid = uuid4().hex
    first = client.stateful(control, actions_prefix=merged[:3].tolist(), request_id=rid)
    again = client.stateful(control, actions_prefix=merged[:3].tolist(), request_id=rid)
    np.testing.assert_array_equal(np.asarray(first["actions"]), np.asarray(again["actions"]))
    print("retry replay ok: identical response for a repeated request_id")

    payload = {"cmd": "predict", **observation(), "request_id": uuid4().hex,
               "stateful_tactile": {"op": "refine", "plan_id": "nope", "action_offset": 1, "tactile_seq": 0}}
    client._socket.send(msgpack.packb(payload, use_bin_type=True))
    resp = msgpack.unpackb(client._socket.recv(), raw=False)
    assert resp["status"] == "error" and "plan_id" in resp["message"], resp
    print("stale plan_id rejected as expected")

    client.request({"cmd": "reset"})
    client.close()
    print("STATEFUL TACTILE SMOKE OK")


if __name__ == "__main__":
    main()
