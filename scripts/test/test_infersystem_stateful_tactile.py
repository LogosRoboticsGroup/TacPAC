"""Server side of the InferSystem stateful tactile protocol (prepare / refine).

Mirrors what the robot-side `TactilePlanWorker` sends: `cmd=predict` plus a
`stateful_tactile` control block, a `request_id` for idempotent retries, and — on refine —
the already-executed action prefix.
"""

from __future__ import annotations

from collections import OrderedDict
import unittest

import cv2
import numpy as np

from deployment.model_server.infersystem_protocol import InferSystemAdapterConfig
from deployment.model_server.server_infersystem import InferSystemPolicyServer


CAMERAS = ("third_view", "left_wrist_view", "left_tactile", "right_tactile")
HORIZON, ACTION_DIM = 6, 4


def _png(value: int) -> bytes:
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    return cv2.imencode(".png", image)[1].tobytes()


def _request(op: str, plan_id: str, *, offset: int = 0, seq: int = 0, request_id: str = "rid") -> dict:
    msg = {
        "cmd": "predict",
        "state": [0.1] * ACTION_DIM,
        "prompt": "plug the board",
        "request_id": request_id,
        "stateful_tactile": {
            "op": op,
            "plan_id": plan_id,
            "action_offset": offset,
            "tactile_seq": seq,
        },
    }
    for index, name in enumerate(CAMERAS):
        msg[name] = _png(10 * index + 5)
    if op == "refine":
        msg["actions"] = np.zeros((offset, ACTION_DIM), dtype=np.float32).tolist()
    return msg


class FakePolicy:
    """Records what the server hands to the model, in model (normalized) space."""

    n_tactile_views = 2
    stat_key = "flexiv_tac"

    def __init__(self):
        self.predict_calls = 0
        self.correct_calls: list[dict] = []
        self.postprocess_inputs: list[dict] = []
        self.plan = np.arange(HORIZON * ACTION_DIM, dtype=np.float32).reshape(HORIZON, ACTION_DIM)
        self.tactile_prefill_pending = False
        self.prefill_calls = 0

    def preprocess(self, payload, stat_key=None):
        return {"batch_images": payload["batch_images"]}

    def predict_action(self, **_kwargs):
        self.predict_calls += 1
        self.tactile_prefill_pending = True
        return {"normalized_actions": self.plan[None]}

    def prefill_tactile_cache(self):
        self.prefill_calls += 1
        self.tactile_prefill_pending = False

    def correct_action(self, batch_tactile_images, offset):
        self.correct_calls.append({"images": batch_tactile_images, "offset": offset})
        chunk = self.plan.copy()
        chunk[offset:] += 100.0
        return {
            "normalized_actions": chunk[offset:][None],
            "normalized_chunk": chunk[None],
            "delta": np.full((1, HORIZON - offset, ACTION_DIM), 100.0, dtype=np.float32),
            "offset": offset,
        }

    def postprocess(self, output, input_data, stat_key=None):
        # State-relative un-normalization: the refine path must reuse the prepare observation.
        self.postprocess_inputs.append(input_data)
        state = np.asarray(input_data["state"][0], dtype=np.float32).reshape(-1)
        return {"actions": np.asarray(output["normalized_actions"], dtype=np.float32) + state[0]}

    def reset(self):
        self.plan = self.plan * 0 + 1


def _server(policy) -> InferSystemPolicyServer:
    server = InferSystemPolicyServer.__new__(InferSystemPolicyServer)
    server._policy = policy
    server._adapter_config = InferSystemAdapterConfig(
        camera_order=CAMERAS,
        image_resize_size=(16, 16),
        n_tactile_views=int(getattr(policy, "n_tactile_views", 0)),
    )
    server._log_timing_every = 0
    server._log_actions_every = 0
    server._infer_count = 0
    server._save_generated_video_dir = None
    server._save_generated_video_every = 1
    server._save_generated_video_fps = None
    server._plan_state = None
    server._replay_cache = OrderedDict()
    return server


class PrepareRefineFlowTest(unittest.TestCase):
    def test_prepare_returns_chunk_and_opens_a_plan(self):
        policy = FakePolicy()
        server = _server(policy)
        reply = server._handle_stateful_tactile(_request("prepare", "plan-a", seq=0))

        self.assertEqual(reply["status"], "ok")
        self.assertEqual(policy.predict_calls, 1)
        self.assertEqual(
            reply["plan"],
            {"plan_id": "plan-a", "op": "prepare", "action_offset": 0, "tactile_seq": 0, "action_horizon": HORIZON},
        )
        self.assertEqual(server._plan_state["plan_id"], "plan-a")

    def test_refine_ticks_tactile_views_only_and_returns_full_chunk(self):
        policy = FakePolicy()
        server = _server(policy)
        server._handle_stateful_tactile(_request("prepare", "plan-a", request_id="r0"))
        reply = server._handle_stateful_tactile(
            _request("refine", "plan-a", offset=2, seq=1, request_id="r1")
        )

        self.assertEqual(policy.predict_calls, 1)  # refine must not re-run the slow path
        self.assertEqual(len(policy.correct_calls), 1)
        tick = policy.correct_calls[0]
        self.assertEqual(tick["offset"], 2)
        self.assertEqual(len(tick["images"][0]), 2)  # only the two tactile cameras
        self.assertEqual(reply["plan"]["action_horizon"], HORIZON)
        self.assertEqual(np.asarray(reply["actions"]).shape[-2], HORIZON)

    def test_refine_unnormalizes_against_the_prepare_observation(self):
        policy = FakePolicy()
        server = _server(policy)
        prepare_msg = _request("prepare", "plan-a", request_id="r0")
        server._handle_stateful_tactile(prepare_msg)
        refine_msg = _request("refine", "plan-a", offset=1, request_id="r1")
        refine_msg["state"] = [9.9] * ACTION_DIM  # the robot has moved since the plan was made
        server._handle_stateful_tactile(refine_msg)

        # Both calls must un-normalize against the prepare state, or absolute actions jump.
        self.assertEqual(len(policy.postprocess_inputs), 2)
        for input_data in policy.postprocess_inputs:
            self.assertAlmostEqual(float(np.asarray(input_data["state"][0]).reshape(-1)[0]), 0.1, places=5)

    def test_refine_without_a_matching_plan_is_rejected(self):
        policy = FakePolicy()
        server = _server(policy)
        no_plan = server._handle_stateful_tactile(_request("refine", "plan-a", offset=1))
        self.assertEqual(no_plan["status"], "error")

        server._handle_stateful_tactile(_request("prepare", "plan-a", request_id="r0"))
        stale = server._handle_stateful_tactile(_request("refine", "plan-old", offset=1, request_id="r1"))
        self.assertEqual(stale["status"], "error")
        self.assertEqual(policy.correct_calls, [])

    def test_unknown_op_is_rejected(self):
        server = _server(FakePolicy())
        self.assertEqual(server._handle_stateful_tactile(_request("nope", "plan-a"))["status"], "error")


class ReplayAndResetTest(unittest.TestCase):
    def test_retry_with_same_request_id_replays_instead_of_rerunning(self):
        policy = FakePolicy()
        server = _server(policy)
        first = server._handle_stateful_tactile(_request("prepare", "plan-a", request_id="rid-1"))
        again = server._handle_stateful_tactile(_request("prepare", "plan-a", request_id="rid-1"))

        self.assertIs(first, again)
        self.assertEqual(policy.predict_calls, 1)

    def test_replay_cache_is_bounded(self):
        policy = FakePolicy()
        server = _server(policy)
        for index in range(InferSystemPolicyServer._REPLAY_CACHE_SIZE + 3):
            server._handle_stateful_tactile(_request("prepare", "plan-a", request_id=f"rid-{index}"))
        self.assertEqual(len(server._replay_cache), InferSystemPolicyServer._REPLAY_CACHE_SIZE)

    def test_reset_drops_the_plan_and_replay_cache(self):
        policy = FakePolicy()
        server = _server(policy)
        server._handle_stateful_tactile(_request("prepare", "plan-a", request_id="rid-1"))
        server._handle_reset({"cmd": "reset"})

        self.assertIsNone(server._plan_state)
        self.assertEqual(len(server._replay_cache), 0)
        self.assertEqual(server._handle_stateful_tactile(_request("refine", "plan-a", offset=1))["status"], "error")


class MetadataTest(unittest.TestCase):
    def test_flag_tracks_policy_capability(self):
        with_tactile = _server(FakePolicy())
        self.assertTrue(with_tactile._metadata()["supports_stateful_tactile"])

        class PlainPolicy:
            stat_key = "flexiv"

        self.assertFalse(_server(PlainPolicy())._metadata()["supports_stateful_tactile"])


if __name__ == "__main__":
    unittest.main()
