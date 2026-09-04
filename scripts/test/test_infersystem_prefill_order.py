from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from deployment.model_server.server_infersystem import InferSystemPolicyServer


class FakeSocket:
    """REP socket stub that records the interleaving of sends and policy work."""

    def __init__(self, events: list[str], requests: list[bytes]):
        self._events = events
        self._requests = list(requests)

    def recv(self) -> bytes:
        if not self._requests:
            raise KeyboardInterrupt  # the run() loop treats this as a clean shutdown
        return self._requests.pop(0)

    def send(self, _payload: bytes) -> None:
        self._events.append("send")

    def setsockopt(self, *_args) -> None:
        pass

    def bind(self, *_args) -> None:
        pass

    def close(self) -> None:
        pass


class FakeContext:
    def __init__(self, socket: FakeSocket):
        self._socket = socket

    def socket(self, _socket_type):
        return self._socket

    def term(self) -> None:
        pass


class FakeTactilePolicy:
    def __init__(self, events: list[str], fail: bool = False):
        self._events = events
        self._fail = fail
        self.tactile_prefill_pending = False

    def predict(self) -> None:
        self._events.append("predict")
        self.tactile_prefill_pending = True

    def prefill_tactile_cache(self) -> None:
        self._events.append("prefill")
        if self._fail:
            raise RuntimeError("prefill boom")
        self.tactile_prefill_pending = False


def _run_server(policy, events: list[str], requests: list[bytes]) -> InferSystemPolicyServer:
    """Drive the real `run()` loop over a stub socket."""
    server = InferSystemPolicyServer.__new__(InferSystemPolicyServer)
    server._policy = policy
    server._bind = "inproc://test"
    server._log_timing_every = 0
    server._infer_count = 0
    server._context = None
    server._socket = None

    def handle(_raw: bytes) -> dict:
        policy.predict()
        return {"status": "ok"}

    server._handle_raw_request = handle
    socket = FakeSocket(events, requests)
    with mock.patch(
        "deployment.model_server.server_infersystem.zmq.Context",
        return_value=FakeContext(socket),
    ):
        server.run()
    return server


class PrefillAfterReplyTest(unittest.TestCase):
    def test_chunk_is_sent_before_prefill_runs(self):
        events: list[str] = []
        policy = FakeTactilePolicy(events)
        _run_server(policy, events, [b"req", b"req"])
        self.assertEqual(events, ["predict", "send", "prefill", "predict", "send", "prefill"])
        self.assertFalse(policy.tactile_prefill_pending)

    def test_noop_without_a_fresh_plan(self):
        events: list[str] = []
        policy = FakeTactilePolicy(events)
        server = _run_server(policy, events, [b"req"])
        server._prefill_tactile_cache_after_reply()  # nothing pending anymore
        self.assertEqual(events.count("prefill"), 1)

    def test_policy_without_tactile_expert_is_untouched(self):
        events: list[str] = []
        _run_server(SimpleNamespace(predict=lambda: events.append("predict")), events, [b"req"])
        self.assertEqual(events, ["predict", "send"])

    def test_prefill_failure_does_not_kill_the_server_loop(self):
        events: list[str] = []
        policy = FakeTactilePolicy(events, fail=True)
        with self.assertLogs("deployment.model_server.server_infersystem", level="ERROR"):
            _run_server(policy, events, [b"req", b"req"])
        self.assertEqual(events, ["predict", "send", "prefill", "predict", "send", "prefill"])


if __name__ == "__main__":
    unittest.main()
