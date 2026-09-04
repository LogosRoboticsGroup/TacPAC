from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional
import time

import numpy as np
from PIL import Image

from deployment.model_server.adaptive_ensemble import AdaptiveEnsembler
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


@dataclass
class QueuedChunkAction:
    action: np.ndarray
    step_id: int
    chunk_start_step: int


class M1Inference:
    def __init__(
        self,
        host="0.0.0.0",
        port=10095,
        execution_steps: int = 16,
        prefix_steps: int = 0,
        adaptive_prefix: bool = False,
        fps: int = 30,
        rtc=False,
        action_ensemble=False,
        action_ensemble_horizon=7,
        adaptive_ensemble_alpha=0.1,
        online_state_rollout: bool = False,
        server_rtc: bool = False,
        verbose=False,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port, verbose=verbose)
        self.metadata = self.client.get_metadata()
        self.stat_key = str(self.metadata["stat_key"])
        self.tactile_view_indices = set(self.metadata.get("tactile_view_indices", ()))

        self.task_description = None

        self.execution_steps = execution_steps
        self.max_prefix_steps = prefix_steps
        self.prefix_steps = prefix_steps
        self.adaptive_prefix = adaptive_prefix
        self.rtc = rtc
        self.fps = fps
        self.action_ensemble = action_ensemble
        self.online_state_rollout = bool(online_state_rollout)
        self.server_rtc = bool(server_rtc)
        self.verbose = verbose

        self.action_queue = deque()
        self._last_consumed_step_id = -1
        self._queue_generation = 0
        self.online_needs_new_observation = True
        self.online_chunk_step_count = 0

        if self.online_state_rollout:
            self.infer_func = self.infer_online_sync
        elif self.rtc:
            self.last_vla_input = None
            self.worker_busy = False
            self.infer_func = self.infer_async
            self.start_infer_async_thread()
        else:
            self.infer_func = self.infer_sync

        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

    @staticmethod
    def _format_timing_parts(timing: Dict[str, Any], keys: list[str]) -> str:
        parts = []
        for key in keys:
            if key in timing:
                parts.append(f"{key}={float(timing[key]):.1f}ms")
        return " ".join(parts)

    def _print_response_timing(self, label: str, response: Dict[str, Any], *, client_encode_ms: float = 0.0) -> None:
        if not self.verbose:
            return
        client_timing = response.get("client_timing") or {}
        server_timing = response.get("timing") or {}
        server_total_ms = float(server_timing.get("server_total_ms", 0.0))
        roundtrip_ms = float(client_timing.get("roundtrip_ms", 0.0))
        network_ms = max(0.0, roundtrip_ms - server_total_ms)
        client_parts = []
        if client_encode_ms > 0:
            client_parts.append(f"encode={client_encode_ms:.1f}ms")
        if client_timing:
            client_parts.append(self._format_timing_parts(client_timing, ["pack_ms", "roundtrip_ms", "unpack_ms"]))
        server_parts = self._format_timing_parts(
            server_timing,
            ["decode_ms", "preprocess_ms", "predict_ms", "postprocess_ms", "server_total_ms"],
        )
        msg = f"[{label}] est_network={network_ms:.1f}ms"
        if client_parts:
            msg += " client: " + " ".join(part for part in client_parts if part)
        if server_parts:
            msg += " server: " + server_parts
        print(msg)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.clear_queue()
        self._last_consumed_step_id = -1
        if self.rtc:
            while True:
                with self.lock:
                    worker_busy = self.worker_busy
                if not worker_busy:
                    break
                if not self.thread.is_alive():
                    raise RuntimeError("RTC inference worker exited.")
                time.sleep(0.01)
        self.client.reset(
            instruction=task_description,
            clear_replay=False,
            clear_registry=False,
            server_rtc=getattr(self, "server_rtc", None),
            execution_steps=getattr(self, "execution_steps", None),
            prefix_steps=getattr(self, "prefix_steps", None),
        )
        if self.action_ensemble:
            self.action_ensembler.reset()
        if self.online_state_rollout:
            self.online_needs_new_observation = True
            self.online_chunk_step_count = 0

    def clear_queue(self) -> None:
        if self.rtc:
            with self.lock:
                self.action_queue.clear()
                self.last_vla_input = None
                self._queue_generation += 1
        else:
            self.action_queue.clear()
            self._queue_generation += 1

    def start_infer_async_thread(self):
        self.stop_event = Event()
        self.lock = Lock()
        self.thread = Thread(target=self.run_infer_async_loop, args=(), name=f"{self}_run_infer_async_loop")
        self.thread.daemon = True
        self.thread.start()

    def _finalize_request_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = dict(payload)
        if "batch_images" in request_payload:
            request_payload["batch_images"] = [self._encode_images(images) for images in request_payload["batch_images"]]
        return request_payload

    def _make_queue_record(
        self,
        *,
        action: np.ndarray,
        step_id: int,
        chunk_start_step: int,
        response_data: Dict[str, Any],
    ) -> QueuedChunkAction:
        del response_data
        return QueuedChunkAction(
            action=action,
            step_id=step_id,
            chunk_start_step=chunk_start_step,
        )

    def _enqueue_response(
        self,
        response_data: Dict[str, Any],
        *,
        request_payload: Dict[str, Any],
    ) -> None:
        if "actions" not in response_data:
            raise KeyError("Inference response requires 'actions'.")
        actions = np.asarray(response_data["actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"Expected inference response actions with shape (1, T, D), got {actions.shape}.")
        actions = np.array(actions[0], copy=True)

        if "step_id_start" in response_data:
            action_start = int(response_data["step_id_start"])
        else:
            if "step_idx" not in request_payload:
                raise KeyError("Inference request requires 'step_idx' when response omits 'step_id_start'.")
            if request_payload.get("actions") is not None:
                if "delay" not in request_payload:
                    raise KeyError("Inference request with prefix actions requires 'delay'.")
                delay = int(request_payload["delay"])
                if delay < 0:
                    raise ValueError(f"Expected non-negative prefix delay, got {delay}.")
                if delay >= actions.shape[0]:
                    raise ValueError(
                        f"Inference response length {actions.shape[0]} is not larger than prefix delay {delay}."
                    )
                actions = actions[delay:]
            else:
                delay = 0
            action_start = int(request_payload["step_idx"]) + delay

        if self.action_ensemble:
            actions = np.asarray(self.action_ensembler.ensemble_action(actions), dtype=np.float32)[None]

        actions = actions[: self.execution_steps]
        if actions.shape[0] == 0:
            return
        min_allowed = int(self._last_consumed_step_id + 1)
        if action_start + actions.shape[0] - 1 < min_allowed:
            return
        if action_start < min_allowed:
            actions = actions[min_allowed - action_start :]
            action_start = min_allowed

        while len(self.action_queue) > 0 and self.action_queue[-1].step_id >= action_start:
            self.action_queue.pop()
        records = [
            self._make_queue_record(
                action=np.array(action, copy=True),
                step_id=action_start + i,
                chunk_start_step=action_start,
                response_data=response_data,
            )
            for i, action in enumerate(actions)
        ]
        self.action_queue.extend(records)

    def run_infer_async_loop(self):
        while not self.stop_event.is_set():
            request_payload = None
            request_generation = None
            with self.lock:
                if self.last_vla_input is not None and not self.worker_busy:
                    latest_vla_input, request_generation = self.last_vla_input
                    self.last_vla_input = None
                    self.worker_busy = True
                    request_payload = dict(latest_vla_input)
                    if len(self.action_queue) > 0:
                        request_payload["step_idx"] = int(self.action_queue[0].step_id)
                    else:
                        request_payload["step_idx"] = int(self._last_consumed_step_id + 1)
                    delay = min(len(self.action_queue), self.prefix_steps)
                    if delay > 0:
                        request_payload["actions"] = [
                            np.array([self.action_queue[i].action for i in range(delay)], copy=True)
                        ]
                    else:
                        request_payload.pop("actions", None)
                    request_payload["delay"] = delay
            if request_payload is None:
                time.sleep(0.001)
                continue
            request_payload = self._finalize_request_payload(request_payload)
            start_time = time.time()
            response = self.client.infer(request_payload)
            elapsed = time.time() - start_time
            with self.lock:
                if request_generation == self._queue_generation:
                    self._enqueue_response(response["data"], request_payload=request_payload)
                    if self.adaptive_prefix:
                        self.prefix_steps = min(self.max_prefix_steps, int(elapsed * self.fps) + 1)
                        if self.verbose:
                            print(
                                f"Infer RTC elapsed: {int(elapsed * 1000)}ms, "
                                f"Prefix_steps is adapted to {self.prefix_steps}",
                                flush=True,
                            )
                    elif self.verbose:
                        print(f"Infer RTC elapsed: {int(elapsed * 1000)}ms", flush=True)
                self.worker_busy = False

    def infer_async(self, vla_input):
        with self.lock:
            if len(self.action_queue) <= self.prefix_steps:
                self.last_vla_input = (vla_input.copy(), self._queue_generation)
        while True:
            with self.lock:
                if len(self.action_queue) > 0:
                    break
            if not self.thread.is_alive():
                raise RuntimeError("RTC inference worker exited.")
            time.sleep(0.001)

    def infer_sync(self, vla_input):
        if len(self.action_queue) == 0:
            encode_started_at = time.perf_counter()
            request_payload = self._finalize_request_payload(vla_input)
            client_encode_ms = (time.perf_counter() - encode_started_at) * 1000
            start_time = time.time()
            response = self.client.infer(request_payload)
            elapsed = time.time() - start_time
            if self.verbose:
                print(f"[infer] total_client_wait={elapsed * 1000:.1f}ms")
            self._print_response_timing("infer", response, client_encode_ms=client_encode_ms)
            self._enqueue_response(response["data"], request_payload=request_payload)

    def encode_image(self, img: np.ndarray, quality=70, image_format="JPEG"):
        buf = BytesIO()
        save_kwargs = {"quality": int(quality)} if image_format == "JPEG" else {}
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format=image_format, **save_kwargs)
        return buf.getvalue()

    def _encode_images(self, images):
        return [
            self.encode_image(image, image_format="PNG" if index in self.tactile_view_indices else "JPEG")
            if isinstance(image, np.ndarray)
            else image
            for index, image in enumerate(images)
        ]

    def _prepare_online_request_payload(self, vla_input: Dict[str, Any]) -> Dict[str, Any]:
        if "state" not in vla_input:
            raise KeyError("online_state_rollout requires 'state' on every step.")

        if self.online_needs_new_observation:
            required = ("batch_images", "instructions", "fps", "view_mask")
            missing = [key for key in required if key not in vla_input]
            if missing:
                raise KeyError(f"online_state_rollout start requires new observation keys: {missing}")
            request_payload = dict(vla_input)
            return self._finalize_request_payload(request_payload)

        drop_keys = {"actions", "delay"}
        if not getattr(self, "server_rtc", False):
            drop_keys.update({"batch_images", "instructions", "fps", "view_mask"})
        request_payload = {key: value for key, value in vla_input.items() if key not in drop_keys}
        if getattr(self, "server_rtc", False) and "batch_images" in request_payload:
            return self._finalize_request_payload(request_payload)
        return request_payload

    def _advance_online_rollout_state(self) -> None:
        self.online_chunk_step_count += 1
        if self.online_chunk_step_count >= self.execution_steps:
            self.online_needs_new_observation = True
            self.online_chunk_step_count = 0
        else:
            self.online_needs_new_observation = False

    def infer_online_sync(self, vla_input: Dict[str, Any]) -> None:
        if len(self.action_queue) > 0:
            return
        encode_started_at = time.perf_counter()
        request_payload = self._prepare_online_request_payload(vla_input)
        client_encode_ms = (time.perf_counter() - encode_started_at) * 1000
        start = time.perf_counter()
        response = self.client.infer(request_payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 1 and self.verbose:
            print(f"[online_rollout] total_client_wait={elapsed_ms:.1f}ms")
        self._print_response_timing("online_rollout", response, client_encode_ms=client_encode_ms)
        self.action_queue.clear()
        self._enqueue_response(response["data"], request_payload=request_payload)
        if len(self.action_queue) != 1:
            raise ValueError(
                f"online_state_rollout expects one action per infer call, got {len(self.action_queue)} records."
            )
        self._advance_online_rollout_state()

    def step_native(
        self,
        vla_input: Dict[str, Any],
        task_description: Optional[str] = None,
        return_record: bool = False,
    ) -> Dict[str, Any]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        if "step_idx" not in vla_input:
            vla_input = dict(vla_input)
            vla_input["step_idx"] = int(self._last_consumed_step_id + 1)
        start = time.perf_counter()
        self.infer_func(vla_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 1 and self.verbose:
            print(f"Step elapsed: {elapsed_ms:.1f}ms", flush=True)

        if self.rtc:
            with self.lock:
                record = self.action_queue.popleft()
                self._last_consumed_step_id = record.step_id
        else:
            record = self.action_queue.popleft()
            self._last_consumed_step_id = record.step_id

        if return_record:
            return {"action": record.action, "record": record}
        return {"action": record.action}

    def step(
        self,
        images=None,
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        step: int = 0,
        stat_key: Optional[str] = None,
        fps: Optional[int] = None,
        view_mask=None,
        ee: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        del step
        kwargs.pop("goal", None)
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        vla_input = {}
        if images is not None:
            vla_input["batch_images"] = [list(images)]
        effective_task = self.task_description if task_description is None else task_description
        if effective_task is not None:
            vla_input["instructions"] = [effective_task]
        if state is not None:
            vla_input["state"] = [state]
        if stat_key is not None:
            vla_input["stat_key"] = stat_key
        if fps is not None:
            vla_input["fps"] = [fps]
        if view_mask is not None:
            vla_input["view_mask"] = [view_mask]
        if ee is not None:
            vla_input["ee"] = [ee]
        if q is not None:
            vla_input["q"] = [q]
        if mode is not None:
            vla_input["mode"] = str(mode)
        for key, value in kwargs.items():
            vla_input[key] = value
        return self.step_native(vla_input, task_description=None, return_record=False)

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join()
