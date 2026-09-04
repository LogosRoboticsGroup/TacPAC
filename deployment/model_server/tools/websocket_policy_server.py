# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import inspect
import logging
import time
import traceback
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames
from PIL import Image

from . import image_tools
from . import msgpack_numpy


def _config_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _config_get_nested(obj, *keys, default=None):
    current = obj
    for key in keys:
        current = _config_get(current, key, default=None)
        if current is None:
            return default
    return current


def _resolve_policy_metadata(policy) -> dict:
    if hasattr(policy, "get_metadata"):
        return policy.get_metadata()
    stat_key = getattr(policy, "stat_key", None)
    if stat_key is None:
        raise AttributeError(f"{type(policy).__name__} must set stat_key before serving metadata.")
    stat_key = str(stat_key)
    return {
        "stat_key": stat_key,
    }


def _resolve_warmup_policy(policy):
    candidate = getattr(policy, "framework", None)
    if candidate is not None and hasattr(candidate, "predict_action"):
        return candidate
    candidate = getattr(policy, "backend", None)
    if candidate is not None and hasattr(candidate, "predict_action"):
        return candidate
    return policy


class WebsocketPolicyServer:
    """Serve a policy using the LogosVLA websocket protocol."""

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,
        enable_compile: bool = False,
        infer_executor_workers: int = 1,
        log_timing_every: int = 0,
        metadata: dict | None = None,
    ) -> None:
        if int(infer_executor_workers) != 1:
            raise ValueError("WebsocketPolicyServer requires infer_executor_workers=1.")
        del metadata
        self._policy = policy
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._infer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-infer")
        self._last_active = time.time()
        self._log_timing_every = int(log_timing_every)
        self._infer_count = 0
        self._warmup_iters = 2
        self._warmup_fps = 24.0
        self._warmup_instruction = "pick up the object"
        logging.getLogger("websockets.server").setLevel(logging.WARNING)
        if enable_compile:
            self._compile_policy()
            self._warmup_policy()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=None,
            ) as server:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
        finally:
            self.close()

    def close(self) -> None:
        self._infer_executor.shutdown(wait=False)

    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info("Idle timeout (%ss) reached, shutting down server.", self._idle_timeout)
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.debug("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = await self._route_message_async(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _compile_policy(self) -> None:
        if not hasattr(self._policy, "compile"):
            raise AttributeError("Policy has no compile method.")
        logging.info("Compiling policy with framework defaults.")
        self._policy.eval()
        self._policy.compile()
        logging.info("Policy compile done.")

    def _warmup_policy(self) -> None:
        if not hasattr(self._policy, "predict_action"):
            raise AttributeError("Policy has no predict_action method for warmup.")
        if self._warmup_iters <= 0:
            return
        warmup_policy = _resolve_warmup_policy(self._policy)
        cfg = getattr(warmup_policy, "config", None)
        data_config = getattr(self._policy, "data_config", None)
        if data_config is None:
            data_config = getattr(warmup_policy, "data_config", None)
        datasets_cfg = _config_get(cfg, "datasets")
        framework_cfg = _config_get(cfg, "framework")
        vla_data_cfg = _config_get(datasets_cfg, "vla_data")
        action_model_cfg = _config_get(framework_cfg, "action_model")
        if datasets_cfg is None or vla_data_cfg is None or action_model_cfg is None or data_config is None:
            logging.info("Skipping warmup inference because policy metadata is unavailable.")
            return
        image_size = _config_get(vla_data_cfg, "image_size", [224, 224])
        h, w = int(image_size[0]), int(image_size[1])
        state_dim = int(_config_get(action_model_cfg, "state_dim", 0))
        if state_dim <= 0:
            logging.info("Skipping warmup inference because state_dim is unavailable.")
            return
        num_views = len(getattr(data_config, "video_keys", []) or [])
        if num_views <= 0:
            runtime_video_keys = getattr(warmup_policy, "runtime_video_keys", None)
            num_views = len(runtime_video_keys or [])
        if num_views <= 0:
            image_keys = _config_get(vla_data_cfg, "image_keys", None)
            num_views = len(image_keys or [])
        if num_views <= 0:
            num_views = 1

        batch_images = [[np.zeros((h, w, 3), dtype=np.uint8) for _ in range(num_views)]]
        view_mask = [torch.ones(num_views, dtype=torch.bool)]
        instructions = [self._warmup_instruction]
        fps = [float(self._warmup_fps)]
        state = [np.zeros((1, state_dim), dtype=np.float32)]
        warmup_stat_key = str(_resolve_policy_metadata(self._policy)["stat_key"])

        logging.info("Running warmup inference: iters=%d, views=%d", self._warmup_iters, num_views)
        with torch.inference_mode():
            for _ in range(self._warmup_iters):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                payload = {
                    "batch_images": batch_images,
                    "view_mask": view_mask,
                    "instructions": instructions,
                    "fps": fps,
                    "state": state,
                }
                payload["stat_key"] = warmup_stat_key
                self._run_infer_path(payload)
        if hasattr(warmup_policy, "reset"):
            warmup_policy.reset()
        logging.info("Warmup inference done.")

    def decode_image(self, buf):
        if isinstance(buf, bytes):
            return np.asarray(Image.open(BytesIO(buf)).convert("RGB"))
        return buf

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000

    def _run_infer_path(self, payload: dict) -> tuple[dict, dict]:
        total_started_at = time.perf_counter()
        decode_ms = 0.0
        if "batch_images" in payload:
            decode_started_at = time.perf_counter()
            payload["batch_images"] = [[self.decode_image(image) for image in images] for images in payload["batch_images"]]
            payload["batch_images"] = image_tools.to_pil_preserve(payload["batch_images"])
            decode_ms = self._elapsed_ms(decode_started_at)
        preprocess_started_at = time.perf_counter()
        payload_preprocessed = self._policy.preprocess(payload, stat_key=payload.get("stat_key"))
        preprocess_ms = self._elapsed_ms(preprocess_started_at)
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        predict_started_at = time.perf_counter()
        output_dict = self._policy.predict_action(**payload_preprocessed)
        predict_ms = self._elapsed_ms(predict_started_at)
        postprocess_started_at = time.perf_counter()
        output_dict = self._policy.postprocess(output_dict, payload, stat_key=payload.get("stat_key"))
        postprocess_ms = self._elapsed_ms(postprocess_started_at)
        timing = {
            "phase": "infer",
            "decode_ms": round(decode_ms, 3),
            "preprocess_ms": round(preprocess_ms, 3),
            "predict_ms": round(predict_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "server_total_ms": round(self._elapsed_ms(total_started_at), 3),
        }
        return output_dict, timing

    def _call_policy_reset(self, payload: dict) -> dict:
        if hasattr(self._policy, "configure_inference_session"):
            self._policy.configure_inference_session(
                server_rtc=bool(payload.get("server_rtc", False)),
                execution_steps=payload.get("execution_steps"),
                prefix_steps=payload.get("prefix_steps"),
            )
        if hasattr(self._policy, "handle_request"):
            result = self._policy.handle_request("reset", payload)
            if isinstance(result, dict):
                return result
            return {"data": result}
        if not hasattr(self._policy, "reset"):
            raise AttributeError("Policy has no reset method.")
        reset_fn = self._policy.reset
        signature = inspect.signature(reset_fn)
        supported_kwargs = {}
        for key, value in payload.items():
            if key in signature.parameters:
                supported_kwargs[key] = value
        reset_result = reset_fn(**supported_kwargs)
        if isinstance(reset_result, dict):
            return reset_result
        return {"data": reset_result}

    async def _route_message_async(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")
        payload = msg.get("payload", msg)
        if mtype == "infer":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._infer_executor, self._route_infer_message, payload, req_id)
        return self._route_non_infer_message(mtype, payload, req_id)

    def _route_message(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")
        payload = msg.get("payload", msg)
        if mtype == "infer":
            return self._route_infer_message(payload, req_id)
        return self._route_non_infer_message(mtype, payload, req_id)

    def _route_infer_message(self, payload: dict, req_id: str) -> dict:
        if not isinstance(payload, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
            }
        try:
            save_video_dir = payload.get("_save_generated_video_dir")
            save_video = bool(save_video_dir)
            if save_video:
                payload = dict(payload)
                payload["return_video"] = True
            infer_payload = dict(payload)
            infer_payload.pop("_save_generated_video_dir", None)
            output_dict, timing = self._run_infer_path(infer_payload)
            if save_video:
                self._save_generated_video(output_dict, payload)
                output_dict = dict(output_dict)
                output_dict.pop("video", None)
        except Exception as e:
            logging.exception("Policy inference error (request_id=%s)", req_id)
            logging.exception(e)
            return {
                "status": "error",
                "ok": False,
                "type": "inference_result",
                "request_id": req_id,
                "error": {"message": str(e)},
            }
        self._infer_count += 1
        if self._log_timing_every > 0 and self._infer_count % self._log_timing_every == 0:
            logging.info(
                "TIMING infer_count=%d decode_ms=%.3f preprocess_ms=%.3f predict_ms=%.3f postprocess_ms=%.3f server_total_ms=%.3f",
                self._infer_count,
                float(timing["decode_ms"]),
                float(timing["preprocess_ms"]),
                float(timing["predict_ms"]),
                float(timing["postprocess_ms"]),
                float(timing["server_total_ms"]),
            )
        return {
            "status": "ok",
            "ok": True,
            "type": "inference_result",
            "request_id": req_id,
            "data": output_dict,
            "timing": timing,
        }

    def _save_generated_video(self, output: dict, payload: dict) -> None:
        video = output.get("video")
        if video is None:
            logging.warning("Generated-video saving is enabled, but policy output has no `video` field.")
            return
        try:
            import imageio.v2 as imageio

            frames = self._video_to_uint8_frames(video)
            fps = self._resolve_save_video_fps(payload)
            save_dir = payload.get("_save_generated_video_dir")
            path = Path(save_dir) / "generated_video.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(path), frames, fps=fps, quality=8, macro_block_size=1)
            logging.info("Saved generated video to %s (frames=%d, fps=%.3f)", path, len(frames), fps)
        except Exception:
            logging.exception("Failed to save generated video; continuing inference.")

    def _resolve_save_video_fps(self, payload: dict) -> float:
        fps = payload.get("fps", [self._warmup_fps])
        if isinstance(fps, (list, tuple)):
            fps = fps[0] if fps else self._warmup_fps
        stride = _config_get_nested(getattr(self._policy, "config", None), "datasets", "vla_data", "future_frame_stride", default=1)
        try:
            return max(float(fps) / max(int(stride), 1), 0.001)
        except (TypeError, ValueError):
            return 8.0

    @staticmethod
    def _video_to_uint8_frames(video) -> list[np.ndarray]:
        arr = video.detach().to(device="cpu", dtype=torch.float32).numpy()
        arr = np.transpose(arr[0], (1, 2, 3, 0))  # [B,C,T,H,W] -> [T,H,W,C]
        arr = (arr.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        return [np.ascontiguousarray(frame) for frame in arr]

    def _route_non_infer_message(self, mtype: str, payload: dict, req_id: str) -> dict:
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype == "reset":
            try:
                result = self._call_policy_reset(payload)
            except Exception as e:
                logging.exception("Policy reset error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
            if isinstance(result, dict) and "status" in result and "ok" in result:
                if "request_id" not in result:
                    result["request_id"] = req_id
                if "type" not in result:
                    result["type"] = "reset"
                return result
            return {
                "status": "ok",
                "ok": True,
                "type": "reset",
                "request_id": req_id,
                "data": result,
            }

        if mtype == "metadata":
            try:
                metadata = _resolve_policy_metadata(self._policy)
            except Exception as e:
                logging.exception("Policy metadata error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "metadata",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
            return {
                "status": "ok",
                "ok": True,
                "type": "metadata",
                "request_id": req_id,
                "metadata": metadata,
            }

        if hasattr(self._policy, "handle_request"):
            try:
                result = self._policy.handle_request(mtype, payload, request_id=req_id)
            except Exception as e:
                logging.exception("Policy request error (type=%s, request_id=%s)", mtype, req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": mtype,
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
            if isinstance(result, dict) and "status" in result and "ok" in result:
                if "request_id" not in result:
                    result["request_id"] = req_id
                if "type" not in result:
                    result["type"] = mtype
                return result
            return {
                "status": "ok",
                "ok": True,
                "type": mtype,
                "request_id": req_id,
                "data": result,
            }

        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "error": {"message": f"Unsupported message type '{mtype}'"},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise NotImplementedError("This module is not intended to be run directly.")
