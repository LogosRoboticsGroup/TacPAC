# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import inspect
import itertools
import logging
import os
import time
from threading import Lock
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

import websockets.exceptions
import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Synchronous websocket client for the LogosVLA policy protocol."""

    _RETRY_ON_RECV_REQUEST_TYPES = {"infer", "metadata", "ping", "trainer_status"}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 10093,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._uri = self._build_ws_uri(host=host, port=port)
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._verbose = bool(verbose)
        self._ws = self._wait_for_server()
        self._request_lock = Lock()
        self._request_counter = itertools.count()

    @staticmethod
    def _expected_response_type(request_type: str) -> str:
        if request_type == "infer":
            return "inference_result"
        return request_type

    def _next_request_id(self, request_type: str) -> str:
        return f"{request_type}-{next(self._request_counter)}"

    def _reconnect(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = self._wait_for_server()

    @staticmethod
    def _is_legacy_response(request_type: str, response: Dict) -> bool:
        if request_type == "metadata":
            return "metadata" not in response and "type" not in response and "status" not in response
        if request_type == "infer":
            return "actions" in response and "type" not in response and "status" not in response
        return False

    @staticmethod
    def _wrap_legacy_response(request_type: str, response: Dict, request_id: str) -> Dict:
        if request_type == "metadata":
            return {
                "status": "ok",
                "ok": True,
                "type": "metadata",
                "request_id": request_id,
                "metadata": response,
            }
        if request_type == "infer":
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": request_id,
                "data": response,
            }
        return response

    @staticmethod
    def _build_ws_uri(host: str, port: Optional[int]) -> str:
        host = str(host or "127.0.0.1").strip()
        if host in {"0.0.0.0", "localhost:0"}:
            host = "127.0.0.1"

        if "://" in host:
            parsed = urlparse(host)
            scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
        else:
            parsed = urlparse(f"ws://{host}")
            scheme = "ws"

        has_proxy_path = parsed.path not in {"", "/"}
        has_explicit_port = parsed.port is not None
        if port is not None and not has_proxy_path and not has_explicit_port:
            parsed = parsed._replace(netloc=f"{parsed.netloc}:{int(port)}")

        return urlunparse(parsed._replace(scheme=scheme))

    def _wait_for_server(self, timeout: float = 600) -> websockets.sync.client.ClientConnection:
        logging.debug("Waiting for server at %s", self._uri)
        start_time = time.time()

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connect_kwargs = dict(
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                )
                if "ping_interval" in inspect.signature(websockets.sync.client.connect).parameters:
                    connect_kwargs["ping_interval"] = None
                return websockets.sync.client.connect(self._uri, **connect_kwargs)
            except ConnectionRefusedError:
                logging.debug("Still waiting for server %s ...", self._uri)
                time.sleep(2)

    def init_device(self, device: str = "cuda") -> Dict:
        return self.request("ping", {"device": device})

    def request(self, request_type: str, payload: Optional[Dict] = None, request_id: Optional[str] = None) -> Dict:
        if request_id is None:
            request_id = self._next_request_id(request_type)
        pack_started_at = time.perf_counter()
        query_info = {
            "payload": payload or {},
            "type": request_type,
            "request_id": request_id,
        }
        data = self._packer.pack(query_info)
        pack_ms = (time.perf_counter() - pack_started_at) * 1000
        with self._request_lock:
            roundtrip_started_at = time.perf_counter()
            try:
                self._ws.send(data)
            except websockets.exceptions.ConnectionClosed as exc:
                logging.warning(
                    "Websocket connection closed before sending %s request (%s); reconnecting once.",
                    request_type,
                    exc,
                )
                self._reconnect()
                self._ws.send(data)
            retried_after_recv_close = False
            while True:
                try:
                    response = self._ws.recv()
                except websockets.exceptions.ConnectionClosed as exc:
                    if retried_after_recv_close or request_type not in self._RETRY_ON_RECV_REQUEST_TYPES:
                        raise
                    logging.warning(
                        "Websocket connection closed while waiting for %s response (%s); reconnecting and retrying once.",
                        request_type,
                        exc,
                    )
                    self._reconnect()
                    self._ws.send(data)
                    retried_after_recv_close = True
                    roundtrip_started_at = time.perf_counter()
                    continue
                roundtrip_ms = (time.perf_counter() - roundtrip_started_at) * 1000
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                unpack_started_at = time.perf_counter()
                unpacked = msgpack_numpy.unpackb(response)
                unpack_ms = (time.perf_counter() - unpack_started_at) * 1000
                if not isinstance(unpacked, dict):
                    raise RuntimeError(f"{request_type} request returned non-dict response: {type(unpacked)!r}")
                if self._is_legacy_response(request_type, unpacked):
                    unpacked = self._wrap_legacy_response(request_type, unpacked, request_id)
                actual_request_id = unpacked.get("request_id")
                if actual_request_id is not None and actual_request_id != request_id:
                    logging.warning(
                        "Discarding out-of-order websocket response for %s: expected request_id=%r, got %r.",
                        request_type,
                        request_id,
                        actual_request_id,
                    )
                    continue
                expected_type = self._expected_response_type(request_type)
                actual_type = unpacked.get("type")
                if actual_type is not None and actual_type != expected_type:
                    raise RuntimeError(
                        f"{request_type} response type mismatch: expected {expected_type!r}, got {actual_type!r}."
                    )
                if not bool(unpacked.get("ok", True)):
                    error = unpacked.get("error", {})
                    message = error.get("message", "Unknown server error.")
                    raise RuntimeError(f"{request_type} request failed: {message}")
                unpacked["client_timing"] = {
                    "pack_ms": round(pack_ms, 3),
                    "roundtrip_ms": round(roundtrip_ms, 3),
                    "unpack_ms": round(unpack_ms, 3),
                }
                if self._verbose and request_type == "infer":
                    server_timing = unpacked.get("timing") or {}
                    server_total_ms = float(server_timing.get("server_total_ms", 0.0))
                    network_ms = max(0.0, roundtrip_ms - server_total_ms)
                    print(
                        f"[ws:{request_type}] pack={pack_ms:.1f}ms roundtrip={roundtrip_ms:.1f}ms "
                        f"unpack={unpack_ms:.1f}ms est_network={network_ms:.1f}ms"
                    )
                return unpacked

    @override
    def infer(self, obs: Dict) -> Dict:
        return self.request("infer", obs)

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        return self.infer(query_info)

    @override
    def get_metadata(self) -> Dict:
        return self.request("metadata")["metadata"]

    def get_server_metadata(self) -> Dict:
        return self.get_metadata()

    @override
    def reset(
        self,
        instruction=None,
        *,
        clear_replay: bool = False,
        clear_registry: bool = True,
        server_rtc: Optional[bool] = None,
        execution_steps: Optional[int] = None,
        prefix_steps: Optional[int] = None,
    ) -> Dict:
        payload = {}
        if instruction is not None:
            payload["instruction"] = instruction
        payload["clear_replay"] = bool(clear_replay)
        payload["clear_registry"] = bool(clear_registry)
        if server_rtc is not None:
            payload["server_rtc"] = bool(server_rtc)
        if execution_steps is not None:
            payload["execution_steps"] = int(execution_steps)
        if prefix_steps is not None:
            payload["prefix_steps"] = int(prefix_steps)
        return self.request("reset", payload)

    def trainer_status(self) -> Dict:
        return self.request("trainer_status")

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
