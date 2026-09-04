#!/usr/bin/env python3
"""Replay EEF-pose CSV rows through the legacy OpenPI websocket protocol.

This matches openpi_client.websocket_client_policy.WebsocketClientPolicy from
Kai0/kai0_inference: the server sends metadata immediately after connection,
then each client message is a raw observation dict and each reply is a raw dict
with top-level ``actions``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.exceptions
from websockets.sync.server import serve


LOGGER = logging.getLogger("openpi_eef_pose_replay")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = (
    "data.csv"
)

ACTION_COLUMNS = [
    "left_x",
    "left_y",
    "left_z",
    "left_r1",
    "left_r2",
    "left_r3",
    "left_r4",
    "left_r5",
    "left_r6",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_r1",
    "right_r2",
    "right_r3",
    "right_r4",
    "right_r5",
    "right_r6",
    "right_gripper",
]


def pack_array(obj: Any) -> Any:
    """Same NumPy msgpack extension used by openpi_client.msgpack_numpy."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: dict[Any, Any]) -> Any:
    """Same NumPy msgpack extension used by openpi_client.msgpack_numpy."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_eef_pose_csv(csv_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header.")
        missing = [name for name in ACTION_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in ACTION_COLUMNS]
            except Exception as exc:
                raise ValueError(f"Failed to parse action row at line {line_no}: {exc}") from exc
            rows.append(values)

    if not rows:
        raise ValueError(f"{csv_path} contains no action rows.")
    actions = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(actions).all():
        bad = np.argwhere(~np.isfinite(actions))[0]
        raise ValueError(f"{csv_path} contains non-finite value at row={bad[0]} col={bad[1]}.")
    return actions


class EefPoseReplayState:
    def __init__(
        self,
        actions: np.ndarray,
        *,
        csv_path: Path,
        chunk_size: int,
        stride: int,
        start_index: int,
        warmup_requests: int,
        loop: bool,
    ) -> None:
        if actions.ndim != 2 or actions.shape[1] != 20:
            raise ValueError(f"Expected actions with shape [N, 20], got {actions.shape}.")
        self.actions = actions
        self.csv_path = csv_path
        self.chunk_size = max(1, int(chunk_size))
        self.stride = max(1, int(stride))
        self.start_index = int(np.clip(start_index, 0, len(actions) - 1))
        self.warmup_requests = max(0, int(warmup_requests))
        self.loop = bool(loop)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.cursor = self.start_index
            self.infer_count = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "server": "openpi_eef_pose_csv_replay",
            "protocol": "legacy_openpi_websocket",
            "action_format": "absolute_eef_20d_xyz_rot6d_gripper_left_right",
            "action_dim": 20,
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "warmup_requests": self.warmup_requests,
            "loop": self.loop,
            "episode_length": int(len(self.actions)),
            "csv_path": str(self.csv_path),
            "note": "Use with ctrl_type=delta_eef; client delta_eef_publish applies these as absolute EEF targets.",
        }

    def next_reply(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        with self._lock:
            self.infer_count += 1
            infer_count = self.infer_count
            warmup = infer_count <= self.warmup_requests
            start = int(self.cursor)
            chunk, indices = self._chunk_from(start)
            if not warmup:
                self.cursor = self._advance_cursor(self.cursor)
            next_cursor = int(self.cursor)

        first = chunk[0]
        LOGGER.info(
            "infer=%06d warmup=%s cursor=%d next_cursor=%d csv_idx=%d..%d "
            "left_xyz=(%.4f,%.4f,%.4f) left_grip=%.6f "
            "right_xyz=(%.4f,%.4f,%.4f) right_grip=%.6f",
            infer_count,
            warmup,
            start,
            next_cursor,
            int(indices[0]),
            int(indices[-1]),
            float(first[0]),
            float(first[1]),
            float(first[2]),
            float(first[9]),
            float(first[10]),
            float(first[11]),
            float(first[12]),
            float(first[19]),
        )
        return {
            "actions": chunk,
            "metadata": {
                "infer_count": infer_count,
                "warmup": warmup,
                "cursor": next_cursor,
                "csv_start_index": int(indices[0]),
                "csv_end_index": int(indices[-1]),
                "done": bool((not self.loop) and next_cursor >= len(self.actions) - 1),
            },
            "infer_time_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        }

    def _chunk_from(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        if self.loop:
            indices = (np.arange(self.chunk_size, dtype=np.int64) + start) % len(self.actions)
            return self.actions[indices].copy(), indices

        raw_indices = np.arange(start, start + self.chunk_size, dtype=np.int64)
        indices = np.clip(raw_indices, 0, len(self.actions) - 1)
        return self.actions[indices].copy(), indices

    def _advance_cursor(self, cursor: int) -> int:
        if self.loop:
            return int((cursor + self.stride) % len(self.actions))
        return int(min(cursor + self.stride, len(self.actions) - 1))


class LegacyOpenPIReplayServer:
    def __init__(self, state: EefPoseReplayState, *, host: str, port: int, reset_on_connect: bool) -> None:
        self.state = state
        self.host = host
        self.port = int(port)
        self.reset_on_connect = bool(reset_on_connect)
        self.packer = Packer()

    def serve_forever(self) -> None:
        LOGGER.info("Starting legacy OpenPI replay server on ws://%s:%d", self.host, self.port)
        LOGGER.info("Advertised metadata: %s", self.state.metadata())
        with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            server.serve_forever()

    def _handler(self, websocket) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOGGER.info("Client connected: %s", peer)
        if self.reset_on_connect:
            self.state.reset()
        websocket.send(self.packer.pack(self.state.metadata()))

        while True:
            try:
                raw = websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                LOGGER.info("Client disconnected: %s", peer)
                return

            if isinstance(raw, str):
                websocket.send(f"Expected binary msgpack observation, got text: {raw[:200]}")
                continue

            try:
                msg = unpackb(raw)
            except Exception as exc:
                LOGGER.exception("Failed to decode client msgpack.")
                websocket.send(f"Failed to decode msgpack observation: {exc}")
                continue

            if isinstance(msg, dict) and msg.get("cmd") == "reset":
                self.state.reset()
                websocket.send(self.packer.pack({"status": "ok", "metadata": self.state.metadata()}))
                continue

            websocket.send(self.packer.pack(self.state.next_reply()))


def local_ip() -> str:
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return "127.0.0.1"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="EEF-pose CSV path. Relative paths use repo root.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="CSV frames to advance after each non-warmup infer.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Initial infer requests that return a chunk without advancing. The Agilex script discards one warmup infer.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-reset-on-connect", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    csv_path = resolve_csv_path(args.csv_path)
    actions = load_eef_pose_csv(csv_path)
    LOGGER.info("Loaded %s actions from %s", actions.shape, csv_path)
    LOGGER.info("Remote clients can use --host %s --port %d", local_ip(), args.port)

    state = EefPoseReplayState(
        actions,
        csv_path=csv_path,
        chunk_size=args.chunk_size,
        stride=args.stride,
        start_index=args.start_index,
        warmup_requests=args.warmup_requests,
        loop=args.loop,
    )
    server = LegacyOpenPIReplayServer(
        state,
        host=args.host,
        port=args.port,
        reset_on_connect=not args.no_reset_on_connect,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
