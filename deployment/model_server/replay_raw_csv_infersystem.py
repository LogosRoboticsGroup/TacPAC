"""Replay raw joint-position CSV actions through the InferSystem ZMQ protocol."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import pandas as pd
import zmq


logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    "playground/Datasets/neoteai/raw/wipe_the_white_plastic_board_Aloha/PiPER/20260615/"
    "leijunyang/episode_0000_20260616-014321_leijunyang_aloha_none/"
    "actions.joint_position/data.csv"
)

ACTION_COLUMNS = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
    "left_gripper",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_j6",
    "right_gripper",
]


def load_actions(csv_path: str | Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[ACTION_COLUMNS].to_numpy(dtype=np.float32)


class RawCsvReplayServer:
    def __init__(
        self,
        actions: np.ndarray,
        *,
        bind: str,
        chunk_size: int = 10,
        pause_every: int = 10,
        pause_seconds: float = 0.1,
        loop: bool = False,
    ) -> None:
        self.actions = actions
        self.bind = bind
        self.chunk_size = int(chunk_size)
        self.pause_every = int(pause_every)
        self.pause_seconds = float(pause_seconds)
        self.loop = loop
        self.cursor = 0
        self.sent_count = 0
        self._sleep_after_send = 0.0

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.bind)
        logger.info("Raw CSV replay server listening on %s", self.bind)

        try:
            while True:
                msg = msgpack.unpackb(socket.recv(), raw=False, strict_map_key=False)
                reply = self._handle(msg)
                socket.send(msgpack.packb(reply, use_bin_type=True))
                self._pause_if_needed()
        except KeyboardInterrupt:
            logger.info("Raw CSV replay server stopped.")
        finally:
            socket.close()
            context.term()

    def _handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        cmd = msg.get("cmd", "predict")
        if cmd == "reset":
            self.cursor = 0
            self.sent_count = 0
            return {"status": "ok"}
        if cmd == "metadata":
            return {"status": "ok", "metadata": self._metadata()}
        if cmd != "predict":
            return {"status": "error", "message": f"unknown cmd: {cmd}"}

        t0 = time.perf_counter()
        chunk, csv_steps = self._next_chunk()
        self._print_grippers(chunk, csv_steps)

        return {
            "status": "ok",
            "actions": chunk.tolist(),
            "infer_time_ms": round((time.perf_counter() - t0) * 1000, 3),
            "metadata": {
                **self._metadata(),
                "cursor": self.cursor,
                "done": self.cursor >= len(self.actions),
            },
        }

    def _next_chunk(self) -> tuple[np.ndarray, list[int]]:
        if self.loop:
            idx = (np.arange(self.chunk_size) + self.cursor) % len(self.actions)
            self.cursor = (self.cursor + self.chunk_size) % len(self.actions)
            return self.actions[idx], idx.astype(int).tolist()

        end = min(self.cursor + self.chunk_size, len(self.actions))
        csv_steps = list(range(self.cursor, end))
        chunk = self.actions[self.cursor:end]
        self.cursor = end

        if len(chunk) < self.chunk_size:
            pad_n = self.chunk_size - len(chunk)
            chunk = np.concatenate([chunk, np.repeat(self.actions[-1][None], pad_n, axis=0)])
            csv_steps.extend([len(self.actions) - 1] * pad_n)
        return chunk, csv_steps

    def _print_grippers(self, chunk: np.ndarray, csv_steps: list[int]) -> None:
        for action, csv_step in zip(chunk, csv_steps):
            self.sent_count += 1
            print(
                f"send={self.sent_count:06d} csv_step={csv_step:06d} "
                f"left_gripper={action[6]:.6f} right_gripper={action[13]:.6f}",
                flush=True,
            )
            if self.pause_every > 0 and self.sent_count % self.pause_every == 0:
                self._sleep_after_send += self.pause_seconds

    def _pause_if_needed(self) -> None:
        if self._sleep_after_send <= 0:
            return
        time.sleep(self._sleep_after_send)
        self._sleep_after_send = 0.0

    def _metadata(self) -> dict[str, Any]:
        return {
            "env": "starvla_raw_csv_replay_server",
            "episode_length": int(len(self.actions)),
            "action_dim": int(self.actions.shape[1]),
            "chunk_size": self.chunk_size,
            "pause_every": self.pause_every,
            "pause_seconds": self.pause_seconds,
            "loop": self.loop,
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay raw joint-position CSV actions over InferSystem ZMQ/msgpack.")
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--bind", default="tcp://*:5555")
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--pause_every", type=int, default=10)
    parser.add_argument("--pause_seconds", type=float, default=0.1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    actions = load_actions(args.csv_path)
    logger.info("Loaded %s: actions=%s", args.csv_path, tuple(actions.shape))

    server = RawCsvReplayServer(
        actions,
        bind=args.bind,
        chunk_size=args.chunk_size,
        pause_every=args.pause_every,
        pause_seconds=args.pause_seconds,
        loop=args.loop,
    )
    server.run()


if __name__ == "__main__":
    main()
