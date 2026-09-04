"""Debug client for deployment/model_server/server_infersystem.py.

This tool speaks the same ZMQ/msgpack wire protocol as InferSystem's
InferenceClient, but uses synthetic images and state for server smoke tests.
"""

from __future__ import annotations

import argparse
from io import BytesIO

import msgpack
import numpy as np
from PIL import Image
import zmq


def _parse_camera_order(value: str) -> list[str]:
    cameras = [part.strip() for part in value.split(",") if part.strip()]
    if not cameras:
        raise argparse.ArgumentTypeError("--camera_order must contain at least one camera name.")
    if len(set(cameras)) != len(cameras):
        raise argparse.ArgumentTypeError(f"duplicate camera names in {value!r}.")
    return cameras


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def _build_predict_payload(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    payload: dict = {
        "cmd": "predict",
        "state": np.zeros((args.state_dim,), dtype=np.float32).tolist(),
        "prompt": args.prompt,
    }
    if args.stat_key:
        payload["extra"] = {"stat_key": args.stat_key}
    for camera in args.camera_order:
        image = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
        payload[camera] = _encode_jpeg(image, args.jpeg_quality)
    return payload


def _request(server: str, payload: dict, timeout_ms: int) -> dict:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://{server}")
        socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = socket.recv()
        return msgpack.unpackb(raw, raw=False)
    finally:
        socket.close()
        context.term()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InferSystem ZMQ protocol smoke-test client.")
    parser.add_argument("--server", default="127.0.0.1:5555", help="Server address without tcp:// prefix.")
    parser.add_argument("--camera_order", type=_parse_camera_order, required=True)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--prompt", default="pick up the cup")
    parser.add_argument("--stat_key", default="")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--timeout_ms", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cmd", choices=("metadata", "reset", "predict"), default="predict")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "predict":
        payload = _build_predict_payload(args)
    else:
        payload = {"cmd": args.cmd}

    response = _request(args.server, payload, args.timeout_ms)
    print(f"status={response.get('status')}")
    if response.get("status") == "ok" and "actions" in response:
        actions = np.asarray(response["actions"], dtype=np.float32)
        print(f"actions_shape={tuple(actions.shape)}")
        print(f"first_action={actions[0].tolist() if actions.size else []}")
    else:
        print(response)


if __name__ == "__main__":
    main()
