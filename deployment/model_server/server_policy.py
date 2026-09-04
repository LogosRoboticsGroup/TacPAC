# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

import torch

from deployment.model_server.inference_loader import load_framework_for_inference
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    if args.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    vla = load_framework_for_inference(args.ckpt_path)
    vla.set_dataconfig(args.stat_key)

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        enable_compile=args.compile,
        log_timing_every=args.log_timing_every,
    )
    logging.info("server running on ws://%s:%d ... metadata=%s", local_ip, args.port, vla.get_metadata())
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--stat_key", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log_timing_every", type=int, default=0)
    return parser


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
