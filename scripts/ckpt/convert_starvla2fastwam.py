#!/usr/bin/env python3
"""Convert a StarVLA flat checkpoint to FastWAM payload format.

StarVLA WanMoT checkpoints are saved as a flat state_dict with keys like
``mot.mixtures.video...`` and ``proprio_encoder.weight``. FastWAM expects a
payload with top-level keys like ``mot`` and ``proprio_encoder``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}: {path}")
    return obj


def _infer_torch_dtype(payload: dict[str, Any]) -> str:
    for value in payload.values():
        if isinstance(value, torch.Tensor):
            return str(value.dtype)
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, torch.Tensor):
                    return str(nested.dtype)
    return "torch.bfloat16"


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def convert_checkpoint(src: Path, dst: Path, step: int | None, overwrite: bool) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser()

    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and dst.resolve() == src:
        raise ValueError(f"Refusing to overwrite source checkpoint: {dst}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists; pass --overwrite to replace it: {dst}")

    state_dict = _torch_load(src)

    mot = _strip_prefix(state_dict, "mot.")
    source_prefix = "mot."
    if not mot:
        mot = _strip_prefix(state_dict, "dit.")
        source_prefix = "dit."
    if not mot:
        sample = list(state_dict.keys())[:20]
        raise ValueError(f"No mot.* or dit.* keys found in {src}. First keys: {sample}")

    proprio_encoder = _strip_prefix(state_dict, "proprio_encoder.")

    payload: dict[str, Any] = {
        "mot": mot,
        "step": step,
        "torch_dtype": _infer_torch_dtype(mot),
    }
    if proprio_encoder:
        payload["proprio_encoder"] = proprio_encoder

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"[load] {src}", flush=True)
    print(
        f"[convert] source_prefix={source_prefix} "
        f"mot_keys={len(mot)} proprio_keys={len(proprio_encoder)} "
        f"torch_dtype={payload['torch_dtype']}",
        flush=True,
    )
    print(f"[save] {tmp}", flush=True)
    torch.save(payload, tmp)
    os.replace(tmp, dst)
    print(f"[done] {dst}", flush=True)


def update_symlink(link: Path, target: Path, overwrite_link: bool) -> None:
    link = link.expanduser()
    target = target.expanduser().resolve()

    if link.exists() or link.is_symlink():
        if not link.is_symlink() and not overwrite_link:
            raise FileExistsError(f"Link path exists and is not a symlink: {link}")
        link.unlink()

    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    print(f"[link] {link} -> {target}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="StarVLA flat .pt checkpoint.")
    parser.add_argument("dst", type=Path, help="Output FastWAM-format .pt checkpoint.")
    parser.add_argument("--step", type=int, default=None, help="Optional step value to store in the payload.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dst if it already exists.")
    parser.add_argument(
        "--link",
        type=Path,
        default=None,
        help="Optional symlink path to update so FastWAM can keep using an existing ckpt path.",
    )
    parser.add_argument(
        "--overwrite-link",
        action="store_true",
        help="Allow --link to replace a regular file. Existing symlinks are always replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(args.src, args.dst, step=args.step, overwrite=args.overwrite)
    if args.link is not None:
        update_symlink(args.link, args.dst, overwrite_link=args.overwrite_link)


if __name__ == "__main__":
    main()
