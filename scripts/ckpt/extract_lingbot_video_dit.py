#!/usr/bin/env python
"""Extract LingBot-VA video DiT weights for starVLA wan_video/wan_mot.

LingBot-VA stores video and action parameters in one shared transformer. This
script writes a starVLA-compatible video-only state_dict that can be used as:

  framework.video_model.dit_path: path/to/lingbot_video_dit.pt
  framework.wan_video.dit_path: path/to/lingbot_video_dit.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_SOURCE = "playground/Pretrained_models/lingbot-va-base"
DEFAULT_OUTPUT = "playground/Pretrained_models/lingbot-va-base-video-dit.pt"


class TensorReader:
    def __init__(self, path: Path):
        self.path = path
        self.state_dict = None
        self.weight_map = None
        self.base_dir = None
        self.single_safetensors = None

        if path.is_dir():
            index_path = path / "diffusion_pytorch_model.safetensors.index.json"
            single_path = path / "diffusion_pytorch_model.safetensors"
            if index_path.exists():
                with index_path.open("r", encoding="utf-8") as f:
                    index = json.load(f)
                self.weight_map = dict(index["weight_map"])
                self.base_dir = path
                self.keys = set(self.weight_map.keys())
                return
            if single_path.exists():
                self.single_safetensors = single_path
                self.keys = self._safetensors_keys(single_path)
                return
            raise FileNotFoundError(f"No transformer safetensors found in {path}")

        if path.suffix == ".safetensors":
            self.single_safetensors = path
            self.keys = self._safetensors_keys(path)
            return

        if path.suffix in {".pt", ".pth"}:
            try:
                self.state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except (TypeError, RuntimeError):
                self.state_dict = torch.load(path, map_location="cpu")
            self.keys = set(self.state_dict.keys())
            return

        raise ValueError(f"Unsupported input format: {path}")

    @staticmethod
    def _safetensors_keys(path: Path) -> set[str]:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            return set(f.keys())

    def get(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            raise KeyError(key)
        if self.state_dict is not None:
            return self.state_dict[key]

        from safetensors import safe_open

        if self.weight_map is not None:
            tensor_path = self.base_dir / self.weight_map[key]
        else:
            tensor_path = self.single_safetensors
        with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def resolve_transformer_path(source: str) -> Path:
    path = Path(source)
    transformer = path / "transformer"
    if transformer.is_dir():
        return transformer
    return path


def read_num_layers(transformer_path: Path, keys: Iterable[str]) -> int:
    config_path = transformer_path / "config.json" if transformer_path.is_dir() else None
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if "num_layers" in config:
            return int(config["num_layers"])

    layer_ids = []
    for key in keys:
        match = re.match(r"^blocks\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        raise ValueError("Could not infer number of transformer layers.")
    return max(layer_ids) + 1


def require_lingbot_signature(keys: set[str]) -> None:
    required = {
        "condition_embedder.text_embedder.linear_1.weight",
        "condition_embedder.time_embedder.linear_1.weight",
        "proj_out.weight",
        "action_embedder.weight",
        "blocks.0.attn1.to_q.weight",
        "blocks.0.attn2.to_q.weight",
    }
    missing = sorted(required - keys)
    if missing:
        raise ValueError(
            "Input does not look like a LingBot-VA shared transformer; "
            f"missing keys: {missing}"
        )


def build_key_map(num_layers: int) -> dict[str, str]:
    key_map = {
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",
        "scale_shift_table": "head.modulation",
    }

    per_block = {
        "attn1.to_q.weight": "self_attn.q.weight",
        "attn1.to_q.bias": "self_attn.q.bias",
        "attn1.to_k.weight": "self_attn.k.weight",
        "attn1.to_k.bias": "self_attn.k.bias",
        "attn1.to_v.weight": "self_attn.v.weight",
        "attn1.to_v.bias": "self_attn.v.bias",
        "attn1.to_out.0.weight": "self_attn.o.weight",
        "attn1.to_out.0.bias": "self_attn.o.bias",
        "attn1.norm_q.weight": "self_attn.norm_q.weight",
        "attn1.norm_k.weight": "self_attn.norm_k.weight",
        "attn2.to_q.weight": "cross_attn.q.weight",
        "attn2.to_q.bias": "cross_attn.q.bias",
        "attn2.to_k.weight": "cross_attn.k.weight",
        "attn2.to_k.bias": "cross_attn.k.bias",
        "attn2.to_v.weight": "cross_attn.v.weight",
        "attn2.to_v.bias": "cross_attn.v.bias",
        "attn2.to_out.0.weight": "cross_attn.o.weight",
        "attn2.to_out.0.bias": "cross_attn.o.bias",
        "attn2.norm_q.weight": "cross_attn.norm_q.weight",
        "attn2.norm_k.weight": "cross_attn.norm_k.weight",
        "norm2.weight": "norm3.weight",
        "norm2.bias": "norm3.bias",
        "ffn.net.0.proj.weight": "ffn.0.weight",
        "ffn.net.0.proj.bias": "ffn.0.bias",
        "ffn.net.2.weight": "ffn.2.weight",
        "ffn.net.2.bias": "ffn.2.bias",
        "scale_shift_table": "modulation",
    }

    for layer_idx in range(num_layers):
        for src_suffix, dst_suffix in per_block.items():
            key_map[f"blocks.{layer_idx}.{src_suffix}"] = f"blocks.{layer_idx}.{dst_suffix}"
    return key_map


def extract_video_dit(reader: TensorReader, num_layers: int) -> dict[str, torch.Tensor]:
    require_lingbot_signature(reader.keys)
    output = {}

    if "patch_embedding_mlp.weight" not in reader.keys:
        raise ValueError("LingBot-VA checkpoint is missing `patch_embedding_mlp.weight`.")
    if "patch_embedding.weight" not in reader.keys:
        raise ValueError("LingBot-VA checkpoint is missing `patch_embedding.weight` for shape inference.")

    patch_mlp_weight = reader.get("patch_embedding_mlp.weight")
    patch_conv_shape = reader.get("patch_embedding.weight").shape
    output["patch_embedding.weight"] = patch_mlp_weight.reshape(patch_conv_shape).contiguous()

    if "patch_embedding_mlp.bias" in reader.keys:
        output["patch_embedding.bias"] = reader.get("patch_embedding_mlp.bias")

    key_map = build_key_map(num_layers)
    missing = sorted(src for src in key_map if src not in reader.keys)
    if missing:
        raise ValueError(f"Missing required video DiT keys: {missing[:20]}")

    for src, dst in key_map.items():
        output[dst] = reader.get(src)

    bad_prefixes = ("action_", "condition_embedder_action", "tactile", "local_tactile", "contact_gate")
    leaked = [key for key in output if key.startswith(bad_prefixes)]
    if leaked:
        raise RuntimeError(f"Unexpected action/tactile keys leaked into output: {leaked[:20]}")

    return output


def save_state_dict(state_dict: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, str(output_path))
        return
    if output_path.suffix not in {".pt", ".pth"}:
        raise ValueError("Output must end with .pt, .pth, or .safetensors")
    torch.save(state_dict, output_path)


def write_metadata(
    metadata_path: Path,
    *,
    source: Path,
    transformer_path: Path,
    output_path: Path,
    num_layers: int,
    input_key_count: int,
    output_key_count: int,
    dropped_groups: list[str],
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source),
        "transformer_path": str(transformer_path),
        "output_path": str(output_path),
        "format": "starVLA wan_video/wan_mot video DiT state_dict",
        "num_layers": num_layers,
        "input_key_count": input_key_count,
        "output_key_count": output_key_count,
        "dropped_groups_present_in_source": dropped_groups,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="LingBot-VA root or transformer directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .pt/.pth/.safetensors path.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON path. Defaults to '<output>.metadata.json'.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing weights.")
    return parser.parse_args()


def summarize_dropped_groups(input_keys: set[str], output_keys: set[str]) -> list[str]:
    groups = []
    group_patterns = {
        "action_embedder": lambda key: key.startswith("action_embedder."),
        "action_proj_out": lambda key: key.startswith("action_proj_out."),
        "condition_embedder_action": lambda key: key.startswith("condition_embedder_action."),
        "tactile": lambda key: "tactile" in key,
        "contact_gate": lambda key: "contact_gate" in key or key.startswith("contact_gate."),
    }
    for name, predicate in group_patterns.items():
        if any(predicate(key) for key in input_keys) and not any(predicate(key) for key in output_keys):
            groups.append(name)
    return groups


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    transformer_path = resolve_transformer_path(args.source)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(str(output_path) + ".metadata.json")

    reader = TensorReader(transformer_path)
    num_layers = read_num_layers(transformer_path, reader.keys)
    state_dict = extract_video_dit(reader, num_layers)
    dropped_groups = summarize_dropped_groups(reader.keys, set(state_dict.keys()))

    print(f"source: {source}")
    print(f"transformer: {transformer_path}")
    print(f"input keys: {len(reader.keys)}")
    print(f"output keys: {len(state_dict)}")
    print(f"layers: {num_layers}")
    print(f"output: {output_path}")

    if args.dry_run:
        print("dry run: not writing output")
        return

    save_state_dict(state_dict, output_path)
    write_metadata(
        metadata_path,
        source=source,
        transformer_path=transformer_path,
        output_path=output_path,
        num_layers=num_layers,
        input_key_count=len(reader.keys),
        output_key_count=len(state_dict),
        dropped_groups=dropped_groups,
    )
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
