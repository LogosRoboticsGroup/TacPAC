import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open

from starVLA.model.modules.wan_mot.action_expert import ActionExpert


DEFAULT_CONFIG = "starVLA/config/training/vla/starvla_wam.yaml"
DEFAULT_PRETRAINED_DIR = "playground/Pretrained_models/Wan2.2-TI2V-5B"
DEFAULT_OUTPUT = "results/Checkpoints/WanMoT/ActionExpert_linear_interp_Wan22_alphascale_1024hdim.pt"


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected float32, float16, or bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _resolve_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    cfg = OmegaConf.load(str(config_path))
    if "framework" not in cfg:
        raise ValueError(f"`{config_path}` must contain top-level `framework` config.")

    action_cfg = OmegaConf.to_container(cfg.framework.action_model.config, resolve=True)
    video_cfg = OmegaConf.to_container(cfg.framework.video_model.config, resolve=True)
    if not isinstance(action_cfg, dict) or not isinstance(video_cfg, dict):
        raise ValueError("`framework.action_model.config` and `framework.video_model.config` must be dicts.")

    pretrained_dir = str(OmegaConf.select(cfg, "framework.video_model.dit_path") or DEFAULT_PRETRAINED_DIR)
    output_path = str(OmegaConf.select(cfg, "framework.action_model.model_path") or DEFAULT_OUTPUT)
    return action_cfg, video_cfg, pretrained_dir, output_path


def _normalize_action_config(action_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(action_cfg)
    int_fields = [
        "hidden_dim",
        "action_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
    ]
    for key in int_fields:
        normalized[key] = int(normalized[key])
    normalized["eps"] = float(normalized["eps"])
    normalized["use_gradient_checkpointing"] = _parse_bool(str(normalized.get("use_gradient_checkpointing", False)))
    return normalized


def _target_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


class TensorSource:
    def __init__(self, path: Path):
        self.path = path
        self.base_dir: Path | None = None
        self.weight_map: dict[str, str] | None = None
        self.single_safetensors: Path | None = None
        self.state_dict: dict[str, torch.Tensor] | None = None

        if path.is_dir():
            index_path = path / "diffusion_pytorch_model.safetensors.index.json"
            single_path = path / "diffusion_pytorch_model.safetensors"
            if index_path.exists():
                with index_path.open("r", encoding="utf-8") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict):
                    raise ValueError(f"`weight_map` missing from {index_path}")
                self.base_dir = path
                self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
                self.keys = set(self.weight_map.keys())
                return
            if single_path.exists():
                self.single_safetensors = single_path
                self.keys = self._safetensors_keys(single_path)
                return
            raise FileNotFoundError(f"No diffusion safetensors index or single file found in {path}")

        if path.suffix == ".safetensors":
            self.single_safetensors = path
            self.keys = self._safetensors_keys(path)
            return

        if path.suffix in {".pt", ".pth"}:
            try:
                self.state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except (TypeError, RuntimeError):
                self.state_dict = torch.load(path, map_location="cpu")
            if not isinstance(self.state_dict, dict):
                raise ValueError(f"Expected a state_dict from {path}, got {type(self.state_dict)}")
            self.keys = set(self.state_dict.keys())
            return

        raise ValueError(f"Unsupported pretrained source format: {path}")

    @staticmethod
    def _safetensors_keys(path: Path) -> set[str]:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return set(f.keys())

    def iter_grouped_tensors(self, keys: list[str]):
        if self.state_dict is not None:
            def read_state_dict():
                for key in keys:
                    yield key, self.state_dict[key]

            yield str(self.path), read_state_dict()
            return

        if self.single_safetensors is not None:
            def read_single():
                with safe_open(str(self.single_safetensors), framework="pt", device="cpu") as f:
                    for key in keys:
                        yield key, f.get_tensor(key)

            yield str(self.single_safetensors), read_single()
            return

        if self.weight_map is None or self.base_dir is None:
            raise RuntimeError("Invalid TensorSource state.")

        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            keys_by_shard[self.weight_map[key]].append(key)

        for shard_name in sorted(keys_by_shard):
            shard_path = self.base_dir / shard_name
            shard_keys = sorted(keys_by_shard[shard_name])

            def read_shard(shard_path=shard_path, shard_keys=shard_keys):
                if not shard_path.exists():
                    raise FileNotFoundError(f"Missing pretrained DiT shard: {shard_path}")
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in shard_keys:
                        yield key, f.get_tensor(key)

            yield str(shard_path), read_shard()


def _build_payload(
    action_cfg: dict[str, Any],
    video_cfg: dict[str, Any],
    pretrained_dir: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
    apply_alpha_scaling: bool,
) -> dict[str, Any]:
    if int(action_cfg["num_heads"]) != int(video_cfg["num_heads"]):
        raise ValueError("ActionExpert `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_cfg["attn_head_dim"]):
        raise ValueError("ActionExpert `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(video_cfg["num_layers"]):
        raise ValueError("ActionExpert `num_layers` must match video expert.")

    action_expert = ActionExpert(**action_cfg).to(device="cpu", dtype=torch_dtype)
    action_state = action_expert.state_dict()
    backbone_keys = sorted(ActionExpert.backbone_key_set(action_state.keys()))
    tensor_source = TensorSource(pretrained_dir)

    missing = [key for key in backbone_keys if key not in tensor_source.keys]
    if missing:
        raise ValueError(
            "Pretrained DiT state dict is missing ActionExpert backbone keys: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    processed = 0
    total = len(backbone_keys)

    with torch.no_grad():
        for source_name, tensor_iter in tensor_source.iter_grouped_tensors(backbone_keys):
            print(f"[INFO] Processing {source_name}.", flush=True)
            for key, src_cpu in tensor_iter:
                target = action_state[key]
                src = src_cpu.to(device=device, dtype=torch_dtype)
                target_shape = tuple(int(dim) for dim in target.shape)

                if tuple(src.shape) == target_shape:
                    value = src
                    copied += 1
                else:
                    value = _resize_tensor_to_shape(src, target_shape)
                    if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
                        alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
                        value = value.to(torch.float32) * alpha
                    interpolated += 1

                backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()
                processed += 1
                if processed % 50 == 0 or processed == total:
                    print(f"[INFO] Processed {processed}/{total} tensors.", flush=True)
                del src, value
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    skipped = len(action_state) - len(backbone_keys)
    print(
        f"[INFO] Completed interpolation (copied={copied}, interpolated={interpolated}, skipped={skipped}).",
        flush=True,
    )

    return {
        "policy": {
            "skip_prefixes": list(ActionExpert.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }


def _verify_payload(path: Path, action_cfg: dict[str, Any], torch_dtype: torch.dtype) -> None:
    ActionExpert.from_pretrained(
        action_dit_config=action_cfg,
        action_dit_pretrained_path=str(path),
        skip_dit_load_from_pretrain=False,
        device="cpu",
        torch_dtype=torch_dtype,
    )
    print(f"[INFO] Verified ActionExpert.from_pretrained can load {path}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate a Wan2.2-TI2V video DiT backbone into a 1024-hidden ActionExpert payload."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="WanMoT training config path.")
    parser.add_argument(
        "--pretrained-dir",
        default=None,
        help="Local Wan2.2-TI2V-5B directory or converted video DiT .pt/.pth/.safetensors file.",
    )
    parser.add_argument("--output", default=None, help="Output .pt path for the ActionExpert backbone payload.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Apply alpha=sqrt(source_last_dim/target_last_dim) after last-dim resize.",
    )
    parser.add_argument("--skip-verify-load", action="store_true", help="Skip final ActionExpert load check.")
    args = parser.parse_args()

    config_path = Path(args.config)
    action_cfg, video_cfg, config_pretrained_dir, config_output = _resolve_config(config_path)
    action_cfg = _normalize_action_config(action_cfg)
    pretrained_dir = Path(args.pretrained_dir or config_pretrained_dir or DEFAULT_PRETRAINED_DIR)
    output_path = Path(args.output or config_output or DEFAULT_OUTPUT)
    device = _target_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    if not pretrained_dir.exists():
        raise FileNotFoundError(f"Local pretrained DiT source does not exist: {pretrained_dir}")

    print(
        f"[INFO] Loading config={config_path}, pretrained_dir={pretrained_dir}, "
        f"output={output_path}, device={device}, dtype={torch_dtype}, "
        f"apply_alpha_scaling={apply_alpha_scaling}.",
        flush=True,
    )

    payload = _build_payload(
        action_cfg=action_cfg,
        video_cfg=video_cfg,
        pretrained_dir=pretrained_dir,
        device=device,
        torch_dtype=torch_dtype,
        apply_alpha_scaling=apply_alpha_scaling,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)
    print(f"[INFO] Saved ActionExpert backbone payload to {output_path}.", flush=True)

    if not args.skip_verify_load:
        _verify_payload(output_path, action_cfg, torch_dtype)


if __name__ == "__main__":
    main()
