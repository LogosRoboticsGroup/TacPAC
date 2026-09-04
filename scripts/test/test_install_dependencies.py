from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencySpec:
    package: str
    module: str
    group: str
    required: bool
    note: str = ""


GROUP_LABELS = {
    "core": "Core Training / Runtime",
    "deploy": "Deploy Common",
    "optional": "Optional Accelerators / Hardware",
}


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("torch", "torch", "core", True),
    DependencySpec("torchvision", "torchvision", "core", True),
    DependencySpec("transformers", "transformers", "core", True),
    DependencySpec("accelerate", "accelerate", "core", True),
    DependencySpec("tiktoken", "tiktoken", "core", True),
    DependencySpec("einops", "einops", "core", True),
    DependencySpec("transformers_stream_generator", "transformers_stream_generator", "core", True),
    DependencySpec("scipy", "scipy", "core", True),
    DependencySpec("Pillow", "PIL", "core", True),
    DependencySpec("psutil", "psutil", "core", True),
    DependencySpec("tensorboard", "tensorboard", "core", True),
    DependencySpec("matplotlib", "matplotlib", "core", True),
    DependencySpec("msgpack", "msgpack", "core", True),
    DependencySpec("websockets", "websockets", "core", True),
    DependencySpec("websocket-client", "websocket", "core", True),
    DependencySpec("albumentations", "albumentations", "core", True),
    DependencySpec("pipablepytorch3d", "pytorch3d", "core", True),
    DependencySpec("pydantic", "pydantic", "core", True),
    DependencySpec("pyarrow", "pyarrow", "core", True),
    DependencySpec("fastparquet", "fastparquet", "core", True),
    DependencySpec("av", "av", "core", True),
    DependencySpec("numpydantic", "numpydantic", "core", True),
    DependencySpec("deepspeed", "deepspeed", "core", True),
    DependencySpec("qwen-vl-utils", "qwen_vl_utils", "core", True),
    DependencySpec("omegaconf", "omegaconf", "core", True),
    DependencySpec("numpy", "numpy", "core", True),
    DependencySpec("rich", "rich", "core", True),
    DependencySpec("diffusers", "diffusers", "core", True),
    DependencySpec("timm", "timm", "core", True),
    DependencySpec("torchcodec", "torchcodec", "core", True),
    DependencySpec("datasets", "datasets", "core", True),
    DependencySpec("pin", "pinocchio", "core", True),
    DependencySpec("yourdfpy", "yourdfpy", "core", True),
    DependencySpec("draccus", "draccus", "deploy", True, "deployment config registry"),
    DependencySpec("opencv-python", "cv2", "deploy", True, "model server / camera wrappers"),
    DependencySpec("pyzmq", "zmq", "deploy", True, "InferSystem ZMQ policy server"),
    DependencySpec("pynput", "pynput", "deploy", True, "keyboard control input"),
    DependencySpec("pyserial", "serial", "deploy", True, "Robotiq serial utilities"),
    DependencySpec("imageio-ffmpeg", "imageio_ffmpeg", "deploy", True, "video export / encoding"),
    DependencySpec("flash-attn", "flash_attn", "optional", False, "attention acceleration"),
    DependencySpec("decord", "decord", "optional", False, "video dataloading"),
    DependencySpec("lerobot", "lerobot", "optional", False, "collect / replay / teleoperate entrypoints"),
    DependencySpec("pyrealsense2", "pyrealsense2", "optional", False, "RealSense cameras"),
    DependencySpec("ur-rtde", "rtde_control", "optional", False, "UR robot control"),
    DependencySpec("ur-rtde", "rtde_receive", "optional", False, "UR robot state receive"),
    DependencySpec("dynamixel-sdk", "dynamixel_sdk", "optional", False, "Gello teleoperation"),
    DependencySpec("viser", "viser", "optional", False, "VR / visualization tools"),
)


def _metadata_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_dependency(spec: DependencySpec, mode: str) -> tuple[bool, str]:
    if mode == "spec":
        module_spec = importlib.util.find_spec(spec.module)
        if module_spec is None:
            return False, "module spec not found"
        version = _metadata_version(spec.package)
        location = module_spec.origin or "namespace-package"
        if version is not None:
            return True, f"version={version}, origin={location}"
        return True, f"origin={location}"

    try:
        module = importlib.import_module(spec.module)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    version = _metadata_version(spec.package)
    location = getattr(module, "__file__", None) or "built-in"
    if version is not None:
        return True, f"version={version}, origin={location}"
    return True, f"origin={location}"


def _selected_dependencies(groups: list[str]) -> list[DependencySpec]:
    return [spec for spec in DEPENDENCIES if spec.group in groups]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether LogosVLA dependencies are installed. "
            "Default mode uses importlib.find_spec() so headless-only packages such as pynput "
            "can still be validated without opening a display connection."
        )
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(GROUP_LABELS),
        default=["core", "deploy", "optional"],
        help="Dependency groups to check.",
    )
    parser.add_argument(
        "--mode",
        choices=("spec", "import"),
        default="spec",
        help="Probe via module spec lookup or real import.",
    )
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="Fail when optional packages are missing.",
    )
    args = parser.parse_args()

    missing_required: list[DependencySpec] = []
    missing_optional: list[DependencySpec] = []

    for group in args.groups:
        specs = [spec for spec in _selected_dependencies(args.groups) if spec.group == group]
        if not specs:
            continue
        print(f"== {GROUP_LABELS[group]} ==")
        for spec in specs:
            ok, detail = _probe_dependency(spec, args.mode)
            note = f" [{spec.note}]" if spec.note else ""
            if ok:
                print(f"OK       {spec.package:<24} module={spec.module:<20} {detail}{note}")
                continue

            print(f"MISSING  {spec.package:<24} module={spec.module:<20} {detail}{note}")
            if spec.required:
                missing_required.append(spec)
            else:
                missing_optional.append(spec)
        print()

    if missing_required:
        print("Missing required dependencies:")
        for spec in missing_required:
            print(f"- {spec.package} (module `{spec.module}`)")

    if missing_optional:
        print("Missing optional dependencies:")
        for spec in missing_optional:
            print(f"- {spec.package} (module `{spec.module}`)")

    if missing_required or (args.strict_optional and missing_optional):
        sys.exit(1)

    print("Dependency check passed.")


if __name__ == "__main__":
    main()
