"""Create a repaired copy of the Flexiv LeRobot dataset with actions shifted forward.

The affected dataset was collected with joint actions equal to the same-frame
joint state.  For closed-loop policy learning we want observation at frame t to
supervise the command for the next frame, so this script writes:

    action[t] = observation.state[t + shift]
    actions.joint_position[t] = observation.state.joint_position[t + shift]

The final ``shift`` rows are padded with the last available state to keep the
LeRobot episode lengths aligned with the existing videos and metadata.

Non-parquet files are hardlinked by default so videos/latents are not duplicated.
Old cache/stat files are skipped because their action statistics are stale.
Matching StarVLA caches in ``playground/meta_data`` are removed so training
recomputes dataset statistics from the repaired parquet files.
If a dataset only has split joint columns, missing ``observation.state`` and
``action`` columns are created from ``observation.state.joint_position`` and
``actions.joint_position`` before shifting.

Usage:
    # 只预览将要处理的路径和 episode 数量，不写入任何文件。
    python scripts/data/shift_flexiv_lerobot_actions.py --dry-run

    # 创建默认修复后的数据集：
    # playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621_action_shifted
    python scripts/data/shift_flexiv_lerobot_actions.py

    # 如果目标目录已经存在，先删除旧目录再重新生成。
    # 同时会清理 playground/meta_data 中该目标数据集对应的 StarVLA stats cache。
    python scripts/data/shift_flexiv_lerobot_actions.py --overwrite

    # 直接覆盖原数据集路径；原目录会先移动到：
    # playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621.before_action_shift.bak
    python scripts/data/shift_flexiv_lerobot_actions.py --in-place

    # 指定源数据集和自定义输出目录。
    python scripts/data/shift_flexiv_lerobot_actions.py \\
        --src playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621 \\
        --dst playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621_shift1 \\
        --overwrite

    # 对视频和 latent 使用完整复制；默认是 hardlink，速度更快且不额外占用大容量空间。
    python scripts/data/shift_flexiv_lerobot_actions.py --link-mode copy
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SRC = Path("playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621")
SKIP_NAMES = {".cache", "norm_stats.json"}
REPORT_NAME = "action_shift_repair_report.json"
STARVLA_CACHE_DIR = Path("playground/meta_data")


@dataclass(frozen=True)
class EpisodeRepairResult:
    parquet: str
    frames: int
    shift: int
    before_max_abs_action_state: float
    after_max_abs_action_state: float
    after_max_abs_action_next_state: float
    padded_tail_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source LeRobot dataset root. Default: {DEFAULT_SRC}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination dataset root. Default: <src>_action_shifted",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Positive frame offset used for action[t] = state[t + shift]. Default: 1",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to mirror non-parquet files. Default: hardlink",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of parquet rewrite workers. Default: min(16, cpu_count)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing destination or in-place backup before writing.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace --src with the repaired dataset after creating a sibling backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned work without writing files.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_paths(src: Path, dst: Path, overwrite: bool, dry_run: bool, in_place: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src}")
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {src / 'meta' / 'info.json'}")
    if src.resolve() == dst.resolve() and not in_place:
        raise ValueError("Destination must be different from source.")
    if is_relative_to(dst, src):
        raise ValueError("Destination cannot be inside the source directory.")
    if dst.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Destination already exists: {dst}. Use --overwrite to replace it.")


def in_place_paths(src: Path) -> tuple[Path, Path]:
    tmp_dst = src.with_name(f"{src.name}.action_shift_tmp")
    backup_dst = src.with_name(f"{src.name}.before_action_shift.bak")
    return tmp_dst, backup_dst


def finalize_in_place(src: Path, tmp_dst: Path, backup_dst: Path, overwrite: bool) -> None:
    if backup_dst.exists():
        if not overwrite:
            raise FileExistsError(f"In-place backup already exists: {backup_dst}. Use --overwrite to replace it.")
        shutil.rmtree(backup_dst)
    if not tmp_dst.exists():
        raise FileNotFoundError(f"Temporary repaired dataset does not exist: {tmp_dst}")
    src.rename(backup_dst)
    tmp_dst.rename(src)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_NAMES for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def mirror_static_file(src_file: Path, dst_file: Path, link_mode: str) -> None:
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        os.link(src_file, dst_file)
    elif link_mode == "symlink":
        os.symlink(src_file.resolve(), dst_file)
    else:
        shutil.copy2(src_file, dst_file)


def mirror_non_parquet_files(src: Path, dst: Path, link_mode: str) -> int:
    count = 0
    for src_file in iter_files(src):
        rel = src_file.relative_to(src)
        if rel.parts[0] == "data" and src_file.suffix == ".parquet":
            continue
        if rel.name == REPORT_NAME:
            continue
        mirror_static_file(src_file, dst / rel, link_mode)
        count += 1
    return count


def starvla_cache_hashes_for_path(path: Path) -> list[str]:
    variants = [str(path)]
    resolved = str(path.resolve())
    if resolved not in variants:
        variants.append(resolved)
    return sorted({hashlib.sha1(variant.encode("utf-8")).hexdigest()[:6] for variant in variants})


def find_starvla_cache_files(path: Path) -> list[Path]:
    if not STARVLA_CACHE_DIR.exists():
        return []
    cache_files: list[Path] = []
    for root_hash in starvla_cache_hashes_for_path(path):
        cache_files.extend(STARVLA_CACHE_DIR.glob(f"*_{root_hash}_vla_dataset_info_cache.json"))
        cache_files.extend(STARVLA_CACHE_DIR.glob(f"*_{root_hash}_*_vla_dataset_statistic.json"))
    return sorted(set(cache_files))


def remove_starvla_cache_files(path: Path) -> list[Path]:
    removed: list[Path] = []
    for cache_file in find_starvla_cache_files(path):
        cache_file.unlink()
        removed.append(cache_file)
    return removed


def ensure_aggregate_features(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return []

    with open(info_path) as f:
        info = json.load(f)
    features = info.setdefault("features", {})
    added: list[str] = []

    if "observation.state" not in features and "observation.state.joint_position" in features:
        features["observation.state"] = copy.deepcopy(features["observation.state.joint_position"])
        added.append("observation.state")
    if "action" not in features and "actions.joint_position" in features:
        features["action"] = copy.deepcopy(features["actions.joint_position"])
        added.append("action")

    if added:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
    return added


def shifted_rows(values: np.ndarray, shift: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {values.shape}")
    out = np.empty_like(values)
    if values.shape[0] <= shift:
        out[:] = values[-1]
        return out
    out[:-shift] = values[shift:]
    out[-shift:] = values[-1]
    return out


def to_object_series(values: np.ndarray) -> list[np.ndarray]:
    return [np.array(row, dtype=np.float32, copy=True) for row in values]


def stack_column(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"Missing required column {key!r}")
    return np.stack(df[key].to_numpy()).astype(np.float32, copy=False)


def repair_parquet(src_file: Path, dst_file: Path, src_root: Path, shift: int) -> EpisodeRepairResult:
    df = pd.read_parquet(src_file)

    state_joint = stack_column(df, "observation.state.joint_position")
    action_joint = stack_column(df, "actions.joint_position")
    if "observation.state" in df.columns:
        state = stack_column(df, "observation.state")
    else:
        state = np.array(state_joint, copy=True)
        df["observation.state"] = to_object_series(state)
    if "action" in df.columns:
        action = stack_column(df, "action")
    else:
        action = np.array(action_joint, copy=True)
        df["action"] = to_object_series(action)

    if state.shape != action.shape:
        raise ValueError(f"{src_file}: observation.state shape {state.shape} != action shape {action.shape}")
    if state_joint.shape != action_joint.shape:
        raise ValueError(
            f"{src_file}: observation.state.joint_position shape {state_joint.shape} "
            f"!= actions.joint_position shape {action_joint.shape}"
        )
    if state.shape[0] == 0:
        raise ValueError(f"{src_file}: empty episode")

    shifted_state = shifted_rows(state, shift)
    shifted_joint = shifted_rows(state_joint, shift)

    before_max = float(np.max(np.abs(action - state)))
    after_same = float(np.max(np.abs(shifted_state - state)))
    if state.shape[0] > shift:
        after_next = float(np.max(np.abs(shifted_state[:-shift] - state[shift:])))
    else:
        after_next = 0.0

    df["action"] = to_object_series(shifted_state)
    df["actions.joint_position"] = to_object_series(shifted_joint)

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_file, index=False)

    return EpisodeRepairResult(
        parquet=str(src_file.relative_to(src_root)),
        frames=int(state.shape[0]),
        shift=int(shift),
        before_max_abs_action_state=before_max,
        after_max_abs_action_state=after_same,
        after_max_abs_action_next_state=after_next,
        padded_tail_rows=min(int(shift), int(state.shape[0])),
    )


def write_report(
    dst: Path,
    *,
    src: Path,
    shift: int,
    link_mode: str,
    static_files: int,
    removed_starvla_caches: list[Path],
    added_info_features: list[str],
    results: list[EpisodeRepairResult],
) -> None:
    before = [r.before_max_abs_action_state for r in results]
    after_next = [r.after_max_abs_action_next_state for r in results]
    report = {
        "source": str(src),
        "destination": str(dst),
        "shift": shift,
        "link_mode": link_mode,
        "static_files_mirrored": static_files,
        "removed_starvla_caches": [str(path) for path in removed_starvla_caches],
        "added_info_features": added_info_features,
        "parquet_files_rewritten": len(results),
        "notes": [
            "Rewrote action and actions.joint_position from future joint states.",
            "Added observation.state/action from joint-position columns when missing.",
            "Kept actions.eef_pose unchanged because it is not identical to observation.state.eef_pose in the source.",
            "Skipped .cache and norm_stats.json because old statistics are stale after action repair.",
            "Removed matching playground/meta_data StarVLA caches so training recomputes statistics.",
        ],
        "summary": {
            "total_frames": int(sum(r.frames for r in results)),
            "max_before_abs_action_state": float(max(before)) if before else None,
            "max_after_abs_action_next_state": float(max(after_next)) if after_next else None,
        },
        "episodes": [asdict(r) for r in sorted(results, key=lambda item: item.parquet)],
    }
    with open(dst / REPORT_NAME, "w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()
    src = args.src
    if args.in_place and args.dst is not None:
        raise ValueError("--in-place cannot be combined with --dst.")
    tmp_dst = None
    backup_dst = None
    if args.in_place:
        dst, backup_dst = in_place_paths(src)
        tmp_dst = dst
        final_dst = src
    else:
        dst = args.dst if args.dst is not None else src.with_name(f"{src.name}_action_shifted")
        final_dst = dst
    shift = int(args.shift)
    if shift <= 0:
        raise ValueError(f"--shift must be positive, got {shift}")

    validate_paths(src, dst, args.overwrite, args.dry_run, args.in_place)
    parquet_files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {src / 'data'}")

    if args.dry_run:
        if args.in_place:
            print(f"Would create temporary repaired dataset: {tmp_dst}")
            print(f"Would move original dataset to backup: {backup_dst}")
            print(f"Would move repaired dataset back to original path: {src}")
        else:
            print(f"Would create repaired dataset: {dst}")
        print(f"Would mirror non-parquet files with {args.link_mode}, skipping {sorted(SKIP_NAMES)}")
        print(f"Would rewrite {len(parquet_files)} parquet files with shift={shift}")
        caches = find_starvla_cache_files(final_dst)
        print(
            f"Would remove {len(caches)} matching playground/meta_data StarVLA cache files "
            f"for destination hashes {starvla_cache_hashes_for_path(final_dst)}."
        )
        return

    if args.in_place and backup_dst is not None and backup_dst.exists() and not args.overwrite:
        raise FileExistsError(f"In-place backup already exists: {backup_dst}. Use --overwrite to replace it.")

    removed_starvla_caches = remove_starvla_cache_files(final_dst)
    if removed_starvla_caches:
        print(f"Removed {len(removed_starvla_caches)} stale StarVLA cache files for {final_dst}.")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"Mirroring non-parquet files to {dst} using {args.link_mode}...")
    static_files = mirror_non_parquet_files(src, dst, args.link_mode)
    added_info_features = ensure_aggregate_features(dst)
    if added_info_features:
        print(f"Added aggregate features to meta/info.json: {added_info_features}")

    print(f"Rewriting {len(parquet_files)} parquet files with action shift={shift}...")
    results: list[EpisodeRepairResult] = []
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                repair_parquet,
                src_file,
                dst / src_file.relative_to(src),
                src,
                shift,
            ): src_file
            for src_file in parquet_files
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    write_report(
        dst,
        src=src,
        shift=shift,
        link_mode=args.link_mode,
        static_files=static_files,
        removed_starvla_caches=removed_starvla_caches,
        added_info_features=added_info_features,
        results=results,
    )
    if args.in_place:
        finalize_in_place(src, dst, backup_dst, args.overwrite)
        print(f"Done. Repaired dataset is now at original path: {src}")
        print(f"Original dataset backup: {backup_dst}")
        print(f"Report: {src / REPORT_NAME}")
    else:
        print(f"Done. Repaired dataset: {dst}")
        print(f"Report: {dst / REPORT_NAME}")


if __name__ == "__main__":
    main()
