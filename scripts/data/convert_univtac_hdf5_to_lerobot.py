"""Convert UniVTAC official HDF5 episodes to LeRobot datasets.

The converter writes one LeRobot dataset per UniVTAC task under ``--dst-root``:

    playground/Datasets/UniVTAC/lerobot/<task>_joint

It keeps StarVLA training semantics simple: ``observation.state`` is
``joint[t, :8]`` and ``action`` is ``joint[t + 1, :8]``. Tactile images stay as
raw marker RGB videos; StarVLA's dataloader applies tactile residuals at
training time.

Usage:
    # Smoke test one task with two episodes.
    python scripts/data/convert_univtac_hdf5_to_lerobot.py \
        --src playground/Datasets/UniVTAC/hdf5 \
        --dst-root playground/Datasets/UniVTAC/lerobot \
        --tasks lift_can \
        --limit 2 \
        --episode-workers 1 \
        --video-encode-workers 1 \
        --overwrite

    # Convert all official tasks.
    python scripts/data/convert_univtac_hdf5_to_lerobot.py \
        --src playground/Datasets/UniVTAC/hdf5 \
        --dst-root playground/Datasets/UniVTAC/lerobot \
        --tasks all \
        --episode-workers 8 \
        --video-encode-workers 4 \
        --overwrite
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import tqdm
from torchvision.transforms import functional as TF


TASK_INSTRUCTIONS = {
    "grasp_classify": "grasp the object and classify it by tactile feedback",
    "insert_HDMI": "insert the HDMI connector into the port",
    "insert_hole": "insert the peg into the hole",
    "insert_tube": "insert the tube into the fixture",
    "lift_bottle": "lift the bottle",
    "lift_can": "lift the can",
    "pull_out_key": "pull the key out of the lock",
    "put_bottle_in_shelf": "put the bottle in the shelf",
}

VIDEO_KEY_CANDIDATES = {
    "observation.images.head": ("observation/head/rgb",),
    "observation.images.wrist": ("observation/wrist/rgb",),
    "observation.images.left_tactile": (
        "tactile/left_gsmini/rgb_marker",
        "tactile/left_tactile/rgb_marker",
    ),
    "observation.images.right_tactile": (
        "tactile/right_gsmini/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ),
}

VIDEO_KEYS = tuple(VIDEO_KEY_CANDIDATES)
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_FPS = 10
ROBOT_TYPE = "Franka Panda"


@dataclass(frozen=True)
class UniVTACEpisodeJob:
    hdf5_path: str
    episode_index: int
    start_frame_index: int
    episode_length: int
    task: str
    task_index: int
    source_video_keys: dict[str, str]
    root: str
    data_file_path: str
    video_file_paths: dict[str, str]
    hf_features: Any
    features: dict[str, Any]
    fps: int
    image_size: tuple[int, int]
    video_encode_workers: int
    video_crf: int
    video_gop: int | None


@dataclass(frozen=True)
class UniVTACEpisodeResult:
    episode_index: int
    episode_length: int
    task: str
    stats: dict[str, dict]
    timings: dict[str, float]


def _load_arx_tools():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import convert_arx_to_lerobot as arx

    return arx


def _normalize_image_size(image_size: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("--image-size expects HEIGHT WIDTH")
    return int(image_size[0]), int(image_size[1])


def _resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize the short edge, then center-crop to ``image_size``."""
    image = torch.from_numpy(image).movedim(-1, 0)
    image = TF.center_crop(TF.resize(image, min(image_size)), image_size)
    return image.movedim(0, -1).contiguous().numpy()


def _decode_jpeg(value: Any, image_size: tuple[int, int]) -> np.ndarray:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, np.void):
        data = bytes(value)
    else:
        data = np.asarray(value).tobytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode UniVTAC JPEG frame")
    return _resize_image(image, image_size)


def _discover_tasks(src: Path, raw_tasks: list[str]) -> list[str]:
    requested = []
    for item in raw_tasks:
        requested.extend(part.strip() for part in item.split(",") if part.strip())
    if not requested or requested == ["all"]:
        return [task for task in TASK_INSTRUCTIONS if (src / task / "clean").is_dir()]
    if "all" in requested:
        return [task for task in TASK_INSTRUCTIONS if (src / task / "clean").is_dir()]
    return requested


def _discover_episodes(task_dir: Path, limit: int | None) -> list[Path]:
    episodes = sorted((task_dir / "clean").glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return episodes[:limit] if limit is not None else episodes


def _resolve_video_keys(h5: h5py.File) -> dict[str, str]:
    resolved = {}
    for out_key, candidates in VIDEO_KEY_CANDIDATES.items():
        for candidate in candidates:
            if candidate in h5:
                resolved[out_key] = candidate
                break
    return resolved


def _episode_preflight(path: Path) -> tuple[int, dict[str, str]]:
    with h5py.File(path, "r") as h5:
        if "embodiment/joint" not in h5:
            return 0, {}
        joint_len = int(h5["embodiment/joint"].shape[0])
        source_video_keys = _resolve_video_keys(h5)
        if len(source_video_keys) != len(VIDEO_KEYS):
            return 0, source_video_keys
        video_len = min(int(h5[src_key].shape[0]) for src_key in source_video_keys.values())
        return min(joint_len - 1, video_len), source_video_keys


def _build_features(image_size: tuple[int, int]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": (image_size[0], image_size[1], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _create_lerobot_dataset(repo_id: str, root: Path, fps: int, features: dict[str, Any]):
    os.environ["HF_LEROBOT_HOME"] = str(root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = {
        "repo_id": repo_id,
        "fps": fps,
        "robot_type": ROBOT_TYPE,
        "features": features,
        "use_videos": True,
    }
    try:
        params = inspect.signature(LeRobotDataset.create).parameters
    except (TypeError, ValueError):
        params = {}
    if "root" in params:
        kwargs["root"] = root / repo_id
    for key, value in {
        "tolerance_s": 0.1,
        "video_codec": "h264",
        "image_writer_processes": 0,
        "image_writer_threads": 0,
    }.items():
        if not params or key in params:
            kwargs[key] = value
    return LeRobotDataset.create(**kwargs)


def _read_episode(job: UniVTACEpisodeJob) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    with h5py.File(job.hdf5_path, "r") as h5:
        joints = h5["embodiment/joint"][: job.episode_length + 1, :8].astype(np.float32)
        timestamps = np.arange(job.episode_length, dtype=np.float32) / job.fps
        episode_buffer = {
            "observation.state": joints[:-1],
            "action": joints[1:],
            "timestamp": timestamps,
            "frame_index": np.arange(job.episode_length, dtype=np.int64),
            "episode_index": np.full((job.episode_length,), job.episode_index, dtype=np.int64),
            "index": np.arange(
                job.start_frame_index,
                job.start_frame_index + job.episode_length,
                dtype=np.int64,
            ),
            "task_index": np.full((job.episode_length,), job.task_index, dtype=np.int64),
        }
        frames = {
            out_key: [
                _decode_jpeg(h5[src_key][i], job.image_size)
                for i in range(job.episode_length)
            ]
            for out_key, src_key in job.source_video_keys.items()
        }
    return episode_buffer, frames


def _process_episode_job(job: UniVTACEpisodeJob) -> UniVTACEpisodeResult:
    arx = _load_arx_tools()
    root = Path(job.root)
    timings = {name: 0.0 for name in ("read_numeric", "decode", "table", "stats", "video", "total")}
    started = time.perf_counter()

    read_started = time.perf_counter()
    episode_buffer, cam_frames = _read_episode(job)
    timings["read_numeric"] = time.perf_counter() - read_started

    table_started = time.perf_counter()
    arx._save_episode_table_direct(root, job.data_file_path, episode_buffer, job.hf_features)
    timings["table"] = time.perf_counter() - table_started

    stats_started = time.perf_counter()
    ep_stats = arx._compute_episode_stats_direct(episode_buffer, cam_frames, job.features)
    timings["stats"] = time.perf_counter() - stats_started

    video_started = time.perf_counter()
    encode_workers = arx._effective_worker_count(job.video_encode_workers, len(VIDEO_KEYS), default_cap=4)

    def encode_one(video_key: str) -> None:
        arx._encode_video_arrays(
            cam_frames[video_key],
            root / job.video_file_paths[video_key],
            fps=job.fps,
            crf=job.video_crf,
            gop=job.video_gop,
        )

    if encode_workers == 1:
        for video_key in VIDEO_KEYS:
            encode_one(video_key)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers) as executor:
            list(executor.map(encode_one, VIDEO_KEYS))
    timings["video"] = time.perf_counter() - video_started
    timings["total"] = time.perf_counter() - started

    return UniVTACEpisodeResult(
        episode_index=job.episode_index,
        episode_length=job.episode_length,
        task=job.task,
        stats=ep_stats,
        timings=timings,
    )


def _run_episode_jobs(jobs: list[UniVTACEpisodeJob], episode_workers: int) -> dict[int, UniVTACEpisodeResult]:
    if episode_workers == 1:
        results = {}
        for job in tqdm.tqdm(jobs, desc="Converting episodes"):
            result = _process_episode_job(job)
            results[result.episode_index] = result
        return results

    results: dict[int, UniVTACEpisodeResult] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(episode_workers, len(jobs))) as executor:
        futures = {executor.submit(_process_episode_job, job): job for job in jobs}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting episodes"):
            result = future.result()
            results[result.episode_index] = result
    return results


def _build_jobs(
    episodes: list[Path],
    dataset: Any,
    task_instruction: str,
    fps: int,
    image_size: tuple[int, int],
    video_encode_workers: int,
    video_crf: int,
    video_gop: int | None,
    skipped: list[tuple[Path, str]],
) -> list[UniVTACEpisodeJob]:
    arx = _load_arx_tools()
    task_index = dataset.meta.get_task_index(task_instruction)
    if task_index is None:
        dataset.meta.add_task(task_instruction)
        task_index = dataset.meta.get_task_index(task_instruction)

    jobs = []
    next_frame_index = 0
    root = Path(dataset.root)
    for path in episodes:
        episode_length, source_video_keys = _episode_preflight(path)
        if episode_length < 1:
            skipped.append((path, "missing required fields or too short"))
            continue
        missing = [key for key in VIDEO_KEYS if key not in source_video_keys]
        if missing:
            skipped.append((path, f"missing video keys: {missing}"))
            continue
        episode_index = len(jobs)
        jobs.append(
            UniVTACEpisodeJob(
                hdf5_path=str(path),
                episode_index=episode_index,
                start_frame_index=next_frame_index,
                episode_length=episode_length,
                task=task_instruction,
                task_index=task_index,
                source_video_keys=source_video_keys,
                root=str(root),
                data_file_path=arx._path_relative_to_root(arx._get_data_file_path(dataset, episode_index), root),
                video_file_paths={
                    video_key: arx._path_relative_to_root(arx._get_video_file_path(dataset, episode_index, video_key), root)
                    for video_key in VIDEO_KEYS
                },
                hf_features=dataset.hf_features,
                features=dataset.features,
                fps=fps,
                image_size=image_size,
                video_encode_workers=video_encode_workers,
                video_crf=video_crf,
                video_gop=video_gop,
            )
        )
        next_frame_index += episode_length
    return jobs


def convert_task(
    src: Path,
    dst_root: Path,
    task: str,
    limit: int | None,
    fps: int,
    image_size: tuple[int, int],
    overwrite: bool,
    episode_workers: int,
    video_encode_workers: int,
    video_crf: int,
    video_gop: int | None,
    profile: bool,
) -> None:
    arx = _load_arx_tools()
    repo_id = f"{task}_joint"
    ds_path = dst_root / repo_id
    if ds_path.exists():
        if not overwrite:
            raise SystemExit(f"{ds_path} exists; pass --overwrite to replace it")
        shutil.rmtree(ds_path)

    episodes = _discover_episodes(src / task, limit)
    if not episodes:
        raise SystemExit(f"No HDF5 episodes found for task {task!r}")

    dataset = _create_lerobot_dataset(
        repo_id=repo_id,
        root=dst_root,
        fps=fps,
        features=_build_features(image_size),
    )
    skipped: list[tuple[Path, str]] = []
    jobs = _build_jobs(
        episodes=episodes,
        dataset=dataset,
        task_instruction=TASK_INSTRUCTIONS.get(task, task.replace("_", " ")),
        fps=fps,
        image_size=image_size,
        video_encode_workers=video_encode_workers,
        video_crf=video_crf,
        video_gop=video_gop,
        skipped=skipped,
    )
    if not jobs:
        raise SystemExit(f"No convertible episodes for task {task!r}")

    print(f"\n[{task}] {len(jobs)} episodes -> {ds_path}")
    started = time.perf_counter()
    results = _run_episode_jobs(jobs, episode_workers=episode_workers)
    timings = arx._finalize_metadata(dataset, jobs, results)
    wall_s = time.perf_counter() - started

    if profile:
        for job in jobs:
            result = results[job.episode_index]
            tqdm.tqdm.write(
                f"{Path(job.hdf5_path).name}: frames={result.episode_length} "
                f"table={result.timings['table']:.2f}s "
                f"stats={result.timings['stats']:.2f}s "
                f"video={result.timings['video']:.2f}s "
                f"total={result.timings['total']:.2f}s"
            )

    print(f"[{task}] converted={len(jobs)} skipped={len(skipped)} wall={wall_s:.2f}s")
    for name in ("read_numeric", "table", "stats", "video", "total"):
        print(f"  worker_{name:12s}: {timings[name]:.2f}s")
    for path, reason in skipped[:10]:
        print(f"  skipped {path.name}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("playground/Datasets/UniVTAC/hdf5"))
    parser.add_argument("--dst-root", type=Path, default=Path("playground/Datasets/UniVTAC/lerobot"))
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--episode-workers", type=int, default=8)
    parser.add_argument("--video-workers", type=int, default=1, help="Accepted for CLI symmetry; HDF5 decode is per episode")
    parser.add_argument("--video-encode-workers", type=int, default=4)
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-gop", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    del args.video_workers

    dst_root = args.dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    image_size = _normalize_image_size(args.image_size)
    tasks = _discover_tasks(args.src, args.tasks)
    for task in tasks:
        convert_task(
            src=args.src,
            dst_root=dst_root,
            task=task,
            limit=args.limit,
            fps=args.fps,
            image_size=image_size,
            overwrite=args.overwrite,
            episode_workers=args.episode_workers,
            video_encode_workers=args.video_encode_workers,
            video_crf=args.video_crf,
            video_gop=args.video_gop,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
