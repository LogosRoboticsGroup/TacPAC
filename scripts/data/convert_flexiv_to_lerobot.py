"""Convert raw Flexiv episodes to LeRobot v2.1 format.

Source schema (raw episode dirs, e.g.
``playground/Datasets/neoteai/raw/FLEXIV-PLUG-0701/<date>/<user>/episode_*/``):

    episode_*/
        metadata.json
        actions.eef_pose/data.csv
        observation.state.eef_pose/data.csv
        observation.state.joint_position/data.csv
        observation.image.third_view/{video.mp4,timestamps.csv}
        observation.image.second_third_view/{video.mp4,timestamps.csv}
        observation.image.left_wrist_view/{video.mp4,timestamps.csv}
        observation.image.left_wrist_left_tactile/{video.mp4,timestamps.csv}
        observation.image.left_wrist_right_tactile/{video.mp4,timestamps.csv}

Output features match the Flexiv LeRobot layout used by
``playground/Datasets/neoteai/lerobot/plug_outlet_flexiv_0621/meta/info.json``:

    observation.state.eef_pose          float32 (10,)
    observation.state.joint_position    float32 (8,)
    actions.eef_pose                    float32 (10,)
    actions.joint_position              float32 (8,)  joint state[t + action_shift]
    observation.state                   float32 (8,)  alias of joint_position
    action                              float32 (8,)  alias of actions.joint_position
    observation.images.*                video

The raw Flexiv ``actions.eef_pose`` gripper command is stored as ``gripper.pos``
in approximately 0..1 units, while state gripper values are in approximately
0..0.1 units.  By default this script scales the action gripper by 0.1 to match
the existing LeRobot feature convention.  Override with
``--action-gripper-scale 1.0`` if your raw action CSV is already in state units.

Usage:
    export HF_LEROBOT_HOME=playground/Datasets/neoteai/lerobot
    python scripts/data/convert_flexiv_to_lerobot.py \\
        --src playground/Datasets/neoteai/raw/FLEXIV-PLUG-0701 \\
        --repo-id plug_outlet_flexiv_0701 \\
        --task "pick up the cord from the base and then plug into the outlet" \\
        --date 20260701 --limit 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import tqdm

from lerobot.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import check_timestamps_sync, get_episode_data_index

import convert_arx_to_lerobot as arx

from starVLA.dataloader.vla.image_fit import resize_crop_geometry


FPS = 30
DEFAULT_IMAGE_SIZE = arx.DEFAULT_IMAGE_SIZE
VIDEO_MODES = arx.VIDEO_MODES
DEFAULT_VIDEO_MODE = arx.DEFAULT_VIDEO_MODE

EEF_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]

STATE_EEF_COLS = EEF_NAMES
STATE_JOINT_COLS = JOINT_NAMES
ACTION_EEF_COLS = [
    "tcp.x",
    "tcp.y",
    "tcp.z",
    "tcp.r1",
    "tcp.r2",
    "tcp.r3",
    "tcp.r4",
    "tcp.r5",
    "tcp.r6",
    "gripper.pos",
]

RGB_CAMERA_MAP = {
    "observation.images.third_view": "observation.image.third_view",
    "observation.images.second_third_view": "observation.image.second_third_view",
    "observation.images.left_wrist_view": "observation.image.left_wrist_view",
}
TACTILE_CAMERA_MAP = {
    "observation.images.left_wrist_left_tactile": "observation.image.left_wrist_left_tactile",
    "observation.images.left_wrist_right_tactile": "observation.image.left_wrist_right_tactile",
}
DEFAULT_CAMERA_MAP = {**RGB_CAMERA_MAP, **TACTILE_CAMERA_MAP}


@dataclass(frozen=True)
class FlexivEpisodeJob:
    ep_dir: str
    episode_index: int
    start_frame_index: int
    episode_length: int
    task: str
    task_index: int
    camera_map: dict[str, str]
    root: str
    data_file_path: str
    video_file_paths: dict[str, str]
    hf_features: Any
    features: dict[str, Any]
    fps: int
    action_shift: int
    action_gripper_scale: float
    video_mode: str
    image_size: tuple[int, int]
    video_workers: int | None
    video_encode_workers: int
    video_crf: int
    video_gop: int | None


@dataclass(frozen=True)
class FlexivEpisodeResult:
    episode_index: int
    episode_length: int
    task: str
    stats: dict[str, dict]
    timings: dict[str, float]


def _read_array_csv(path: Path, cols: list[str]) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i] = [float(row[col]) for col in cols]
    return out


def _read_action_eef_csv(path: Path, gripper_scale: float) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = np.zeros((len(rows), len(EEF_NAMES)), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, :9] = [float(row[col]) for col in ACTION_EEF_COLS[:9]]
        out[i, 9] = float(row[ACTION_EEF_COLS[9]]) * float(gripper_scale)
    return out


def _shift_rows(values: np.ndarray, shift: int) -> np.ndarray:
    if shift < 0:
        raise ValueError(f"action_shift must be >= 0, got {shift}")
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {values.shape}")
    if shift == 0:
        return np.array(values, copy=True)

    out = np.empty_like(values)
    if values.shape[0] <= shift:
        out[:] = values[-1]
        return out
    out[:-shift] = values[shift:]
    out[-shift:] = values[-1]
    return out


def _read_numeric_episode(
    ep_dir: Path,
    episode_length: int,
    action_shift: int,
    action_gripper_scale: float,
) -> dict[str, np.ndarray]:
    state_eef = _read_array_csv(ep_dir / "observation.state.eef_pose" / "data.csv", STATE_EEF_COLS)
    state_joint = _read_array_csv(
        ep_dir / "observation.state.joint_position" / "data.csv",
        STATE_JOINT_COLS,
    )
    action_eef = _read_action_eef_csv(ep_dir / "actions.eef_pose" / "data.csv", action_gripper_scale)

    for key, array in {
        "observation.state.eef_pose": state_eef,
        "observation.state.joint_position": state_joint,
        "actions.eef_pose": action_eef,
    }.items():
        if len(array) < episode_length:
            raise ValueError(f"{ep_dir}: {key} has {len(array)} rows, expected {episode_length}")

    state_eef = state_eef[:episode_length]
    state_joint = state_joint[:episode_length]
    action_eef = action_eef[:episode_length]
    action_joint = _shift_rows(state_joint, action_shift)

    return {
        "observation.state.eef_pose": state_eef,
        "observation.state.joint_position": state_joint,
        "actions.eef_pose": action_eef,
        "actions.joint_position": action_joint,
        "observation.state": state_joint,
        "action": action_joint,
    }


def _csv_row_count(path: Path) -> int:
    return arx._count_joint_csv_rows(path)


def _metadata_task(ep_dir: Path) -> str | None:
    metadata_path = ep_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    task = metadata.get("language_instruction")
    return task if isinstance(task, str) and task.strip() else None


def _discover_episodes(src: Path, limit: int | None, dates: set[str] | None = None) -> list[Path]:
    if (src / "metadata.json").exists() and (src / "observation.state.joint_position" / "data.csv").exists():
        episodes = [src]
    elif dates is None:
        episodes = sorted(src.glob("*/*/episode_*"))
    else:
        episodes = sorted(ep for date in dates for ep in (src / date).glob("*/episode_*"))

    kept = [
        ep
        for ep in episodes
        if (ep / "observation.state.eef_pose" / "data.csv").exists()
        and (ep / "observation.state.joint_position" / "data.csv").exists()
        and (ep / "actions.eef_pose" / "data.csv").exists()
    ]
    date_msg = "all dates" if dates is None else ", ".join(sorted(dates))
    print(f"Scanned {len(episodes)} raw episodes for {date_msg}, kept {len(kept)} Flexiv episodes.")
    if limit is not None:
        kept = kept[:limit]
    return kept


def _build_camera_map(include_tactile: bool, include_second_third_view: bool) -> dict[str, str]:
    camera_map = dict(RGB_CAMERA_MAP)
    if not include_second_third_view:
        camera_map.pop("observation.images.second_third_view", None)
    if include_tactile:
        camera_map.update(TACTILE_CAMERA_MAP)
    return camera_map


def _build_features(camera_map: dict[str, str], camera_hw: dict[str, tuple[int, int]]) -> dict:
    features: dict[str, Any] = {
        "observation.state.eef_pose": {
            "dtype": "float32",
            "shape": (len(EEF_NAMES),),
            "names": [EEF_NAMES],
        },
        "observation.state.joint_position": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
        "actions.eef_pose": {
            "dtype": "float32",
            "shape": (len(EEF_NAMES),),
            "names": [EEF_NAMES],
        },
        "actions.joint_position": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": [JOINT_NAMES],
        },
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
    for out_key in camera_map:
        height, width = camera_hw.get(out_key, DEFAULT_IMAGE_SIZE)
        features[out_key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _create_lerobot_dataset(repo_id: str, robot_type: str, features: dict) -> LeRobotDataset:
    kwargs = {
        "repo_id": repo_id,
        "fps": FPS,
        "robot_type": robot_type,
        "features": features,
        "use_videos": True,
    }
    optional_kwargs = {
        "tolerance_s": 0.1,
        "video_codec": "h264",
        "image_writer_processes": 0,
        "image_writer_threads": 0,
    }

    try:
        params = inspect.signature(LeRobotDataset.create).parameters
    except (TypeError, ValueError):
        params = {}
    for key, value in optional_kwargs.items():
        if not params or key in params:
            kwargs[key] = value

    return LeRobotDataset.create(**kwargs)


def _infer_camera_hw(
    episodes: list[Path],
    camera_map: dict[str, str],
    video_mode: str,
    image_size: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    if video_mode == "reencode":
        return {out_key: image_size for out_key in camera_map}

    camera_hw: dict[str, tuple[int, int]] = {}
    for out_key, src_key in camera_map.items():
        for ep_dir in episodes:
            hw = arx._probe_video_hw(ep_dir / src_key / "video.mp4")
            if hw is not None:
                camera_hw[out_key] = hw
                break
        camera_hw.setdefault(out_key, image_size)
    return camera_hw


def _validate_episode_videos(
    ep_dir: Path,
    camera_map: dict[str, str],
    expected_frames: int,
    num_samples: int,
) -> None:
    for src_key in camera_map.values():
        arx._validate_video_decode_with_decord(
            ep_dir / src_key / "video.mp4",
            expected_frames=expected_frames,
            num_samples=num_samples,
        )


def _decode_resize_crop(
    vid_path: Path,
    expected: int,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Decode ``vid_path`` at an aspect-preserving size, center crop to ``image_size``.

    Unlike a direct resize to ``image_size``, this keeps the source aspect ratio so the
    stored frames match what the dataloader feeds the model.  Pads or truncates to
    exactly ``expected`` frames so all cameras align with the state/action CSV length.
    """
    import decord

    src_hw = arx._probe_video_hw(vid_path)
    if src_hw is None:
        raise RuntimeError(f"{vid_path}: failed to probe video size")
    target_h, target_w = image_size
    resize_h, resize_w, top, left = resize_crop_geometry(src_hw, image_size)

    frames: list[np.ndarray] = []
    video_reader = decord.VideoReader(
        str(vid_path),
        width=resize_w,
        height=resize_h,
        num_threads=arx.DECORD_DECODE_THREADS,
    )
    frame_count = min(int(expected), len(video_reader))
    for start in range(0, frame_count, arx.DECORD_BATCH_SIZE):
        stop = min(start + arx.DECORD_BATCH_SIZE, frame_count)
        batch = video_reader.get_batch(list(range(start, stop))).asnumpy()
        # Contiguous copy: av.VideoFrame.from_ndarray rejects sliced views.
        frames.extend(
            np.ascontiguousarray(batch[i, top : top + target_h, left : left + target_w])
            for i in range(len(batch))
        )
    if not frames:
        raise RuntimeError(f"{vid_path}: decoded 0 frames")
    while len(frames) < expected:
        frames.append(frames[-1])
    return frames


def _decode_episode_cameras(
    ep_dir: Path,
    camera_map: dict[str, str],
    expected_frames: int,
    image_size: tuple[int, int],
    video_workers: int | None,
) -> dict[str, list[np.ndarray]]:
    workers = arx._effective_worker_count(video_workers, len(camera_map), default_cap=4)

    def decode_one(item: tuple[str, str]) -> tuple[str, list[np.ndarray]]:
        out_key, src_key = item
        vid_path = ep_dir / src_key / "video.mp4"
        if out_key in TACTILE_CAMERA_MAP:
            # Tactile frames are a full sensor readout, not a composition: stretch to
            # image_size rather than cropping away part of the contact surface.
            frames = arx._decode_and_resize(vid_path, expected_frames, image_size)
        else:
            frames = _decode_resize_crop(vid_path, expected_frames, image_size)
        return out_key, frames

    items = list(camera_map.items())
    if workers == 1:
        return dict(decode_one(item) for item in items)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(decode_one, items))


def _process_episode_job(job: FlexivEpisodeJob) -> FlexivEpisodeResult:
    ep_dir = Path(job.ep_dir)
    root = Path(job.root)
    timings = {
        "read_numeric": 0.0,
        "decode": 0.0,
        "table": 0.0,
        "stats": 0.0,
        "video": 0.0,
        "total": 0.0,
    }
    episode_t0 = time.perf_counter()

    numeric_t0 = time.perf_counter()
    numeric = _read_numeric_episode(
        ep_dir,
        episode_length=job.episode_length,
        action_shift=job.action_shift,
        action_gripper_scale=job.action_gripper_scale,
    )
    episode_length = len(numeric["observation.state.joint_position"])
    if episode_length != job.episode_length:
        raise ValueError(
            f"{ep_dir}: expected {job.episode_length} frames from preflight, got {episode_length}"
        )
    timings["read_numeric"] = time.perf_counter() - numeric_t0

    timestamps = np.arange(episode_length, dtype=np.float32) / job.fps
    episode_buffer = {
        **numeric,
        "timestamp": timestamps,
        "frame_index": np.arange(episode_length, dtype=np.int64),
        "episode_index": np.full((episode_length,), job.episode_index, dtype=np.int64),
        "index": np.arange(
            job.start_frame_index,
            job.start_frame_index + episode_length,
            dtype=np.int64,
        ),
        "task_index": np.full((episode_length,), job.task_index, dtype=np.int64),
    }

    cam_frames: dict[str, list[np.ndarray]] | None = None
    if job.video_mode == "reencode":
        decode_t0 = time.perf_counter()
        cam_frames = _decode_episode_cameras(
            ep_dir=ep_dir,
            camera_map=job.camera_map,
            expected_frames=episode_length,
            image_size=job.image_size,
            video_workers=job.video_workers,
        )
        timings["decode"] = time.perf_counter() - decode_t0

    table_t0 = time.perf_counter()
    arx._save_episode_table_direct(root, job.data_file_path, episode_buffer, job.hf_features)
    timings["table"] = time.perf_counter() - table_t0

    stats_t0 = time.perf_counter()
    ep_stats = arx._compute_episode_stats_direct(episode_buffer, cam_frames, job.features)
    timings["stats"] = time.perf_counter() - stats_t0

    video_t0 = time.perf_counter()
    if job.video_mode == "copy":
        for out_key, src_key in job.camera_map.items():
            src_video = ep_dir / src_key / "video.mp4"
            dst_video = root / job.video_file_paths[out_key]
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            if src_video.resolve() != dst_video.resolve():
                shutil.copy2(src_video, dst_video)
    else:
        assert cam_frames is not None
        encode_workers = arx._effective_worker_count(
            job.video_encode_workers,
            len(job.camera_map),
            default_cap=4,
        )

        def encode_one(out_key: str) -> None:
            arx._encode_video_arrays(
                cam_frames[out_key],
                root / job.video_file_paths[out_key],
                fps=job.fps,
                crf=job.video_crf,
                gop=job.video_gop,
            )

        out_keys = list(job.camera_map)
        if encode_workers == 1:
            for out_key in out_keys:
                encode_one(out_key)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers) as executor:
                list(executor.map(encode_one, out_keys))
    timings["video"] = time.perf_counter() - video_t0
    timings["total"] = time.perf_counter() - episode_t0

    return FlexivEpisodeResult(
        episode_index=job.episode_index,
        episode_length=episode_length,
        task=job.task,
        stats=ep_stats,
        timings=timings,
    )


def _run_episode_jobs(jobs: list[FlexivEpisodeJob], episode_workers: int) -> dict[int, FlexivEpisodeResult]:
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")

    results: dict[int, FlexivEpisodeResult] = {}
    if episode_workers == 1:
        for job in tqdm.tqdm(jobs, desc="Converting episodes"):
            result = _process_episode_job(job)
            results[result.episode_index] = result
        return results

    workers = min(episode_workers, len(jobs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_episode_job, job): job for job in jobs}
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Converting episodes",
        ):
            result = future.result()
            results[result.episode_index] = result
    return results


def _build_episode_jobs(
    episodes: list[Path],
    dataset: LeRobotDataset,
    camera_map: dict[str, str],
    task: str,
    use_metadata_task: bool,
    action_shift: int,
    action_gripper_scale: float,
    video_mode: str,
    image_size: tuple[int, int],
    validate_videos: bool,
    validate_video_samples: int,
    video_workers: int | None,
    video_encode_workers: int,
    video_crf: int,
    video_gop: int | None,
    skipped: list[tuple[Path, str]],
) -> list[FlexivEpisodeJob]:
    root = Path(dataset.root)
    jobs: list[FlexivEpisodeJob] = []
    next_frame_index = 0

    for ep_dir in episodes:
        try:
            ep_task = _metadata_task(ep_dir) if use_metadata_task else None
            job_task = ep_task or task
            task_index = dataset.meta.get_task_index(job_task)
            if task_index is None:
                dataset.meta.add_task(job_task)
                task_index = dataset.meta.get_task_index(job_task)
            if task_index is None:
                raise RuntimeError(f"failed to register task: {job_task}")

            required_csvs = [
                ep_dir / "observation.state.eef_pose" / "data.csv",
                ep_dir / "observation.state.joint_position" / "data.csv",
                ep_dir / "actions.eef_pose" / "data.csv",
            ]
            missing_csv = next((path for path in required_csvs if not path.exists()), None)
            if missing_csv is not None:
                skipped.append((ep_dir, f"missing {missing_csv.relative_to(ep_dir)}"))
                continue

            row_counts = {str(path.relative_to(ep_dir)): _csv_row_count(path) for path in required_csvs}
            if len(set(row_counts.values())) != 1:
                skipped.append((ep_dir, f"numeric row count mismatch: {row_counts}"))
                continue
            episode_length = next(iter(row_counts.values()))
            if episode_length < 2:
                skipped.append((ep_dir, f"only {episode_length} frames"))
                continue

            missing_cam = next(
                (src_key for src_key in camera_map.values() if not (ep_dir / src_key / "video.mp4").exists()),
                None,
            )
            if missing_cam is not None:
                skipped.append((ep_dir, f"missing {missing_cam}"))
                continue
            if validate_videos:
                _validate_episode_videos(
                    ep_dir=ep_dir,
                    camera_map=camera_map,
                    expected_frames=episode_length,
                    num_samples=validate_video_samples,
                )

            episode_index = len(jobs)
            jobs.append(
                FlexivEpisodeJob(
                    ep_dir=str(ep_dir),
                    episode_index=episode_index,
                    start_frame_index=next_frame_index,
                    episode_length=episode_length,
                    task=job_task,
                    task_index=task_index,
                    camera_map=camera_map,
                    root=str(root),
                    data_file_path=arx._path_relative_to_root(
                        arx._get_data_file_path(dataset, episode_index),
                        root,
                    ),
                    video_file_paths={
                        out_key: arx._path_relative_to_root(
                            arx._get_video_file_path(dataset, episode_index, out_key),
                            root,
                        )
                        for out_key in camera_map
                    },
                    hf_features=dataset.hf_features,
                    features=dataset.features,
                    fps=dataset.fps,
                    action_shift=action_shift,
                    action_gripper_scale=action_gripper_scale,
                    video_mode=video_mode,
                    image_size=image_size,
                    video_workers=video_workers,
                    video_encode_workers=video_encode_workers,
                    video_crf=video_crf,
                    video_gop=video_gop,
                )
            )
            next_frame_index += episode_length
        except Exception as exc:
            skipped.append((ep_dir, f"{type(exc).__name__}: {exc}"))
            continue
    return jobs


def _finalize_metadata(
    dataset: LeRobotDataset,
    jobs: list[FlexivEpisodeJob],
    results: dict[int, FlexivEpisodeResult],
) -> dict[str, float]:
    timings = {
        "read_numeric": 0.0,
        "decode": 0.0,
        "table": 0.0,
        "stats": 0.0,
        "video": 0.0,
        "total": 0.0,
    }
    if dataset.meta.video_keys:
        dataset.meta.update_video_info()

    for job in jobs:
        result = results[job.episode_index]
        dataset.meta.save_episode(
            result.episode_index,
            result.episode_length,
            [result.task],
            result.stats,
        )
        timestamps = np.arange(result.episode_length, dtype=np.float32) / dataset.fps
        episode_indices = np.full((result.episode_length,), result.episode_index, dtype=np.int64)
        ep_data_index = get_episode_data_index(dataset.meta.episodes, [result.episode_index])
        ep_data_index_np = {key: tensor.numpy() for key, tensor in ep_data_index.items()}
        check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            dataset.fps,
            dataset.tolerance_s,
        )
        for name, value in result.timings.items():
            timings[name] += value
    return timings


def convert(
    src: Path,
    repo_id: str,
    task: str,
    limit: int | None,
    include_tactile: bool,
    include_second_third_view: bool,
    use_metadata_task: bool,
    robot_type: str | None,
    dates: set[str] | None = None,
    action_shift: int = 1,
    action_gripper_scale: float = 0.1,
    video_mode: str = DEFAULT_VIDEO_MODE,
    image_size: tuple[int, int] | list[int] = DEFAULT_IMAGE_SIZE,
    validate_videos: bool = True,
    validate_video_samples: int = 3,
    episode_workers: int = 8,
    video_workers: int | None = 8,
    video_encode_workers: int = 4,
    video_crf: int = 23,
    video_gop: int | None = 2,
    profile: bool = False,
) -> None:
    src = Path(src)
    if video_mode not in VIDEO_MODES:
        raise ValueError(f"video_mode must be one of {VIDEO_MODES}, got {video_mode!r}")
    image_size = arx._normalize_image_size(image_size)
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")
    if action_shift < 0:
        raise ValueError(f"action_shift must be >= 0, got {action_shift}")

    camera_map = _build_camera_map(include_tactile, include_second_third_view)
    robot_type = robot_type or ("flexiv_tactile" if include_tactile else "flexiv")

    episodes = _discover_episodes(src, limit, dates)
    if not episodes:
        raise SystemExit(f"No episodes under {src}")
    print(f"Found {len(episodes)} episodes; converting with {len(camera_map)} cameras:")
    for out_key, src_key in camera_map.items():
        print(f"  - {out_key} <- {src_key}")
    print(f"Robot type: {robot_type}")
    print(f"Action shift: {action_shift}")
    print(f"Action eef gripper scale: {action_gripper_scale:g}")
    print(f"Video mode: {video_mode}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(
        "Parallelism: "
        f"episode_workers={episode_workers}, "
        f"video_workers={arx._effective_worker_count(video_workers, len(camera_map), default_cap=4)}, "
        f"video_encode_workers={arx._effective_worker_count(video_encode_workers, len(camera_map), default_cap=4)}, "
        f"video_crf={video_crf}, "
        f"video_gop={video_gop}"
    )
    if validate_videos:
        print(f"Video validation: decord sampled decode ({validate_video_samples} frames/video)")

    ds_path = LEROBOT_HOME / repo_id
    if ds_path.exists():
        print(f"Removing existing dataset at {ds_path}")
        shutil.rmtree(ds_path)

    camera_hw = _infer_camera_hw(episodes, camera_map, video_mode, image_size)
    if video_mode == "copy":
        print("Copying source mp4 files directly; no video decode/resize/re-encode.")
        for out_key in camera_map:
            height, width = camera_hw[out_key]
            print(f"  {out_key}: {height}x{width}")
    else:
        print(
            f"Re-encoding videos at {image_size[0]}x{image_size[1]}: "
            "rgb = aspect-preserving resize + center crop (matching pixel_transforms_resize), "
            "tactile = direct resize."
        )

    dataset = _create_lerobot_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        features=_build_features(camera_map, camera_hw),
    )

    skipped: list[tuple[Path, str]] = []
    jobs = _build_episode_jobs(
        episodes=episodes,
        dataset=dataset,
        camera_map=camera_map,
        task=task,
        use_metadata_task=use_metadata_task,
        action_shift=action_shift,
        action_gripper_scale=action_gripper_scale,
        video_mode=video_mode,
        image_size=image_size,
        validate_videos=validate_videos,
        validate_video_samples=validate_video_samples,
        video_workers=video_workers,
        video_encode_workers=video_encode_workers,
        video_crf=video_crf,
        video_gop=video_gop,
        skipped=skipped,
    )
    if not jobs:
        raise SystemExit(f"No convertible episodes under {src}")

    run_t0 = time.perf_counter()
    results = _run_episode_jobs(jobs, episode_workers=episode_workers)
    timings = _finalize_metadata(dataset, jobs, results)
    wall_s = time.perf_counter() - run_t0

    if profile:
        for job in jobs:
            result = results[job.episode_index]
            tqdm.tqdm.write(
                f"{Path(job.ep_dir).name}: frames={result.episode_length} "
                f"numeric={result.timings['read_numeric']:.2f}s "
                f"decode={result.timings['decode']:.2f}s "
                f"table={result.timings['table']:.2f}s "
                f"stats={result.timings['stats']:.2f}s "
                f"video={result.timings['video']:.2f}s "
                f"worker_total={result.timings['total']:.2f}s"
            )

    video_files = list(Path(dataset.root).rglob("*.mp4"))
    assert len(video_files) == len(jobs) * len(dataset.meta.video_keys)

    parquet_files = list(Path(dataset.root).rglob("*.parquet"))
    assert len(parquet_files) == len(jobs)

    print(f"\nDataset written to {ds_path}")
    print(f"  Converted: {len(jobs)}")
    print(f"  Skipped:   {len(skipped)}")
    print("Timing totals:")
    for name in ("read_numeric", "decode", "table", "stats", "video", "total"):
        print(f"  worker_{name:12s}: {timings[name]:.2f}s")
    print(f"  wall_clock    : {wall_s:.2f}s")
    for ep, why in skipped[:20]:
        print(f"    - {ep.name}: {why}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True, help="Root of raw Flexiv episodes")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task language instruction used when --use-metadata-task is not set or metadata is missing",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional subset for smoke testing")
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Date directory to include under --src. Can be repeated or comma-separated. Defaults to all dates.",
    )
    parser.add_argument("--no-tactile", action="store_true", help="Skip tactile cameras")
    parser.add_argument(
        "--no-second-third-view",
        action="store_true",
        help="Skip observation.images.second_third_view",
    )
    parser.add_argument(
        "--use-metadata-task",
        action="store_true",
        help="Use metadata.json language_instruction per episode when present",
    )
    parser.add_argument("--robot-type", type=str, default=None, help="Value written to meta/info.json robot_type")
    parser.add_argument(
        "--action-shift",
        type=int,
        default=1,
        help="Frame offset for actions.joint_position/action. Default: 1",
    )
    parser.add_argument(
        "--action-gripper-scale",
        type=float,
        default=0.1,
        help="Scale applied to actions.eef_pose gripper.pos. Default: 0.1",
    )
    parser.add_argument(
        "--video-mode",
        choices=VIDEO_MODES,
        default=DEFAULT_VIDEO_MODE,
        help=(
            "reencode: decode/resize/re-encode videos (rgb keeps aspect ratio and is center "
            "cropped, tactile is resized directly); copy: directly copy source mp4 files"
        ),
    )
    parser.add_argument(
        "--image-size",
        "--image_size",
        type=int,
        nargs=2,
        default=list(DEFAULT_IMAGE_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Output video image size for --video-mode reencode, as HEIGHT WIDTH. "
            "RGB frames keep their aspect ratio and are center cropped to this size; "
            "tactile frames are resized directly."
        ),
    )
    parser.add_argument(
        "--validate-videos",
        dest="validate_videos",
        action="store_true",
        default=True,
        help="Use decord to skip episodes with short/corrupt source videos before conversion",
    )
    parser.add_argument(
        "--no-validate-videos",
        dest="validate_videos",
        action="store_false",
        help="Skip decord video validation for faster conversion",
    )
    parser.add_argument(
        "--validate-video-samples",
        type=int,
        default=3,
        help="Number of evenly spaced frames to decode per video when --validate-videos is set",
    )
    parser.add_argument("--episode-workers", type=int, default=8, help="Parallel episode worker processes")
    parser.add_argument("--video-workers", type=int, default=8, help="Parallel camera decode workers per episode")
    parser.add_argument(
        "--video-encode-workers",
        type=int,
        default=4,
        help="Parallel direct MP4 encode workers per episode",
    )
    parser.add_argument("--video-crf", type=int, default=23, help="CRF used by h264 encoding in reencode mode")
    parser.add_argument("--video-gop", type=int, default=2, help="GOP size used by h264 encoding in reencode mode")
    parser.add_argument("--profile", action="store_true", help="Print per-episode timing breakdown")
    args = parser.parse_args()

    convert(
        src=args.src,
        repo_id=args.repo_id,
        task=args.task,
        limit=args.limit,
        include_tactile=not args.no_tactile,
        include_second_third_view=not args.no_second_third_view,
        use_metadata_task=args.use_metadata_task,
        robot_type=args.robot_type,
        dates=arx._normalize_dates(args.date),
        action_shift=args.action_shift,
        action_gripper_scale=args.action_gripper_scale,
        video_mode=args.video_mode,
        image_size=args.image_size,
        validate_videos=args.validate_videos,
        validate_video_samples=args.validate_video_samples,
        episode_workers=args.episode_workers,
        video_workers=args.video_workers,
        video_encode_workers=args.video_encode_workers,
        video_crf=args.video_crf,
        video_gop=args.video_gop,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
