"""Convert raw ARX5 dual-arm NAS episodes (with tactile) to LeRobot v2.1 format.

Source schema (raw NAS episode dirs, e.g.
``/gpfs/EmbodiedAI/vtla/data/vtla_raw/verified/pull_cups/<date>/<user>/episode_XXXX/``):

    episode_XXXX/
        metadata.json
        meta/episode.json
        observation.image.third_view/{video.mp4,timestamps.csv}
        observation.image.left_wrist_view/{video.mp4,timestamps.csv}
        observation.image.right_wrist_view/{video.mp4,timestamps.csv}
        observation.image.left_wrist_left_tactile/{video.mp4,timestamps.csv}
        observation.image.left_wrist_right_tactile/{video.mp4,timestamps.csv}
        observation.image.right_wrist_left_tactile/{video.mp4,timestamps.csv}
        observation.image.right_wrist_right_tactile/{video.mp4,timestamps.csv}
        observation.state.joint_position/data.csv   (timestamp_ms, left_j1..j6,
                                                     left_gripper, right_j1..j6,
                                                     right_gripper)

Output: LeRobot v2.1 dataset under ``$HF_LEROBOT_HOME/<repo_id>/`` with:

    features:
      observation.state                                float32 (14,)  [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
      action                                           float32 (14,)  state[t+1]
      observation.image.third_view                     video   resized image_size
      observation.image.left_wrist_view                video   resized image_size
      observation.image.right_wrist_view               video   resized image_size
      observation.image.left_wrist_left_tactile        video   resized image_size
      observation.image.left_wrist_right_tactile       video   resized image_size
      observation.image.right_wrist_left_tactile       video   resized image_size
      observation.image.right_wrist_right_tactile      video   resized image_size

By default videos are decoded, resized to ``--image-size`` (224x224 by default),
and re-encoded into the LeRobot video layout.  Use ``--video-mode copy`` to
directly copy source mp4 files without resizing.

By default, conversion uses decord to filter out episodes whose source mp4
files are too short or fail sampled reads.  Use ``--no-validate-videos`` to
skip this check.

Usage:
    cd /gpfs/EmbodiedAI/vtla/code/neoVTLA
    source /gpfs/cessharedroot/hhr/hengzzzhou/miniconda3/etc/profile.d/conda.sh
    conda activate openpi
    export HF_LEROBOT_HOME=/gpfs/EmbodiedAI/vtla/data/lerobot_home
    python scripts/convert_arx5_to_lerobot.py \\
        --src /gpfs/EmbodiedAI/vtla/data/vtla_raw/verified/pull_cups \\
        --repo-id local/arx5_pull_cups_tactile \\
        --task "pull cups" \\
        --date 20240501 --date 20240502 \\
        --limit 50    # optional subset for smoke-testing
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import inspect
from dataclasses import dataclass
import os
import shutil
import time
from pathlib import Path
from typing import Any

import av
import datasets
import numpy as np
import tqdm

from lerobot.constants import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import check_timestamps_sync, get_episode_data_index
import lerobot.datasets.lerobot_dataset as _lds


# Canonical camera keys.
RGB_CAMERAS = (
    "observation.image.third_view",
    "observation.image.left_wrist_view",
    "observation.image.right_wrist_view",
)
TACTILE_CAMERAS = (
    "observation.image.left_wrist_left_tactile",
    "observation.image.left_wrist_right_tactile",
    "observation.image.right_wrist_left_tactile",
    "observation.image.right_wrist_right_tactile",
)
ALL_CAMERAS = RGB_CAMERAS + TACTILE_CAMERAS

JOINT_NAMES = [
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
]

DEFAULT_IMAGE_SIZE = (224, 224)  # (height, width)
FPS = 30
VIDEO_MODES = ("copy", "reencode")
DEFAULT_VIDEO_MODE = "reencode"
DECORD_DECODE_THREADS = 1
DECORD_BATCH_SIZE = 256


DUAL_ARM_COLS = (
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper",
    "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper",
)


@dataclass(frozen=True)
class EpisodeJob:
    ep_dir: str
    episode_index: int
    start_frame_index: int
    episode_length: int
    task: str
    task_index: int
    cameras: list[str]
    root: str
    data_file_path: str
    video_file_paths: dict[str, str]
    hf_features: Any
    features: dict[str, Any]
    fps: int
    video_mode: str
    image_size: tuple[int, int]
    video_workers: int | None
    video_encode_workers: int
    video_crf: int
    video_gop: int | None


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    episode_length: int
    task: str
    stats: dict[str, dict]
    timings: dict[str, float]


def _has_dual_arm_schema(csv_path: Path) -> bool:
    """Return True iff the header has the 14-col dual-arm schema (left_j1 present)."""
    try:
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
    except OSError:
        return False
    cols = set(header)
    return "left_j1" in cols


def _read_joint_csv(path: Path) -> np.ndarray:
    """Load joint_position CSV and return shape (T, 14) in JOINT_NAMES order."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = np.zeros((len(rows), 14), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = [float(r[c]) for c in DUAL_ARM_COLS]
    return out


def _count_joint_csv_rows(path: Path) -> int:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty csv") from exc
        return sum(1 for _ in reader)


def _decode_and_resize(
    vid_path: Path,
    expected: int,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Decode ``vid_path`` with decord resize, return list of (H,W,3) uint8.

    Pads or truncates to exactly ``expected`` frames so all cameras align with
    the state/action CSV length.
    """
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("reencode video mode requires decord in the active environment") from exc

    target_hw = _normalize_image_size(image_size)
    frames: list[np.ndarray] = []
    video_reader = decord.VideoReader(
        str(vid_path),
        width=target_hw[1],
        height=target_hw[0],
        num_threads=DECORD_DECODE_THREADS,
    )
    frame_count = min(int(expected), len(video_reader))
    for start in range(0, frame_count, DECORD_BATCH_SIZE):
        stop = min(start + DECORD_BATCH_SIZE, frame_count)
        batch = video_reader.get_batch(list(range(start, stop))).asnumpy()
        frames.extend(batch[i] for i in range(len(batch)))
    if not frames:
        raise RuntimeError(f"{vid_path}: decoded 0 frames")
    while len(frames) < expected:
        frames.append(frames[-1])
    return frames


def _effective_worker_count(
    requested: int | None,
    item_count: int,
    default_cap: int,
) -> int:
    if item_count <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    if requested is None or requested < 1:
        requested = min(default_cap, cpu_count)
    return max(1, min(int(requested), item_count))


def _decode_episode_cameras(
    ep_dir: Path,
    cameras: list[str],
    expected_frames: int,
    image_size: tuple[int, int],
    video_workers: int | None,
) -> dict[str, list[np.ndarray]]:
    workers = _effective_worker_count(video_workers, len(cameras), default_cap=4)

    def decode_one(cam: str) -> tuple[str, list[np.ndarray]]:
        return cam, _decode_and_resize(ep_dir / cam / "video.mp4", expected_frames, image_size)

    if workers == 1:
        return dict(decode_one(cam) for cam in cameras)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(decode_one, cameras))


def _mux_encoded_packets(output: av.container.OutputContainer, packets: Any) -> None:
    if packets is None:
        return
    if isinstance(packets, (list, tuple)):
        for packet in packets:
            output.mux(packet)
    else:
        output.mux(packets)


def _encode_video_arrays(
    frames: list[np.ndarray],
    video_path: Path,
    fps: int,
    crf: int,
    gop: int | None,
) -> None:
    if not frames:
        raise ValueError(f"No frames to encode for {video_path}")

    first_shape = frames[0].shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError(f"Expected RGB frames with shape (H, W, 3), got {first_shape}")
    height, width = first_shape[:2]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, str] = {"crf": str(crf)}
    if gop is not None:
        options["g"] = str(gop)

    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("h264", fps, options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = int(width)
        stream.height = int(height)
        for arr in frames:
            if arr.shape != first_shape:
                raise ValueError(f"{video_path}: frame shape changed from {first_shape} to {arr.shape}")
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            _mux_encoded_packets(output, stream.encode(frame))
        _mux_encoded_packets(output, stream.encode())

    if not video_path.exists():
        raise OSError(f"Video encoding did not create {video_path}")


def _sample_frame_indices(data_len: int) -> list[int]:
    if data_len <= 0:
        return []
    min_num_samples = min(100, data_len)
    num_samples = max(min_num_samples, min(int(data_len**0.75), 10_000))
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _auto_downsample_chw(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    downsample_factor = max(1, downsample_factor)
    return img[:, ::downsample_factor, ::downsample_factor]


def _feature_stats(array: np.ndarray, axis: int | tuple[int, ...], keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _compute_episode_stats_direct(
    episode_buffer: dict[str, np.ndarray],
    cam_frames: dict[str, list[np.ndarray]] | None,
    features: dict[str, Any],
) -> dict[str, dict]:
    ep_stats: dict[str, dict] = {}

    for key, data in episode_buffer.items():
        if features[key]["dtype"] == "string":
            continue
        keepdims = data.ndim == 1
        ep_stats[key] = _feature_stats(data, axis=0, keepdims=keepdims)

    if not cam_frames:
        return ep_stats

    for key, frames in cam_frames.items():
        sampled_indices = _sample_frame_indices(len(frames))
        if not sampled_indices:
            continue

        sampled_frames = []
        for idx in sampled_indices:
            chw = np.asarray(frames[idx], dtype=np.uint8).transpose(2, 0, 1)
            sampled_frames.append(_auto_downsample_chw(chw))
        video_array = np.stack(sampled_frames, axis=0)
        stats = _feature_stats(video_array, axis=(0, 2, 3), keepdims=True)
        ep_stats[key] = {
            stat_key: value if stat_key == "count" else np.squeeze(value / 255.0, axis=0)
            for stat_key, value in stats.items()
        }
    return ep_stats


def _save_episode_table_direct(
    root: Path,
    data_file_path: str,
    episode_buffer: dict[str, np.ndarray],
    hf_features: Any,
) -> None:
    episode_dict = {key: episode_buffer[key] for key in hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=hf_features, split="train")
    ep_dataset = _lds.embed_images(ep_dataset)
    ep_data_path = root / data_file_path
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def _validation_frame_indices(total_frames: int, num_samples: int) -> list[int]:
    """Mirror BaseVideoDataset.validation_frame_indices for conversion-time checks."""
    safe_total_frames = max(1, int(total_frames))
    num_points = min(safe_total_frames, max(1, int(num_samples)))
    return sorted(set(np.linspace(0, safe_total_frames - 1, num=num_points, dtype=int).tolist()))


def _validate_video_decode_with_decord(
    video_path: Path,
    expected_frames: int,
    num_samples: int,
    num_threads: int = 1,
) -> None:
    """Validate that decord can read sampled frames up to ``expected_frames - 1``.

    This follows the same policy as BaseVideoDataset.validate_video_decode:
    construct a decord VideoReader, check that the requested clip range fits in
    the video, then decode sampled frames from that range.
    """
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("--validate-videos requires decord in the active environment") from exc

    video_reader = decord.VideoReader(str(video_path), num_threads=num_threads)
    total_video_frames = len(video_reader)
    if total_video_frames <= 0:
        raise ValueError(f"empty video: {video_path}")

    clip_end = int(expected_frames) - 1
    if clip_end < 0:
        raise ValueError(f"invalid expected frame count {expected_frames} for {video_path}")
    if clip_end >= total_video_frames:
        raise ValueError(
            f"video has {total_video_frames} frames, shorter than joint CSV length {expected_frames}"
        )

    indices = _validation_frame_indices(expected_frames, num_samples)
    video_reader.get_batch(indices)


def _validate_episode_videos(
    ep_dir: Path,
    cameras: list[str],
    expected_frames: int,
    num_samples: int,
) -> None:
    for cam in cameras:
        _validate_video_decode_with_decord(
            ep_dir / cam / "video.mp4",
            expected_frames=expected_frames,
            num_samples=num_samples,
        )


def _probe_video_hw(vid_path: Path) -> tuple[int, int] | None:
    """Read video height/width from container metadata without decoding frames."""
    try:
        with av.open(str(vid_path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return None
            codec_ctx = getattr(stream, "codec_context", None)
            height = getattr(codec_ctx, "height", 0) or getattr(stream, "height", 0)
            width = getattr(codec_ctx, "width", 0) or getattr(stream, "width", 0)
    except Exception:
        return None

    if not height or not width:
        return None
    return int(height), int(width)


def _normalize_image_size(image_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError(f"image_size must have exactly 2 values, got {image_size!r}")
    height, width = (int(image_size[0]), int(image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size values must be positive, got {image_size!r}")
    return height, width


def _infer_camera_hw(
    episodes: list[Path],
    cameras: list[str],
    video_mode: str,
    image_size: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    """Infer per-camera video feature shapes for copy mode; use image_size when re-encoding."""
    if video_mode == "reencode":
        return {cam: image_size for cam in cameras}

    camera_hw: dict[str, tuple[int, int]] = {}
    for cam in cameras:
        for ep_dir in episodes:
            hw = _probe_video_hw(ep_dir / cam / "video.mp4")
            if hw is not None:
                camera_hw[cam] = hw
                break
        camera_hw.setdefault(cam, image_size)
    return camera_hw


def _get_video_file_path(dataset: LeRobotDataset, episode_index: int, video_key: str) -> Path:
    """Return the on-disk LeRobot video path across minor API differences."""
    meta = dataset.meta
    if hasattr(meta, "get_video_file_path"):
        return Path(dataset.root) / meta.get_video_file_path(episode_index, video_key)

    info = meta.info
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    video_path = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return Path(dataset.root) / video_path.format(
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        episode_index=episode_index,
        file_index=episode_index,
        video_key=video_key,
    )


def _get_data_file_path(dataset: LeRobotDataset, episode_index: int) -> Path:
    """Return the on-disk LeRobot parquet path across minor API differences."""
    meta = dataset.meta
    if hasattr(meta, "get_data_file_path"):
        return Path(dataset.root) / meta.get_data_file_path(ep_index=episode_index)

    info = meta.info
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return Path(dataset.root) / data_path.format(
        episode_chunk=episode_chunk,
        chunk_index=episode_chunk,
        episode_index=episode_index,
        file_index=episode_index,
    )


def _normalize_dates(raw_dates: list[str] | None) -> set[str] | None:
    """Normalize repeated/comma-separated --date values.

    Returns None when no date filter is requested, which means all dates.
    """
    if raw_dates is None:
        return None

    dates: set[str] = set()
    for raw in raw_dates:
        for date in raw.split(","):
            date = date.strip()
            if date:
                dates.add(date)
    return dates or None


def _discover_episodes(src: Path, limit: int | None, dates: set[str] | None = None) -> list[Path]:
    """Walk ``src`` looking for ``*/*/episode_*`` dirs whose joint_position CSV
    has the 14-col dual-arm schema (left_j1 present).
    """
    if dates is None:
        all_eps = sorted(src.glob("*/*/episode_*"))
    else:
        all_eps = sorted(
            ep
            for date in dates
            for ep in (src / date).glob("*/episode_*")
        )
    kept: list[Path] = []
    for ep in all_eps:
        jp = ep / "observation.state.joint_position" / "data.csv"
        if jp.exists() and _has_dual_arm_schema(jp):
            kept.append(ep)
    date_msg = "all dates" if dates is None else ", ".join(sorted(dates))
    print(f"Scanned {len(all_eps)} raw episodes for {date_msg}, kept {len(kept)} dual-arm (14-dim).")
    if limit is not None:
        kept = kept[:limit]
    return kept


def _build_features(cameras: list[str], camera_hw: dict[str, tuple[int, int]]) -> dict:
    features: dict = {
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
    for cam in cameras:
        height, width = camera_hw.get(cam, DEFAULT_IMAGE_SIZE)
        features[cam] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _create_lerobot_dataset(
    repo_id: str,
    robot_type: str,
    features: dict,
) -> LeRobotDataset:
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


def _path_relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _process_episode_job(job: EpisodeJob) -> EpisodeResult:
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
    states = _read_joint_csv(ep_dir / "observation.state.joint_position" / "data.csv")
    episode_length = len(states)
    if episode_length != job.episode_length:
        raise ValueError(
            f"{ep_dir}: expected {job.episode_length} frames from preflight, got {episode_length}"
        )
    actions = np.concatenate([states[1:], states[-1:]], axis=0)
    timings["read_numeric"] = time.perf_counter() - numeric_t0

    timestamps = np.arange(episode_length, dtype=np.float32) / job.fps
    episode_buffer = {
        "observation.state": states,
        "action": actions,
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
            cameras=job.cameras,
            expected_frames=episode_length,
            image_size=job.image_size,
            video_workers=job.video_workers,
        )
        timings["decode"] = time.perf_counter() - decode_t0

    table_t0 = time.perf_counter()
    _save_episode_table_direct(root, job.data_file_path, episode_buffer, job.hf_features)
    timings["table"] = time.perf_counter() - table_t0

    stats_t0 = time.perf_counter()
    ep_stats = _compute_episode_stats_direct(episode_buffer, cam_frames, job.features)
    timings["stats"] = time.perf_counter() - stats_t0

    video_t0 = time.perf_counter()
    if job.video_mode == "copy":
        for cam in job.cameras:
            src_video = ep_dir / cam / "video.mp4"
            dst_video = root / job.video_file_paths[cam]
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            if src_video.resolve() != dst_video.resolve():
                shutil.copy2(src_video, dst_video)
    else:
        assert cam_frames is not None
        encode_workers = _effective_worker_count(
            job.video_encode_workers,
            len(job.cameras),
            default_cap=4,
        )

        def encode_one(cam: str) -> None:
            _encode_video_arrays(
                cam_frames[cam],
                root / job.video_file_paths[cam],
                fps=job.fps,
                crf=job.video_crf,
                gop=job.video_gop,
            )

        if encode_workers == 1:
            for cam in job.cameras:
                encode_one(cam)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers) as executor:
                list(executor.map(encode_one, job.cameras))
    timings["video"] = time.perf_counter() - video_t0
    timings["total"] = time.perf_counter() - episode_t0

    return EpisodeResult(
        episode_index=job.episode_index,
        episode_length=episode_length,
        task=job.task,
        stats=ep_stats,
        timings=timings,
    )


def _run_episode_jobs(jobs: list[EpisodeJob], episode_workers: int) -> dict[int, EpisodeResult]:
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")

    results: dict[int, EpisodeResult] = {}
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
    cameras: list[str],
    task: str,
    video_mode: str,
    image_size: tuple[int, int],
    validate_videos: bool,
    validate_video_samples: int,
    video_workers: int | None,
    video_encode_workers: int,
    video_crf: int,
    video_gop: int | None,
    skipped: list[tuple[Path, str]],
) -> list[EpisodeJob]:
    task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        dataset.meta.add_task(task)
        task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        raise RuntimeError(f"failed to register task: {task}")

    jobs: list[EpisodeJob] = []
    next_frame_index = 0
    root = Path(dataset.root)

    for ep_dir in episodes:
        try:
            jp_csv = ep_dir / "observation.state.joint_position" / "data.csv"
            if not jp_csv.exists():
                skipped.append((ep_dir, "no joint_position csv"))
                continue
            episode_length = _count_joint_csv_rows(jp_csv)
            if episode_length < 2:
                skipped.append((ep_dir, f"only {episode_length} frames"))
                continue

            missing_cam = next(
                (cam for cam in cameras if not (ep_dir / cam / "video.mp4").exists()),
                None,
            )
            if missing_cam is not None:
                skipped.append((ep_dir, f"missing {missing_cam}"))
                continue
            if validate_videos:
                _validate_episode_videos(
                    ep_dir=ep_dir,
                    cameras=cameras,
                    expected_frames=episode_length,
                    num_samples=validate_video_samples,
                )

            episode_index = len(jobs)
            jobs.append(
                EpisodeJob(
                    ep_dir=str(ep_dir),
                    episode_index=episode_index,
                    start_frame_index=next_frame_index,
                    episode_length=episode_length,
                    task=task,
                    task_index=task_index,
                    cameras=cameras,
                    root=str(root),
                    data_file_path=_path_relative_to_root(
                        _get_data_file_path(dataset, episode_index),
                        root,
                    ),
                    video_file_paths={
                        cam: _path_relative_to_root(_get_video_file_path(dataset, episode_index, cam), root)
                        for cam in cameras
                    },
                    hf_features=dataset.hf_features,
                    features=dataset.features,
                    fps=dataset.fps,
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
    jobs: list[EpisodeJob],
    results: dict[int, EpisodeResult],
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
    robot_type: str | None,
    dates: set[str] | None = None,
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
    image_size = _normalize_image_size(image_size)
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")

    cameras = list(RGB_CAMERAS)
    if include_tactile:
        cameras += list(TACTILE_CAMERAS)
    robot_type = robot_type or ("arx5_dual_tactile" if include_tactile else "arx5_dual")

    episodes = _discover_episodes(src, limit, dates)
    if not episodes:
        raise SystemExit(f"No episodes under {src}")
    print(f"Found {len(episodes)} episodes; converting with {len(cameras)} cameras:")
    for c in cameras:
        print(f"  - {c}")
    print(f"Robot type: {robot_type}")
    print(f"Video mode: {video_mode}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(
        "Parallelism: "
        f"episode_workers={episode_workers}, "
        f"video_workers={_effective_worker_count(video_workers, len(cameras), default_cap=4)}, "
        f"video_encode_workers={_effective_worker_count(video_encode_workers, len(cameras), default_cap=4)}, "
        f"video_crf={video_crf}, "
        f"video_gop={video_gop}"
    )
    if validate_videos:
        print(f"Video validation: decord sampled decode ({validate_video_samples} frames/video)")

    ds_path = LEROBOT_HOME / repo_id
    if ds_path.exists():
        print(f"Removing existing dataset at {ds_path}")
        shutil.rmtree(ds_path)

    camera_hw = _infer_camera_hw(episodes, cameras, video_mode, image_size)
    if video_mode == "copy":
        print("Copying source mp4 files directly; no video decode/resize/re-encode.")
        for cam in cameras:
            height, width = camera_hw[cam]
            print(f"  {cam}: {height}x{width}")
    else:
        print(f"Re-encoding videos at {image_size[0]}x{image_size[1]}.")

    dataset = _create_lerobot_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        features=_build_features(cameras, camera_hw),
    )

    skipped: list[tuple[Path, str]] = []
    jobs = _build_episode_jobs(
        episodes=episodes,
        dataset=dataset,
        cameras=cameras,
        task=task,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Root of NAS episodes (e.g. .../pull_cups)")
    ap.add_argument("--repo-id", type=str, required=True)
    ap.add_argument("--task", type=str, required=True,
                    help="Task language instruction, e.g. 'pull cups'")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only convert the first N episodes (smoke test)")
    ap.add_argument("--date", action="append", default=None,
                    help="Date directory to include under --src. Can be repeated or comma-separated. Defaults to all dates.")
    ap.add_argument("--no-tactile", action="store_true",
                    help="Skip tactile cameras (RGB-only ablation baseline)")
    ap.add_argument("--robot-type", type=str, default=None,
                    help="Value written to meta/info.json robot_type")
    ap.add_argument("--video-mode", choices=VIDEO_MODES, default=DEFAULT_VIDEO_MODE,
                    help="reencode: decode/resize/re-encode videos; copy: directly copy source mp4 files")
    ap.add_argument("--image-size", "--image_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE),
                    metavar=("HEIGHT", "WIDTH"),
                    help="Output video image size for --video-mode reencode, as HEIGHT WIDTH")
    ap.add_argument("--validate-videos", dest="validate_videos", action="store_true", default=True,
                    help="Use decord to skip episodes with short/corrupt source videos before conversion")
    ap.add_argument("--no-validate-videos", dest="validate_videos", action="store_false",
                    help="Skip decord video validation for faster conversion")
    ap.add_argument("--validate-video-samples", type=int, default=3,
                    help="Number of evenly spaced frames to decode per video when --validate-videos is set")
    ap.add_argument("--episode-workers", type=int, default=8,
                    help="Parallel episode worker processes. Use 1 for serial episode processing.")
    ap.add_argument("--video-workers", type=int, default=8,
                    help="Parallel camera decode workers inside each episode worker")
    ap.add_argument("--video-encode-workers", type=int, default=4,
                    help="Parallel direct MP4 encode workers inside each episode worker")
    ap.add_argument("--video-crf", type=int, default=23,
                    help="CRF used by h264 video encoding in --video-mode reencode")
    ap.add_argument("--video-gop", type=int, default=2,
                    help="GOP size used by h264 video encoding in --video-mode reencode")
    ap.add_argument("--profile", action="store_true",
                    help="Print per-episode timing breakdown")
    args = ap.parse_args()
    convert(
        src=args.src,
        repo_id=args.repo_id,
        task=args.task,
        limit=args.limit,
        include_tactile=not args.no_tactile,
        robot_type=args.robot_type,
        dates=_normalize_dates(args.date),
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
