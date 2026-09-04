from __future__ import annotations

import json
import os
import random
import traceback
from pathlib import Path
from typing import Callable, Iterable, Sequence

import decord
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from starVLA.utils.system_utils import zero_rank_print


class BaseVideoDataset(Dataset):
    cache_version = "v0"

    def __init__(
        self,
        config,
        data_name: str,
        data_root: Path,
        data_config,
        data_type: str,
        data_id: int = 0,
        video_keys: Sequence[str] | None = None,
        cache_name: str | None = None,
        force_rebuild_cache: bool = False,
        validate_video_paths_per_episode: bool = False,
        validate_cache_decode_per_episode: bool = False,
        validate_cache_decode_num_frames: int = 3,
        mode: str = "train",
        **_,
    ):
        self.config = config
        self.mode = mode
        self.data_name = data_name
        self.data_root = Path(data_root)
        self.data_config = data_config
        self.data_type = data_type
        self.data_id = data_id
        self.video_keys_override = [key for key in (video_keys or []) if key is not None]
        self.cache_name = cache_name or data_name or self.data_root.name
        self.dataset = {"data": []}
        self.force_rebuild_cache = bool(force_rebuild_cache or getattr(config, "force_rebuild_cache", False))
        self.validate_video_paths_per_episode = bool(validate_video_paths_per_episode)
        self.validate_cache_decode_per_episode = bool(
            validate_cache_decode_per_episode or getattr(config, "validate_cache_decode_per_episode", False)
        )
        self.validate_cache_decode_num_frames = max(
            1,
            int(validate_cache_decode_num_frames or getattr(config, "validate_cache_decode_num_frames", 3)),
        )
        self.bad_indices = set()
        self.max_getitem_retries = max(1, int(getattr(config, "getitem_retry_limit", 4)))
        self.bad_index_failure_counts: dict[int, int] = {}
        self.cache_validation_failures = 0
        self.cache_validation_failure_examples: list[str] = []

        self.merge_views = getattr(config, "merge_views", None)
        self.target_fps = float(getattr(config, "target_fps"))
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive for dataset {self.data_name}, got {self.target_fps}")
        self.num_frames = int(getattr(config, "num_frames"))
        self.image_size = tuple(getattr(config, "image_size"))
        resize_method = getattr(config, "resize_method", None)
        if resize_method is not None:
            self.data_config.resize_method = str(resize_method)
        self.data_config.set_image_size(self.image_size)

    @property
    def dataset_name(self):
        return self.data_name

    def cache_path(self) -> Path:
        return Path("playground/meta_data") / f"{self.cache_name}_video_dataset_info_cache_{self.cache_version}.json"

    def load_cache(self) -> bool:
        if self.force_rebuild_cache:
            return False
        cache_path = self.cache_path()
        if not cache_path.exists():
            return False
        zero_rank_print(f"Load Cache Dataset Information from {cache_path}")
        with open(cache_path, "r") as f:
            self.dataset = json.load(f)
        return True

    def save_cache(self):
        cache_path = self.cache_path()
        os.makedirs(cache_path.parent, exist_ok=True)
        zero_rank_print(f"Save Cache Dataset Information to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(self.dataset, f, indent=2)

    def finalize_dataset(self, step_getter: Callable):
        self.filter_invalid_entries()
        self.apply_episode_split()
        self.total_episodes = len(self.dataset["data"])
        if self.total_episodes == 0:
            raise ValueError(
                f"[{self.data_name}] No valid video entries remain after target_fps={self.target_fps} filtering"
            )
        self.total_steps = sum(step_getter(item) for item in self.dataset["data"])
        zero_rank_print(f"[{self.data_name}] Total episodes: {self.total_episodes}, Total steps: {self.total_steps}")
        if self.cache_validation_failures > 0:
            zero_rank_print(
                f"[{self.data_name}] Skipped {self.cache_validation_failures} invalid entries during cache rebuild"
            )
            for example in self.cache_validation_failure_examples:
                zero_rank_print(f"[{self.data_name}]   {example}")

    def resolve_video_keys(self, discovered_keys: Iterable[str], override_keys: Sequence[str] | None = None):
        return self.data_config.resolve_video_keys(discovered_keys, override_keys or self.video_keys_override)

    def entry_clip_length(self, entry: dict) -> int:
        if "total_frames" in entry:
            return int(entry["total_frames"])
        if "clip_start" in entry and "clip_end" in entry:
            return int(entry["clip_end"]) - int(entry["clip_start"]) + 1
        if "index_start" in entry and "index_end" in entry:
            return int(entry["index_end"]) - int(entry["index_start"]) + 1
        raise KeyError(f"Unable to resolve clip length for dataset entry in {self.data_name}")

    def filter_invalid_entries(self):
        entries = self.dataset.get("data")
        if not isinstance(entries, list) or not entries:
            return

        kept_entries = []
        for entry in entries:
            source_fps = float(entry.get("fps", 0.0))
            clip_length = self.entry_clip_length(entry)
            clip_duration_s = (clip_length - 1) / source_fps if source_fps > 0 else -1.0
            required_duration_s = self.num_frames / self.target_fps
            if source_fps >= self.target_fps and clip_duration_s >= required_duration_s:
                kept_entries.append(entry)

        if len(kept_entries) != len(entries):
            removed_count = len(entries) - len(kept_entries)
            zero_rank_print(
                f"[{self.data_name}] Filtered {removed_count} / {len(entries)} entries for target_fps={self.target_fps}"
            )
            self.dataset["data"] = kept_entries

    def apply_episode_split(self):
        if getattr(self.config, "split_strategy", "none") != "episode_ratio":
            return

        entries = self.dataset["data"]
        eval_ratio = float(getattr(self.config, "eval_ratio", 0.0))
        if eval_ratio <= 0 or len(entries) < 2:
            return

        rng = random.Random(f"{getattr(self.config, 'split_seed', 42)}:{self.data_name}")
        indices = list(range(len(entries)))
        rng.shuffle(indices)

        n_eval = max(1, int(round(len(entries) * eval_ratio)))
        n_eval = min(n_eval, len(entries) - 1)
        eval_indices = set(indices[:n_eval])

        if self.mode == "eval":
            self.dataset["data"] = [entry for i, entry in enumerate(entries) if i in eval_indices]
        else:
            self.dataset["data"] = [entry for i, entry in enumerate(entries) if i not in eval_indices]

    def sample_frame_indices(self, total_frames: int, fps: float, clip_start: int = 0, clip_end: int | None = None):
        if clip_end is None:
            clip_end = clip_start + int(total_frames) - 1

        source_fps = float(fps)
        if source_fps <= 0:
            raise ValueError(f"Invalid fps={fps} for dataset {self.data_name}")
        if source_fps < self.target_fps:
            raise ValueError(
                f"source fps {source_fps} below target_fps {self.target_fps} for dataset {self.data_name}"
            )

        clip_start = int(clip_start)
        clip_end = int(clip_end)
        clip_length = clip_end - clip_start + 1
        if clip_length <= 0:
            raise ValueError(f"Invalid clip range [{clip_start}, {clip_end}] for dataset {self.data_name}")

        required_points = self.num_frames + 1
        required_duration_s = self.num_frames / self.target_fps
        clip_duration_s = (clip_length - 1) / source_fps
        if clip_duration_s < required_duration_s:
            raise ValueError(
                f"Clip duration {clip_duration_s:.4f}s shorter than required {required_duration_s:.4f}s "
                f"for dataset {self.data_name}"
            )

        max_start_s = clip_duration_s - required_duration_s
        start_s = random.uniform(0.0, max_start_s) if max_start_s > 0 else 0.0
        target_times_s = start_s + np.arange(required_points, dtype=np.float64) / self.target_fps
        relative_indices = np.floor(target_times_s * source_fps + 0.5).astype(np.int64)
        relative_indices = np.clip(relative_indices, a_min=0, a_max=clip_length - 1)
        return (relative_indices + clip_start).tolist()

    def sample_video_keys(self, available_video_keys: Sequence[str]) -> list[str]:
        available = list(available_video_keys)
        if not available:
            raise ValueError(f"No video keys available for dataset {self.data_name}")

        return available

    def decode_video(
        self,
        video_path: str | Path,
        indices,
        source_height: int | None = None,
        source_width: int | None = None,
        num_threads: int = 1,
        force_native_resolution: bool = False,
    ) -> torch.Tensor:
        reader_kwargs = {"num_threads": num_threads}
        if not force_native_resolution and source_height and source_width:
            resize_h, resize_w = self.data_config.get_decode_resize_shape(source_height, source_width)
            reader_kwargs["height"] = resize_h
            reader_kwargs["width"] = resize_w
        video_reader = decord.VideoReader(str(video_path), **reader_kwargs)
        video = video_reader.get_batch(indices)
        return self.data_config.format_video(video)

    def validation_frame_indices(self, total_frames: int):
        safe_total_frames = max(1, int(total_frames))
        num_points = min(safe_total_frames, self.validate_cache_decode_num_frames)
        return sorted(set(np.linspace(0, safe_total_frames - 1, num=num_points, dtype=int).tolist()))

    def validate_video_decode(
        self,
        video_path: str | Path,
        clip_start: int = 0,
        clip_end: int | None = None,
        num_threads: int = 1,
    ):
        video_reader = decord.VideoReader(str(video_path), num_threads=num_threads)
        total_video_frames = len(video_reader)
        if total_video_frames <= 0:
            raise ValueError(f"Empty video: {video_path}")

        clip_start = max(0, int(clip_start))
        if clip_end is None:
            clip_end = total_video_frames - 1
        clip_end = int(clip_end)
        if clip_start > clip_end:
            raise ValueError(f"Invalid clip range [{clip_start}, {clip_end}] for {video_path}")
        if clip_end >= total_video_frames:
            raise ValueError(
                f"Clip end {clip_end} out of range for {video_path} (video has {total_video_frames} frames)"
            )

        indices = [clip_start + offset for offset in self.validation_frame_indices(clip_end - clip_start + 1)]
        video_reader.get_batch(indices)

    def validate_multi_video_decode(
        self,
        video_paths: Sequence[str | Path],
        clip_start: int = 0,
        clip_end: int | None = None,
        num_threads: int = 1,
    ):
        for video_path in video_paths:
            self.validate_video_decode(
                video_path=video_path,
                clip_start=clip_start,
                clip_end=clip_end,
                num_threads=num_threads,
            )

    def record_cache_validation_failure(self, identifier: str, exc: Exception):
        self.cache_validation_failures += 1
        if len(self.cache_validation_failure_examples) < 10:
            self.cache_validation_failure_examples.append(f"{identifier}: {exc}")

    def decode_multi_view(
        self,
        video_keys: Sequence[str],
        path_fn: Callable[[str], str | Path],
        indices,
        source_height: int | None = None,
        source_width: int | None = None,
        num_threads: int = 1,
        force_native_resolution: bool = False,
    ):
        selected_video_keys = self.sample_video_keys(video_keys)
        videos = [
            self.decode_video(
                path_fn(video_key),
                indices,
                source_height=source_height,
                source_width=source_width,
                num_threads=num_threads,
                force_native_resolution=force_native_resolution,
            )
            for video_key in selected_video_keys
        ]
        return self.pack_videos(videos)

    def pack_videos(self, video_list: Sequence[torch.Tensor]):
        if not video_list:
            raise ValueError(f"Empty video list for dataset {self.data_name}")

        packed = list(video_list)
        view_mask = [True] * len(packed)
        if self.merge_views == "horizontal":
            videos = torch.cat(packed, dim=-1).unsqueeze(1)
            return videos, torch.tensor([True], dtype=torch.bool)
        elif self.merge_views == "vertical":
            videos = torch.cat(packed, dim=-2).unsqueeze(1)
            return videos, torch.tensor([True], dtype=torch.bool)
        else:
            videos = torch.stack(packed, dim=1)
            return videos, torch.tensor(view_mask, dtype=torch.bool)

    def _metadata_from_feature(self, info: dict, video_key: str):
        feature_info = info.get("features", {}).get(video_key, {})
        if not isinstance(feature_info, dict):
            return None

        fps = None
        for metadata_key in ("info", "video_info"):
            metadata = feature_info.get(metadata_key, {})
            if not isinstance(metadata, dict):
                continue
            height = metadata.get("video.height")
            width = metadata.get("video.width")
            fps = metadata.get("video.fps", fps)
            if height and width and fps:
                return int(height), int(width), float(fps)

        shape = feature_info.get("shape")
        names = feature_info.get("names")
        if isinstance(shape, (list, tuple)) and isinstance(names, (list, tuple)):
            index_map = {str(name): idx for idx, name in enumerate(names)}
            height_index = index_map.get("height")
            if height_index is None:
                height_index = index_map.get("h")
            width_index = index_map.get("width")
            if width_index is None:
                width_index = index_map.get("w")
            if height_index is not None and width_index is not None:
                height = int(shape[height_index])
                width = int(shape[width_index])
                if height > 1 and width > 1 and fps:
                    return height, width, float(fps)
        return None

    def resolve_video_metadata(self, info: dict, video_key: str, fallback_video_path: str | Path):
        metadata = self._metadata_from_feature(info, video_key)
        if metadata is not None:
            return metadata

        video_reader = decord.VideoReader(str(fallback_video_path), num_threads=1)
        frame = video_reader[0]
        return int(frame.shape[0]), int(frame.shape[1]), float(video_reader.get_avg_fps())

    def resolve_episode_video_metadata(
        self,
        info: dict,
        video_keys: Sequence[str],
        video_template: str,
        cached_video_metadata=None,
    ):
        episode_video_paths = [Path(video_template.format(video_key=video_key)) for video_key in video_keys]
        if any(not video_path.exists() for video_path in episode_video_paths):
            return None

        if cached_video_metadata is not None:
            return cached_video_metadata

        try:
            return self.resolve_video_metadata(info, video_keys[0], episode_video_paths[0])
        except Exception:
            return None

    def build_sample(
        self,
        image: torch.Tensor,
        view_mask: torch.Tensor,
        prompt: str,
    ):
        return {
            "image": image,
            "view_mask": view_mask,
            "prompt": prompt,
            "fps": float(self.target_fps),
        }

    def __len__(self):
        return self.total_episodes

    def _next_retry_index(self, current_idx: int, attempted_indices: set[int]):
        if self.total_episodes <= 1:
            return None

        for offset in range(1, self.total_episodes):
            candidate_idx = (int(current_idx) + offset) % self.total_episodes
            if candidate_idx in attempted_indices or candidate_idx in self.bad_indices:
                continue
            return candidate_idx

        return None

    def __getitem__(self, idx):
        current_idx = int(idx) % self.total_episodes
        attempted_indices = set()
        last_exc = None

        while len(attempted_indices) < min(self.max_getitem_retries, self.total_episodes):
            if current_idx in attempted_indices:
                next_idx = self._next_retry_index(current_idx, attempted_indices)
                if next_idx is None:
                    break
                current_idx = next_idx

            attempted_indices.add(current_idx)
            try:
                return self.get_batch(current_idx)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_exc = exc
                failure_count = self.bad_index_failure_counts.get(current_idx, 0) + 1
                self.bad_index_failure_counts[current_idx] = failure_count
                if failure_count == 1:
                    traceback.print_exc()
                self.bad_indices.add(current_idx)
                next_idx = self._next_retry_index(current_idx, attempted_indices)
                if next_idx is None:
                    break
                current_idx = next_idx

        raise RuntimeError(
            f"Failed to load sample from {self.data_name} after {len(attempted_indices)} retries; "
            f"last_idx={current_idx} bad_indices={len(self.bad_indices)}"
        ) from last_exc
