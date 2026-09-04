"""
Multi-LeRobot VLA Dataset

For datasets that contain multiple lerobot sub-datasets under one root directory.
Each sub-directory has its own meta/info.json.
Examples: robocoin, galaxea-open-world-dataset

Structure:
  data_root/
    subdataset_1/
      meta/info.json
      data/chunk-XXX/
      videos/chunk-XXX/
    subdataset_2/
      meta/info.json
      ...
"""

import os
import traceback
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from starVLA.utils.system_utils import zero_rank_print
from starVLA.utils.data_utils import load_jsonl, OnlineStats
from starVLA.utils.video_utils import decode_video_frames
from starVLA.utils.text_embedding_cache import (
    DEFAULT_CONTEXT_LEN,
    DEFAULT_ENCODER_ID,
    maybe_load_text_embedding_cache,
)
from starVLA.dataloader.vla.data_config import BaseDataConfig
from starVLA.dataloader.vla import tactile_stress
from starVLA.dataloader.vla.dataset.base_vla_dataset import (
    StepIndexedVLADatasetMixin,
    build_static_keep_steps,
    select_static_filter_signal,
    split_episode_entries,
)

import decord


def _ensure_2d(arr):
    """Ensure array is 2D (N, D). Scalars per row become (N, 1)."""
    arr = np.stack(arr) if not isinstance(arr, np.ndarray) else arr
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _concat_columns(dataframe, keys):
    """Concatenate multiple parquet columns into a single 2D array."""
    if len(keys) == 1:
        return _ensure_2d(np.stack(dataframe[keys[0]].to_numpy()))
    return np.concatenate(
        [_ensure_2d(np.stack(dataframe[col].to_numpy())) for col in keys], axis=-1
    )


def _select_deterministic_task(episode_data):
    instruction = episode_data.get("instruction")
    if isinstance(instruction, str) and instruction:
        return instruction

    for task in episode_data.get("tasks", []):
        if isinstance(task, str) and task:
            return task
        if task:
            return str(task)
    return ""


def _process_multi_lerobot_episode(args):
    """Process one episode for statistics computation."""
    idx, episode_entry, action_horizon, data_config, state_keys, action_keys, keep_steps = args

    parquet_path = episode_entry[1]
    total_frames = episode_entry[6]

    dataframe = pd.read_parquet(parquet_path)
    total_frames = min(total_frames, len(dataframe))

    state_data = None
    if not data_config.disable_state:
        state_data = _concat_columns(dataframe, state_keys)
    action_data = _concat_columns(dataframe, action_keys)
    if keep_steps is not None:
        keep_steps = np.asarray(keep_steps, dtype=np.int64)
        if state_data is not None:
            state_data = state_data[keep_steps]
        action_data = action_data[keep_steps]
        total_frames = keep_steps.size

    data = {
        "actions": action_data,
    }
    if state_data is not None:
        data["state"] = state_data

    states = []
    actions = []
    for step_idx in range(total_frames):
        data_transformed = data_config.input_transform_dataloader(data, step_idx, action_horizon, inplace=False)
        if 'state' in data_transformed and data_transformed['state'] is not None:
            states.append(data_transformed['state'])
        actions.append(data_transformed['actions'])

    return states, actions


def _is_lerobot_dataset(path):
    """Check if a directory is a lerobot dataset (contains meta/info.json)."""
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def _find_subdatasets(data_root):
    """
    Recursively find all subdirectories containing meta/info.json.
    Once a directory is confirmed as a lerobot dataset, it is NOT descended
    into further (avoids traversing .git, data/chunk-*, videos/chunk-*, etc.).
    Returns sorted list of paths to subdataset roots.
    """
    subdatasets = []
    visited = set()
    to_visit = [os.path.abspath(data_root)]

    while to_visit:
        current = to_visit.pop()

        # Resolve to real path to detect symlink loops
        try:
            real_path = os.path.realpath(current)
        except OSError:
            continue
        if real_path in visited:
            continue
        visited.add(real_path)

        if _is_lerobot_dataset(current):
            # Found a dataset — record it and do NOT descend further
            subdatasets.append(current)
            continue

        # Not a dataset — descend into child directories
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if not entry.is_dir(follow_symlinks=True):
                        continue
                    # Skip hidden directories (.git, etc.) for efficiency
                    if entry.name.startswith("."):
                        continue
                    to_visit.append(entry.path)
        except OSError:
            continue

    return sorted(subdatasets)


class MultiLeRobotV2VLADataset(StepIndexedVLADatasetMixin, Dataset):
    """
    Multi-LeRobot VLA Dataset.

    Auto-discovers sub-datasets with meta/info.json under the root directory.
    Each sub-dataset is treated as a standard lerobot v2.x dataset.

    Per-episode stores:
        [video_path_template, parquet_path, prompt, video_keys, state_keys, action_keys, total_frames, robot_type, fps]
    """

    def __init__(
        self,
        config,
        data_root: str,
        data_name: str,
        data_config: BaseDataConfig,
        data_type: str = "multi_lerobot",
        video_keys: list = None,
        mode: str = "train",
        **kwargs,
    ):
        zero_rank_print(f"Loading dataset {data_root} (MultiLeRobot VLA)...")
        self.config = config
        self.data_root = data_root
        self.data_name = data_name
        self.data_config = data_config
        self.data_config.disable_state = getattr(config, 'disable_state', True)
        self.data_type = data_type
        self.mode = mode
        self.filter_static_steps = bool(getattr(config, 'filter_static_steps', False)) and self.mode == "train"
        self.static_filter_key = str(getattr(config, 'static_filter_key', 'action'))
        self.static_filter_eps = float(getattr(config, 'static_filter_eps', 1e-4))
        self.video_keys = video_keys if video_keys is not None else data_config.video_keys
        self.text_embedding_cache_dir = getattr(config, 'text_embedding_cache_dir', None)
        self.text_context_len = int(getattr(config, 'text_context_len', DEFAULT_CONTEXT_LEN))
        self.text_cache_encoder_id = str(getattr(config, 'text_cache_encoder_id', DEFAULT_ENCODER_ID))
        self.require_text_embedding_cache = bool(getattr(config, 'require_text_embedding_cache', False))
        self.text_context_prompt_template = getattr(config, 'text_context_prompt_template', None)
        self.tactile_key_substring = str(getattr(config, 'tactile_key_substring', 'tactile'))
        default_tactile_residual = any(
            self.tactile_key_substring in str(key)
            for key in (video_keys if video_keys is not None else data_config.video_keys)
        )
        self.enable_tactile_residual = bool(getattr(config, 'enable_tactile_residual', default_tactile_residual))
        self.tactile_residual_mode = str(getattr(config, 'tactile_residual_mode', 'signed'))
        self.tactile_stress_gain = float(getattr(config, 'tactile_stress_gain', tactile_stress.DEFAULT_GAIN))
        self.tactile_stress_baseline_frames = int(
            getattr(config, 'tactile_stress_baseline_frames', tactile_stress.DEFAULT_BASELINE_FRAMES)
        )
        self._tactile_baseline_cache = {}

        self.data_hash_name = (
            data_name + "_" + hashlib.sha1(data_root.encode("utf-8")).hexdigest()[:6]
        )
        statistics_cache_key = getattr(data_config, "statistics_cache_key", data_type)
        split_strategy = getattr(config, "split_strategy", "none")
        if split_strategy == "episode_ratio":
            statistics_cache_key = (
                f"{statistics_cache_key}_split_{split_strategy}"
                f"_eval_{float(getattr(config, 'eval_ratio', 0.0)):g}"
                f"_seed_{getattr(config, 'split_seed', 42)}"
            )
        if bool(getattr(config, 'filter_static_steps', False)):
            statistics_cache_key = f"{statistics_cache_key}_static_{self.static_filter_key}_eps_{self.static_filter_eps:g}"
        self.statistics_hash_name = (
            self.data_hash_name
            + "_"
            + hashlib.sha1(statistics_cache_key.encode("utf-8")).hexdigest()[:6]
        )

        dataset_info_cache_path = os.path.join(
            "playground/meta_data",
            self.data_hash_name + "_vla_dataset_info_cache.json",
        )

        dataset_info_cache_version = 3
        self.dataset = None
        if os.path.exists(dataset_info_cache_path):
            with open(dataset_info_cache_path, "r") as f:
                cached = json.load(f)
            expected_episode_len = 9
            if cached.get("cache_version") == dataset_info_cache_version and cached.get("data") and all(
                len(episode) == expected_episode_len for episode in cached["data"]
            ):
                zero_rank_print(
                    f"Load Cache Dataset Information from {dataset_info_cache_path}"
                )
                self.dataset = cached
                for episode in self.dataset["data"]:
                    episode[3] = list(self.video_keys)
            else:
                zero_rank_print("Cache format outdated, rebuilding...")

        if self.dataset is None:
            self.dataset = {"cache_version": dataset_info_cache_version, "data": []}

            subdataset_roots = _find_subdatasets(data_root)
            zero_rank_print(
                f"Found {len(subdataset_roots)} sub-datasets in {data_root}"
            )

            episode_data_list = []
            for sub_root in tqdm(
                subdataset_roots, desc="Scanning sub-datasets"
            ):
                try:
                    meta_folder = os.path.join(sub_root, "meta")
                    with open(
                        os.path.join(meta_folder, "info.json"), "r"
                    ) as f:
                        metainfo = json.load(f)

                    chunks_size = metainfo["chunks_size"]
                    features = metainfo["features"]
                    robot_type = metainfo.get("robot_type", os.path.relpath(sub_root, data_root))

                    video_keys = list(self.video_keys)
                    if not all(k in features and features[k]["dtype"] == "video" for k in video_keys):
                        raise ValueError(
                            f"Some video keys {video_keys} not in features {features.keys()}"
                        )

                    # Discover state and action keys from features
                    # Try standard lerobot naming first
                    if "observation.state" in features:
                        state_keys = ["observation.state"]
                    else:
                        # Try RoboMIND-style naming
                        state_keys = sorted(
                            [
                                k
                                for k in features
                                if k.startswith("observation.state")
                                and features[k]["dtype"] != "video"
                                and features[k].get("shape") is not None
                            ]
                        )
                    if not state_keys:
                        state_keys = ["observation.state"]  # fallback

                    if "action" in features:
                        action_keys = ["action"]
                    else:
                        action_keys = sorted(
                            [
                                k
                                for k in features
                                if k.startswith("action")
                                and features[k]["dtype"] != "video"
                                and features[k].get("shape") is not None
                            ]
                        )
                    if not action_keys:
                        action_keys = ["action"]  # fallback

                    # Path templates
                    data_path_template = metainfo.get(
                        "data_path",
                        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    )
                    video_path_template = metainfo.get(
                        "video_path",
                        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                    )

                    tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
                    task_str_dict = {}
                    if os.path.exists(tasks_jsonl):
                        task_index_task_str = load_jsonl(tasks_jsonl)
                        for item in task_index_task_str:
                            task_str_dict[item["task_index"]] = item["task"]

                    episodes_jsonl = os.path.join(
                        meta_folder, "episodes.jsonl"
                    )
                    if not os.path.exists(episodes_jsonl):
                        continue
                    episodes_data = load_jsonl(episodes_jsonl)

                    episode_data_list.append(
                        (
                            episodes_data,
                            sub_root,
                            chunks_size,
                            video_keys,
                            state_keys,
                            action_keys,
                            data_path_template,
                            video_path_template,
                            task_str_dict,
                            robot_type,
                        )
                    )
                except Exception as e:
                    zero_rank_print(
                        f"Warning: Skipping sub-dataset {sub_root}: {e}"
                    )
                    continue

            # Flatten all episodes across all sub-datasets
            all_episode_args = []
            for (
                episodes_data,
                sub_root,
                chunks_size,
                video_keys,
                state_keys,
                action_keys,
                data_path_template,
                video_path_template,
                task_str_dict,
                robot_type,
            ) in episode_data_list:
                for episode_data in episodes_data:
                    all_episode_args.append((
                        episode_data,
                        sub_root,
                        chunks_size,
                        video_keys,
                        state_keys,
                        action_keys,
                        data_path_template,
                        video_path_template,
                        task_str_dict,
                        robot_type,
                    ))

            def _process_episode(args):
                (
                    episode_data,
                    sub_root,
                    chunks_size,
                    video_keys,
                    state_keys,
                    action_keys,
                    data_path_template,
                    video_path_template,
                    task_str_dict,
                    robot_type,
                ) = args
                try:
                    episode_index = episode_data["episode_index"]
                    meta_length = int(episode_data["length"])
                    episode_chunk = int(episode_index // chunks_size)

                    video_path = os.path.join(
                        sub_root,
                        video_path_template.format(
                            episode_chunk=episode_chunk,
                            episode_index=episode_index,
                            video_key="{}",
                        ),
                    )
                    parquet_path = os.path.join(
                        sub_root,
                        data_path_template.format(
                            episode_chunk=episode_chunk,
                            episode_index=episode_index,
                        ),
                    )

                    if not os.path.exists(parquet_path):
                        return None
                    data = pd.read_parquet(parquet_path)
                    tasks = episode_data.get("tasks", [])
                    if "task_index" in data:
                        task_index = data["task_index"].iloc[0]
                        task = task_str_dict.get(task_index, tasks[0] if tasks else "")
                    else:
                        task = _select_deterministic_task(episode_data)
                    if not task:
                        return None

                    video_frame_counts = [
                        len(decord.VideoReader(video_path.format(video_key), num_threads=1))
                        for video_key in video_keys
                    ]
                    if any(frame_count != meta_length for frame_count in video_frame_counts):
                        return None

                    # Validate first video exists and get fps
                    vr = decord.VideoReader(
                        video_path.format(video_keys[0]), num_threads=1
                    )
                    fps = vr.get_avg_fps()

                    return [
                        video_path,
                        parquet_path,
                        task,
                        video_keys,
                        state_keys,
                        action_keys,
                        len(data),
                        robot_type,
                        fps,
                    ]
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=32) as ex:
                for info in tqdm(
                    ex.map(_process_episode, all_episode_args),
                    total=len(all_episode_args),
                    desc="Processing episodes",
                ):
                    if info is not None:
                        self.dataset["data"].append(info)

            zero_rank_print(
                f"Save Cache Dataset Information to {dataset_info_cache_path}"
            )
            os.makedirs(os.path.dirname(dataset_info_cache_path), exist_ok=True)
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f, indent=2)

        self.dataset["data"] = split_episode_entries(
            self.dataset["data"],
            config=self.config,
            data_name=self.data_name,
            mode=self.mode,
        )
        self.total_episodes = len(self.dataset["data"])
        if self.total_episodes == 0:
            raise ValueError(f"[{self.data_name}] No VLA episodes remain after applying mode={self.mode!r} split.")
        episode_lengths = [sample[6] for sample in self.dataset["data"]]
        self.episode_keep_steps = self._build_episode_keep_steps() if self.filter_static_steps else None
        self.initialize_step_indexing(
            episode_lengths,
            mode=self.mode,
            episode_step_indices=self.episode_keep_steps,
        )

        zero_rank_print(
            f"[MultiLeRobot] Total episodes: {self.total_episodes}, Total steps: {self.total_steps}"
        )

        self.compute_statistics()

        self.image_size = config.image_size

        # Get video properties from first episode (for resize shape only)
        example_data = self.dataset["data"][0]
        example_video_keys = example_data[3]
        video_reader = decord.VideoReader(
            example_data[0].format(example_video_keys[0])
        )
        h, w = video_reader[0].shape[:2]
        scale = min(self.image_size) / min(h, w)
        self.decord_resize_shape = (int(scale * h), int(scale * w))

        self.jitter_transform = transforms.ColorJitter(
            brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1
        )

        # Set image size for data_config
        self.data_config.set_image_size(self.image_size)
        self.data_config.state_pad_size = self.config.state_pad_size
        self.data_config.action_pad_size = self.config.action_pad_size
        self.use_future_frames = self.config.use_future_frames
        self.num_future_frames = self.config.num_future_frames
        self.future_frame_stride = max(1, int(getattr(config, 'future_frame_stride', 1)))
        self.use_decord = getattr(config, 'use_decord', True)

    @property
    def data_key(self):
        return f"{self.data_type}"

    def _build_episode_keep_steps(self):
        zero_rank_print(
            f"Building static-step filter from {self.static_filter_key} "
            f"with static_filter_eps={self.static_filter_eps:g}..."
        )
        keep_steps_by_episode = [None] * self.total_episodes

        def _process(idx):
            episode_data = self.dataset["data"][idx]
            parquet_path = episode_data[1]
            state_keys = episode_data[4]
            action_keys = episode_data[5]
            total_frames = int(episode_data[6])
            dataframe = pd.read_parquet(parquet_path)
            total_frames_local = min(total_frames, len(dataframe))
            state_data = None
            if not self.data_config.disable_state or self.static_filter_key == "state":
                state_data = _concat_columns(dataframe.iloc[:total_frames_local], state_keys)
            action_data = _concat_columns(dataframe.iloc[:total_frames_local], action_keys)
            signal = select_static_filter_signal(
                self.data_config,
                state=state_data,
                action=action_data,
                filter_key=self.static_filter_key,
            )
            return idx, build_static_keep_steps(signal, self.static_filter_eps)

        num_workers = min(self.total_episodes, min(32, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process, idx) for idx in range(self.total_episodes)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Static-step filter"):
                idx, keep_steps = fut.result()
                keep_steps_by_episode[idx] = keep_steps

        raw_steps = sum(int(sample[6]) for sample in self.dataset["data"])
        kept_steps = sum(int(steps.size) for steps in keep_steps_by_episode)
        zero_rank_print(
            f"Static-step filter kept {kept_steps}/{raw_steps} steps "
            f"({kept_steps / max(raw_steps, 1):.4f})."
        )
        return keep_steps_by_episode

    def compute_statistics(self):
        statistic_path = os.path.join(
            "playground/meta_data",
            self.statistics_hash_name + "_vla_dataset_statistic.json",
        )
        if os.path.exists(statistic_path):
            zero_rank_print(f"Load Dataset Statistics from {statistic_path}")
            with open(statistic_path, "r") as f:
                self.dataset_statistics = json.load(f)
            if self.data_config.disable_state or "state" in self.dataset_statistics:
                return
            zero_rank_print("State statistics missing from cache, recomputing...")

        zero_rank_print("Computing dataset statistics...")

        num_workers = min(32, os.cpu_count() or 1)
        state_stats = None
        action_stats = None
        args_iter = [
            (
                idx,
                self.dataset["data"][idx],
                self.config.action_horizon,
                self.data_config,
                self.dataset["data"][idx][4],  # state_keys
                self.dataset["data"][idx][5],  # action_keys
                None if self.episode_keep_steps is None else self.episode_keep_steps[idx],
            )
            for idx in range(len(self.dataset["data"]))
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(_process_multi_lerobot_episode, args)
                for args in args_iter
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Stats",
            ):
                try:
                    states, actions = fut.result()
                except Exception:
                    traceback.print_exc()
                    continue

                if not actions:
                    continue

                action_dim = actions[0].shape[1]
                if action_stats is None:
                    action_stats = OnlineStats(action_dim)
                elif action_dim != action_stats.mean.shape[-1]:
                    raise ValueError(
                        "MultiLeRobot single-statistics mode requires consistent "
                        f"action dimensions, got {action_dim} and {action_stats.mean.shape[-1]}."
                    )
                action_stats.update_batch(np.concatenate(actions, axis=0))

                if len(states) > 0 and states[0] is not None:
                    state_dim = states[0].shape[-1]
                    if state_stats is None:
                        state_stats = OnlineStats(state_dim)
                    elif state_dim != state_stats.mean.shape[-1]:
                        raise ValueError(
                            "MultiLeRobot single-statistics mode requires consistent "
                            f"state dimensions, got {state_dim} and {state_stats.mean.shape[-1]}."
                        )
                    state_stats.update_batch(np.concatenate(states, axis=0))

        if action_stats is None:
            raise RuntimeError(
                "All episodes failed during statistics computation. "
                "Check dataset integrity and error logs above."
            )

        action_res = action_stats.finalize()
        self.dataset_statistics = {
            "action": {
                "mean": action_res["mean"].tolist(),
                "std": action_res["std"].tolist(),
                "min": action_res["min"].tolist(),
                "max": action_res["max"].tolist(),
                "q01": None,
                "q99": None,
            },
        }
        if state_stats is not None:
            state_res = state_stats.finalize()
            self.dataset_statistics["state"] = {
                "mean": state_res["mean"].tolist(),
                "std": state_res["std"].tolist(),
                "min": state_res["min"].tolist(),
                "max": state_res["max"].tolist(),
                "q01": None,
                "q99": None,
            }
        zero_rank_print(
            f"  [all] {int(action_stats.count)} samples, action_dim={action_res['mean'].shape[0]}"
        )

        with open(statistic_path, "w") as f:
            json.dump(self.dataset_statistics, f, indent=2)
        zero_rank_print(f"Saved dataset statistics to {statistic_path}")

    def save_dataset_statistics(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dataset_statistics = {self.data_name: self.dataset_statistics}
        with open(save_path, "w") as f:
            json.dump(dataset_statistics, f, indent=2)

    def _decode_frames(self, video_path, indexes, indexes_s):
        """Read video frames. Returns (N, C, H, W) tensor."""
        if self.use_decord:
            h, w = self.decord_resize_shape
            vr = decord.VideoReader(video_path, height=h, width=w, num_threads=1)
            batch = vr.get_batch(indexes.tolist())
            if not isinstance(batch, torch.Tensor):
                batch = torch.from_numpy(batch.asnumpy())
            batch = batch.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            return batch
        else:
            return decode_video_frames(video_path, indexes_s, tolerance_s=1e-4)

    def _is_tactile_video_key(self, video_key):
        return self.enable_tactile_residual and self.tactile_key_substring in str(video_key)

    def _maybe_apply_tactile_residual(self, video_key, video_path, frames, fps):
        if not self._is_tactile_video_key(video_key):
            return frames, False

        if self.tactile_residual_mode == "stress":
            baseline, noise_floor = tactile_stress.video_stress_baseline(
                self._decode_frames,
                video_path,
                self.tactile_stress_baseline_frames,
                self._tactile_baseline_cache,
                fps,
            )
            return tactile_stress.stress_view(frames, baseline, noise_floor, self.tactile_stress_gain), True

        reference = self._decode_frames(
            video_path,
            np.array([0], dtype=np.int64),
            [0.0],
        )[0].float()
        residual = (frames.float() - reference.unsqueeze(0)) / 255.0
        return residual.clamp(-1.0, 1.0), True

    def _get_image_indexes(self, episode_idx, step_idx, total_frames):
        if self.episode_step_indices is None:
            if not self.use_future_frames:
                return np.array([step_idx], dtype=np.int64)
            indexes = np.array(
                [step_idx + i * self.future_frame_stride for i in range(self.num_future_frames + 1)],
                dtype=np.int64,
            )
            return np.clip(indexes, a_min=0, a_max=total_frames - 1)

        keep_steps = self.episode_step_indices[int(episode_idx)]
        compact_step_idx = self.get_compact_step_index(episode_idx, step_idx)
        if not self.use_future_frames:
            compact_indexes = np.array([compact_step_idx], dtype=np.int64)
        else:
            compact_indexes = np.array(
                [compact_step_idx + i * self.future_frame_stride for i in range(self.num_future_frames + 1)],
                dtype=np.int64,
            )
            compact_indexes = np.clip(compact_indexes, a_min=0, a_max=keep_steps.size - 1)
        return keep_steps[compact_indexes]

    def _get_selected_video_keys(self, video_keys):
        return list(video_keys)

    def get_sample_at_raw_step(self, episode_idx, step_idx):
        episode_data = self.dataset["data"][episode_idx]
        video_path, parquet_path, prompt, video_keys, state_keys, action_keys, total_frames, robot_type, fps = (
            episode_data
        )
        stats = self.dataset_statistics

        # Load parquet first to get actual frame count
        dataframe = pd.read_parquet(parquet_path)
        actual_frames = len(dataframe)
        total_frames = min(total_frames, actual_frames)
        if step_idx < 0 or step_idx >= total_frames:
            raise IndexError(
                f"step_idx={step_idx} out of range for episode_idx={episode_idx} with total_frames={total_frames}"
            )

        indexes = self._get_image_indexes(episode_idx, step_idx, total_frames)
        indexes_s = (indexes / fps).tolist()

        chosen_keys = self._get_selected_video_keys(video_keys)

        images = []
        view_masks = []
        view_is_tactile = []
        for vk in chosen_keys:
            video_file = video_path.format(vk)
            try:
                frame = self._decode_frames(
                    video_file, indexes, indexes_s,
                )
            except Exception as e:
                print(video_file)
                raise e
            frame, is_tactile = self._maybe_apply_tactile_residual(
                video_key=vk,
                video_path=video_file,
                frames=frame,
                fps=fps,
            )
            if not self.use_future_frames:
                frame = frame[0]
            else:
                frame = frame.permute(1, 0, 2, 3)
            images.append(frame)
            view_masks.append(True)
            view_is_tactile.append(is_tactile)

        state_data = None
        if not self.data_config.disable_state:
            state_data = _concat_columns(dataframe, state_keys)
        action_data = _concat_columns(dataframe, action_keys)
        transform_step_idx = step_idx
        if self.episode_step_indices is not None:
            keep_steps = self.episode_step_indices[episode_idx]
            transform_step_idx = self.get_compact_step_index(episode_idx, step_idx)
            if state_data is not None:
                state_data = state_data[keep_steps]
            action_data = action_data[keep_steps]

        input_data = {
            "image": images,
            "view_mask": view_masks,
            "lang": prompt,
            "fps": fps,
            "actions": action_data,
            "image_is_tactile": torch.tensor(view_is_tactile, dtype=torch.bool),
            "image_video_keys": chosen_keys,
        }
        if state_data is not None:
            input_data["state"] = state_data
        data_unnormalized = self.data_config.input_transform_dataloader(input_data, transform_step_idx, self.config.action_horizon)
        try:
            data = self.data_config.normalize_data(data_unnormalized, stats)
        except:
            print(video_path)
            raise
        data = self.data_config.pad_data(data)
        if self.text_embedding_cache_dir:
            cached_text = maybe_load_text_embedding_cache(
                self.text_embedding_cache_dir,
                prompt,
                context_len=self.text_context_len,
                encoder_id=self.text_cache_encoder_id,
                required=self.require_text_embedding_cache,
                prompt_template=self.text_context_prompt_template,
            )
            if cached_text is not None:
                context, context_mask = cached_text
                data["context"] = context
                data["context_mask"] = context_mask
        return data
