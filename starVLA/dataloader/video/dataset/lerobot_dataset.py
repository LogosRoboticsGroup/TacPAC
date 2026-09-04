import json
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from starVLA.dataloader.video.dataset.base_video_dataset import BaseVideoDataset
from starVLA.utils.data_utils import load_jsonl
from starVLA.utils.system_utils import zero_rank_print

class LeRobotV2VideoDataset(BaseVideoDataset):
    def __init__(
        self,
        config,
        data_name: str,
        data_root: Path,
        data_config,
        data_type: str,
        data_id: int = 0,
        video_keys=None,
        cache_name=None,
        **kwargs,
    ):
        zero_rank_print(f"Loading dataset {data_root} (LeRobot Video)...")
        super().__init__(
            config=config,
            data_name=data_name,
            data_root=data_root,
            data_config=data_config,
            data_type=data_type,
            data_id=data_id,
            video_keys=video_keys,
            cache_name=cache_name,
            **kwargs,
        )

        if not self.load_cache():
            self.build_dataset()
            self.save_cache()

        self.finalize_dataset(
            lambda item: (
                item["clip_end"] - item["clip_start"] + 1
                if "clip_end" in item
                else item["index_end"] - item["index_start"] + 1
            )
        )
        
    def build_dataset(self):
        meta_folder = self.data_root / "meta"
        with open(meta_folder / "info.json", "r") as f:
            info = json.load(f)
            
        features = info["features"]
        discovered_video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
        resolved_video_keys = self.resolve_video_keys(discovered_video_keys)
        if not resolved_video_keys:
            raise ValueError(f"No valid video keys found for dataset {self.data_name}")
            
        if (meta_folder / "episodes.jsonl").exists():
            self._build_v21_dataset(info, resolved_video_keys)
            
    def _load_task_map_v21(self, meta_folder: Path):
        task_map = {}
        tasks_jsonl = meta_folder / "tasks.jsonl"
        if not tasks_jsonl.exists():
            return task_map
        for item in load_jsonl(str(tasks_jsonl)):
            task_map[item["task_index"]] = item["task"]
        return task_map
          
    def _resolve_episode_prompt(self, tasks, episode_data, task_map):
        if tasks:
            task = tasks[np.random.randint(len(tasks))]
            if isinstance(task, str) and task:
                return task
            if isinstance(task, (int, np.integer)):
                return task_map.get(int(task), "")
            if isinstance(task, dict):
                if task.get("task"):
                    return str(task["task"])
                if task.get("task_index") is not None:
                    return task_map.get(int(task["task_index"]), "")
        instruction = episode_data.get("instruction")
        if instruction:
            return str(instruction)
        return ""
            
    def _build_v21_dataset(self, info, resolved_video_keys):
        meta_folder = self.data_root / "meta"
        task_map = self._load_task_map_v21(meta_folder)
        episodes_data = load_jsonl(str(meta_folder / "episodes.jsonl"))
        chunks_size = info["chunks_size"]
        video_path_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

        cached_video_metadata = None

        for episode_data in tqdm(episodes_data, desc=f"Scanning {self.data_name}"):
            prompt = self._resolve_episode_prompt(episode_data.get("tasks", []), episode_data, task_map)
            if not prompt:
                continue

            episode_index = int(episode_data["episode_index"])
            episode_chunk = int(episode_index // chunks_size)
            video_template = str(
                self.data_root
                / video_path_template.format(
                    episode_chunk=episode_chunk,
                    chunk_index=episode_chunk,
                    episode_index=episode_index,
                    file_index=episode_index,
                    video_key="{video_key}",
                )
            )
            cached_video_metadata = self.resolve_episode_video_metadata(
                info=info,
                video_keys=resolved_video_keys,
                video_template=video_template,
                cached_video_metadata=cached_video_metadata,
            )
            if cached_video_metadata is None:
                continue

            height, width, fps = cached_video_metadata
            length = int(episode_data["length"])
            if self.validate_cache_decode_per_episode:
                try:
                    self.validate_multi_video_decode(
                        [video_template.format(video_key=video_key) for video_key in resolved_video_keys],
                        clip_end=max(0, length - 1),
                        num_threads=1,
                    )
                except Exception as exc:
                    self.record_cache_validation_failure(
                        f"{self.data_root.name}/episode_{episode_index:06d}",
                        exc,
                    )
                    continue
            self.dataset["data"].append(
                {
                    "video_path": video_template,
                    "prompt": prompt,
                    "video_keys": list(resolved_video_keys),
                    "fps": float(fps),
                    "height": int(height),
                    "width": int(width),
                    "clip_start": 0,
                    "clip_end": max(0, length - 1),
                }
            )
            
    def get_batch(self, idx):
        entry = self.dataset["data"][idx % self.total_episodes]
        indices = self.sample_frame_indices(
            total_frames=entry["clip_end"] - entry["clip_start"] + 1,
            fps=entry["fps"],
            clip_start=entry["clip_start"],
            clip_end=entry["clip_end"],
        )
        image, view_mask = self.decode_multi_view(
            entry["video_keys"],
            lambda video_key: entry["video_path"].format(video_key=video_key),
            indices,
            source_height=entry["height"],
            source_width=entry["width"],
            num_threads=1,
        )
        return self.build_sample(image=image, view_mask=view_mask, prompt=entry["prompt"])
