import os
import random
import traceback
import zipfile
from typing import Sequence, Tuple

import numpy as np

from starVLA.utils.system_utils import zero_rank_print


def build_step_offsets(lengths: Sequence[int]) -> np.ndarray:
    episode_lengths = np.asarray([int(length) for length in lengths], dtype=np.int64)
    if episode_lengths.ndim != 1:
        raise ValueError(f"episode lengths must be 1D, got shape {episode_lengths.shape}")
    if episode_lengths.size == 0:
        return np.zeros(1, dtype=np.int64)
    if np.any(episode_lengths <= 0):
        invalid = episode_lengths[episode_lengths <= 0][:8].tolist()
        raise ValueError(f"episode lengths must all be positive, got examples {invalid}")

    offsets = np.zeros(episode_lengths.shape[0] + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(episode_lengths)
    return offsets


def resolve_step_index(step_offsets: np.ndarray, index: int) -> Tuple[int, int]:
    """
    Map a flattened global step index to (episode_idx, step_idx) using cumulative episode offsets.
    """
    if step_offsets.ndim != 1 or step_offsets.size == 0:
        raise ValueError("step_offsets must be a non-empty 1D array")
    total_steps = int(step_offsets[-1])
    if index < 0 or index >= total_steps:
        raise IndexError(f"step index {index} out of range for total_steps={total_steps}")
    episode_idx = int(np.searchsorted(step_offsets, index, side="right") - 1)
    step_idx = int(index - step_offsets[episode_idx])
    return episode_idx, step_idx


def is_eval_mode(mode: str) -> bool:
    return str(mode).lower() in {"eval", "val", "validation"}


def split_episode_entries(entries: Sequence, config, data_name: str, mode: str):
    if getattr(config, "split_strategy", "none") != "episode_ratio":
        return list(entries)

    entries = list(entries)
    eval_ratio = float(getattr(config, "eval_ratio", 0.0))
    if eval_ratio <= 0 or len(entries) < 2:
        return entries

    rng = random.Random(f"{getattr(config, 'split_seed', 42)}:{data_name}")
    indices = list(range(len(entries)))
    rng.shuffle(indices)

    n_eval = max(1, int(round(len(entries) * eval_ratio)))
    n_eval = min(n_eval, len(entries) - 1)
    eval_indices = set(indices[:n_eval])

    if is_eval_mode(mode):
        split_entries = [entry for i, entry in enumerate(entries) if i in eval_indices]
    else:
        split_entries = [entry for i, entry in enumerate(entries) if i not in eval_indices]

    split_name = "eval" if is_eval_mode(mode) else "train"
    zero_rank_print(
        f"[{data_name}] episode_ratio split mode={split_name}: "
        f"{len(split_entries)}/{len(entries)} episodes "
        f"(eval_ratio={eval_ratio}, split_seed={getattr(config, 'split_seed', 42)})"
    )
    return split_entries


def build_static_keep_steps(signal: np.ndarray, static_filter_eps: float) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.shape[0] == 0:
        raise ValueError("Cannot build static-step filter for an empty episode.")
    if signal.shape[0] == 1:
        return np.zeros(1, dtype=np.int64)

    signal_flat = signal.reshape(signal.shape[0], -1)
    motion = np.max(np.abs(np.diff(signal_flat, axis=0)), axis=1)
    keep_mask = np.concatenate([np.ones(1, dtype=bool), motion > float(static_filter_eps)])
    return np.flatnonzero(keep_mask).astype(np.int64)

def select_static_filter_signal(
    data_config,
    *,
    state: np.ndarray | None = None,
    action: np.ndarray | None = None,
    filter_key: str = "action",
) -> np.ndarray:
    filter_key = str(filter_key)
    if filter_key == "state":
        if state is None:
            raise KeyError("static_filter_key='state' requires state data.")
        return select_static_filter_state(data_config, state)
    if filter_key != "action":
        raise ValueError(f"Unsupported static_filter_key={filter_key!r}; expected 'action' or 'state'.")
    if action is None:
        raise KeyError("static_filter_key='action' requires action data.")
    action = np.asarray(action)
    if getattr(data_config, "use_all_columns", False):
        return action
    action_ids = getattr(data_config, "action_ids", None)
    if action_ids is None or action.shape[-1] == len(action_ids):
        return action
    return action[..., action_ids]

def select_static_filter_state(data_config, state: np.ndarray) -> np.ndarray:
    if hasattr(data_config, "select_state_columns"):
        return data_config.select_state_columns(state)
    return np.asarray(state)


class StepIndexedVLADatasetMixin:
    """
    Step-indexing 变量定义：
    - episode_step_indices: 每个 episode 中允许采样的 raw step 下标列表（经过 static filter后）；为 None 时使用全部原始 step。
    - episode_step_offsets: 各 episode 在展平后的全局 step 空间中的累计偏移。
    """
    def initialize_step_indexing(
        self,
        episode_lengths: Sequence[int],
        mode: str = "train",
        episode_step_indices: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.mode = mode
        self.deterministic_sampling = is_eval_mode(self.mode)

        self.episode_step_indices = None
        if episode_step_indices is not None:
            if len(episode_step_indices) != len(episode_lengths):
                raise ValueError(
                    f"episode_step_indices length {len(episode_step_indices)} does not match "
                    f"episode_lengths length {len(episode_lengths)}."
                )
            validated = []
            for episode_idx, (steps, episode_length) in enumerate(zip(episode_step_indices, episode_lengths)):
                steps = np.asarray(steps, dtype=np.int64)
                if steps.ndim != 1 or steps.size == 0:
                    raise ValueError(f"episode_step_indices[{episode_idx}] must be a non-empty 1D array.")
                if steps[0] < 0 or steps[-1] >= int(episode_length):
                    raise ValueError(
                        f"episode_step_indices[{episode_idx}] out of range for episode length {episode_length}."
                    )
                if np.any(np.diff(steps) <= 0):
                    raise ValueError(f"episode_step_indices[{episode_idx}] must be strictly increasing.")
                validated.append(steps)
            self.episode_step_indices = validated
            step_lengths = [steps.size for steps in self.episode_step_indices]
        else:
            step_lengths = episode_lengths
        
        self.episode_step_offsets = build_step_offsets(step_lengths)
        self.episode_lengths = self.episode_step_offsets[1:] - self.episode_step_offsets[:-1]
        self.total_steps = int(self.episode_step_offsets[-1])

    def resolve_step_index(self, index: int) -> Tuple[int, int]:
        episode_idx, step_idx = resolve_step_index(self.episode_step_offsets, int(index))
        if self.episode_step_indices is None:
            return episode_idx, step_idx
        return episode_idx, int(self.episode_step_indices[episode_idx][step_idx])

    def get_compact_step_index(self, episode_idx: int, raw_step_idx: int) -> int:
        if self.episode_step_indices is None:
            return int(raw_step_idx)
        steps = self.episode_step_indices[int(episode_idx)]
        compact_idx = int(np.searchsorted(steps, int(raw_step_idx)))
        if compact_idx >= steps.size or int(steps[compact_idx]) != int(raw_step_idx):
            raise ValueError(f"raw_step_idx={raw_step_idx} is not sampleable for episode_idx={episode_idx}.")
        return compact_idx

    def sample_random_step_index(self) -> int:
        if self.total_steps <= 0:
            raise ValueError("Cannot sample from an empty dataset.")
        return random.randrange(self.total_steps)

    def get_sample_by_step_index(self, index: int):
        episode_idx, step_idx = self.resolve_step_index(index)
        return self.get_sample_at_raw_step(episode_idx, step_idx)

    def __len__(self):
        return self.total_steps

    def __getitem__(self, idx):
        # validation 时方便复现结果
        if self.deterministic_sampling:
            return self.get_sample_by_step_index(idx)

        while True:
            try:
                return self.get_sample_by_step_index(self.sample_random_step_index())
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()
