
import os
import traceback
import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from starVLA.utils.system_utils import zero_rank_print
from starVLA.utils.data_utils import load_jsonl

import decord

from starVLA.dataloader.vla.dataset.base_vla_dataset import build_step_offsets, is_eval_mode, resolve_step_index


class MixDataset(Dataset):
    def __init__(
        self,
        data_mixture,
        mode,
        balance_dataset_weights: bool = True,
        sampling_temperature: float = 2.0,
        stats_merge_strategy: str = "average",
        stats_references: dict | None = None,
    ):
        datasets = []
        dataset_sampling_weights = []
        for dataset, weight in data_mixture:
            if len(dataset) == 0:
                zero_rank_print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.sampling_temperature = sampling_temperature
        self.mode = str(mode)
        if self.mode not in {"train", "eval", "val", "validation"}:
            raise ValueError(
                f"Unsupported VLA mix dataset mode={self.mode!r}; expected 'train', 'eval', or 'val'."
            )
        self.deterministic_sampling = is_eval_mode(self.mode)

        self._dataset_episodes = np.array([dataset.total_episodes for dataset in self.datasets])
        self._dataset_steps = np.array([dataset.total_steps for dataset in self.datasets])
        self.total_episodes = self._dataset_episodes.sum()
        self.total_steps = self._dataset_steps.sum()
        self._dataset_step_offsets = build_step_offsets(self._dataset_steps.tolist())
        zero_rank_print(f"Dataset episodes: {self._dataset_episodes}")
        zero_rank_print(f"Dataset steps: {self._dataset_steps}")
        zero_rank_print(f"Total mixed dataset episodes: {self.total_episodes}")
        zero_rank_print(f"Total mixed dataset steps: {self.total_steps}")

        self._dataset_sampling_weights = np.array(dataset_sampling_weights)

        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_steps ** (1.0 / self.sampling_temperature)
            
        if np.any(self._dataset_sampling_weights <= 0):
            assert False, "Dataset sampling weights must be positive after balancing."
            
        weights_sum = self._dataset_sampling_weights.sum()
        self._dataset_sampling_weights /= weights_sum

        zero_rank_print(f"Sampling temperature: {self.sampling_temperature}")
        for i, dataset in enumerate(self.datasets):
            zero_rank_print(
                f"  [{dataset.data_name}] steps={self._dataset_steps[i]}, "
                f"sampling_prob={self._dataset_sampling_weights[i]:.4f}"
            )
        
        if stats_merge_strategy not in {"average", "reference_by_data_key"}:
            raise ValueError(
                f"Unsupported stats_merge_strategy={stats_merge_strategy!r}; "
                "expected 'average' or 'reference_by_data_key'."
            )
        
        # Merge dataset statistics: average stats that share the same merged_key,
        # then write back to sub-datasets so all datasets with the same key use unified statistics.
        self.merged_dataset_statistics = {}
        self._nested_data_keys = set()  # Track which data_keys are nested
        _stats_collection = {}  # merged_key -> list of stats dicts to average
        datasets_for_statistics = self.datasets

        if stats_merge_strategy == "reference_by_data_key":
            stats_references = stats_references or {}
            datasets_for_statistics = []
            for data_key in sorted({dataset.data_key for dataset in self.datasets}):
                reference_name = stats_references.get(data_key)
                if reference_name is None:
                    raise ValueError(
                        f"stats_merge_strategy='reference_by_data_key' requires "
                        f"stats_references[{data_key!r}]."
                    )
                matches = [
                    dataset for dataset in self.datasets
                    if dataset.data_key == data_key and dataset.data_name == reference_name
                ]
                if len(matches) != 1:
                    names = [
                        dataset.data_name for dataset in self.datasets
                        if dataset.data_key == data_key
                    ]
                    raise ValueError(
                        f"stats_references[{data_key!r}]={reference_name!r} must match exactly "
                        f"one dataset in {names}."
                    )
                datasets_for_statistics.append(matches[0])
                zero_rank_print(f"Using statistics from [{reference_name}] for data_key={data_key}")

        for dataset in datasets_for_statistics:
            stats = dataset.dataset_statistics
            is_nested = isinstance(stats, dict) and "action" not in stats and "state" not in stats
            if is_nested:
                self._nested_data_keys.add(dataset.data_key)
                for key, value in stats.items():
                    merged_key = f"{dataset.data_key}_{key}"
                    _stats_collection.setdefault(merged_key, []).append(value)
            else:
                _stats_collection.setdefault(dataset.data_key, []).append(stats)

        for merged_key, stats_list in _stats_collection.items():
            if len(stats_list) == 1:
                self.merged_dataset_statistics[merged_key] = stats_list[0]
            else:
                self.merged_dataset_statistics[merged_key] = self._average_stats(stats_list)
                zero_rank_print(f"Averaged {len(stats_list)} statistics for merged key '{merged_key}'")

        # Write merged statistics back to sub-datasets for unified normalization
        for dataset in self.datasets:
            if dataset.data_key in self._nested_data_keys:
                new_stats = {}
                for key in dataset.dataset_statistics:
                    merged_key = f"{dataset.data_key}_{key}"
                    new_stats[key] = self.merged_dataset_statistics[merged_key]
                dataset.dataset_statistics = new_stats
            else:
                dataset.dataset_statistics = self.merged_dataset_statistics[dataset.data_key]

    @staticmethod
    def _average_stats(stats_list):
        """Average a list of statistics dicts/values element-wise."""
        if not stats_list:
            return None
        if len(stats_list) == 1:
            return stats_list[0]
        first = stats_list[0]
        if isinstance(first, dict):
            result = {}
            for key in first:
                values = [s[key] for s in stats_list if key in s]
                result[key] = MixDataset._average_stats(values)
            return result
        elif isinstance(first, list):
            n = len(stats_list)
            return [sum(s[i] for s in stats_list) / n for i in range(len(first))]
        elif first is None:
            return None
        else:
            return sum(stats_list) / len(stats_list)
    
    def save_dataset_statistics(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.merged_dataset_statistics, f, indent=2)
        zero_rank_print(f"Saved mixed dataset statistics to {save_path}")
            
    def __len__(self):
        return self.total_steps

    def _resolve_dataset_step_index(self, index):
        dataset_idx, local_step_idx = resolve_step_index(self._dataset_step_offsets, int(index))
        return self.datasets[dataset_idx], local_step_idx
    
    def __getitem__(self, index):
        if self.deterministic_sampling:
            dataset, step_index = self._resolve_dataset_step_index(index)
            return dataset.get_sample_by_step_index(step_index)

        dataset = random.choices(self.datasets, weights=self._dataset_sampling_weights)[0]
        step_index = random.randrange(dataset.total_steps)
        return dataset.get_sample_by_step_index(step_index)
