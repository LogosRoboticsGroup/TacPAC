import random

import numpy as np
from torch.utils.data.dataset import Dataset

from starVLA.utils.system_utils import zero_rank_print


class MixVideoDataset(Dataset):
    def __init__(
        self,
        data_mixture,
        mode,
        balance_dataset_weights: bool = True,
        sampling_temperature: float = 2.0,
    ):
        datasets = []
        dataset_sampling_weights = []
        for dataset, weight in data_mixture:
            if len(dataset) == 0:
                zero_rank_print(f"Warning: Skipping empty dataset {dataset.data_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if not datasets:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.sampling_temperature = sampling_temperature
        self.mode = mode

        self._dataset_episodes = np.array([dataset.total_episodes for dataset in self.datasets])
        self._dataset_steps = np.array([dataset.total_steps for dataset in self.datasets])
        self.total_episodes = int(self._dataset_episodes.sum())
        self.total_steps = int(self._dataset_steps.sum())
        zero_rank_print(f"Dataset episodes: {self._dataset_episodes}")
        zero_rank_print(f"Dataset steps: {self._dataset_steps}")
        zero_rank_print(f"Total mixed dataset episodes: {self.total_episodes}")
        zero_rank_print(f"Total mixed dataset steps: {self.total_steps}")

        self._dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float64)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_steps ** (1.0 / self.sampling_temperature)

        if np.any(self._dataset_sampling_weights <= 0):
            raise ValueError("Dataset sampling weights must be positive after balancing.")

        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        zero_rank_print(f"Sampling temperature: {self.sampling_temperature}")
        for i, dataset in enumerate(self.datasets):
            zero_rank_print(
                f"  [{dataset.data_name}] steps={self._dataset_steps[i]}, "
                f"sampling_prob={self._dataset_sampling_weights[i]:.4f}"
            )

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, index):
        dataset = random.choices(self.datasets, weights=self._dataset_sampling_weights)[0]
        episode_index = random.randint(0, dataset.total_episodes - 1)
        return dataset[episode_index]
