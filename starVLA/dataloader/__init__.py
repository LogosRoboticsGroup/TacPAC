import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
import gc

def _gc_disable_worker_init(worker_id):
    gc.disable()

logger = get_logger(__name__)


def _build_loader_kwargs(num_workers, *, prefetch_factor=None, worker_init_fn=None):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if worker_init_fn is not None:
        kwargs["worker_init_fn"] = worker_init_fn
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe", mode="train"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "vla_datasets":
        from starVLA.dataloader.vla_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            mode=mode,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
        )
        vla_batch_size = (
            getattr(vla_dataset_cfg, "eval_per_device_batch_size", vla_dataset_cfg.per_device_batch_size)
            if mode in {"eval", "val", "validation"}
            else vla_dataset_cfg.per_device_batch_size
        )
        dataloader_kwargs = _build_loader_kwargs(
            getattr(cfg.datasets.vla_data, "num_workers", 12),
            prefetch_factor=4,
        )
        if mode == "eval":
            eval_generator = torch.Generator()
            eval_generator.manual_seed(int(getattr(vla_dataset_cfg, "split_seed", 42)))
            dataloader_kwargs["generator"] = eval_generator
        
        vla_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs,
            shuffle=True,
        )        
        if mode == "train" and (not dist.is_initialized() or dist.get_rank() == 0):
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "video_datasets":
        from starVLA.dataloader.video_datasets import get_video_dataset, collate_fn
        video_dataset_cfg = cfg.datasets.video_data
        
        video_dataset = get_video_dataset(data_cfg=video_dataset_cfg, mode=mode, balance_dataset_weights=True)
        video_batch_size = (
            getattr(video_dataset_cfg, "eval_per_device_batch_size", video_dataset_cfg.per_device_batch_size)
            if mode == "eval"
            else video_dataset_cfg.per_device_batch_size
        )
        
        video_num_workers = int(getattr(video_dataset_cfg, "num_workers", 8))
        video_pin_memory = bool(getattr(video_dataset_cfg, "pin_memory", True))
        video_persistent_workers = bool(getattr(video_dataset_cfg, "persistent_workers", True))
        video_timeout = float(getattr(video_dataset_cfg, "timeout", 120))
        
        dataloader_kwargs = {
            "collate_fn": collate_fn,
            "batch_size": video_batch_size,
            "num_workers": video_num_workers,
            "pin_memory": video_pin_memory,
            "timeout": video_timeout if video_num_workers > 0 else 0,
        }
        if video_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = video_persistent_workers
            dataloader_kwargs["worker_init_fn"] = _gc_disable_worker_init
            dataloader_kwargs["prefetch_factor"] = int(getattr(video_dataset_cfg, "prefetch_factor", 4))
            
        dataloader_config_message = (
            "Video DataLoader config: "
            f"num_workers={video_num_workers} "
            f"pin_memory={video_pin_memory} "
            f"persistent_workers={dataloader_kwargs.get('persistent_workers', False)} "
            f"prefetch_factor={dataloader_kwargs.get('prefetch_factor')} "
            f"timeout={dataloader_kwargs['timeout']}"
        )
        try:
            logger.info(dataloader_config_message)
        except RuntimeError:
            print(dataloader_config_message)

        video_train_dataloader = DataLoader(
            video_dataset,
            **dataloader_kwargs,
            # shuffle=True
        )
        return video_train_dataloader
