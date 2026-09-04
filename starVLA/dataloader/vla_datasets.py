from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.vla.dataset import LeRobotV2Dataset, MultiLeRobotV2VLADataset
from starVLA.dataloader.vla.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.vla.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.vla.mix_dataset import MixDataset
import copy

def collate_fn(batch):
    return batch

def make_VLASingleDataset(
    data_cfg: dict,
    data_name: str,
    data_type: str,
    data_class: str,
    mode: str = "train",
    **kwargs,
):
    
    # Allow overriding video_keys from mixture spec
    video_keys = kwargs.pop("video_keys", None)

    if data_class == "lerobot_vla":
        data_config = ROBOT_TYPE_CONFIG_MAP[data_type]
        return LeRobotV2Dataset(
            config=data_cfg,
            data_name=data_name,
            data_config=data_config,
            data_type=data_type,
            video_keys=video_keys,
            mode=mode,
            **kwargs
        )
    elif data_class == "multi_lerobot_vla":
        data_config = ROBOT_TYPE_CONFIG_MAP[data_type]
        return MultiLeRobotV2VLADataset(
            config=data_cfg,
            data_name=data_name,
            data_config=data_config,
            data_type=data_type,
            video_keys=video_keys,
            mode=mode,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported data_class: {data_class}")

def _stats_references_from_config(data_cfg) -> dict:
    stats_references = getattr(data_cfg, "stats_references", None)
    if stats_references is None:
        return {}
    if OmegaConf.is_config(stats_references):
        return OmegaConf.to_container(stats_references, resolve=True)
    return dict(stats_references)


def _build_vla_mix(
    data_cfg: dict,
    data_mix: str,
    mode: str,
    balance_dataset_weights: bool = True,
):
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    dataset_mixture = []
    for d_name, d_kwargs in mixture_spec.items():
        dataset_mixture.append((make_VLASingleDataset(data_cfg, d_name, mode=mode, **d_kwargs), d_kwargs["data_weight"]))
    return MixDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        sampling_temperature=getattr(data_cfg, "sampling_temperature", 2.0),
        stats_merge_strategy=getattr(data_cfg, "stats_merge_strategy", "average"),
        stats_references=_stats_references_from_config(data_cfg),
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = True,
):
    """
    Get a LeRobotMixtureDataset object.
    """
    return _build_vla_mix(
        data_cfg=data_cfg,
        data_mix=data_cfg.data_mix,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
    )


if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/vla/starvla_wam.yaml", help="Path to YAML config")
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_dim = 10
    cfg.framework.action_model.state_dim = 7

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_mix = args.data_mix if args.data_mix else "libero_all"
    vla_dataset_cfg.use_future_frames = True
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()) if isinstance(sample, dict) else type(sample))
    if isinstance(sample, dict):
        for k, v in sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")

    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4,
        # shuffle=True
    )        
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(train_dataloader, desc="Processing Batches")):
        # breakpoint()
        # [b['data_id'] for b in batch]
        if i >= 10:
            break
