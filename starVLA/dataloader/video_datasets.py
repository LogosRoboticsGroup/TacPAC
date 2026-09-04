import copy
import json
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.video.data_config import VIDEO_TYPE_CONFIG_MAP
from starVLA.dataloader.video.mix_dataset import MixVideoDataset
from starVLA.dataloader.video.mixtures import DATASET_NAMED_MIXTURES


def collate_fn(batch):
    return batch


def make_VideoSingleDataset(
    data_cfg: dict,
    data_name: str,
    data_type: str,
    data_class: str,
    **kwargs,
):
    data_config = copy.deepcopy(VIDEO_TYPE_CONFIG_MAP[data_type])
    
    if data_class == "lerobot_video":
        from starVLA.dataloader.video.dataset.lerobot_dataset import LeRobotV2VideoDataset
        
        return LeRobotV2VideoDataset(
            config=data_cfg,
            data_name=data_name,
            data_config=data_config,
            data_type=data_type,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Unsupported data_class: {data_class}")
        


def get_video_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    **kwargs: dict,
):
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    seen_signatures = set()
    dataset_mixture = []
    for data_name, data_kwargs in mixture_spec.items():
        signature = (
            data_kwargs["data_root"],
            data_kwargs["data_class"],
            data_kwargs["data_type"],
            json.dumps(data_kwargs.get("video_keys"), sort_keys=True),
            data_kwargs.get("cache_name"),
            json.dumps(data_kwargs.get("subdataset_filter"), sort_keys=True),
            data_kwargs.get("validate_video_paths_per_episode"),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        dataset_mixture.append(
            (
                make_VideoSingleDataset(data_cfg, data_name, mode=mode, **data_kwargs),
                data_kwargs["data_weight"],
            )
        )

    return MixVideoDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        sampling_temperature=getattr(data_cfg, "sampling_temperature", 2.0),
    )
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/video/starvla_video.yaml",
        help="Path to YAML config"
    )
    parser.add_argument("--data_mix", type=str, default="all")
    args, _ = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.video_data.data_mix = args.data_mix
    
    dataset = get_video_dataset(data_cfg=cfg.datasets.video_data)
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
