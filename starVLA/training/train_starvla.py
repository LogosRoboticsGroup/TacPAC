# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from einops import rearrange

# Third-Party Libraries
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler
from diffusers.utils import export_to_video

# Local Modules
from starVLA.training.trainer_utils import should_enable_progress_bar
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args, TrainerUtils, build_param_lr_groups, resolve_gradient_accumulation_steps
from starVLA.model.framework.base_framework import build_framework

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
torch._dynamo.config.recompile_limit = 64


def apply_data_config_dims(cfg):
    vla_data_cfg = cfg.datasets.vla_data

    from starVLA.dataloader.vla.data_config import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.vla.mixtures import DATASET_NAMED_MIXTURES

    data_mix = DATASET_NAMED_MIXTURES[str(vla_data_cfg.data_mix)]
    data_type = next(iter(data_mix.values()))["data_type"]
    data_config = ROBOT_TYPE_CONFIG_MAP[data_type]
    cfg.framework.action_model.action_dim = len(data_config.action_ids)
    cfg.framework.action_model.state_dim = len(data_config.state_ids)
    return cfg


def build_accelerator(cfg) -> Accelerator:
    # PartialState exposes the world size before the Accelerator locks in grad accum.
    grad_accum = resolve_gradient_accumulation_steps(cfg, PartialState().num_processes)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logger.info(accelerator.state)
    return accelerator


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "validation", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir):
    """prepare training data"""
    # VLA data loader
    logger.info(f"\nCreating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="train")
    vla_eval_dataloader = None
    if cfg.datasets.vla_data.get("split_strategy", "none") == "episode_ratio":
        vla_eval_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="eval")

    accelerator.dataloader_config.dispatch_batches = False
    accelerator.wait_for_everyone()

    return vla_train_dataloader, vla_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg, accelerator):
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        foreach=False,
        fused=True,
    )

    # print optimizer group info
    if not dist.is_initialized() or dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"\nLR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Accelerate advances prepared schedulers once per process when split_batches=False.
    scheduler_step_scale = accelerator.num_processes

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps * scheduler_step_scale,
        num_training_steps=cfg.trainer.max_train_steps * scheduler_step_scale,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vla_eval_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # load model weights from a previous stage (model only, no optimizer/scheduler/training state)
        if hasattr(self.config.trainer, "init_from_stage_checkpoint") and self.config.trainer.init_from_stage_checkpoint:
            self.model = self._load_stage_checkpoint(self.model, self.config.trainer.init_from_stage_checkpoint)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        self._maybe_compile_model()

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        self._init_tensorboard()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
    
    def _init_tensorboard(self):
        """initialize TensorBoard"""
        if self.accelerator.is_main_process:
            tensorboard_path = self.config.output_dir
            self.tb_writer = SummaryWriter(tensorboard_path)
            logger.info(f"Tensorboard logs is saved in {tensorboard_path}")
            logger.info(f"Run 'tensorboard --logdir {tensorboard_path}' to view the Tensorboard logs.")

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        resume_from = getattr(self.config.trainer, "resume_from_checkpoint", None)
        if not resume_from:
            return

        if resume_from == "latest":
            # Rank 0 resolves the latest checkpoint, then broadcasts to all ranks
            # to avoid NFS cache inconsistency across nodes.
            resolved = [None]
            if self.accelerator.is_main_process:
                dirs = [
                    d for d in os.listdir(self.checkpoint_dir)
                    if d.startswith("steps_") and os.path.isdir(os.path.join(self.checkpoint_dir, d))
                ]
                if dirs:
                    latest = max(dirs, key=lambda x: int(x.split("_")[1]))
                    resolved[0] = os.path.join(self.checkpoint_dir, latest)

            # Broadcast resolved path from rank 0 to all processes
            if dist.is_initialized():
                dist.broadcast_object_list(resolved, src=0)
            resume_from = resolved[0]

            if resume_from is None:
                logger.info("No checkpoint found, starting from scratch.")
                return

        self.accelerator.wait_for_everyone()
        self._load_checkpoint(resume_from)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint and restore training state"""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)

        meta_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                training_state = json.load(f)
            self.completed_steps = training_state["completed_steps"]
            self.vla_epoch_count = training_state.get("vla_epoch_count", 0)
            logger.info(f"Restored: step={self.completed_steps}, epoch={self.vla_epoch_count}")
        else:
            logger.warning("training_state.json not found, starting from step 0")

    def _save_checkpoint(self):
        """save current training state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_path)

        if self.accelerator.is_main_process:
            training_state = {
                "completed_steps": self.completed_steps,
                "vla_epoch_count": getattr(self, "vla_epoch_count", 0),
            }
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)

            # Save standalone model weights (for inference/evaluation)
            state_dict = self._get_clean_state_dict_for_save()
            model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"
            torch.save(state_dict, model_checkpoint_path)

            # Append to summary log
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            logger.info(f"Checkpoint saved at {model_checkpoint_path}")

        self.accelerator.wait_for_everyone()

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("\n***** Training Configuration *****")
            logger.info(f"\n  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"\n  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"\n  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"\n  Total batch size = {self.total_batch_size}")

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(
                    self.completed_steps
                    * self.accelerator.gradient_accumulation_steps
                    / len(self.vla_train_dataloader),
                    2,
                )

                # record to TensorBoard
                for k, v in metrics.items():
                    self.tb_writer.add_scalar("train/"+k, v, self.completed_steps)
                # debug output
                logger.info(f"\nStep {self.completed_steps}, Loss: {metrics})")

    def _maybe_compile_model(self):
        enable_compile = bool(getattr(self.config.trainer, "enable_compile", False))
        if not enable_compile:
            return
        if not hasattr(self.model, "compile"):
            logger.warning("Model has no compile method; skipping.")
            return
        logger.info("\nEnable framework compile.")
        self.model.compile()

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _get_next_eval_batch(self):
        if self.vla_eval_dataloader is None:
            return self._get_next_batch()
        if not hasattr(self, "vla_eval_iter"):
            self.vla_eval_iter = iter(self.vla_eval_dataloader)
        try:
            return next(self.vla_eval_iter)
        except StopIteration:
            self.vla_eval_iter = iter(self.vla_eval_dataloader)
            return next(self.vla_eval_iter)

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self._get_clean_state_dict_for_save()
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"\nTraining complete. Final model saved at {final_checkpoint}")

        self.accelerator.wait_for_everyone()

    def _get_clean_state_dict_for_save(self):
        model_to_save = self.model
        while hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        raw_state_dict = model_to_save.state_dict()
        cleaned_state_dict = {}
        for key, value in raw_state_dict.items():
            clean_key = key.replace("._orig_mod.", ".")
            if clean_key.startswith("_orig_mod."):
                clean_key = clean_key[len("_orig_mod.") :]
            cleaned_state_dict[clean_key] = value
        return cleaned_state_dict

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not should_enable_progress_bar(self.accelerator.is_local_main_process),
        )
        last_val_score = -1

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

                if self.accelerator.is_local_main_process:
                    progress_postfix = {
                        "loss": f"{step_metrics['total_loss']:.3f}",
                        "val_score": f"{last_val_score:.3f}",
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                    action_loss = step_metrics.get("action_loss")
                    if action_loss is not None:
                        progress_postfix["action_loss"] = f"{float(action_loss):.3f}"
                    video_loss = step_metrics.get("video_loss")
                    if video_loss is not None:
                        progress_postfix["video_loss"] = f"{float(video_loss):.3f}"
                    progress_bar.set_postfix(progress_postfix)

                # evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    if self.vla_eval_dataloader is not None:
                        examples = self._get_next_eval_batch() if self.accelerator.is_main_process else None
                    else:
                        examples = batch_vla
                    step_metrics = self.eval_action_model(step_metrics, examples=examples)
                    step_metrics = self.eval_video_model(step_metrics, examples=examples)
                    if self.accelerator.is_main_process:
                        last_val_score = step_metrics['mse_score']

                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()
            
            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None, examples=None) -> float:
        """Evaluate the action model by computing MSE between predicted and ground truth actions."""
        if step_metrics is None:
            step_metrics = {}

        if self.accelerator.is_main_process:

            if examples is None:
                examples = self._get_next_eval_batch()

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            actions = [example["actions"].cpu().numpy() for example in examples]  # label
            action_mask = [example["action_mask"].cpu().numpy() for example in examples]
            view_mask = [example["view_mask"] if "view_mask" in example else [True] * len(examples[0]['image']) for example in examples]
            image_is_tactile = [example["image_is_tactile"] for example in examples] if "image_is_tactile" in examples[0] else None
            fps = [example["fps"] if "fps" in example else 30 for example in examples]
            data_id = [example["data_id"] if "data_id" in example else 0 for example in examples]
            has_context = ["context" in example for example in examples]
            has_context_mask = ["context_mask" in example for example in examples]
            if any(context_present != mask_present for context_present, mask_present in zip(has_context, has_context_mask)):
                raise ValueError("Eval cached text inputs require both `context` and `context_mask` in each sample.")
            context = None
            context_mask = None
            if any(has_context):
                if not all(has_context):
                    raise ValueError("Eval cached text inputs must be present for every sample in the batch.")
                context = [example["context"] for example in examples]
                context_mask = [example["context_mask"] for example in examples]

            # Predict actions using the unwrapped model (safe for DDP/DeepSpeed + submodule-compile setup)
            eval_model = self.model
            while hasattr(eval_model, "module"):
                eval_model = eval_model.module
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            was_training = self.model.training
            try:
                output_dict = eval_model.predict_action(
                    batch_images=batch_images, instructions=instructions, state=state, view_mask=view_mask, fps=fps,
                    data_id=data_id, context=context, context_mask=context_mask, image_is_tactile=image_is_tactile
                )
            finally:
                if was_training:
                    self.model.train()

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            actions = np.array(actions)
            action_mask = np.array(action_mask).astype(np.float32)
            num_pots = action_mask.sum().item()
            score = TrainerUtils.euclidean_distance(normalized_actions * action_mask, actions * action_mask)
            average_score = score / num_pots
            step_metrics["mse_score"] = float(average_score)
        self.accelerator.wait_for_everyone()
        return step_metrics
    
    def eval_video_model(self, step_metrics: dict = None, examples=None) -> dict:
        """Generate validation videos with framework.predict_video and save to local disk."""
        if step_metrics is None:
            step_metrics = {}

        if self.accelerator.is_main_process:
            eval_model = self.model
            while hasattr(eval_model, "module"):
                eval_model = eval_model.module

            if hasattr(eval_model, "predict_video"):
                if examples is None:
                    examples = self._get_next_eval_batch()

                batch_images = [example["image"] for example in examples]
                view_mask = [
                    example["view_mask"] if "view_mask" in example else [True] * len(example["image"])
                    for example in examples
                ]
                image_is_tactile = [
                    example["image_is_tactile"] for example in examples
                ] if "image_is_tactile" in examples[0] else None
                instructions = [
                    example["lang"] if "lang" in example else example.get("prompt", "")
                    for example in examples
                ]
                fps = [example["fps"] if "fps" in example else 30 for example in examples]
                has_context = ["context" in example for example in examples]
                has_context_mask = ["context_mask" in example for example in examples]
                if any(context_present != mask_present for context_present, mask_present in zip(has_context, has_context_mask)):
                    raise ValueError("Eval cached text inputs require both `context` and `context_mask` in each sample.")
                context = None
                context_mask = None
                if any(has_context):
                    if not all(has_context):
                        raise ValueError("Eval cached text inputs must be present for every sample in the batch.")
                    context = [example["context"] for example in examples]
                    context_mask = [example["context_mask"] for example in examples]

                was_training = self.model.training
                self.model.eval()
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.inference_mode():
                    output_dict = eval_model.predict_video(
                        batch_images=batch_images,
                        view_mask=view_mask,
                        instructions=instructions,
                        fps=fps,
                        disable_view_embed=False,
                        context=context,
                        context_mask=context_mask,
                        image_is_tactile=image_is_tactile,
                    )
                if was_training:
                    self.model.train()

                video = output_dict["video"]
                batch_size = len(examples)
                if video.ndim == 5:
                    video = rearrange(video, "(b v) c t h w -> b v c t h w", b=batch_size)
                elif video.ndim != 6:
                    raise ValueError(f"Unexpected video shape from predict_video: {tuple(video.shape)}")

                keep = min(10, video.shape[0])
                video = video[:keep].float().clamp(0, 1)
                video_gt = self._prepare_eval_video_gt(
                    eval_model=eval_model,
                    batch_images=batch_images,
                    view_mask=view_mask,
                    target_video=video,
                    image_is_tactile=image_is_tactile,
                )[:keep]

                n_frames = min(video.shape[3], video_gt.shape[3])
                video = video[:, :, :, :n_frames]
                video_gt = video_gt[:, :, :, :n_frames]
                videos = torch.stack([video, video_gt], dim=2)
                videos = rearrange(videos, "b v p c t h w -> t (b p h) (v w) c")
                videos = videos.cpu().numpy()

                save_fps = int(max(1, round(float(fps[0])))) if len(fps) > 0 else 30
                save_path = Path(self.config.output_dir) / "validation" / f"{self.completed_steps:06d}.mp4"
                export_to_video(videos, save_path, fps=save_fps)

        self.accelerator.wait_for_everyone()
        return step_metrics

    def _prepare_eval_video_gt(
        self,
        eval_model,
        batch_images,
        view_mask,
        target_video: torch.Tensor,
        image_is_tactile=None,
    ) -> torch.Tensor:
        """Return GT video as [B, V, C, T, H, W] in 0-1, aligned to the predicted layout."""
        if hasattr(eval_model, "_prepare_batch_images"):
            video_gt = eval_model._prepare_batch_images(batch_images)
            if hasattr(eval_model, "_resize_views_if_needed"):
                video_gt = eval_model._resize_views_if_needed(video_gt)
            video_gt = self._visualize_eval_video_gt_views(video_gt, image_is_tactile, eval_model)
            if target_video.shape[1] == 1 and hasattr(eval_model, "_prepare_view_mask") and hasattr(eval_model, "_concat_views"):
                gt_mask = eval_model._prepare_view_mask(view_mask, video_gt.shape[0], video_gt.shape[1], video_gt.device)
                video_gt = eval_model._concat_views(video_gt, gt_mask).unsqueeze(1)
        else:
            video_gt = torch.stack(
                [
                    torch.stack([image if torch.is_tensor(image) else torch.as_tensor(image) for image in images], dim=0)
                    for images in batch_images
                ],
                dim=0,
            )
            if video_gt.ndim == 5:
                video_gt = video_gt.unsqueeze(3)
            video_gt = self._visualize_eval_video_gt_views(video_gt, image_is_tactile, eval_model)

        if video_gt.ndim != 6:
            raise ValueError(f"Unexpected GT video shape: {tuple(video_gt.shape)}")

        video_gt = video_gt.to(device=target_video.device, dtype=torch.float32)
        if video_gt.shape[1] != target_video.shape[1]:
            if video_gt.shape[1] > target_video.shape[1]:
                video_gt = video_gt[:, : target_video.shape[1]]
            elif video_gt.shape[1] == 1:
                video_gt = video_gt.expand(-1, target_video.shape[1], -1, -1, -1, -1)
            else:
                raise ValueError(
                    f"Cannot align GT views {video_gt.shape[1]} to prediction views {target_video.shape[1]}."
                )

        if video_gt.shape[-2:] != target_video.shape[-2:]:
            b, v, c, t, _, _ = video_gt.shape
            video_gt = F.interpolate(
                video_gt.reshape(b * v, c, t, video_gt.shape[-2], video_gt.shape[-1]),
                size=(t, target_video.shape[-2], target_video.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).reshape(b, v, c, t, target_video.shape[-2], target_video.shape[-1])

        return video_gt.clamp(0, 1)

    def _visualize_eval_video_gt_views(self, video_gt: torch.Tensor, image_is_tactile, eval_model) -> torch.Tensor:
        video_gt = video_gt.float()
        rgb_gt = video_gt / 255.0 if video_gt.max().item() > 1.5 else video_gt
        if image_is_tactile is None:
            return rgb_gt

        mask = eval_model._prepare_image_is_tactile(
            image_is_tactile,
            video_gt.shape[0],
            video_gt.shape[1],
            video_gt.device,
        )
        tactile_gt = video_gt.clamp(-1.0, 1.0) * 0.5 + 0.5
        return torch.where(mask[:, :, None, None, None, None], tactile_gt, rgb_gt)

    def _train_step(self, batch_vla):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                output_dict = self.model(batch_vla)

                total_loss = output_dict["total_loss"]

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Collect all losses for logging
        log_dict = {"total_loss": total_loss.item()}
        for k, v in output_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                log_dict[k] = v.detach().item()
            elif isinstance(v, (int, float, np.number)):
                log_dict[k] = float(v)
        return log_dict

def main(cfg) -> None:
    accelerator = build_accelerator(cfg)
    logger.info("\nVLA Training :: Warming Up")
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_eval_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg, accelerator=accelerator)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_eval_dataloader=vla_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    # And... we're done!
    logger.info("\nTraining complete!")
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/vla/starvla_vla.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_data_config_dims(cfg)

    if cfg.is_debug and (not dist.is_initialized() or dist.get_rank() == 0):
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...", flush=True)
        debugpy.wait_for_client()

    main(cfg)
