# Copyright 2025 LogosVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
LogosVLA's trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
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
import time
from einops import rearrange

# Third-Party Libraries
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler
from diffusers.utils import export_to_video

from starVLA.training.trainer_utils import should_enable_progress_bar
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, resolve_gradient_accumulation_steps

import decord
decord.bridge.set_bridge('torch')

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
torch._dynamo.config.recompile_limit = 64


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
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "validation", exist_ok=True)
        
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    return output_dir


from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator):
    """prepare training data"""
    logger.info(f"\nCreating Video Dataset with Mixture `{cfg.datasets.video_data.data_mix}`")
    video_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.video_data.dataset_py, mode="train")
    video_eval_dataloader = None
    if cfg.datasets.video_data.get("split_strategy", "none") != "none":
        video_eval_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.video_data.dataset_py, mode="eval")

    accelerator.dataloader_config.dispatch_batches = False
    accelerator.wait_for_everyone()
    
    return video_train_dataloader, video_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg, accelerator):
    """set optimizer and scheduler"""
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
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"\nLR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    
    # Accelerate advances prepared schedulers once per process when split_batches=False.
    scheduler_step_scale = accelerator.num_processes

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps * scheduler_step_scale,
        num_training_steps=cfg.trainer.max_train_steps * scheduler_step_scale,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VideoTrainer(TrainerUtils):
    def __init__(self, cfg, model, video_train_dataloader, video_eval_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.video_train_dataloader = video_train_dataloader
        self.video_eval_dataloader = video_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        # training status tracking
        self.completed_steps = 0
        self.video_epoch_count = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 42
        set_seed(seed)
        
        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        
        self.print_trainable_parameters(self.model)
        
        self._maybe_compile_model()
        
        # initialize distributed training components
        self.model, self.optimizer, self.video_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.video_train_dataloader,
            self.lr_scheduler,
        )
        
        self._init_tensorboard()
        self._init_checkpointing()
        
        
    def _maybe_compile_model(self):
        enable_compile = bool(getattr(self.config.trainer, "enable_compile", False))
        if not enable_compile:
            return
        if not hasattr(self.model, "compile"):
            logger.warning("Model has no compile method; skipping.")
            return
        logger.info("\nEnable framework compile.")
        self.model.compile()
        
    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.video_data.per_device_batch_size
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
            resolved = [None]
            if self.accelerator.is_main_process:
                dirs = [
                    d for d in os.listdir(self.checkpoint_dir)
                    if d.startswith("steps_") and os.path.isdir(os.path.join(self.checkpoint_dir, d))
                ]
                if dirs:
                    latest = max(dirs, key=lambda x: int(x.split("_")[1]))
                    resolved[0] = os.path.join(self.checkpoint_dir, latest)

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
            self.video_epoch_count = training_state.get("video_epoch_count", 0)
            logger.info(
                f"Restored: step={self.completed_steps}, epoch={self.video_epoch_count}"
            )
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
                "video_epoch_count": self.video_epoch_count,
            }
            with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)

            state_dict = self._get_clean_state_dict_for_save()
            model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"
            torch.save(state_dict, model_checkpoint_path)

            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            logger.info(f"Checkpoint saved at {checkpoint_path}; model weights saved at {model_checkpoint_path}")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(
                    self.completed_steps
                    * self.accelerator.gradient_accumulation_steps
                    / len(self.video_train_dataloader),
                    2,
                )

                # record to TensorBoard
                for k, v in metrics.items():
                    self.tb_writer.add_scalar("train/"+k, v, self.completed_steps)
                logger.info(f"\nStep {self.completed_steps}, Loss: {metrics})")
                
    def _create_data_iterators(self):
        """create data iterators"""
        if hasattr(self.video_train_dataloader, "sampler") and callable(
            getattr(self.video_train_dataloader.sampler, "set_epoch", None)
        ):
            self.video_train_dataloader.sampler.set_epoch(self.video_epoch_count)
        self.video_iter = iter(self.video_train_dataloader)
        
    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_video = next(self.video_iter)
        except StopIteration:
            if not hasattr(self, "video_epoch_count"):
                self.video_epoch_count = 0
            self.video_iter, self.video_epoch_count = TrainerUtils._reset_dataloader(
                self.video_train_dataloader, self.video_epoch_count
            )
            logger.info("Reset video dataloader iterator at epoch boundary: epoch=%s", self.video_epoch_count)
            batch_video = next(self.video_iter)

        return batch_video

    def _get_next_eval_batch(self):
        if not hasattr(self, "video_eval_iter"):
            self.video_eval_iter = iter(self.video_eval_dataloader)
        try:
            return next(self.video_eval_iter)
        except StopIteration:
            self.video_eval_iter = iter(self.video_eval_dataloader)
            return next(self.video_eval_iter)
    
    def train(self):
        """execute training loop"""
        self._log_training_config()
        self._create_data_iterators()
        
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not should_enable_progress_bar(self.accelerator.is_local_main_process),
        )
        
        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_video = self._get_next_batch()
            t_end_data = time.perf_counter()
            
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_video)
            t_end_model = time.perf_counter()
            
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

                # evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    examples = None if self.video_eval_dataloader is not None else batch_video
                    step_metrics = self.eval_video_model(step_metrics, examples=examples)
                    
                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)
                
                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()
                    
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
            
        self._finalize_training()
        
    def eval_video_model(self, step_metrics: dict = None, examples=None) -> dict:
        """Generate validation videos with predict_video and save to disk.
        
        Layout: t (v h) (b w) c - views stacked vertically, samples horizontally.
        Top V rows = predicted, bottom V rows = GT. Total 2V rows, B columns.
        """
        if step_metrics is None:
            step_metrics = {}
            
        if self.accelerator.is_main_process:
            if examples is None:
                if self.video_eval_dataloader is not None:
                    examples = self._get_next_eval_batch()
                else:
                    examples = self._get_next_batch()
                
            prompt = [example["prompt"] for example in examples]
            fps = [example["fps"] for example in examples]
            B = len(examples)
            
            eval_model = self.model
            while hasattr(eval_model, "module"):
                eval_model = eval_model.module
                
            # image: [B, C, V, 1, H, W], view_mask: [B, V]
            image = torch.stack([example["image"][:, :, :1] for example in examples])
            view_mask = torch.stack([example["view_mask"] for example in examples])
            n_view = image.shape[2]
            
            was_training = self.model.training
            self.model.eval()
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                output_dict = eval_model.predict_video(
                    image=image, view_mask=view_mask, prompt=prompt, fps=fps)
            if was_training:
                self.model.train()
                eval_model.vae.eval()
                eval_model.text_encoder.eval()
                
            keep = 10
            video = output_dict["video"]
            video = rearrange(video, '(b v) c t h w -> b c v t h w', b=B, v=n_view)[:keep]
            video_gt = torch.stack([example["image"] for example in examples], dim=0).to(video.device)[:keep]
            # Trim GT to prediction length (dataset loads num_frames+1; VAE decodes num_frames)
            T_pred = video.shape[3]
            video_gt = video_gt[:, :, :, :T_pred]
            
            B = video.shape[0]
            videos = rearrange(video, 'b c v t h w -> t (v h) (b w) c', b=B)
            videos_gt = rearrange(video_gt, 'b c v t h w -> t (v h) (b w) c', b=B)
            videos = videos.float().clamp(0, 1)
            videos_gt = (videos_gt.float() / 255.0).clamp(0, 1)
            videos = torch.cat([videos, videos_gt], dim=1).cpu().numpy()

            val_dir = Path(self.config.output_dir) / "validation"
            os.makedirs(val_dir, exist_ok=True)
            save_fps = int(max(1, round(float(fps[0])))) if len(fps) > 0 else 30
            export_to_video(videos, val_dir / f"{self.completed_steps:06d}.mp4", fps=save_fps)

        self.accelerator.wait_for_everyone()
        return step_metrics
       
    def _train_step(self, batch_video):
        """execute single training step""" 
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                output_dict = self.model(batch_video)
                
                video_loss = output_dict["video_loss"]
                total_loss = video_loss
            
            self.accelerator.backward(total_loss)
            
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
        log_dict = {"video_loss": video_loss.item()}
        for k, v in output_dict.items():
            if k != "video_loss" and isinstance(v, torch.Tensor) and v.dim() == 0:
                log_dict[k] = v.item()
        return log_dict
        
    def _finalize_training(self):
        """training end processing"""
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
        
    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("\n***** Training Configuration *****")
            logger.info(f"\n  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"\n  Per device batch size = {self.config.datasets.video_data.per_device_batch_size}")
            logger.info(f"\n  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"\n  Total batch size = {self.total_batch_size}")


def main(cfg) -> None:
    logger.info("\nVideo Training :: Warming Up")
    
    accelerator = build_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    video_model = build_framework(cfg)
    video_train_dataloader, video_eval_dataloader = prepare_data(cfg=cfg, accelerator=accelerator)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=video_model, cfg=cfg, accelerator=accelerator)
    
    trainer = VideoTrainer(
        cfg=cfg,
        model=video_model,
        video_train_dataloader=video_train_dataloader,
        video_eval_dataloader=video_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    
    trainer.prepare_training()
    trainer.train()
    
    logger.info("\nTraining complete!")
    accelerator.wait_for_everyone()
    if dist.is_initialized():
        dist.destroy_process_group()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/video/starvla_video.yaml")
    args, cliargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(cliargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    if cfg.is_debug and (not dist.is_initialized() or dist.get_rank() == 0):
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        logger.info("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
