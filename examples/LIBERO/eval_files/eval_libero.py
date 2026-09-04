from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np
import torch
import tqdm
import tyro
from accelerate import Accelerator
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image, ImageDraw
from rich.console import Console
from rich.logging import RichHandler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from deployment.model_server.inferencer import M1Inference


accelerator = Accelerator()
_console = Console(force_terminal=True, stderr=True)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DATE_TIME = dt.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


def _sanitize_task_description(task_description: str) -> str:
    return task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]


def _frame_to_video_image(frame) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for key, value in frame.items():
            value_array = np.asarray(value.convert("RGB")) if isinstance(value, Image.Image) else np.asarray(value)
            pil_img = Image.fromarray(np.array(value_array, copy=True))
            ImageDraw.Draw(pil_img).text((10, 10), str(key), fill=(255, 255, 255))
            images.append(np.asarray(pil_img))
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    return np.asarray(frame)


def _write_video(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    imageio.mimwrite(path, [_frame_to_video_image(frame) for frame in frames], fps=fps)
    print(f"Saved rollout MP4 at path {path}", flush=True)


class _AsyncVideoWriter:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-video")
        self._futures = []

    def submit(self, path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
        self._futures.append(self._executor.submit(_write_video, path, frames, fps))
        if len(self._futures) >= 2:
            self._futures.pop(0).result()

    def close(self) -> None:
        try:
            for future in self._futures:
                future.result()
        finally:
            self._executor.shutdown(wait=True)


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]
    action_horizon: int = 32
    num_server_ports: int = 0
    rtc: bool = False
    prefix_steps: int = 0
    adaptive_prefix: bool = False
    verbose_timing: bool = False

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: int = 0

    out_path: str = "experiments/libero/logs"
    log_path: str = ""
    save_video: bool = True

    seed: int = 42
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test"


def eval_libero(args: Args) -> None:
    handlers = []
    handler = RichHandler(
        console=_console,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True,
        show_level=True,
        show_path=True,
        show_time=True,
    )
    handler.setFormatter(
        logging.Formatter(f"| rank={accelerator.process_index} | %(message)s", datefmt="%m/%d [%H:%M:%S]")
    )
    handlers.append(handler)
    if args.log_path:
        log_path = pathlib.Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s | %(levelname)s | rank={accelerator.process_index} | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    if accelerator.is_local_main_process:
        logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))

    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    selected_task_ids = list(range(num_tasks_in_suite))
    if args.max_tasks > 0:
        selected_task_ids = selected_task_ids[: args.max_tasks]
    if not selected_task_ids:
        raise ValueError("No LIBERO tasks selected for evaluation.")
    if accelerator.is_local_main_process:
        logging.info(
            "Task suite: %s full_tasks=%d selected_tasks=%d",
            args.task_suite_name,
            num_tasks_in_suite,
            len(selected_task_ids),
        )

    out_path = pathlib.Path(args.out_path)
    video_out_path = out_path / "video"
    out_path.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        video_out_path.mkdir(parents=True, exist_ok=True)
    video_writer = _AsyncVideoWriter() if args.save_video else None

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    total_successes = 0
    num_processes = accelerator.state.num_processes
    local_rank = accelerator.local_process_index
    show_terminal_progress = accelerator.is_local_main_process and num_processes == 1

    episodes_per_process = int(math.ceil(args.num_trials_per_task / num_processes))
    start_episode_idx = local_rank * episodes_per_process
    end_episode_idx = start_episode_idx + episodes_per_process
    num_server_ports = args.num_server_ports if args.num_server_ports > 0 else num_processes
    if num_server_ports <= 0:
        raise ValueError(f"Expected positive num_server_ports, got {num_server_ports}.")
    server_rank = local_rank % num_server_ports
    server_port = args.port + server_rank

    model = M1Inference(
        execution_steps=args.action_horizon,
        host=args.host,
        port=server_port,
        rtc=args.rtc,
        prefix_steps=args.prefix_steps,
        adaptive_prefix=args.adaptive_prefix,
        fps=20,
        verbose=args.verbose_timing,
    )

    task_successes_list = []

    for task_id in tqdm.tqdm(selected_task_ids, desc="Task", disable=not show_terminal_progress):
        episode_list = list(range(args.num_trials_per_task))[start_episode_idx:end_episode_idx]
        task_episodes, task_successes = args.num_trials_per_task, 0
        if not episode_list:
            if num_processes > 1:
                accelerator.wait_for_everyone()
                task_successes = int(
                    accelerator.gather(torch.tensor(task_successes, device=accelerator.device, dtype=torch.long))
                    .sum()
                    .item()
                )
            task_successes_list.append(task_successes)
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in tqdm.tqdm(episode_list, desc="Episode", disable=not show_terminal_progress, leave=False):
            episode_started_at = time.time()
            model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            step = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                if args.save_video:
                    replay_images.append({"image": img.copy(), "wrist_image": wrist_img.copy()})

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {
                    "observation.primary": np.expand_dims(img, axis=0),
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),
                    "observation.state": np.expand_dims(state[None], axis=0),
                    "instruction": [str(task_description)],
                }

                obs_input = {
                    "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "state": observation["observation.state"][0],
                    "task_description": observation["instruction"][0],
                    "step": step,
                    "fps": 20,
                    "view_mask": [True, True],
                }

                response = model.step(**obs_input)
                raw_action = response["action"]

                world_vector_delta = np.asarray(raw_action[:3], dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action[3:6], dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action[6:7], dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    break
                t += 1
                step += 1

            if args.save_video:
                task_segment = _sanitize_task_description(task_description)
                video_writer.submit(
                    pathlib.Path(video_out_path)
                    / (
                        f"{DATE_TIME}--episode=task{task_id}_trial{episode_idx}"
                        f"--success={bool(done)}--task={task_segment}.mp4"
                    ),
                    replay_images,
                    fps=24,
                )
            logging.info(
                "rank=%d task_id=%d episode=%d success=%s elapsed=%.3f steps=%d server_port=%d",
                local_rank,
                task_id,
                episode_idx,
                done,
                time.time() - episode_started_at,
                step,
                server_port,
            )

        env.close()
        if num_processes > 1:
            accelerator.wait_for_everyone()
            task_successes = int(
                accelerator.gather(torch.tensor(task_successes, device=accelerator.device, dtype=torch.long))
                .sum()
                .item()
            )
        total_successes += task_successes
        task_successes_list.append(task_successes)

        if accelerator.is_local_main_process:
            logging.info(
                "Current task success rate: %.4f, Current total success rate: %.4f",
                float(task_successes) / float(task_episodes),
                float(total_successes) / float(len(task_successes_list) * args.num_trials_per_task),
            )

    if video_writer is not None:
        video_writer.close()

    if num_processes > 1:
        accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        total_episodes = len(selected_task_ids) * args.num_trials_per_task
        final_results = {
            "task_suite_name": args.task_suite_name,
            "full_task_count": num_tasks_in_suite,
            "selected_task_count": len(selected_task_ids),
            "num_trials_per_task": args.num_trials_per_task,
            "num_processes": num_processes,
            "num_server_ports": num_server_ports,
            "action_horizon": args.action_horizon,
            "rtc": args.rtc,
            "prefix_steps": args.prefix_steps,
            "save_video": args.save_video,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_success_rate": float(total_successes) / float(total_episodes),
            "task_successes_list": task_successes_list,
            "selected_task_ids": selected_task_ids,
        }
        results_file = out_path / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)

        logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
        logging.info("Total episodes: %s", total_episodes)
        logging.info("Results saved to %s", results_file)
    model.client.close()


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero, console_outputs=False)
