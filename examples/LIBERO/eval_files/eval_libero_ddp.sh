#!/bin/bash

set -o pipefail
set -u

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-$(cd "${STARVLA_DIR}/.." && pwd)/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
default_ckpt="${STARVLA_DIR}/results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt"

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}
export PYTHONPATH=${STARVLA_DIR}:${PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_GL=${MUJOCO_GL:-egl}
LIBERO_ENV_LIB="$("${LIBERO_PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.executable).resolve().parent.parent / "lib")')"
export LD_LIBRARY_PATH="${LIBERO_ENV_LIB}:${LD_LIBRARY_PATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

num_processes=${1:?Usage: bash examples/LIBERO/eval_files/eval_libero_ddp.sh <num_processes> [checkpoint] [suite]}
your_ckpt=${2:-${CKPT:-${default_ckpt}}}
your_folder=$(dirname "$(dirname "$your_ckpt")")
ckpt_name=$(basename "$your_ckpt" .pt)
run_root=${OUT_ROOT:-"${your_folder}/results/${ckpt_name}/libero"}

task_suite_name=${3:-${TASK_SUITE:-libero_goal}}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
max_tasks=${MAX_TASKS:-0}
seed=${SEED:-42}
num_server_ports=${NUM_SERVER_PORTS:-$num_processes}
action_horizon=${ACTION_HORIZON:-32}
save_video=${SAVE_VIDEO:-1}
rtc=${RTC:-0}
prefix_steps=${PREFIX_STEPS:-0}
adaptive_prefix=${ADAPTIVE_PREFIX:-0}
verbose_timing=${VERBOSE_TIMING:-0}
eval_use_cpu=${EVAL_USE_CPU:-0}
out_path="${run_root}/${task_suite_name}"
suite_log="${run_root}/${task_suite_name}.log"

host=${EVAL_HOST:-127.0.0.1}
base_port=${PORT:-6694}

mkdir -p "$run_root"
: > "$suite_log"

extra_args=()
if [ "$save_video" = "0" ]; then
    extra_args+=(--args.no-save-video)
fi
if [ "$rtc" = "1" ]; then
    extra_args+=(--args.rtc)
fi
if [ "$adaptive_prefix" = "1" ]; then
    extra_args+=(--args.adaptive-prefix)
fi
if [ "$verbose_timing" = "1" ]; then
    extra_args+=(--args.verbose-timing)
fi

cd "${STARVLA_DIR}"
eval_args=(
    ./examples/LIBERO/eval_files/eval_libero.py
    --args.pretrained-path "$your_ckpt"
    --args.host "$host"
    --args.port "$base_port"
    --args.task-suite-name "$task_suite_name"
    --args.num-trials-per-task "$num_trials_per_task"
    --args.max-tasks "$max_tasks"
    --args.seed "$seed"
    --args.num-server-ports "$num_server_ports"
    --args.action-horizon "$action_horizon"
    --args.prefix-steps "$prefix_steps"
    --args.out-path "$out_path"
    --args.log-path "$suite_log"
    "${extra_args[@]}"
)

if [ "$eval_use_cpu" = "1" ]; then
    ACCELERATE_USE_CPU=true "${LIBERO_PYTHON}" -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$num_processes" \
        "${eval_args[@]}"
elif [ "$num_processes" = "1" ]; then
    "${LIBERO_PYTHON}" -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node 1 \
        "${eval_args[@]}"
else
    "${LIBERO_PYTHON}" -m accelerate.commands.launch \
        --main_process_port "$((10000 + RANDOM % 50000))" \
        --multi_gpu \
        --num_processes "$num_processes" \
        "${eval_args[@]}"
fi
