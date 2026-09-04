#!/bin/bash

set -o pipefail
set -u

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-$(cd "${STARVLA_DIR}/.." && pwd)/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"

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

default_ckpt="${STARVLA_DIR}/results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt"
your_ckpt=${1:-${CKPT:-${default_ckpt}}}
your_folder=$(dirname "$(dirname "$your_ckpt")")
ckpt_name=$(basename "$your_ckpt" .pt)
run_root=${OUT_ROOT:-"${your_folder}/results/${ckpt_name}/libero"}

task_suite_name=${2:-${TASK_SUITE:-libero_goal}}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
max_tasks=${MAX_TASKS:-0}
seed=${SEED:-42}
action_horizon=${ACTION_HORIZON:-32}
save_video=${SAVE_VIDEO:-1}
rtc=${RTC:-0}
prefix_steps=${PREFIX_STEPS:-0}
adaptive_prefix=${ADAPTIVE_PREFIX:-0}
verbose_timing=${VERBOSE_TIMING:-0}
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
"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "$your_ckpt" \
    --args.host "$host" \
    --args.port "$base_port" \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.max-tasks "$max_tasks" \
    --args.seed "$seed" \
    --args.action-horizon "$action_horizon" \
    --args.prefix-steps "$prefix_steps" \
    --args.out-path "$out_path" \
    --args.log-path "$suite_log" \
    "${extra_args[@]}"
