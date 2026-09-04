#!/bin/bash

set -u

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
default_ckpt="${STARVLA_DIR}/results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt"
your_ckpt=${1:-${CKPT:-${default_ckpt}}}
base_port=${PORT:-6694}
gpu_id=${GPU_ID:-0}

extra_args=()
if [ "${COMPILE:-0}" = "1" ]; then
    extra_args+=(--compile)
fi
if [ -n "${STAT_KEY:-}" ]; then
    extra_args+=(--stat_key "$STAT_KEY")
fi
if [ "${LOG_TIMING_EVERY:-0}" != "0" ]; then
    extra_args+=(--log_timing_every "$LOG_TIMING_EVERY")
fi
if [ "${USE_BF16:-1}" = "1" ]; then
    extra_args+=(--use_bf16)
fi

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

CUDA_VISIBLE_DEVICES="$gpu_id" "${STARVLA_PYTHON}" -u deployment/model_server/server_policy.py \
    --ckpt_path "$your_ckpt" \
    --port "$base_port" \
    "${extra_args[@]}"
