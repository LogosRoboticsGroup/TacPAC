#!/bin/bash

set -u

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
default_ckpt="${STARVLA_DIR}/results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt"

num_gpus=${1:?Usage: bash examples/LIBERO/eval_files/run_policy_server_ddp.sh <num_gpus> [checkpoint] [replicas_per_gpu]}
your_ckpt=${2:-${CKPT:-${default_ckpt}}}
replicas_per_gpu=${3:-${REPLICAS_PER_GPU:-1}}
ckpt_name=$(basename "$your_ckpt" .pt)
base_port=${PORT:-6694}
gpu_ids_csv="${GPU_IDS:-}"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="./logs/${ckpt_name}/policy_server_ddp/${timestamp}"
pid_dir="/tmp/starvla_policy_server_ddp"
pid_file="${POLICY_SERVER_PID_FILE:-${pid_dir}/${ckpt_name}.pid}"
latest_pid_file="${pid_dir}/latest.pid"

if [ -n "$gpu_ids_csv" ]; then
    IFS=',' read -r -a gpu_ids <<< "$gpu_ids_csv"
    if [ "${#gpu_ids[@]}" -lt "$num_gpus" ]; then
        echo "GPU_IDS must contain at least ${num_gpus} ids, got: ${gpu_ids_csv}" >&2
        exit 1
    fi
else
    gpu_ids=()
    for local_rank in $(seq 0 $((num_gpus - 1))); do
        gpu_ids+=("$local_rank")
    done
fi

cd "${STARVLA_DIR}"
mkdir -p "$log_dir"
mkdir -p "$pid_dir"
: > "$pid_file"
echo "Server logs will be written to: $log_dir"
echo "Server PIDs will be written to: $pid_file"

for local_rank in $(seq 0 $((num_gpus - 1))); do
    gpu_id="${gpu_ids[$local_rank]}"
    for replica_rank in $(seq 0 $((replicas_per_gpu - 1))); do
        port=$((base_port + local_rank + replica_rank * num_gpus))
        log_file="${log_dir}/gpu_${gpu_id}_rank_${local_rank}_replica_${replica_rank}_port_${port}.log"
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

        echo "Starting server on port $port for GPU $gpu_id (rank $local_rank) replica $replica_rank..."

        nohup setsid env \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}" \
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}" \
            "${STARVLA_PYTHON}" -u deployment/model_server/server_policy.py \
            --ckpt_path "${your_ckpt}" \
            --port "${port}" \
            "${extra_args[@]}" \
            > "$log_file" 2>&1 < /dev/null &
        pid=$!
        echo "${pid} ${gpu_id} ${port} ${log_file}" >> "$pid_file"

        echo "Server started for GPU $gpu_id replica $replica_rank on port $port"
        echo "PID: $pid"
        echo "Log file: $log_file"
    done
done

cp "$pid_file" "$latest_pid_file"
echo "Latest PID file: $latest_pid_file"
echo "All servers started."
