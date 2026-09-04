#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/cfs/miniconda3/envs/starVLA/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/cfs/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/cfs/miniconda3/envs/libero/bin/python}"

PORT="${PORT:-6694}"
NUM_SERVERS="${NUM_SERVERS:-8}"
NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS:-${NUM_SERVERS}}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
GPU_MEM_IDLE_MB="${GPU_MEM_IDLE_MB:-5000}"
GPU_UTIL_IDLE_PCT="${GPU_UTIL_IDLE_PCT:-0}"
SERVER_IDLE_TIMEOUT_SECONDS="${SERVER_IDLE_TIMEOUT_SECONDS:-1800}"
EVAL_HOST="${EVAL_HOST:-127.0.0.1}"
SEED="${SEED:-42}"

# Example:
#   GPU_IDS=0,1,2,3,4,5,6,7 NUM_SERVERS=8 NUM_EVAL_WORKERS=8 GPU_UTIL_IDLE_PCT=0 \
#     bash examples/LIBERO/eval_files/eval_libero_queue_tmux.sh <ckpt_path> <label>
#
# This single entrypoint creates two tmux sessions internally:
#   libero_<label>_server: waits for the checkpoint, idle GPUs, and free ports, then starts policy servers.
#   libero_<label>_sim: waits for server readiness, then runs the LIBERO all-suite simulation.

usage() {
  cat <<'USAGE'
Usage:
  bash examples/LIBERO/eval_files/eval_libero_queue_tmux.sh <ckpt_path> [label]

Queues a LIBERO all-suite evaluation in two tmux sessions:
  <label>_server: waits for checkpoint, idle GPUs, free ports, then starts policy servers
  <label>_sim: waits for server readiness, then runs eval_libero_all_ddp.sh

Useful env vars:
  NUM_SERVERS=8
  NUM_EVAL_WORKERS=8
  GPU_IDS=0,1,2,3,4,5,6,7
  GPU_UTIL_IDLE_PCT=0
  SEED=42
  GPU_MEM_IDLE_MB=5000
  PORT=6694
  CHECK_INTERVAL_SECONDS=60
  SERVER_IDLE_TIMEOUT_SECONDS=1800
USAGE
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$(pwd)" "$1" ;;
  esac
}

shell_quote() {
  printf '%q' "$1"
}

derive_paths() {
  ckpt="${CKPT:?CKPT is required}"
  label="${LABEL:-}"
  ckpt_name="$(basename "${ckpt}" .pt)"
  ckpt_parent="$(dirname "$(dirname "${ckpt}")")"
  if [ -z "${label}" ]; then
    label="$(basename "${ckpt_parent}")"
  fi
  safe_label="$(printf '%s' "${label}" | tr -c 'A-Za-z0-9_' '_' | sed 's/^_\\+//; s/_\\+$//')"
  if [ -z "${safe_label}" ]; then
    safe_label="libero_eval"
  fi
  run_root="${OUT_ROOT:-${ckpt_parent}/results/${ckpt_name}/libero}"
  queue_root="${QUEUE_ROOT:-${ckpt_parent}/results/${ckpt_name}/libero_queue}"
  coord_dir="${COORD_DIR:-${queue_root}/coord}"
  log_dir="${LOG_DIR:-${queue_root}/logs}"
}

log_server() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [server] %s\n' -1 "$*" >&2
}

log_sim() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [sim] %s\n' -1 "$*"
}

idle_gpus() {
  local requested_gpus="${GPU_IDS:-${GPU_ID:-}}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v reqs="${requested_gpus}" -v max_mem="${GPU_MEM_IDLE_MB}" -v max_util="${GPU_UTIL_IDLE_PCT}" '
      BEGIN {
        split(reqs, requested, /[,[:space:]]+/)
        for (idx in requested) {
          if (requested[idx] != "") {
            requested_set[requested[idx]] = 1
            has_requested = 1
          }
        }
      }
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
        if (has_requested && !($1 in requested_set)) next
        if ($2 <= max_mem && $3 <= max_util) print $1
      }
    '
}

wait_for_idle_gpus() {
  local gpus=()
  while true; do
    mapfile -t gpus < <(idle_gpus || true)
    if [ "${#gpus[@]}" -ge "${NUM_SERVERS}" ]; then
      printf '%s\n' "${gpus[@]:0:${NUM_SERVERS}}"
      return 0
    fi
    log_server "waiting for ${NUM_SERVERS} idle GPU(s): memory<=${GPU_MEM_IDLE_MB}MB util<=${GPU_UTIL_IDLE_PCT}%"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
      sed 's/^/[gpu] /' >&2
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

wait_for_checkpoint() {
  while [ ! -s "${ckpt}" ]; do
    log_server "waiting for checkpoint: ${ckpt}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

port_open() {
  local port="$1"
  "${STARVLA_PYTHON}" - "$port" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_for_ports_free() {
  local busy_ports=()
  local server_idx
  local server_port
  while true; do
    busy_ports=()
    for server_idx in $(seq 0 "$((NUM_SERVERS - 1))"); do
      server_port="$((PORT + server_idx))"
      if port_open "${server_port}"; then
        busy_ports+=("${server_port}")
      fi
    done
    if [ "${#busy_ports[@]}" -eq 0 ]; then
      return 0
    fi
    log_server "port(s) already open: ${busy_ports[*]}; waiting"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

all_servers_alive() {
  local pid
  for pid in "$@"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

stop_servers() {
  local pid
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  for pid in "$@"; do
    kill "${pid}" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    local any_alive=0
    for pid in "$@"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    if [ "${any_alive}" -eq 0 ]; then
      return 0
    fi
    sleep 1
  done
  for pid in "$@"; do
    kill -9 "${pid}" 2>/dev/null || true
  done
}

wait_for_servers_ready() {
  local pids=("$@")
  local ready_file="${coord_dir}/server_ready_${safe_label}"
  local server_idx
  local server_port
  local all_ready
  rm -f "${ready_file}"
  while true; do
    all_ready=1
    for server_idx in $(seq 0 "$((NUM_SERVERS - 1))"); do
      server_port="$((PORT + server_idx))"
      if ! port_open "${server_port}"; then
        all_ready=0
        break
      fi
    done
    if [ "${all_ready}" -eq 1 ]; then
      touch "${ready_file}"
      log_server "servers ready on ports ${PORT}-$((PORT + NUM_SERVERS - 1))"
      return 0
    fi
    if ! all_servers_alive "${pids[@]}"; then
      log_server "a server process exited before all ports opened"
      return 1
    fi
    sleep 5
  done
}

run_server_role() {
  derive_paths
  mkdir -p "${coord_dir}" "${log_dir}"
  cd "${STARVLA_DIR}"
  export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

  local done_file="${coord_dir}/eval_done_${safe_label}"
  local fail_file="${coord_dir}/eval_failed_${safe_label}"
  local stopped_file="${coord_dir}/server_stopped_${safe_label}"
  local pid_file="${coord_dir}/server_pid_${safe_label}"
  local server_pids=()

  trap 'stop_servers "${server_pids[@]:-}"' INT TERM EXIT
  rm -f "${done_file}" "${fail_file}" "${stopped_file}" "${pid_file}" "${coord_dir}/server_ready_${safe_label}"

  wait_for_checkpoint
  mapfile -t gpu_ids < <(wait_for_idle_gpus)
  wait_for_ports_free

  log_server "starting ${NUM_SERVERS} policy server(s): label=${safe_label} gpus=${gpu_ids[*]} ports=${PORT}-$((PORT + NUM_SERVERS - 1)) ckpt=${ckpt}"

  for server_idx in $(seq 0 "$((NUM_SERVERS - 1))"); do
    gpu_id="${gpu_ids[$server_idx]}"
    server_port="$((PORT + server_idx))"
    server_log="${log_dir}/server_${safe_label}_port${server_port}.log"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${STARVLA_PYTHON}" -u deployment/model_server/server_policy.py \
      --ckpt_path "${ckpt}" \
      --port "${server_port}" \
      --use_bf16 \
      --idle_timeout "${SERVER_IDLE_TIMEOUT_SECONDS}" >"${server_log}" 2>&1 &
    server_pid="$!"
    server_pids+=("${server_pid}")
    log_server "started server ${server_idx}: gpu=${gpu_id} port=${server_port} pid=${server_pid} log=${server_log}"
  done
  printf '%s\n' "${server_pids[@]}" > "${pid_file}"

  if ! wait_for_servers_ready "${server_pids[@]}"; then
    touch "${fail_file}"
    stop_servers "${server_pids[@]}"
    exit 1
  fi

  log_server "waiting for LIBERO eval completion marker"
  while true; do
    if [ -f "${done_file}" ]; then
      log_server "eval completed; stopping servers"
      break
    fi
    if [ -f "${fail_file}" ]; then
      log_server "eval failed; stopping servers"
      stop_servers "${server_pids[@]}"
      touch "${stopped_file}"
      exit 1
    fi
    if ! all_servers_alive "${server_pids[@]}"; then
      log_server "a server process died during eval"
      touch "${fail_file}"
      exit 1
    fi
    sleep 10
  done

  stop_servers "${server_pids[@]}"
  server_pids=()
  touch "${stopped_file}"
  trap - INT TERM EXIT
  log_server "servers stopped"
}

run_sim_role() {
  derive_paths
  mkdir -p "${coord_dir}" "${log_dir}"

  local done_file="${coord_dir}/eval_done_${safe_label}"
  local fail_file="${coord_dir}/eval_failed_${safe_label}"
  local stopped_file="${coord_dir}/server_stopped_${safe_label}"
  local ready_file="${coord_dir}/server_ready_${safe_label}"
  local sim_log="${log_dir}/sim_${safe_label}.log"

  rm -f "${done_file}" "${fail_file}"
  while [ ! -s "${ckpt}" ]; do
    log_sim "waiting for checkpoint: ${ckpt}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
  while [ ! -f "${ready_file}" ]; do
    if [ -f "${fail_file}" ]; then
      log_sim "server side marked failed before ready"
      exit 1
    fi
    log_sim "waiting for server_ready marker: ${ready_file}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  log_sim "starting LIBERO all-suite eval: ckpt=${ckpt}"
  log_sim "sim log: ${sim_log}"
  set +e
  (
    cd "${STARVLA_DIR}"
    export STARVLA_DIR
    export LIBERO_HOME
    export LIBERO_PYTHON
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
    export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
    export PORT
    export EVAL_HOST
    export SEED
    if [ -n "${OUT_ROOT:-}" ]; then
      export OUT_ROOT
    fi
    export EVAL_USE_CPU="${EVAL_USE_CPU:-1}"
    export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
    export MAX_TASKS="${MAX_TASKS:-0}"
    export ACTION_HORIZON="${ACTION_HORIZON:-32}"
    export SAVE_VIDEO="${SAVE_VIDEO:-1}"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh "${NUM_SERVERS}" "${NUM_EVAL_WORKERS}" "${ckpt}"
  ) 2>&1 | tee "${sim_log}"
  status="${PIPESTATUS[0]}"
  set -e

  if [ "${status}" -ne 0 ]; then
    log_sim "LIBERO eval failed with exit code ${status}"
    touch "${fail_file}"
    exit "${status}"
  fi

  touch "${done_file}"
  log_sim "LIBERO eval completed; waiting for server stop marker"
  while [ ! -f "${stopped_file}" ]; do
    sleep 5
  done
  touch "${coord_dir}/sim_all_done"
  log_sim "all LIBERO simulations completed"
}

start_tmux() {
  if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 2
  fi

  CKPT="$(abs_path "$1")"
  LABEL="${2:-}"
  export CKPT LABEL
  derive_paths

  local script_path
  script_path="$(abs_path "$0")"
  local server_session="libero_${safe_label}_server"
  local sim_session="libero_${safe_label}_sim"
  local server_cmd
  local sim_cmd

  server_cmd="cd $(shell_quote "${STARVLA_DIR}") && CKPT=$(shell_quote "${CKPT}") LABEL=$(shell_quote "${label}") PORT=$(shell_quote "${PORT}") NUM_SERVERS=$(shell_quote "${NUM_SERVERS}") GPU_MEM_IDLE_MB=$(shell_quote "${GPU_MEM_IDLE_MB}") GPU_UTIL_IDLE_PCT=$(shell_quote "${GPU_UTIL_IDLE_PCT}") CHECK_INTERVAL_SECONDS=$(shell_quote "${CHECK_INTERVAL_SECONDS}") SERVER_IDLE_TIMEOUT_SECONDS=$(shell_quote "${SERVER_IDLE_TIMEOUT_SECONDS}") GPU_IDS=$(shell_quote "${GPU_IDS:-}") OUT_ROOT=$(shell_quote "${OUT_ROOT:-}") bash $(shell_quote "${script_path}") --server"
  sim_cmd="cd $(shell_quote "${STARVLA_DIR}") && CKPT=$(shell_quote "${CKPT}") LABEL=$(shell_quote "${label}") PORT=$(shell_quote "${PORT}") NUM_SERVERS=$(shell_quote "${NUM_SERVERS}") NUM_EVAL_WORKERS=$(shell_quote "${NUM_EVAL_WORKERS}") CHECK_INTERVAL_SECONDS=$(shell_quote "${CHECK_INTERVAL_SECONDS}") SEED=$(shell_quote "${SEED}") OUT_ROOT=$(shell_quote "${OUT_ROOT:-}") bash $(shell_quote "${script_path}") --sim"

  tmux new-session -d -s "${server_session}" "${server_cmd}"
  tmux new-session -d -s "${sim_session}" "${sim_cmd}"

  printf 'started tmux sessions:\n'
  printf '  %s\n' "${server_session}"
  printf '  %s\n' "${sim_session}"
  printf 'queue logs: %s\n' "${log_dir}"
  printf 'eval results: %s\n' "${run_root}"
}

case "${1:-}" in
  --server)
    run_server_role
    ;;
  --sim)
    run_sim_role
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    start_tmux "$@"
    ;;
esac
