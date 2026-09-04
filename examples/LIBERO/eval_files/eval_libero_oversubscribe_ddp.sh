#!/bin/bash

set -o pipefail
set -u

num_servers=${1:?Usage: bash examples/LIBERO/eval_files/eval_libero_oversubscribe_ddp.sh <num_servers> <num_eval_workers> [checkpoint] [suite]}
num_eval_workers=${2:?Usage: bash examples/LIBERO/eval_files/eval_libero_oversubscribe_ddp.sh <num_servers> <num_eval_workers> [checkpoint] [suite]}
your_ckpt=${3:-${CKPT:-}}
task_suite_name=${4:-${TASK_SUITE:-libero_goal}}

if [ -n "$your_ckpt" ]; then
    NUM_SERVER_PORTS="$num_servers" \
    EVAL_USE_CPU="${EVAL_USE_CPU:-1}" \
    bash examples/LIBERO/eval_files/eval_libero_ddp.sh "$num_eval_workers" "$your_ckpt" "$task_suite_name"
else
    NUM_SERVER_PORTS="$num_servers" \
    EVAL_USE_CPU="${EVAL_USE_CPU:-1}" \
    bash examples/LIBERO/eval_files/eval_libero_ddp.sh "$num_eval_workers" "" "$task_suite_name"
fi
