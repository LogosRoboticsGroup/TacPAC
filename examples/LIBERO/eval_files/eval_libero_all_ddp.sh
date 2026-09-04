#!/bin/bash

set -o pipefail
set -u

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
default_ckpt="${STARVLA_DIR}/results/Checkpoints/vla/0610_WanMoT_libero_all/final_model/pytorch_model.pt"

if [ "$#" -eq 1 ]; then
    num_servers=${1:?Usage: bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh <num_servers> [num_eval_workers] [checkpoint]}
    num_eval_workers="$num_servers"
    your_ckpt=${CKPT:-${default_ckpt}}
elif [ "$#" -eq 2 ]; then
    num_servers=${1:?Usage: bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh <num_servers> [num_eval_workers] [checkpoint]}
    num_eval_workers="$num_servers"
    your_ckpt=${2:-${CKPT:-${default_ckpt}}}
elif [ "$#" -ge 3 ]; then
    num_servers=${1:?Usage: bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh <num_servers> [num_eval_workers] [checkpoint]}
    num_eval_workers=${2:?Usage: bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh <num_servers> [num_eval_workers] [checkpoint]}
    your_ckpt=${3:-${CKPT:-${default_ckpt}}}
else
    echo "Usage: bash examples/LIBERO/eval_files/eval_libero_all_ddp.sh <num_servers> [num_eval_workers] [checkpoint]"
    exit 1
fi

your_folder=$(dirname "$(dirname "$your_ckpt")")
ckpt_name=$(basename "$your_ckpt" .pt)
run_root=${OUT_ROOT:-"${your_folder}/results/${ckpt_name}/libero"}
summary_log="${run_root}/eval_summary.log"

task_suites=(
  libero_spatial
  libero_object
  libero_goal
  libero_10
)

cd "${STARVLA_DIR}"
mkdir -p "$run_root"
: > "$summary_log"

{
    echo "===== LIBERO all-suite evaluation started at $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "checkpoint: ${your_ckpt}"
    echo "num_servers: ${num_servers}"
    echo "num_eval_workers: ${num_eval_workers}"
    echo "seed: ${SEED:-42}"
} | tee -a "$summary_log"

for task_suite_name in "${task_suites[@]}"; do
    out_path="${run_root}/${task_suite_name}"
    suite_log="${run_root}/${task_suite_name}.log"
    : > "$suite_log"

    {
        echo "===== Evaluating ${task_suite_name} ====="
        echo "log_file: ${suite_log}"
        echo "out_path: ${out_path}"
    } | tee -a "$summary_log"

    bash examples/LIBERO/eval_files/eval_libero_oversubscribe_ddp.sh "$num_servers" "$num_eval_workers" "$your_ckpt" "$task_suite_name"
    suite_status=$?
    if [ "$suite_status" -eq 0 ]; then
        echo "===== ${task_suite_name} finished successfully =====" | tee -a "$summary_log"
    else
        echo "===== ${task_suite_name} failed with exit code ${suite_status} =====" | tee -a "$summary_log"
        exit "$suite_status"
    fi
done

"${LIBERO_PYTHON}" examples/LIBERO/eval_files/aggregate_results.py --results-root "$run_root" | tee -a "$summary_log"

echo "===== LIBERO all-suite evaluation finished at $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "$summary_log"
