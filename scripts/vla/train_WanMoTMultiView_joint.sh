#!/bin/bash
# Multi-GPU WanMoTJoint training script (DeepSpeed ZeRO-2)
# Usage: bash train_WanMoTJoint.sh [data_mix] [run_id_suffix] [extra_args...]
# Example: bash train_WanMoTJoint.sh libero_all "debug" trainer.max_train_steps=1000

set -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

###########################################################################################
# === Configuration ===
Framework_name=WanMoTMultiViewJoint
freeze_module_list=''
config_yaml=starVLA/config/training/vla/starvla_wam.yaml
run_root_dir=./results/Checkpoints/vla
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
###########################################################################################

data_mix=${1:-libero_all}

# $2 is treated as run_id_suffix only if it does NOT contain '='
if [ -n "$2" ] && [[ "$2" != *"="* ]] && [[ "$2" != --* ]]; then
  run_id_suffix=$2
  shift 2 2>/dev/null || shift $#
elif [ -n "$1" ]; then
  run_id_suffix=""
  shift 1
else
  run_id_suffix=""
fi

# Remaining args are hydra-style overrides
EXTRA_ARGS=("$@")

NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

DISTRIBUTED_ARGS="--num_machines ${NNODES} --num_processes ${NUM_PROCESSES}"
if [ "${NNODES}" -gt 1 ]; then
  DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

date_prefix=$(date +%m%d)
if [ -n "${RUN_ID:-}" ]; then
  run_id="${RUN_ID}"
elif [ -z "${run_id_suffix}" ]; then
  run_id="${date_prefix}_${Framework_name}_${data_mix}"
else
  run_id="${date_prefix}_${Framework_name}_${data_mix}_${run_id_suffix}"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
echo "Logging to ${output_dir}/train.log"
compile_cache_root=${TORCH_COMPILE_CACHE_ROOT:-$(pwd)/playground/cache/torch_compile/${Framework_name}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${compile_cache_root}/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${compile_cache_root}/triton}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}
export TORCHINDUCTOR_AUTOGRAD_CACHE=${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}
mkdir -p ${TORCHINDUCTOR_CACHE_DIR} ${TRITON_CACHE_DIR}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  ${DISTRIBUTED_ARGS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.video_model.load_text_encoder false \
  --framework.view_image_size '[224,224]' \
  --framework.action_model.action_horizon 48 \
  --datasets.vla_data.require_text_embedding_cache true \
  --datasets.vla_data.text_embedding_cache_dir data/text_embeds_cache/aloha_wipe_board \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.eval_per_device_batch_size 4 \
  --datasets.vla_data.num_future_frames 12 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.enable_gradient_checkpointing true \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 20000 \
  --trainer.num_warmup_steps 1000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base 1e-4 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.scheduler_specific_kwargs.min_lr 1e-6 \
  --trainer.optimizer.weight_decay 1e-2 \
  --trainer.resume_from_checkpoint latest \
  --trainer.enable_compile true \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  "${EXTRA_ARGS[@]}" 2>&1 | tee ${output_dir}/train.log
