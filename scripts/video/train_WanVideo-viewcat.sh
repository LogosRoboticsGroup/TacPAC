#!/bin/bash
# Multi-GPU WanVideo TI2V training script (DeepSpeed ZeRO-2)
# Usage: bash train_WanVideo.sh <data_mix> [run_id_suffix] [extra_args...]
# Example: bash train_WanVideo.sh robocoin "debug" trainer.max_train_steps=1000

set -e

if [ -z "$1" ]; then
  echo "Usage: bash $0 <data_mix> [run_id_suffix] [extra_args...]"
  echo "  extra_args: hydra-style overrides, e.g. trainer.max_train_steps=1000 datasets.video_data.num_frames=17"
  exit 1
fi

###########################################################################################
# === Configuration ===
Framework_name=WanVideo
freeze_module_list=''
config_yaml=starVLA/config/training/video/starvla_video.yaml
run_root_dir=./results/Checkpoints/video
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

merge_views=${MERGE_VIEWS:-horizontal}
target_height=${TARGET_HEIGHT:-256}
target_width=${TARGET_WIDTH:-512}
###########################################################################################


data_mix=$1

# $2 is treated as run_id_suffix only if it does NOT contain '='
if [ -n "$2" ] && [[ "$2" != *"="* ]]; then
  run_id_suffix=$2
  shift 2 2>/dev/null || shift $#
else
  run_id_suffix=""
  shift 1
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
if [ -z "${run_id_suffix}" ]; then
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
  starVLA/training/train_starvla_video.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --datasets.video_data.data_mix ${data_mix} \
  --datasets.video_data.per_device_batch_size 4 \
  --datasets.video_data.eval_per_device_batch_size 4 \
  --datasets.video_data.num_frames 13 \
  --datasets.video_data.image_size '[256,256]' \
  --datasets.video_data.merge_views "${merge_views}" \
  --framework.wan_video.target_height "${target_height}" \
  --framework.wan_video.target_width "${target_width}" \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.enable_gradient_checkpointing false \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 1e-4 \
  --trainer.resume_from_checkpoint latest \
  --trainer.enable_compile true \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  "${EXTRA_ARGS[@]}" 2>&1 | tee ${output_dir}/train.log
