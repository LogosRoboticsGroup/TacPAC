#!/bin/bash
# Multi-GPU LTXQformerFM LIBERO training script (DeepSpeed ZeRO-2)
# Hyperparameters are aligned with LogosVLA/scripts/experiments/libero/train_LTXQformerFM.sh.
# Usage: bash scripts/vla/train_LTXQformerFM-libero.sh [data_mix] [run_id_suffix] [extra_args...]
# Example: bash scripts/vla/train_LTXQformerFM-libero.sh libero_all_multi debug --trainer.max_train_steps 1000

set -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

###########################################################################################
# === Configuration ===
Framework_name=LTXQformerFM
freeze_module_list=''
config_yaml=${CONFIG_YAML:-starVLA/config/training/vla/starvla_ltx_qformer_fm.yaml}
base_video=${BASE_VIDEO:-playground/Pretrained_models/RobotLTXVIdeo/pretrain_ltxvideo_0223_gpu64.pt}
stage2_checkpoint=${STAGE2_CHECKPOINT:-playground/Pretrained_models/RobotLTXVIdeo/pretrain_ltxqformerfm_wostate.pt}
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints/vla}
qformer_start_layer=${QFORMER_START_LAYER:-0}
qformer_end_layer=${QFORMER_END_LAYER:-28}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
###########################################################################################

NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

DISTRIBUTED_ARGS="--num_machines ${NNODES} --num_processes ${NUM_PROCESSES}"
if [ "${NNODES}" -gt 1 ]; then
  DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

data_mix=${1:-libero_all}
text_embedding_cache_dir=${TEXT_EMBEDDING_CACHE_DIR:-data/text_embeds_cache/${data_mix}_ltx}
require_text_embedding_cache=${REQUIRE_TEXT_EMBEDDING_CACHE:-true}
load_text_encoder=${LOAD_TEXT_ENCODER:-false}

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

date_prefix=$(date +%m%d)
if [ -n "${RUN_ID:-}" ]; then
  run_id="${RUN_ID}"
elif [ -z "${run_id_suffix}" ]; then
  run_id="${date_prefix}_${Framework_name}_${data_mix}"
else
  run_id="${date_prefix}_${Framework_name}_${data_mix}_${run_id_suffix}"
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
echo "Logging to ${output_dir}/train.log"

compile_cache_root=${TORCH_COMPILE_CACHE_ROOT:-$(pwd)/playground/cache/torch_compile/${Framework_name}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${compile_cache_root}/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${compile_cache_root}/triton}
export TORCHINDUCTOR_FX_GRAPH_CACHE=${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}
export TORCHINDUCTOR_AUTOGRAD_CACHE=${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  ${DISTRIBUTED_ARGS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.video_model.load_text_encoder ${load_text_encoder} \
  --framework.video_model.diffusion_model.model_path ${base_video} \
  --framework.video_model.num_frames 8 \
  --framework.action_model.action_horizon 32 \
  --framework.action_model.action_dim 100 \
  --framework.action_model.state_dim 100 \
  --framework.layer_qformer.qformer_start_layer ${qformer_start_layer} \
  --framework.layer_qformer.qformer_end_layer ${qformer_end_layer} \
  --framework.layer_qformer.num_query_tokens 64 \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.eval_per_device_batch_size 4 \
  --datasets.vla_data.image_size '[256,256]' \
  --datasets.vla_data.use_future_frames true \
  --datasets.vla_data.future_frame_stride 4 \
  --datasets.vla_data.disable_state false \
  --datasets.vla_data.use_decord true \
  --datasets.vla_data.require_text_embedding_cache ${require_text_embedding_cache} \
  --datasets.vla_data.text_embedding_cache_dir ${text_embedding_cache_dir} \
  --datasets.vla_data.text_cache_encoder_id ltx-video-t5 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.knowledge_isolation false \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.learning_rate.layer_qformer 1e-4 \
  --trainer.learning_rate.diffusion_model 1e-5 \
  --trainer.enable_compile true \
  --trainer.init_from_stage_checkpoint ${stage2_checkpoint} \
  --trainer.num_warmup_steps 1000 \
  --trainer.resume_from_checkpoint latest \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${output_dir}/train.log"
