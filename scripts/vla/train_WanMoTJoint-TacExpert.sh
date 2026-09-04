#!/bin/bash
# Phase-2 tactile-expert training (WanMoTJointTacExpert, DeepSpeed ZeRO-2).
# The base (video/action experts) is frozen and loaded from a phase-1 WanMoTJoint checkpoint;
# each step runs the frozen 10-step joint denoise (plan + prefill KV) and supervises the
# tactile expert with delta = GT - plan on the not-yet-executed suffix.
#
# Usage: bash train_WanMoTJoint-TacExpert.sh <data_mix> <phase1_ckpt> [extra_args...]
# Example: bash scripts/vla/train_WanMoTJoint-TacExpert.sh flexiv_plug_4views \
#   results/Checkpoints/vla/0721_WanMoTJoint_flexiv_plug_4views/final_model/pytorch_model.pt \
#   trainer.max_train_steps=10000

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

###########################################################################################
# === Configuration ===
Framework_name=WanMoTJointTacExpert
freeze_module_list='mot,proprio_encoder'
config_yaml=starVLA/config/training/vla/starvla_wam.yaml
run_root_dir=./results/Checkpoints/vla
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
###########################################################################################

data_mix=${1:?"data_mix is required (for example: flexiv_plug_4views)"}
stage_ckpt=${2:?"phase-1 WanMoTJoint checkpoint path is required"}
shift 2
text_embedding_cache_dir=${TEXT_EMBEDDING_CACHE_DIR:-data/text_embeds_cache/${data_mix}}
require_text_embedding_cache=${REQUIRE_TEXT_EMBEDDING_CACHE:-true}

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
run_id=${RUN_ID:-"${date_prefix}_${Framework_name}_${data_mix}"}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
echo "Logging to ${output_dir}/train.log"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  ${DISTRIBUTED_ARGS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.video_model.load_text_encoder false \
  --framework.action_model.action_horizon 48 \
  --datasets.vla_data.require_text_embedding_cache ${require_text_embedding_cache} \
  --datasets.vla_data.text_embedding_cache_dir ${text_embedding_cache_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.eval_per_device_batch_size 2 \
  --datasets.vla_data.use_future_frames false \
  --datasets.vla_data.tactile_offsets_per_sample 4 \
  --trainer.init_from_stage_checkpoint "${stage_ckpt}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.enable_gradient_checkpointing false \
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
