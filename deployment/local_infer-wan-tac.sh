#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

export CKPT=${CKPT:?"Set CKPT to a TacPAC checkpoint path"}
export STAT_KEY=${STAT_KEY:-flexiv_tac}
export CAMERA_ORDER=${CAMERA_ORDER:-observation.images.third_view,observation.images.left_wrist_view,observation.images.left_wrist_left_tactile,observation.images.left_wrist_right_tactile}
export PORT=${PORT:-5556}
export SAVE_GENERATED_VIDEO_DIR=${SAVE_GENERATED_VIDEO_DIR:-outputs/infersystem_generated_videos}
export SAVE_GENERATED_VIDEO_EVERY=${SAVE_GENERATED_VIDEO_EVERY:-1}
export SAVE_GENERATED_VIDEO_FPS=${SAVE_GENERATED_VIDEO_FPS:-7.5}
export PRE_RESIZE_IMAGE_SIZE=${PRE_RESIZE_IMAGE_SIZE:-224,224}
export LOG_ACTIONS_EVERY=${LOG_ACTIONS_EVERY:-1}
# 触觉预处理自动跟随 ckpt 训练 config (tactile_residual_mode),无需传参

# python -m deployment.model_server.server_infersystem \
#     --ckpt_path "$CKPT" \
#     --bind "tcp://*:${PORT}" \
#     --stat_key "$STAT_KEY" \
#     --camera_order "$CAMERA_ORDER" \
#     --use_bf16 \
#     --log_timing_every 20 \
#     --log_actions_every "$LOG_ACTIONS_EVERY" \
#     --pre_resize_image_size "$PRE_RESIZE_IMAGE_SIZE" \
#     --save_generated_video_dir "$SAVE_GENERATED_VIDEO_DIR" \
#     --save_generated_video_every "$SAVE_GENERATED_VIDEO_EVERY" \
#     --save_generated_video_fps "$SAVE_GENERATED_VIDEO_FPS"

# 不保存视频
python -m deployment.model_server.server_infersystem \
    --ckpt_path "$CKPT" \
    --bind "tcp://*:${PORT}" \
    --stat_key "$STAT_KEY" \
    --camera_order "$CAMERA_ORDER" \
    --use_bf16 \
    --log_timing_every 20 \
    --log_actions_every "$LOG_ACTIONS_EVERY" \
    --pre_resize_image_size "$PRE_RESIZE_IMAGE_SIZE"
