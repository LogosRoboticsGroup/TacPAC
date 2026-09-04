export CKPT=results/Checkpoints/vla/0625_LTXFM_aloha_wipe_board_notac_64bs/final_model/pytorch_model.pt
export STAT_KEY=aloha_tac
export CAMERA_ORDER=observation.image.third_view,observation.image.left_wrist_view,observation.image.right_wrist_view
export PORT=5556
export SAVE_GENERATED_VIDEO_DIR=${SAVE_GENERATED_VIDEO_DIR:-outputs/infersystem_generated_videos}
export SAVE_GENERATED_VIDEO_EVERY=${SAVE_GENERATED_VIDEO_EVERY:-1}
export SAVE_GENERATED_VIDEO_FPS=${SAVE_GENERATED_VIDEO_FPS:-7.5}
export PRE_RESIZE_IMAGE_SIZE=${PRE_RESIZE_IMAGE_SIZE:-224,224}
export LOG_ACTIONS_EVERY=${LOG_ACTIONS_EVERY:-1}

python -m deployment.model_server.server_infersystem \
    --ckpt_path "$CKPT" \
    --bind "tcp://*:${PORT}" \
    --stat_key "$STAT_KEY" \
    --camera_order "$CAMERA_ORDER" \
    --use_bf16 \
    --log_timing_every 20 \
    --log_actions_every "$LOG_ACTIONS_EVERY" \
    --pre_resize_image_size "$PRE_RESIZE_IMAGE_SIZE" \
    --save_generated_video_dir "$SAVE_GENERATED_VIDEO_DIR" \
    --save_generated_video_every "$SAVE_GENERATED_VIDEO_EVERY" \
    --save_generated_video_fps "$SAVE_GENERATED_VIDEO_FPS"
