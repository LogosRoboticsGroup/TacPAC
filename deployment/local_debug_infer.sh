export STAT_KEY=aloha_tac
export CAMERA_ORDER=observation.image.third_view,observation.image.left_wrist_view,observation.image.right_wrist_view
export PORT=5555

python deployment/model_server/tools/debug_infersystem_client.py \
    --server "127.0.0.1:${PORT}" \
    --camera_order "$CAMERA_ORDER" \
    --state_dim 14 \
    --prompt "wipe_the_white_plastic_board" \
    --stat_key "$STAT_KEY"
