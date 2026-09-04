#!/usr/bin/env bash
EPISODE_ID=${1:?Usage: bash local_replay.sh EPISODE_ID [extra args]}
shift

export DATA_ROOT=${DATA_ROOT:-playground/Datasets/neoteai/lerobot/wipe_the_white_plastic_board_Aloha}
export PORT=${PORT:-5555}
export CHUNK_SIZE=${CHUNK_SIZE:-48}
export LOG_ACTIONS_EVERY=${LOG_ACTIONS_EVERY:-1}

python -m deployment.model_server.replay_infersystem \
    --data_root "$DATA_ROOT" \
    --episode_id "$EPISODE_ID" \
    --bind "tcp://*:${PORT}" \
    --chunk_size "$CHUNK_SIZE" \
    --log_actions_every "$LOG_ACTIONS_EVERY" \
    "$@"
