#!/usr/bin/env bash
# Start the T-Rex GPU inference server for a full-trained or merged-LoRA checkpoint.
set -euo pipefail

: "${TREX_ROOT:?Set TREX_ROOT to the patched T-Rex checkout}"
: "${CHECKPOINT:?Set CHECKPOINT to a full-trained or merged-LoRA checkpoint directory}"

PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}
BIND_HOST=${BIND_HOST:-127.0.0.1}
PORT=${PORT:-5678}

"${PYTHON_BIN}" "$(dirname "$0")/check_deploy_checkpoint.py" \
  --checkpoint "${CHECKPOINT}"

export PYTHONPATH="${TREX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
exec "${PYTHON_BIN}" "${TREX_ROOT}/scripts/test.py" \
  --checkpoint_path "${CHECKPOINT}" \
  --dataset_name rlbench \
  --action_dim 62 --action_chunk 16 \
  --use_robot_state 0 --use_tactile_vec 1 --use_tactile_deform 1 \
  --image_size 384 288 \
  --cuda 0 --bind_host "${BIND_HOST}" --port "${PORT}"
