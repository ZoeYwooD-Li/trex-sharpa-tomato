#!/usr/bin/env bash
# Eight-GPU full-parameter post-training entry point. Defaults to a 20-batch smoke run.
# Install both install_adapter.py and install_lora.py first; rank 0 disables LoRA.
set -euo pipefail

: "${TREX_ROOT:?Set the patched T-Rex checkout path}"
: "${PREPARED_ROOT:?Set the prepared Sharpa cache path}"
: "${ORIGIN_MODEL_PATH:?Set the Qwen3-VL-2B-Instruct directory}"
: "${RESUME_CHECKPOINT:?Set the official T-Rex midtrain checkpoint directory}"
: "${OUTPUT_DIR:?Set a new output directory}"

export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${TREX_ROOT}/config/sft_qwen.yaml}"
export TRAIN_BSZ="${TRAIN_BSZ:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export RUN_NAME="${RUN_NAME:-sharpa_full_smoke}"

if [[ "$GRAD_ACCUM" != 1 ]]; then
  echo "This DeepSpeed training entry point currently requires GRAD_ACCUM=1." >&2
  exit 2
fi

bash "$(dirname "$0")/train_sharpa.sh" \
  --lora_rank 0 \
  --n_epochs "${N_EPOCHS:-1}" \
  --max_train_batches "${MAX_TRAIN_BATCHES:-20}" \
  --save_steps "${SAVE_STEPS:-0}" \
  --max_ckpts "${MAX_CKPTS:-5}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --warmup_rates "${WARMUP_RATES:-0.02}" \
  --train_bsz_per_gpu "$TRAIN_BSZ" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --val_freq "${VAL_FREQ:-10}" \
  --max_val_batches "${MAX_VAL_BATCHES:-2}" \
  "$@"
