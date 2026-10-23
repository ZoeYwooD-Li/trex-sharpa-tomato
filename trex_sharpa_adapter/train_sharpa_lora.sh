#!/usr/bin/env bash
# Single-GPU LoRA entry point. Run after install_adapter.py and install_lora.py.
set -euo pipefail
: "${TREX_ROOT:?Set the patched T-Rex checkout path}"
: "${PREPARED_ROOT:?Set the prepared Sharpa cache path}"
: "${ORIGIN_MODEL_PATH:?Set the Qwen3-VL-2B-Instruct directory}"
: "${RESUME_CHECKPOINT:?Set the midtrain checkpoint directory}"
: "${OUTPUT_DIR:?Set the output directory}"
export PYTHONPATH="${TREX_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "$TREX_ROOT"
"${PYTHON:-python}" scripts/train.py \
  --model_path "$ORIGIN_MODEL_PATH" --data_format sharpa --lerobot_root "$PREPARED_ROOT" \
  --resume_checkpoint "$RESUME_CHECKPOINT" --resume_source midtrain \
  --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR" \
  --experiment_name trex_sharpa_tomato --run_name "${RUN_NAME:-sharpa_lora}" \
  --action_dim 62 --action_chunk 16 --use_robot_state 0 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --deform_encoder_ckpt "${DEFORM_ENCODER_PATH:-}" --tactile_intermediate_size 1536 \
  --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 --flare_loss_weight 0.5 \
  --flare_frame_stride 4 --flare_layer_index -1 --image_size 384 288 \
  --lora_rank "${LORA_RANK:-8}" --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" --lora_gradient_checkpointing 1 \
  --n_epochs "${N_EPOCHS:-1}" --max_train_batches "${MAX_TRAIN_BATCHES:-20}" \
  --save_freq "${SAVE_FREQ:-1}" --learning_rate "${LEARNING_RATE:-1e-4}" \
  --min_lr_ratio 0 --weight_decay 0 \
  --train_bsz_per_gpu "${TRAIN_BSZ:-1}" --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --seed 42 --val_ratio 0.05 --val_freq "${VAL_FREQ:-10}" \
  --max_val_batches "${MAX_VAL_BATCHES:-2}" "$@"
