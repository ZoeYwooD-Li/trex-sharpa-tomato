#!/usr/bin/env bash
# Environment/hardware setup is intentionally supplied by the training machine.
set -euo pipefail
: "${TREX_ROOT:?Set the official T-Rex checkout path}"
: "${PREPARED_ROOT:?Set the prepared adapter manifest directory}"
: "${ORIGIN_MODEL_PATH:?Set local Qwen3-VL-2B-Instruct directory}"
: "${RESUME_CHECKPOINT:?Set the local T-Rex MIDTRAIN checkpoint directory}"
: "${OUTPUT_DIR:?Set training output directory}"
: "${NUM_PROCESSES:?Set training process count for your machine}"
: "${ACCELERATE_CONFIG:?Set the DeepSpeed accelerate YAML for this machine}"
export PYTHONPATH="${TREX_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "${TREX_ROOT}"
accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" \
  scripts/train.py \
  --model_path "${ORIGIN_MODEL_PATH}" --data_format sharpa --lerobot_root "${PREPARED_ROOT}" \
  --resume_checkpoint "${RESUME_CHECKPOINT}" --resume_source midtrain \
  --output_dir "${OUTPUT_DIR}" --log_dir "${OUTPUT_DIR}" \
  --experiment_name trex_sharpa_tomato --run_name "${RUN_NAME:-sharpa_tomato_posttrain}" \
  --action_dim 62 --action_chunk 16 --use_robot_state 0 \
  --use_tactile_vec 1 --use_tactile_deform 1 --use_tactile_vqvae 1 \
  --deform_encoder_ckpt "${DEFORM_ENCODER_PATH:-}" --tactile_intermediate_size 1536 \
  --training_stage 2 --tactile_loss_weight 1.0 \
  --cascaded_total_steps 10 --cascaded_split_step 6 --cascaded_tactile_dropout 0.1 --cascaded_loss_weight 1.0 \
  --use_flare 1 --n_flare_tokens_per_frame 4 --n_flare_steps 8 --flare_loss_weight 0.5 \
  --flare_frame_stride 4 --flare_layer_index -1 --image_size 384 288 \
  --n_epochs 100 --save_freq 50 --learning_rate 1e-4 --min_lr_ratio 0 --weight_decay 0 \
  --train_bsz_per_gpu "${TRAIN_BSZ:?Set per-GPU batch size}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" --seed 42 \
  --val_ratio 0.05 --val_freq 500 --max_val_batches 30 "$@"
