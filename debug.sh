#!/bin/bash


WANDB_MODE=disabled CUDA_VISIBLE_DEVICES='' accelerate launch --mixed_precision="fp16"  \
    image_gen/train/run_text_to_image.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --use_ema \
  --resolution=512 \
  --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-pokemon-model"
