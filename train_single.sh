#!/bin/bash

# 单卡训练
export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    train.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --data_path ./data/train.json \
    --image_folder ./data/images \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --output_dir ./checkpoints/qwenvl-finetune-single \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none \
    --bf16 True \
    --tf32 True \
    --attn_implementation flash_attention_2 \
    --save_total_limit 2