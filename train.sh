#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=2
#SBATCH --partition=gpu_a100
#SBATCH --time=96:00:00  # 调整时间限制
#SBATCH --mem=84G        # 可调整根据节点资源
#SBATCH --exclusive      # 如果需要独占节点，取消注释
#SBATCH --job-name=qwen
#SBATCH -o ./log/a100_test.out # 指定输出文件

module load CUDA/12.4.0
source /home/npu/miniconda3/bin/activate qwen

MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=2                            # Automatically detects available GPUs

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"    # [ModelArguments] Pretrained model path
OUTPUT_DIR="./checkpoints"                  # Directory for saving checkpoints
CACHE_DIR="./cache"                         # [TrainingArguments] Cache directory for models
DATASETS="samm_data"                        # [DataArguments] Dataset with sampling rate

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    qwenvl/train/train_qwen.py \
    --model_name_or_path "$MODEL_PATH" \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-7 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 4096 \
    --data_flatten True \
    --data_packing True \
    --max_pixels 576*28*28 \
    --min_pixels 16*28*28 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --deepspeed zero3.json
