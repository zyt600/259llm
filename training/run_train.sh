#!/bin/bash
# LongBench SFT Training Script - Pure LoRA

# Default parameters
# MODEL_NAME=${MODEL_NAME:-"google/gemma-3-270m-it"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
GPU=${GPU:-"2,3"}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-200} 
EVAL_STEPS=${EVAL_STEPS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-"./training_output9"}  # New directory
TRAIN_DATASETS=${TRAIN_DATASETS:-"qmsum,trivia_qa"}  # qmsum+trivia_qa related to evaluation tasks
# TRAIN_DATASETS=${TRAIN_DATASETS:-"all"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU

# Disable tokenizer parallelism to avoid warnings and potential issues after fork
export TOKENIZERS_PARALLELISM=false

# Print configuration
echo "=========================================="
echo "LongBench SFT Training Config (Pure LoRA)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPU: $GPU"
echo "Training method: Pure LoRA (r=$LORA_R, alpha=$LORA_ALPHA)"
echo "Epochs: $NUM_EPOCHS"
echo "Max steps: $MAX_STEPS"
echo "Eval interval: $EVAL_STEPS steps (full test set)"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Max length: $MAX_LENGTH"
echo "Output directory: $OUTPUT_DIR"
echo "Training datasets: $TRAIN_DATASETS"
echo "=========================================="
echo "Evaluation: TruthfulQA + QMSum full test set"
echo "Combined score = accuracy + rougeL + avg_max_score"
echo "=========================================="

# Build command
CMD="python3 train.py \
    --model_name $MODEL_NAME \
    --gpu $GPU \
    --num_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --eval_steps $EVAL_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --train_datasets $TRAIN_DATASETS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA"

# Execute training
echo ""
echo "Executing command: $CMD"
echo ""
eval $CMD
