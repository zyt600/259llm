#!/bin/bash
# LongBench SFT训练运行脚本 - 使用纯 LoRA

# 默认参数
# MODEL_NAME=${MODEL_NAME:-"google/gemma-3-270m-it"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
GPU=${GPU:-"2,3"}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-3} 
EVAL_STEPS=${EVAL_STEPS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-"./training_output8"}  # 新目录
TRAIN_DATASETS=${TRAIN_DATASETS:-"qmsum"}  # qmsum+trivia_qa 与评估任务相关
# TRAIN_DATASETS=${TRAIN_DATASETS:-"all"}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU

# 禁用tokenizer并行，避免fork后的警告和潜在问题
export TOKENIZERS_PARALLELISM=false

# 打印配置
echo "=========================================="
echo "LongBench SFT训练配置 (纯 LoRA)"
echo "=========================================="
echo "模型: $MODEL_NAME"
echo "GPU: $GPU"
echo "训练方式: 纯 LoRA (r=$LORA_R, alpha=$LORA_ALPHA)"
echo "训练轮数: $NUM_EPOCHS"
echo "最大步数: $MAX_STEPS"
echo "评估间隔: $EVAL_STEPS 步 (完整测试集)"
echo "批处理大小: $BATCH_SIZE"
echo "梯度累积: $GRAD_ACCUM"
echo "最大长度: $MAX_LENGTH"
echo "输出目录: $OUTPUT_DIR"
echo "训练数据集: $TRAIN_DATASETS"
echo "=========================================="
echo "评估: TruthfulQA + QMSum 完整测试集"
echo "综合分数 = accuracy + rougeL + avg_max_score"
echo "=========================================="

# 构建命令
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

# 执行训练
echo ""
echo "执行命令: $CMD"
echo ""
eval $CMD
