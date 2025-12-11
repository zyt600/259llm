#!/bin/bash

cd /root/newtest
source venv/bin/activate

# 设置CUDA环境变量，供FlashInfer JIT编译使用
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH

# 创建FlashInfer期望的nvcc符号链接（如果不存在或指向旧版本）
NVIDIA_BIN_DIR="/root/newtest/venv/lib/python3.10/site-packages/nvidia/bin"
mkdir -p "$NVIDIA_BIN_DIR"
ln -sf /usr/local/cuda-12.4/bin/nvcc "$NVIDIA_BIN_DIR/nvcc"
echo "nvcc版本: $($NVIDIA_BIN_DIR/nvcc --version | grep release)"

# 定义模型列表
MODELS=(
    "google/gemma-3-270m-it"
    "google/gemma-3-1b-it"
    "google/gemma-2b-GGUF"

    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B-Instruct"

    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

    "h2oai/h2o-danube2-1.8b-chat-GGUF"

    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 上下文太短，不能qmsum
    "/root/newtest/training/training_output2/qwen3-0.6b-sft-q8_0.gguf"
    # "unsloth/gpt-oss-20b"
    # "unsloth/Phi-4-mini-reasoning"
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-4B-Instruct-2507-FP8"
    # "Qwen/Qwen3-4B-Thinking-2507"
    # "Qwen/Qwen3-4B-AWQ"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-1.7B-FP8"
    "Qwen/Qwen3-1.7B-GPTQ-Int8"
    # "Qwen/Qwen3-4B-Base"
    # "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
    # "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    # "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
    # "unsloth/Qwen3-14B-GGUF"
    "unsloth/Qwen3-0.6B-GGUF"
    "unsloth/Qwen3-1.7B"
    # "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    # "unsloth/Qwen3-4B-Instruct-2507-GGUF" # 很差
    "unsloth/Llama-3.2-3B-Instruct-GGUF"
    # "unsloth/Phi-4-mini-reasoning-GGUF"
    # "unsloth/gpt-oss-20b-GGUF"
    # "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    "unsloth/Qwen3-1.7B-GGUF"
    "unsloth/Qwen3-1.7B-Q4_0.gguf"
    "unsloth/Qwen3-1.7B-UD-Q8_K_XL.gguf"

    "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF"
)

# 定义温度列表
# TEMPERATURES=(0 0.05 0.1 0.2 0.4 0.5 0.7 0.9 1.0)
# TEMPERATURES=(0 0.05 0.1 0.2)
TEMPERATURES=(0)

# 总实验数
TOTAL=$((${#MODELS[@]} * ${#TEMPERATURES[@]}))
CURRENT=0

echo "============================================================"
echo "开始批量实验"
echo "模型数量: ${#MODELS[@]}"
echo "温度数量: ${#TEMPERATURES[@]}"
echo "总实验数: $TOTAL"
echo "============================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 遍历所有组合
SKIPPED=0
for TEMP in "${TEMPERATURES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # 生成结果文件名（与evaluator.py中的命名方式一致）
        MODEL_CLEAN=$(echo "$MODEL" | sed 's/\//_/g')
        
        # 将温度转换为Python/evaluator保存时的格式（浮点数格式）
        # 例如: 0 -> 0.0, 1 -> 1.0, 0.05 -> 0.05
        TEMP_FORMATTED=$(python3 -c "print(float($TEMP))")
        
        RESULT_FILE_QMSUM="./results/${MODEL_CLEAN}_qmsum_temp${TEMP_FORMATTED}_results.json"
        RESULT_FILE_TRUTHFULQA="./results/${MODEL_CLEAN}_truthfulqa_temp${TEMP_FORMATTED}_results.json"
        
        # 检查两个数据集的结果文件是否都存在
        if [ -f "$RESULT_FILE_QMSUM" ] && [ -f "$RESULT_FILE_TRUTHFULQA" ]; then
            echo ""
            echo "============================================================"
            echo "实验 $CURRENT/$TOTAL - 跳过（结果已存在）"
            echo "模型: $MODEL"
            echo "温度: $TEMP"
            echo "============================================================"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        echo ""
        echo "============================================================"
        echo "实验 $CURRENT/$TOTAL"
        echo "模型: $MODEL"
        echo "温度: $TEMP"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        
        python main.py --model_name "$MODEL" --temperature "$TEMP" --gpu 2,3
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "警告: 实验 $CURRENT 失败 (退出码: $EXIT_CODE)"
        fi
        
        echo ""
        echo "实验 $CURRENT/$TOTAL 完成"
        echo ""
    done
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "所有实验完成!"
echo "总实验数: $TOTAL"
echo "跳过数量: $SKIPPED"
echo "实际运行: $((TOTAL - SKIPPED))"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "结果保存在: ./results/"
echo "============================================================"

# 列出所有结果文件
echo ""
echo "生成的结果文件:"
ls -la ./results/*.json 2>/dev/null | tail -30

