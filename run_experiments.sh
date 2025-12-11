#!/bin/bash

cd /root/newtest
source venv/bin/activate

# Set CUDA environment variables for FlashInfer JIT compilation
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH

# Create nvcc symlink that FlashInfer expects (if not exists or points to old version)
NVIDIA_BIN_DIR="/root/newtest/venv/lib/python3.10/site-packages/nvidia/bin"
mkdir -p "$NVIDIA_BIN_DIR"
ln -sf /usr/local/cuda-12.4/bin/nvcc "$NVIDIA_BIN_DIR/nvcc"
echo "nvcc version: $($NVIDIA_BIN_DIR/nvcc --version | grep release)"

# Define model list
MODELS=(
    "google/gemma-3-270m-it"
    "google/gemma-3-1b-it"

    "unsloth/gemma-3-1b-it-GGUF"
    # "google/gemma-2b-GGUF"

    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.2-3B-Instruct"

    # "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

    # "h2oai/h2o-danube2-1.8b-chat-GGUF"

    # # "TinyLlama/TinyLlama-1.1B-Chat-v1.0" context too short for qmsum
    # "/root/newtest/training/training_output2/qwen3-0.6b-sft-q8_0.gguf"
    # # "unsloth/gpt-oss-20b"
    # # "unsloth/Phi-4-mini-reasoning"
    # # "Qwen/Qwen3-4B-Instruct-2507"
    # # "Qwen/Qwen3-4B-Instruct-2507-FP8"
    # # "Qwen/Qwen3-4B-Thinking-2507"
    # # "Qwen/Qwen3-4B-AWQ"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen3-1.7B-FP8"
    # "Qwen/Qwen3-1.7B-GPTQ-Int8"
    # # "Qwen/Qwen3-4B-Base"
    # # "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
    # # "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF"
    # # "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF"
    # # "unsloth/Qwen3-14B-GGUF"
    # "unsloth/Qwen3-0.6B-GGUF"
    # "unsloth/Qwen3-1.7B"
    # # "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    # # "unsloth/Qwen3-4B-Instruct-2507-GGUF" # very bad
    # "unsloth/Llama-3.2-3B-Instruct-GGUF"
    # # "unsloth/Phi-4-mini-reasoning-GGUF"
    # # "unsloth/gpt-oss-20b-GGUF"
    # # "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    # "unsloth/Qwen3-1.7B-GGUF"
    # "unsloth/Qwen3-1.7B-Q4_0.gguf"
    # "unsloth/Qwen3-1.7B-UD-Q8_K_XL.gguf"

    # "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF"
)

# Define temperature list
# TEMPERATURES=(0 0.05 0.1 0.2 0.4 0.5 0.7 0.9 1.0)
TEMPERATURES=(0.5 0.7 0.9 1.0)
# TEMPERATURES=(0)

# Total experiments
TOTAL=$((${#MODELS[@]} * ${#TEMPERATURES[@]}))
CURRENT=0

echo "============================================================"
echo "Starting batch experiments"
echo "Model count: ${#MODELS[@]}"
echo "Temperature count: ${#TEMPERATURES[@]}"
echo "Total experiments: $TOTAL"
echo "============================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Iterate through all combinations
SKIPPED=0
for TEMP in "${TEMPERATURES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # Generate result filename (consistent with evaluator.py naming)
        MODEL_CLEAN=$(echo "$MODEL" | sed 's/\//_/g')
        
        # Convert temperature to Python/evaluator save format (float format)
        # e.g.: 0 -> 0.0, 1 -> 1.0, 0.05 -> 0.05
        TEMP_FORMATTED=$(python3 -c "print(float($TEMP))")
        
        RESULT_FILE_QMSUM="./results/${MODEL_CLEAN}_qmsum_temp${TEMP_FORMATTED}_results.json"
        RESULT_FILE_TRUTHFULQA="./results/${MODEL_CLEAN}_truthfulqa_temp${TEMP_FORMATTED}_results.json"
        
        # Check if result files for both datasets exist
        if [ -f "$RESULT_FILE_QMSUM" ] && [ -f "$RESULT_FILE_TRUTHFULQA" ]; then
            echo ""
            echo "============================================================"
            echo "Experiment $CURRENT/$TOTAL - Skipping (results exist)"
            echo "Model: $MODEL"
            echo "Temperature: $TEMP"
            echo "============================================================"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        echo ""
        echo "============================================================"
        echo "Experiment $CURRENT/$TOTAL"
        echo "Model: $MODEL"
        echo "Temperature: $TEMP"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
        echo ""
        
        python main.py --model_name "$MODEL" --temperature "$TEMP" --gpu 4,5,6,7
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Warning: Experiment $CURRENT failed (exit code: $EXIT_CODE)"
        fi
        
        echo ""
        echo "Experiment $CURRENT/$TOTAL complete"
        echo ""
    done
done

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "All experiments complete!"
echo "Total experiments: $TOTAL"
echo "Skipped: $SKIPPED"
echo "Actually run: $((TOTAL - SKIPPED))"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Results saved to: ./results/"
echo "============================================================"

# List all result files
echo ""
echo "Generated result files:"
ls -la ./results/*.json 2>/dev/null | tail -30
