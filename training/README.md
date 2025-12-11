# LongBench SFT Training Module

A complete training framework for Supervised Fine-Tuning (SFT) of large language models using the LongBench dataset.

## Features

- ✅ Load all 21 LongBench subsets for training
- ✅ Pure LoRA fine-tuning (memory efficient)
- ✅ Periodic evaluation on full test sets (TruthfulQA + QMSum)
- ✅ Support for any HuggingFace model
- ✅ Automatic checkpoint saving with best model tracking
- ✅ Multi-GPU training support
- ✅ SGLang-based inference for evaluation

## LongBench Dataset

Contains 21 subsets covering various long-context tasks:

| Category | Subsets |
|----------|---------|
| Single-Doc QA | narrativeqa, qasper, multifieldqa_en, multifieldqa_zh |
| Multi-Doc QA | hotpotqa, 2wikimqa, musique |
| Summarization | gov_report, qmsum, multi_news, vcsum |
| Few-shot Learning | trec, triviaqa, samsum, lsht |
| Synthetic Tasks | passage_count, passage_retrieval_en, passage_retrieval_zh |
| Code | lcc, repobench-p |

## Installation

```bash
pip install -r requirements.txt

# Optional: Install Flash Attention for acceleration (recommended)
pip install flash-attn --no-build-isolation
```

## Quick Start

### Using the Shell Script (Recommended)

The easiest way to start training is using the provided shell script:

```bash
cd training
./run_train.sh
```

You can customize the training by setting environment variables:

```bash
MODEL_NAME="Qwen/Qwen3-0.6B" \
GPU="0,1" \
MAX_STEPS=200 \
EVAL_STEPS=20 \
OUTPUT_DIR="./my_output" \
./run_train.sh
```

### Using Python Directly

```bash
python train.py \
    --model_name Qwen/Qwen3-0.6B \
    --gpu 0,1 \
    --num_epochs 3 \
    --eval_steps 50 \
    --output_dir ./training_output
```

## Command Line Arguments

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | (required) | HuggingFace model name or local path |
| `--gpu` | all available | GPU IDs to use, comma-separated (e.g., "0,1,2") |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate` | 5e-6 | Learning rate (lower values recommended for LoRA) |
| `--num_epochs` | 3 | Number of training epochs |
| `--max_steps` | -1 | Maximum training steps (-1 for unlimited) |
| `--batch_size` | 4 | Per-device batch size |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `--max_length` | 4096 | Maximum sequence length |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--weight_decay` | 0.01 | Weight decay |
| `--bf16` | True | Use bf16 mixed precision |
| `--fp16` | False | Use fp16 mixed precision |
| `--seed` | 42 | Random seed |

### LoRA Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |

LoRA is applied to the following modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### Evaluation & Saving

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval_steps` | 500 | Evaluate every N steps (uses full test set) |
| `--save_steps` | 500 | Save checkpoint every N steps |
| `--logging_steps` | 10 | Log every N steps |

### Dataset Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_datasets` | "all" | Training datasets, comma-separated or "all" |
| `--max_train_samples` | None | Maximum training samples (None = use all) |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | ./training_output | Output directory for models and logs |
| `--resume_from_checkpoint` | None | Resume training from checkpoint path |

## Training Examples

### Basic LoRA Training

```bash
python train.py \
    --model_name Qwen/Qwen3-0.6B \
    --gpu 0 \
    --num_epochs 3 \
    --eval_steps 100
```

### Training with Specific Datasets

```bash
python train.py \
    --model_name Qwen/Qwen3-0.6B \
    --gpu 0,1 \
    --train_datasets qmsum,trivia_qa \
    --max_steps 200 \
    --eval_steps 50
```

### Quick Test Run

```bash
python train.py \
    --model_name Qwen/Qwen3-0.6B \
    --gpu 0 \
    --max_train_samples 100 \
    --max_steps 20 \
    --eval_steps 10 \
    --num_epochs 1
```

### Full Training with Custom LoRA Config

```bash
python train.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --gpu 0,1,2,3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 8192 \
    --eval_steps 500 \
    --output_dir ./qwen2_7b_sft
```

## Output Structure

After training, the output directory contains:

```
training_output/
├── final_model/              # Final trained LoRA adapter
├── best_adapter/             # Best performing adapter (by combined score)
├── adapter_step_100/         # Adapter saved at step 100
├── adapter_step_200/         # Adapter saved at step 200
├── checkpoint-*/             # Training checkpoints
├── eval_step_100/            # Evaluation results at step 100
│   ├── *_qmsum_*_results.json
│   ├── *_qmsum_*_predictions.jsonl
│   ├── *_truthfulqa_*_results.json
│   └── *_truthfulqa_*_predictions.jsonl
├── eval_history.json         # Complete evaluation history
└── final_eval_results.json   # Final evaluation results
```

### eval_history.json Format

```json
{
  "best_step": 100,
  "best_score": 1.25,
  "history": [
    {
      "step": 50,
      "timestamp": "2025-01-01T12:00:00",
      "combined_score": 1.15,
      "results": {
        "qmsum": {"rougeL": 0.35, "num_samples": 100, ...},
        "truthfulqa": {"accuracy": 0.45, "avg_max_score": 0.45, ...}
      }
    },
    ...
  ]
}
```

## Evaluation Metrics

During training, the model is evaluated on **full test sets** of:

- **QMSum**: ROUGE-L score (meeting summarization task)
- **TruthfulQA**: Accuracy + Avg Max Score (truthfulness evaluation)

### Combined Score Calculation

```
Combined Score = TruthfulQA_accuracy + QMSum_rougeL + TruthfulQA_avg_max_score
```

The training process automatically tracks the best model based on this combined score.

## Shell Script Configuration

The `run_train.sh` script supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | Qwen/Qwen3-0.6B | Model to train |
| `GPU` | 2,3 | GPUs to use |
| `NUM_EPOCHS` | 3 | Training epochs |
| `MAX_STEPS` | 200 | Maximum training steps |
| `EVAL_STEPS` | 20 | Evaluation interval |
| `BATCH_SIZE` | 4 | Batch size |
| `GRAD_ACCUM` | 8 | Gradient accumulation |
| `MAX_LENGTH` | 10000 | Maximum sequence length |
| `OUTPUT_DIR` | ./training_output9 | Output directory |
| `TRAIN_DATASETS` | qmsum,trivia_qa | Datasets to use |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha |

Example:

```bash
GPU="0,1,2,3" MAX_STEPS=500 EVAL_STEPS=100 ./run_train.sh
```

## Memory Requirements

| Model Size | LoRA Training | Recommended GPUs |
|------------|---------------|------------------|
| 0.5B-1B | ~8GB | 1x RTX 3090/4090 |
| 7B | ~16GB | 1x A100 40GB |
| 13B | ~24GB | 1x A100 80GB |
| 70B | ~80GB | 4x A100 80GB |

Tips for reducing memory usage:
- Decrease `--batch_size`
- Decrease `--max_length`
- Increase `--gradient_accumulation_steps`
- Use `--bf16` (default) for efficient training

## FAQ

**Q: CUDA out of memory error?**
- Reduce `--batch_size` (e.g., from 4 to 2 or 1)
- Reduce `--max_length` (e.g., from 4096 to 2048)
- Increase `--gradient_accumulation_steps` to maintain effective batch size

**Q: How to train on specific datasets only?**
```bash
python train.py --train_datasets qmsum,narrativeqa,hotpotqa ...
```

**Q: How to resume from a checkpoint?**
```bash
python train.py --resume_from_checkpoint ./training_output/checkpoint-500 ...
```

**Q: How to use the trained adapter for inference?**

The trained adapter is saved in `final_model/` or `best_adapter/`. You can load it using:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./training_output/best_adapter")

# Or merge for faster inference
model = model.merge_and_unload()
```

**Q: How to merge LoRA adapter with base model?**

Use the provided merge script:

```bash
python merge_and_export.py \
    --base_model Qwen/Qwen3-0.6B \
    --adapter_path ./training_output/best_adapter \
    --output_path ./merged_model
```

## File Descriptions

| File | Description |
|------|-------------|
| `train.py` | Main training entry point |
| `trainer.py` | SFTTrainer class with evaluation callbacks |
| `train_args.py` | Command-line argument parsing |
| `longbench_loader.py` | LongBench dataset loading utilities |
| `run_train.sh` | Convenient shell script for training |
| `merge_and_export.py` | Merge LoRA adapter with base model |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 configuration |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 configuration |

## License

This project is for research purposes. Please comply with the licenses of the models and datasets you use.
