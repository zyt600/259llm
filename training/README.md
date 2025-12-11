# LongBench SFT训练模块

使用LongBench数据集对大语言模型进行监督微调(SFT)的完整训练框架。

## 功能特点

- ✅ 支持加载LongBench所有21个子集用于训练
- ✅ 支持全参数微调和LoRA高效微调
- ✅ 定期评估：每隔N步自动运行全量测试
- ✅ 支持从HuggingFace加载任意模型
- ✅ 自动保存检查点和评估历史
- ✅ 支持多GPU训练

## LongBench数据集

包含21个子集，覆盖多种长文本任务：

| 类别 | 子集 |
|------|------|
| 单文档QA | narrativeqa, qasper, multifieldqa_en, multifieldqa_zh |
| 多文档QA | hotpotqa, 2wikimqa, musique |
| 摘要 | gov_report, qmsum, multi_news, vcsum |
| Few-shot学习 | trec, triviaqa, samsum, lsht |
| 合成任务 | passage_count, passage_retrieval_en, passage_retrieval_zh |
| 代码 | lcc, repobench-p |

## 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装Flash Attention加速（推荐）
pip install flash-attn --no-build-isolation
```

## 使用方法

### 快速开始（LoRA微调，推荐）

```bash
python train.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --gpu 0 \
    --use_lora \
    --num_epochs 3 \
    --eval_steps 500
```

### 全参数微调（需要更多显存）

```bash
python train.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --gpu 0,1,2,3 \
    --num_epochs 3 \
    --eval_steps 500
```

### 快速测试

```bash
python train.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --gpu 0 \
    --use_lora \
    --max_train_samples 100 \
    --eval_steps 50 \
    --num_epochs 1
```

## 参数说明

### 模型参数
- `--model_name`: HuggingFace模型名称（必需）
- `--gpu`: 使用的GPU编号，逗号分隔（默认：使用所有GPU）
- `--use_lora`: 使用LoRA微调
- `--lora_r`: LoRA秩（默认：16）
- `--lora_alpha`: LoRA alpha（默认：32）

### 训练参数
- `--learning_rate`: 学习率（默认：2e-5）
- `--num_epochs`: 训练轮数（默认：3）
- `--batch_size`: 批处理大小（默认：4）
- `--gradient_accumulation_steps`: 梯度累积步数（默认：8）
- `--max_length`: 最大序列长度（默认：4096）
- `--warmup_ratio`: 预热比例（默认：0.1）

### 评估参数
- `--eval_steps`: 每隔多少步评估一次（默认：500）
- 每次评估都使用 **完整测试集**（TruthfulQA + QMSum）

### 数据参数
- `--train_datasets`: 训练数据集，逗号分隔或"all"（默认：all）
- `--max_train_samples`: 最大训练样本数（默认：全部）

### 输出参数
- `--output_dir`: 输出目录（默认：./training_output）
- `--save_steps`: 检查点保存间隔（默认：500）

## 输出结构

训练完成后，输出目录包含：

```
training_output/
├── final_model/          # 最终训练好的模型
├── checkpoint-500/       # 检查点
├── checkpoint-1000/
├── eval_step_500/        # 每步评估结果（完整测试集）
│   ├── *_qmsum_results.json
│   ├── *_qmsum_predictions.jsonl
│   ├── *_truthfulqa_results.json
│   └── *_truthfulqa_predictions.jsonl
├── eval_history.json     # 评估历史（含综合分数和最佳模型信息）
└── final_eval_results.json  # 最终评估结果
```

### eval_history.json 格式

```json
{
  "best_step": 1000,
  "best_score": 1.25,
  "history": [
    {
      "step": 500,
      "combined_score": 1.15,
      "results": { "qmsum": {...}, "truthfulqa": {...} }
    },
    ...
  ]
}
```

## 评估指标

训练过程中和训练结束后，会在以下数据集上进行 **完整测试集** 评估：

- **QMSum**: ROUGE-L分数（会议摘要）
- **TruthfulQA**: Accuracy + Avg Max Score（真实性问答）

### 综合分数计算

```
综合分数 = TruthfulQA_accuracy + QMSum_rougeL + TruthfulQA_avg_max_score
```

评估历史中会记录每个检查点的综合分数，并自动追踪最佳模型。

## 示例：完整训练流程

```bash
# 1. 进入训练目录
cd training

# 2. LoRA微调Qwen2-7B
python train.py \
    --model_name Qwen/Qwen2-7B-Instruct \
    --gpu 0,1 \
    --use_lora \
    --lora_r 32 \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_length 8192 \
    --eval_steps 500 \
    --save_steps 500 \
    --output_dir ./qwen2_7b_sft

# 3. 查看评估历史
cat ./qwen2_7b_sft/eval_history.json

# 4. 使用训练好的模型进行推理
# 模型保存在 ./qwen2_7b_sft/final_model/
```

## 注意事项

1. **显存需求**：
   - 7B模型LoRA微调：约16GB显存
   - 7B模型全参数微调：约80GB显存（需要多卡）
   - 建议使用LoRA微调以节省显存

2. **训练时间**：
   - 取决于数据量、模型大小和硬件配置
   - 典型情况：7B模型在4xA100上，使用全部LongBench数据训练3轮约需10-20小时

3. **数据长度**：
   - LongBench包含长文本数据，建议设置较大的max_length（4096-8192）
   - 长序列会增加显存消耗，可以通过减小batch_size来平衡

## 常见问题

**Q: CUDA out of memory怎么办？**
- 使用`--use_lora`启用LoRA微调
- 减小`--batch_size`
- 减小`--max_length`
- 增大`--gradient_accumulation_steps`

**Q: 如何只使用部分数据集训练？**
```bash
python train.py --train_datasets qmsum,narrativeqa,hotpotqa ...
```

**Q: 如何从检查点恢复训练？**
```bash
python train.py --resume_from_checkpoint ./training_output/checkpoint-500 ...
```

