"""
训练参数解析模块
"""
import argparse
import os
import torch


def parse_train_args():
    """
    解析训练命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="LongBench SFT训练工具 - 使用LongBench数据集对模型进行监督微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本训练
  python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1
  
  # 指定训练参数
  python train.py --model_name meta-llama/Llama-2-7b-chat-hf --learning_rate 1e-5 --num_epochs 3
  
  # 每500步评估一次
  python train.py --model_name Qwen/Qwen2-7B-Instruct --eval_steps 500
        """
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face上的模型名称，例如: Qwen/Qwen2-7B-Instruct"
    )
    
    # GPU配置
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="使用的GPU编号，逗号分隔。例如: '0,1,2'。默认使用所有可用GPU"
    )
    
    # 训练参数
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="学习率 (默认: 5e-6，QLoRA推荐较低学习率)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="最大训练步数，-1表示不限制 (默认: -1)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="每个设备的批处理大小 (默认: 4)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数 (默认: 8)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="最大序列长度 (默认: 4096)"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="预热比例 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="权重衰减 (默认: 0.01)"
    )
    
    # 评估参数
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="每隔多少步进行一次评估 (默认: 500)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="每隔多少步保存一次模型 (默认: 500)"
    )
    
    # eval_max_samples 已移除 - 每次评估都使用完整测试集
    
    # 数据集参数
    parser.add_argument(
        "--train_datasets",
        type=str,
        default="all",
        help="训练使用的数据集，逗号分隔或'all'使用全部 (默认: all)"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="训练使用的最大样本数 (默认: None使用全部)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_output",
        help="模型和日志输出目录 (默认: ./training_output)"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="每隔多少步记录一次日志 (默认: 10)"
    )
    
    # LoRA参数 (强制使用LoRA以节省显存)
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA秩 (默认: 16)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (默认: 32)"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (默认: 0.05)"
    )
    
    # 混合精度训练
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="使用bf16混合精度训练 (默认: True)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="使用fp16混合精度训练 (默认: False)"
    )
    
    # 其他参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    
    args = parser.parse_args()
    
    # 后处理参数
    args = _process_train_args(args)
    
    return args


def _process_train_args(args):
    """
    处理和验证训练参数
    """
    # 处理GPU参数
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
    else:
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            args.gpu_ids = []
    
    # 处理数据集参数
    if args.train_datasets.lower() == "all":
        args.train_dataset_list = None  # None表示使用全部
    else:
        args.train_dataset_list = [x.strip() for x in args.train_datasets.split(",")]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 强制使用 LoRA
    args.use_lora = True
    
    return args


def print_train_args(args):
    """
    打印训练参数配置
    """
    print("\n" + "=" * 70)
    print("训练配置参数")
    print("=" * 70)
    
    print(f"\n【模型配置】")
    print(f"  模型名称: {args.model_name}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print(f"  训练方式: 纯 LoRA")
    print(f"  LoRA秩: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    
    print(f"\n【训练参数】")
    print(f"  学习率: {args.learning_rate}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"  最大序列长度: {args.max_length}")
    print(f"  预热比例: {args.warmup_ratio}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  BF16: {args.bf16}, FP16: {args.fp16}")
    
    print(f"\n【评估配置】")
    print(f"  评估间隔步数: {args.eval_steps}")
    print(f"  保存间隔步数: {args.save_steps}")
    print(f"  评估数据集: TruthfulQA + QMSum (完整测试集)")
    print(f"  综合分数: accuracy + rougeL + avg_max_score")
    
    print(f"\n【数据集配置】")
    print(f"  训练数据集: {args.train_datasets}")
    print(f"  最大训练样本数: {args.max_train_samples if args.max_train_samples else '全部'}")
    
    print(f"\n【输出配置】")
    print(f"  输出目录: {args.output_dir}")
    print(f"  日志间隔步数: {args.logging_steps}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    args = parse_train_args()
    print_train_args(args)

