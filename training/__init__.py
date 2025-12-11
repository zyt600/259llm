"""
LongBench SFT训练模块

该模块提供使用LongBench数据集对大语言模型进行监督微调(SFT)的功能。

主要组件:
- train_args: 训练参数解析
- longbench_loader: LongBench数据集加载器
- trainer: SFT训练器
- train: 训练主入口

使用示例:
    # 命令行运行
    python -m training.train --model_name Qwen/Qwen2-7B-Instruct --use_lora
    
    # 或直接进入training目录运行
    cd training && python train.py --model_name Qwen/Qwen2-7B-Instruct --use_lora
"""

from .train_args import parse_train_args, print_train_args
from .longbench_loader import (
    load_all_longbench_subsets,
    load_longbench_subset,
    create_sft_dataset,
    ALL_SUBSETS,
    LONGBENCH_SUBSETS
)
from .trainer import SFTTrainer, PeriodicEvalCallback

__all__ = [
    "parse_train_args",
    "print_train_args",
    "load_all_longbench_subsets",
    "load_longbench_subset",
    "create_sft_dataset",
    "ALL_SUBSETS",
    "LONGBENCH_SUBSETS",
    "SFTTrainer",
    "PeriodicEvalCallback",
]

