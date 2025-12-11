"""
LongBench SFT Training Module

This module provides functionality for supervised fine-tuning (SFT) of large language models using LongBench dataset.

Main components:
- train_args: Training argument parsing
- longbench_loader: LongBench dataset loader
- trainer: SFT trainer
- train: Training main entry

Usage examples:
    # Run from command line
    python -m training.train --model_name Qwen/Qwen2-7B-Instruct --use_lora
    
    # Or run directly from training directory
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
