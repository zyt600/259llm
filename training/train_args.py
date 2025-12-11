"""
Training argument parsing module
"""
import argparse
import os
import torch


def parse_train_args():
    """
    Parse training command line arguments
    
    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(
        description="LongBench SFT Training Tool - Fine-tune models using LongBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic training
  python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1
  
  # Specify training parameters
  python train.py --model_name meta-llama/Llama-2-7b-chat-hf --learning_rate 1e-5 --num_epochs 3
  
  # Evaluate every 500 steps
  python train.py --model_name Qwen/Qwen2-7B-Instruct --eval_steps 500
        """
    )
    
    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name on Hugging Face, e.g.: Qwen/Qwen2-7B-Instruct"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU IDs to use, comma-separated. e.g.: '0,1,2'. Default: use all available GPUs"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6, lower learning rate recommended for QLoRA)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps, -1 means unlimited (default: -1)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device (default: 4)"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save model every N steps (default: 500)"
    )
    
    # eval_max_samples removed - always use full test set for evaluation
    
    # Dataset parameters
    parser.add_argument(
        "--train_datasets",
        type=str,
        default="all",
        help="Training datasets, comma-separated or 'all' for all (default: all)"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum training samples (default: None, use all)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./training_output",
        help="Output directory for models and logs (default: ./training_output)"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps (default: 10)"
    )
    
    # LoRA parameters (forced to use LoRA to save VRAM)
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    
    # Mixed precision training
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bf16 mixed precision training (default: True)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 mixed precision training (default: False)"
    )
    
    # Other parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Post-process arguments
    args = _process_train_args(args)
    
    return args


def _process_train_args(args):
    """
    Process and validate training arguments
    """
    # Process GPU arguments
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
    else:
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            args.gpu_ids = []
    
    # Process dataset arguments
    if args.train_datasets.lower() == "all":
        args.train_dataset_list = None  # None means use all
    else:
        args.train_dataset_list = [x.strip() for x in args.train_datasets.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Force use LoRA
    args.use_lora = True
    
    return args


def print_train_args(args):
    """
    Print training configuration
    """
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    
    print(f"\n[Model Configuration]")
    print(f"  Model name: {args.model_name}")
    print(f"  GPU IDs: {args.gpu_ids}")
    print(f"  Training method: Pure LoRA")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    
    print(f"\n[Training Parameters]")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  BF16: {args.bf16}, FP16: {args.fp16}")
    
    print(f"\n[Evaluation Configuration]")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Save steps: {args.save_steps}")
    print(f"  Eval datasets: TruthfulQA + QMSum (full test set)")
    print(f"  Combined score: accuracy + rougeL + avg_max_score")
    
    print(f"\n[Dataset Configuration]")
    print(f"  Training datasets: {args.train_datasets}")
    print(f"  Max training samples: {args.max_train_samples if args.max_train_samples else 'All'}")
    
    print(f"\n[Output Configuration]")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Logging steps: {args.logging_steps}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    args = parse_train_args()
    print_train_args(args)
