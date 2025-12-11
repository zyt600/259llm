#!/usr/bin/env python
"""
LongBench SFT Training Main Program

Usage:
    python train.py --model_name <model_name> [--gpu <GPU_IDs>] [--use_lora]

Examples:
    # Full parameter fine-tuning
    python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1,2,3
    
    # LoRA fine-tuning (recommended, saves VRAM)
    python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0 --use_lora
    
    # Quick test
    python train.py --model_name Qwen/Qwen2-0.5B-Instruct --gpu 0 --use_lora --max_train_samples 100 --eval_steps 50
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from datetime import datetime

from train_args import parse_train_args, print_train_args
from longbench_loader import (
    load_all_longbench_subsets,
    create_sft_dataset,
    list_all_subsets,
    ALL_SUBSETS,
    ALL_SUBSETS_WITH_EXTRA,
    EXTRA_SUBSET_NAMES,
    load_extra_dataset,
)
from trainer import SFTTrainer

# Maximum samples per dataset
MAX_SAMPLES_PER_DATASET = 1300


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_train_args()
    print_train_args(args)
    
    # Record start time
    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Create trainer
        print("\n" + "=" * 60)
        print("Step 1: Initialize Trainer")
        print("=" * 60)
        
        trainer = SFTTrainer(
            model_name=args.model_name,
            output_dir=args.output_dir,
            gpu_ids=args.gpu_ids,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bf16=args.bf16,
            fp16=args.fp16,
            seed=args.seed,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        # Step 2: Load model and tokenizer
        print("\n" + "=" * 60)
        print("Step 2: Load Model and Tokenizer")
        print("=" * 60)
        
        trainer.load_model_and_tokenizer()
        
        # Step 3: Load training data
        print("\n" + "=" * 60)
        print("Step 3: Load Training Data")
        print("=" * 60)
        
        # Determine subsets to use
        use_all_datasets = args.train_dataset_list is None
        
        if use_all_datasets:
            # Use all datasets (including extra datasets)
            subset_names = ALL_SUBSETS
            extra_subset_names = EXTRA_SUBSET_NAMES
            print(f"Using ALL mode, loading all datasets")
            print(f"  Base datasets: {subset_names}")
            print(f"  Extra datasets: {extra_subset_names}")
        else:
            # Check for extra datasets
            subset_names = [s for s in args.train_dataset_list if s not in EXTRA_SUBSET_NAMES]
            extra_subset_names = [s for s in args.train_dataset_list if s in EXTRA_SUBSET_NAMES]
            print(f"Using specified datasets:")
            if subset_names:
                print(f"  Base datasets: {subset_names}")
            if extra_subset_names:
                print(f"  Extra datasets: {extra_subset_names}")
        
        print(f"Loading at most {MAX_SAMPLES_PER_DATASET} samples per dataset")
        
        # Load base datasets
        raw_samples = []
        if subset_names:
            base_samples = load_all_longbench_subsets(
                subset_names=subset_names,
                max_samples_per_subset=MAX_SAMPLES_PER_DATASET,  # Max 1300 per subset
                max_total_samples=None,  # No total limit
                shuffle=True,
                seed=args.seed
            )
            raw_samples.extend(base_samples)
        
        # Load extra datasets
        if use_all_datasets or extra_subset_names:
            datasets_to_load = extra_subset_names if extra_subset_names else EXTRA_SUBSET_NAMES
            print(f"\n{'='*60}")
            print("Loading Extra Datasets")
            print(f"{'='*60}")
            for extra_name in datasets_to_load:
                extra_samples = load_extra_dataset(
                    subset_name=extra_name,
                    max_samples=MAX_SAMPLES_PER_DATASET
                )
                raw_samples.extend(extra_samples)
        
        # If max total samples specified, truncate
        if args.max_train_samples and len(raw_samples) > args.max_train_samples:
            import random
            random.seed(args.seed)
            random.shuffle(raw_samples)
            raw_samples = raw_samples[:args.max_train_samples]
            print(f"\nLimited total samples to {args.max_train_samples}")
        
        print(f"\nTotal loaded {len(raw_samples)} training samples")
        
        # Create SFT dataset
        print("\nCreating SFT training dataset...")
        train_dataset = create_sft_dataset(
            samples=raw_samples,
            tokenizer=trainer.tokenizer,
            max_length=args.max_length,
            truncation_side="left"  # Truncate long text from left, keep recent content
        )
        
        print(f"Training dataset created, {len(train_dataset)} samples total")
        
        # Step 4: Start training
        print("\n" + "=" * 60)
        print("Step 4: Start SFT Training")
        print("=" * 60)
        
        final_results, final_score = trainer.train(train_dataset)
        
        # Print final results
        print("\n" + "=" * 70)
        print("Final Evaluation Results Summary")
        print("=" * 70)
        
        for dataset_name, results in final_results.items():
            print(f"\n{results.get('dataset', dataset_name)}:")
            print(f"  Sample count: {results.get('num_samples', 'N/A')}")
            
            if 'rougeL' in results:
                print(f"  ROUGE-L: {results.get('rougeL', 0):.4f}")
            if 'accuracy' in results:
                print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
            if 'avg_max_score' in results:
                print(f"  Avg Max Score: {results.get('avg_max_score', 0):.4f}")
        
        print(f"\nCombined score (acc + rougeL + avg_max_score): {final_score:.4f}")
        
    except KeyboardInterrupt:
        print("\nUser interrupted training...")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
