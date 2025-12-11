"""
Main program entry - LLM Inference Evaluation Tool

Usage:
    python main.py --model_name <model_name> [--gpu <GPU_IDs>] [--max_samples <sample_count>]

Examples:
    python main.py --model_name meta-llama/Llama-2-7b-chat-hf --gpu 0,1
    python main.py --model_name Qwen/Qwen2-7B-Instruct --max_samples 100
"""
import os
import sys

# Suppress TensorFlow logs (must be set before importing any code that might import TensorFlow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set CUDA library paths (needed for llama.cpp server)
def _setup_cuda_libs():
    cuda_lib_paths = [
        "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib",
        "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + existing_ld_path

_setup_cuda_libs()

# Set CUDA_HOME environment variable (must be before importing torch)
def _setup_cuda_home():
    if "CUDA_HOME" not in os.environ:
        # Try to get CUDA path from nvidia package
        try:
            import nvidia.cuda_runtime
            cuda_path = os.path.dirname(os.path.dirname(nvidia.cuda_runtime.__file__))
            os.environ["CUDA_HOME"] = cuda_path
        except ImportError:
            # Try common paths
            for path in ["/usr/local/cuda", "/usr/local/cuda-12"]:
                if os.path.exists(path):
                    os.environ["CUDA_HOME"] = path
                    break

_setup_cuda_home()

import time
from datetime import datetime

# Import modules
from args_parser import parse_args, print_args
from model_loader import ModelManager
from datasets_loader import load_dataset_by_name, get_dataset_info
from inference import InferenceRunner
from evaluator import (
    evaluate_dataset,
    save_results,
    save_predictions,
    print_summary
)


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    print_args(args)
    
    # Record start time
    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Store samples from all datasets (after inference)
    all_samples = {}
    # Store all evaluation results
    all_results = {}
    
    try:
        # Phase 1: Load model and complete all inference
        print("\n" + "=" * 60)
        print("Phase 1: Model Inference")
        print("=" * 60)
        
        with ModelManager(
            model_name=args.model_name,
            gpu_ids=args.gpu_ids,
            tp_size=args.tp_size,
            mem_fraction=args.mem_fraction
        ) as model_manager:
            
            # Create inference runner
            runner = InferenceRunner(
                engine=model_manager.engine,
                temperature=args.temperature,
                batch_size=args.batch_size,
                backend=model_manager.backend,
                max_tokens=args.max_tokens,
                model_path=args.model_name,
                gpu_ids=args.gpu_ids
            )
            
            # Iterate through each dataset for inference
            for dataset_name in args.dataset_list:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset_name.upper()}")
                print(f"{'='*60}")
                
                # Print dataset info
                dataset_info = get_dataset_info(dataset_name)
                print(f"Task: {dataset_info.get('task', 'N/A')}")
                print(f"Description: {dataset_info.get('description', 'N/A')}")
                
                # Load dataset
                samples = load_dataset_by_name(
                    dataset_name=dataset_name,
                    max_samples=args.max_samples
                )
                
                if not samples:
                    print(f"Warning: Dataset {dataset_name} is empty, skipping")
                    continue
                
                # Run inference
                print(f"\nStarting inference, {len(samples)} samples total...")
                samples = runner.run(samples)
                
                # Save inference results
                all_samples[dataset_name] = samples
        
        # Phase 2: Evaluation (model engine closed, safe to use TensorFlow/BLEURT)
        print("\n" + "=" * 60)
        print("Phase 2: Results Evaluation")
        print("=" * 60)
        
        for dataset_name, samples in all_samples.items():
            print(f"\nEvaluating {dataset_name.upper()} dataset...")
            
            # Evaluate results
            results = evaluate_dataset(dataset_name, samples)
            results["model_name"] = args.model_name
            results["timestamp"] = datetime.now().isoformat()
            
            # Save results
            save_results(
                results=results,
                output_dir=args.output_dir,
                model_name=args.model_name,
                dataset_name=dataset_name,
                temperature=args.temperature
            )
            
            # Save predictions
            save_predictions(
                samples=samples,
                output_dir=args.output_dir,
                model_name=args.model_name,
                dataset_name=dataset_name,
                temperature=args.temperature
            )
            
            all_results[dataset_name] = results
        
        # Print final results summary
        print_summary(all_results)
        
    except KeyboardInterrupt:
        print("\nUser interrupted, exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print total time
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
