"""
Command line argument parsing module
"""
import argparse
import os
import torch


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(
        description="LLM Inference Evaluation Tool - Evaluate LongBench QMSum and TruthfulQA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use specified model on all GPUs
  python main.py --model_name meta-llama/Llama-2-7b-chat-hf
  
  # Use specified GPUs
  python main.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1,2
  
  # Limit evaluation samples (for quick testing)
  python main.py --model_name meta-llama/Llama-2-7b-chat-hf --max_samples 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name on Hugging Face, e.g.: meta-llama/Llama-2-7b-chat-hf"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU IDs to use, comma-separated. e.g.: '0,1,2'. Default: use all available GPUs"
    )
    
    # Inference configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0 for greedy decoding (default: 0.0)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset for evaluation, for quick testing. Default: evaluate all"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    # SGLang configuration
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel size. Default: equals GPU count (SGLang only)"
    )
    
    parser.add_argument(
        "--mem_fraction",
        type=float,
        default=0.5,
        help="GPU memory usage fraction (default: 0.5) (SGLang only)"
    )
    
    # llama.cpp configuration
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512) (llama.cpp/GGUF only)"
    )
    
    args = parser.parse_args()
    
    # Post-process arguments
    args = _process_args(args)
    
    return args


def _process_args(args):
    """
    Process and validate arguments
    
    Args:
        args: Raw argument object
        
    Returns:
        Processed argument object
    """
    # Process GPU arguments
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
        args.gpu_ids = gpu_ids
        # Check if GGUF model (llama.cpp multi-GPU parallel mode)
        # Local paths (starting with ./ or /) are treated as GGUF files
        is_gguf = (args.model_name.startswith('./') or args.model_name.startswith('/') or 
                   args.model_name.lower().endswith('gguf'))
        # Only set CUDA_VISIBLE_DEVICES for non-GGUF models (SGLang)
        # llama.cpp multi-GPU parallel mode needs worker processes to set it themselves
        if not is_gguf:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        # Use all available GPUs
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            args.gpu_ids = []
    
    # Set tensor parallel size
    if args.tp_size is None:
        args.tp_size = len(args.gpu_ids) if args.gpu_ids else 1
    
    # Fixed dataset list
    args.dataset_list = ["qmsum", "truthfulqa"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def print_args(args):
    """
    Print argument configuration
    
    Args:
        args: Argument object
    """
    # Determine backend type
    # Local paths (starting with ./ or /) are treated as GGUF files
    is_gguf = (args.model_name.startswith('./') or args.model_name.startswith('/') or 
               args.model_name.lower().endswith('.gguf') or args.model_name.endswith('-GGUF'))
    backend = "llama.cpp" if is_gguf else "SGLang"
    
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Inference backend: {backend}")
    print(f"GPU IDs: {args.gpu_ids}")
    
    if is_gguf:
        if len(args.gpu_ids) > 1:
            print(f"[Multi-GPU parallel mode] {len(args.gpu_ids)} GPUs for parallel inference")
        print(f"Max generation tokens: {args.max_tokens}")
    else:
        print(f"Tensor parallel size: {args.tp_size}")
        print(f"Memory fraction: {args.mem_fraction}")
    
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Evaluation datasets: QMSum, TruthfulQA")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test argument parsing
    args = parse_args()
    print_args(args)
