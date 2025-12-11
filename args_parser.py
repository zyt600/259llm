"""
命令行参数解析模块
"""
import argparse
import os
import torch


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="大模型推理评估工具 - 评估LongBench QMSum和TruthfulQA数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定模型在所有GPU上运行
  python main.py --model_name meta-llama/Llama-2-7b-chat-hf
  
  # 使用指定GPU
  python main.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1,2
  
  # 限制评估样本数量（用于快速测试）
  python main.py --model_name meta-llama/Llama-2-7b-chat-hf --max_samples 100
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face上的模型名称，例如: meta-llama/Llama-2-7b-chat-hf"
    )
    
    # GPU配置
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="使用的GPU编号，逗号分隔。例如: '0,1,2'。默认使用所有可用GPU"
    )
    
    # 推理配置
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，0表示贪婪解码 (默认: 0.0)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批处理大小 (默认: 8)"
    )
    
    # 评估配置
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个数据集评估的最大样本数，用于快速测试。默认评估全部"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="结果输出目录 (默认: ./results)"
    )
    
    # SGLang配置
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="张量并行大小。默认等于GPU数量 (仅SGLang使用)"
    )
    
    parser.add_argument(
        "--mem_fraction",
        type=float,
        default=0.5,
        help="GPU内存使用比例 (默认: 0.5) (仅SGLang使用)"
    )
    
    # llama.cpp配置
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="最大生成token数 (默认: 512) (仅llama.cpp/GGUF使用)"
    )
    
    args = parser.parse_args()
    
    # 后处理参数
    args = _process_args(args)
    
    return args


def _process_args(args):
    """
    处理和验证参数
    
    Args:
        args: 原始参数对象
        
    Returns:
        处理后的参数对象
    """
    # 处理GPU参数
    if args.gpu is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu.split(",")]
        args.gpu_ids = gpu_ids
        # 判断是否为GGUF模型（llama.cpp多GPU并行模式）
        # 本地路径（以./或/开头）视为GGUF文件
        is_gguf = (args.model_name.startswith('./') or args.model_name.startswith('/') or 
                   args.model_name.lower().endswith('gguf'))
        # 只有非GGUF模型（SGLang）才设置CUDA_VISIBLE_DEVICES
        # llama.cpp多GPU并行模式需要由worker进程自己设置
        if not is_gguf:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        # 使用所有可用GPU
        if torch.cuda.is_available():
            args.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            args.gpu_ids = []
    
    # 设置张量并行大小
    if args.tp_size is None:
        args.tp_size = len(args.gpu_ids) if args.gpu_ids else 1
    
    # 固定数据集列表
    args.dataset_list = ["qmsum", "truthfulqa"]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def print_args(args):
    """
    打印参数配置
    
    Args:
        args: 参数对象
    """
    # 判断后端类型
    # 本地路径（以./或/开头）视为GGUF文件
    is_gguf = (args.model_name.startswith('./') or args.model_name.startswith('/') or 
               args.model_name.lower().endswith('.gguf') or args.model_name.endswith('-GGUF'))
    backend = "llama.cpp" if is_gguf else "SGLang"
    
    print("\n" + "=" * 60)
    print("配置参数")
    print("=" * 60)
    print(f"模型名称: {args.model_name}")
    print(f"推理后端: {backend}")
    print(f"GPU IDs: {args.gpu_ids}")
    
    if is_gguf:
        if len(args.gpu_ids) > 1:
            print(f"[多GPU并行模式] {len(args.gpu_ids)} 个GPU并行推理")
        print(f"最大生成token数: {args.max_tokens}")
    else:
        print(f"张量并行大小: {args.tp_size}")
        print(f"内存使用比例: {args.mem_fraction}")
    
    print(f"批处理大小: {args.batch_size}")
    print(f"温度: {args.temperature}")
    print(f"评估数据集: QMSum, TruthfulQA")
    print(f"最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试参数解析
    args = parse_args()
    print_args(args)

