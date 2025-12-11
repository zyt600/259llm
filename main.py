"""
主程序入口 - 大模型推理评估工具

使用方法:
    python main.py --model_name <模型名称> [--gpu <GPU编号>] [--max_samples <样本数>]

示例:
    python main.py --model_name meta-llama/Llama-2-7b-chat-hf --gpu 0,1
    python main.py --model_name Qwen/Qwen2-7B-Instruct --max_samples 100
"""
import os
import sys

# 抑制TensorFlow日志（必须在任何可能导入TensorFlow的代码之前设置）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 设置CUDA库路径（llama.cpp server需要）
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

# 设置CUDA_HOME环境变量（必须在导入torch之前）
def _setup_cuda_home():
    if "CUDA_HOME" not in os.environ:
        # 尝试从nvidia包中获取CUDA路径
        try:
            import nvidia.cuda_runtime
            cuda_path = os.path.dirname(os.path.dirname(nvidia.cuda_runtime.__file__))
            os.environ["CUDA_HOME"] = cuda_path
        except ImportError:
            # 尝试常见路径
            for path in ["/usr/local/cuda", "/usr/local/cuda-12"]:
                if os.path.exists(path):
                    os.environ["CUDA_HOME"] = path
                    break

_setup_cuda_home()

import time
from datetime import datetime

# 导入各模块
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
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    print_args(args)
    
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 存储所有数据集的样本（推理后）
    all_samples = {}
    # 存储所有评估结果
    all_results = {}
    
    try:
        # 第一阶段：加载模型并完成所有推理
        print("\n" + "=" * 60)
        print("第一阶段：模型推理")
        print("=" * 60)
        
        with ModelManager(
            model_name=args.model_name,
            gpu_ids=args.gpu_ids,
            tp_size=args.tp_size,
            mem_fraction=args.mem_fraction
        ) as model_manager:
            
            # 创建推理运行器
            runner = InferenceRunner(
                engine=model_manager.engine,
                temperature=args.temperature,
                batch_size=args.batch_size,
                backend=model_manager.backend,
                max_tokens=args.max_tokens,
                model_path=args.model_name,
                gpu_ids=args.gpu_ids
            )
            
            # 遍历每个数据集进行推理
            for dataset_name in args.dataset_list:
                print(f"\n{'='*60}")
                print(f"数据集: {dataset_name.upper()}")
                print(f"{'='*60}")
                
                # 打印数据集信息
                dataset_info = get_dataset_info(dataset_name)
                print(f"任务: {dataset_info.get('task', 'N/A')}")
                print(f"描述: {dataset_info.get('description', 'N/A')}")
                
                # 加载数据集
                samples = load_dataset_by_name(
                    dataset_name=dataset_name,
                    max_samples=args.max_samples
                )
                
                if not samples:
                    print(f"警告: 数据集 {dataset_name} 为空，跳过")
                    continue
                
                # 运行推理
                print(f"\n开始推理，共 {len(samples)} 个样本...")
                samples = runner.run(samples)
                
                # 保存推理结果
                all_samples[dataset_name] = samples
        
        # 第二阶段：评估（模型引擎已关闭，可以安全使用TensorFlow/BLEURT）
        print("\n" + "=" * 60)
        print("第二阶段：结果评估")
        print("=" * 60)
        
        for dataset_name, samples in all_samples.items():
            print(f"\n正在评估 {dataset_name.upper()} 数据集...")
            
            # 评估结果
            results = evaluate_dataset(dataset_name, samples)
            results["model_name"] = args.model_name
            results["timestamp"] = datetime.now().isoformat()
            
            # 保存结果
            save_results(
                results=results,
                output_dir=args.output_dir,
                model_name=args.model_name,
                dataset_name=dataset_name,
                temperature=args.temperature
            )
            
            # 保存预测结果
            save_predictions(
                samples=samples,
                output_dir=args.output_dir,
                model_name=args.model_name,
                dataset_name=dataset_name,
                temperature=args.temperature
            )
            
            all_results[dataset_name] = results
        
        # 打印最终结果摘要
        print_summary(all_results)
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

