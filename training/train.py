#!/usr/bin/env python
"""
LongBench SFT训练主程序

使用方法:
    python train.py --model_name <模型名称> [--gpu <GPU编号>] [--use_lora]

示例:
    # 全参数微调
    python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0,1,2,3
    
    # LoRA微调（推荐，节省显存）
    python train.py --model_name Qwen/Qwen2-7B-Instruct --gpu 0 --use_lora
    
    # 快速测试
    python train.py --model_name Qwen/Qwen2-0.5B-Instruct --gpu 0 --use_lora --max_train_samples 100 --eval_steps 50
"""
import os
import sys

# 添加父目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 抑制TensorFlow日志
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

# 每个数据集的最大样本数
MAX_SAMPLES_PER_DATASET = 1300


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_train_args()
    print_train_args(args)
    
    # 记录开始时间
    start_time = time.time()
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 第一步：创建训练器
        print("\n" + "=" * 60)
        print("第一步：初始化训练器")
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
        
        # 第二步：加载模型和分词器
        print("\n" + "=" * 60)
        print("第二步：加载模型和分词器")
        print("=" * 60)
        
        trainer.load_model_and_tokenizer()
        
        # 第三步：加载训练数据
        print("\n" + "=" * 60)
        print("第三步：加载训练数据")
        print("=" * 60)
        
        # 确定要使用的子集
        use_all_datasets = args.train_dataset_list is None
        
        if use_all_datasets:
            # 使用所有数据集（包括额外数据集）
            subset_names = ALL_SUBSETS
            extra_subset_names = EXTRA_SUBSET_NAMES
            print(f"使用 ALL 模式，加载所有数据集")
            print(f"  基础数据集: {subset_names}")
            print(f"  额外数据集: {extra_subset_names}")
        else:
            # 检查是否有额外数据集
            subset_names = [s for s in args.train_dataset_list if s not in EXTRA_SUBSET_NAMES]
            extra_subset_names = [s for s in args.train_dataset_list if s in EXTRA_SUBSET_NAMES]
            print(f"使用指定数据集:")
            if subset_names:
                print(f"  基础数据集: {subset_names}")
            if extra_subset_names:
                print(f"  额外数据集: {extra_subset_names}")
        
        print(f"每个数据集最多加载 {MAX_SAMPLES_PER_DATASET} 条数据")
        
        # 加载基础数据集
        raw_samples = []
        if subset_names:
            base_samples = load_all_longbench_subsets(
                subset_names=subset_names,
                max_samples_per_subset=MAX_SAMPLES_PER_DATASET,  # 每个子集最多1300条
                max_total_samples=None,  # 不限制总数
                shuffle=True,
                seed=args.seed
            )
            raw_samples.extend(base_samples)
        
        # 加载额外数据集
        if use_all_datasets or extra_subset_names:
            datasets_to_load = extra_subset_names if extra_subset_names else EXTRA_SUBSET_NAMES
            print(f"\n{'='*60}")
            print("加载额外数据集")
            print(f"{'='*60}")
            for extra_name in datasets_to_load:
                extra_samples = load_extra_dataset(
                    subset_name=extra_name,
                    max_samples=MAX_SAMPLES_PER_DATASET
                )
                raw_samples.extend(extra_samples)
        
        # 如果指定了最大总样本数，进行截断
        if args.max_train_samples and len(raw_samples) > args.max_train_samples:
            import random
            random.seed(args.seed)
            random.shuffle(raw_samples)
            raw_samples = raw_samples[:args.max_train_samples]
            print(f"\n限制总样本数为 {args.max_train_samples}")
        
        print(f"\n总计加载 {len(raw_samples)} 个训练样本")
        
        # 创建SFT数据集
        print("\n正在创建SFT训练数据集...")
        train_dataset = create_sft_dataset(
            samples=raw_samples,
            tokenizer=trainer.tokenizer,
            max_length=args.max_length,
            truncation_side="left"  # 长文本从左边截断，保留最近的内容
        )
        
        print(f"训练数据集创建完成，共 {len(train_dataset)} 个样本")
        
        # 第四步：开始训练
        print("\n" + "=" * 60)
        print("第四步：开始SFT训练")
        print("=" * 60)
        
        final_results, final_score = trainer.train(train_dataset)
        
        # 打印最终结果
        print("\n" + "=" * 70)
        print("最终评估结果摘要")
        print("=" * 70)
        
        for dataset_name, results in final_results.items():
            print(f"\n{results.get('dataset', dataset_name)}:")
            print(f"  样本数量: {results.get('num_samples', 'N/A')}")
            
            if 'rougeL' in results:
                print(f"  ROUGE-L: {results.get('rougeL', 0):.4f}")
            if 'accuracy' in results:
                print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
            if 'avg_max_score' in results:
                print(f"  Avg Max Score: {results.get('avg_max_score', 0):.4f}")
        
        print(f"\n综合分数 (acc + rougeL + avg_max_score): {final_score:.4f}")
        
    except KeyboardInterrupt:
        print("\n用户中断训练...")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印总耗时
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time/3600:.2f} 小时")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

