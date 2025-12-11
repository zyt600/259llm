"""
评估模块 - 计算模型在各数据集上的评估指标
"""
from typing import List, Dict, Tuple, Optional
import json
import os
from collections import defaultdict
import re


def evaluate_qmsum(samples: List[Dict]) -> Dict[str, float]:
    """
    评估QMSum数据集的结果
    
    使用evaluate库的ROUGE评估器，与官方评测一致
    
    Args:
        samples: 带有预测结果的样本列表
        
    Returns:
        Dict[str, float]: 评估分数
    """
    import evaluate
    
    print("\n正在评估QMSum数据集...")
    
    # 加载ROUGE评估器
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    sample_ids = []
    
    for sample in samples:
        pred = sample.get("prediction", "").strip()
        # 使用reference字段（已处理好的单个参考答案）
        ref = sample.get("reference", "")
        if not ref:
            # 回退到answers字段
            ans = sample.get("answers", [])
            if isinstance(ans, list) and len(ans) > 0:
                ref = ans[0].strip()
            else:
                ref = str(ans).strip() if ans else ""
        
        predictions.append(pred)
        references.append(ref)
        sample_ids.append(sample.get("id", len(sample_ids)))
    
    # 计算聚合ROUGE分数
    aggregated = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    # 计算每个样本的分数
    per_sample = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False
    )
    
    # 打印每个样本的ROUGE-L分数
    print("\n=== ROUGE-L per sample ===")
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        print(f"Sample {idx}: ROUGE-L = {rl:.4f}")
    
    results = {
        "dataset": "QMSum",
        "num_samples": len(samples),
        "rouge1": aggregated["rouge1"],
        "rouge2": aggregated["rouge2"],
        "rougeL": aggregated["rougeL"],
        "rougeLsum": aggregated["rougeLsum"],
        "per_sample_rougeL": list(zip(sample_ids, per_sample["rougeL"])),
        "main_metric": "rougeL",
        "main_score": aggregated["rougeL"],
    }
    
    print(f"\n=== Aggregated ROUGE ===")
    print(f"ROUGE-L (F1): {results['rougeL']:.4f}")
    print(f"ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, ROUGE-Lsum: {results['rougeLsum']:.4f}")
    
    return results


def _bleurt_worker(samples_data: List[Dict], result_queue, gpu_id: int = 7):
    """
    在独立进程中运行 BLEURT 评估
    
    Args:
        samples_data: 样本数据列表
        result_queue: 用于返回结果的队列
        gpu_id: 使用的 GPU ID
    """
    import os
    import numpy as np
    
    # 在子进程中设置 CUDA_VISIBLE_DEVICES，不影响主进程
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    import evaluate
    
    print(f"BLEURT 在 GPU {gpu_id} 上运行")
    print("正在加载BLEURT评估器 (bleurt-large-128)...")
    bleurt = evaluate.load('bleurt', 'bleurt-large-128')
    print("BLEURT评估器加载完成")
    
    n = len(samples_data)
    max_score_arr = []
    acc_score_arr = []
    
    for i, sample in enumerate(samples_data):
        pred = sample.get("prediction", "")
        correct_answers = sample.get("correct_answers", [])
        incorrect_answers = sample.get("incorrect_answers", [])
        
        # 确保是列表
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        if isinstance(incorrect_answers, str):
            incorrect_answers = [incorrect_answers]
        
        # 计算与correct_answers的BLEURT分数
        if correct_answers:
            predictions_true = [pred] * len(correct_answers)
            score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)['scores']
        else:
            score_true = [0.0]
        
        # 计算与incorrect_answers的BLEURT分数
        if incorrect_answers:
            predictions_false = [pred] * len(incorrect_answers)
            score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)['scores']
        else:
            score_false = [0.0]
        
        # 计算指标
        max_score = max(score_true)
        acc_score = int(max(score_true) > max(score_false))
        
        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)
        
        # 打印进度
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  已评估 {i + 1}/{n} 个样本")
    
    # 计算平均值
    avg_max_score = float(np.mean(np.array(max_score_arr)))
    accuracy = sum(acc_score_arr) / n
    
    results = {
        "dataset": "TruthfulQA",
        "num_samples": n,
        "avg_max_score": avg_max_score,
        "accuracy": accuracy,
        "correct_count": sum(acc_score_arr),
        "main_metric": "accuracy",
        "main_score": accuracy,
    }
    
    # 清理
    del bleurt
    
    result_queue.put(results)


def evaluate_truthfulqa(samples: List[Dict], bleurt_gpu: int = 7) -> Dict[str, float]:
    """
    评估TruthfulQA数据集的结果
    
    使用独立子进程运行 BLEURT，避免与主进程的 GPU 冲突
    
    Args:
        samples: 带有预测结果的样本列表
        bleurt_gpu: BLEURT 使用的 GPU ID（默认 7）
        
    Returns:
        Dict[str, float]: 评估分数
    """
    import multiprocessing as mp
    
    print("\n正在评估TruthfulQA数据集...")
    
    # 准备样本数据（只传递必要的字段）
    samples_data = []
    for s in samples:
        samples_data.append({
            "prediction": s.get("prediction", ""),
            "correct_answers": s.get("correct_answers", []),
            "incorrect_answers": s.get("incorrect_answers", []),
        })
    
    # 使用 spawn 方式创建子进程，确保完全独立的环境
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    # 在子进程中运行 BLEURT 评估
    p = ctx.Process(target=_bleurt_worker, args=(samples_data, result_queue, bleurt_gpu))
    p.start()
    
    # 等待子进程完成
    results = result_queue.get()
    p.join()
    
    print(f"\nTruthfulQA评估完成:")
    print(f"  avg max score: {results['avg_max_score']:.6f}")
    print(f"  avg accuracy: {results['accuracy']:.3f}")
    print("  BLEURT子进程已结束，显存已释放")
    
    return results


def evaluate_dataset(dataset_name: str, samples: List[Dict]) -> Dict[str, float]:
    """
    根据数据集名称选择评估方法
    
    Args:
        dataset_name: 数据集名称
        samples: 样本列表
        
    Returns:
        Dict[str, float]: 评估结果
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "qmsum":
        return evaluate_qmsum(samples)
    elif dataset_name == "truthfulqa":
        return evaluate_truthfulqa(samples)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


def save_results(
    results: Dict,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    temperature: float = 0.0
):
    """
    保存评估结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
        model_name: 模型名称
        dataset_name: 数据集名称
        temperature: 推理温度
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理模型名称用于文件名
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")
    
    # 保存JSON结果（文件名包含温度）
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset_name}_temp{temperature}_results.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")


def save_predictions(
    samples: List[Dict],
    output_dir: str,
    model_name: str,
    dataset_name: str,
    temperature: float = 0.0
):
    """
    保存预测结果
    
    Args:
        samples: 样本列表
        output_dir: 输出目录
        model_name: 模型名称
        dataset_name: 数据集名称
        temperature: 推理温度
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset_name}_temp{temperature}_predictions.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            # 只保存关键字段
            output_sample = {
                "id": sample.get("id"),
                "prediction": sample.get("prediction", ""),
            }
            
            if dataset_name == "qmsum":
                output_sample["reference"] = sample.get("reference", "")
                output_sample["answers"] = sample.get("answers", [])
            elif dataset_name == "truthfulqa":
                output_sample["question"] = sample.get("question", "")
                output_sample["best_answer"] = sample.get("best_answer", "")
                output_sample["correct_answers"] = sample.get("correct_answers", [])
            
            f.write(json.dumps(output_sample, ensure_ascii=False) + "\n")
    
    print(f"预测结果已保存到: {output_file}")


def print_summary(all_results: Dict[str, Dict]):
    """
    打印评估结果摘要
    
    Args:
        all_results: 所有数据集的评估结果
    """
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{results.get('dataset', dataset_name)}:")
        print(f"  样本数量: {results.get('num_samples', 'N/A')}")
        
        if dataset_name == "qmsum":
            print(f"  ROUGE-L (F1): {results.get('rougeL', 0):.4f}")
            print(f"  ROUGE-1: {results.get('rouge1', 0):.4f}")
            print(f"  ROUGE-2: {results.get('rouge2', 0):.4f}")
            print(f"  ROUGE-Lsum: {results.get('rougeLsum', 0):.4f}")
        
        if dataset_name == "truthfulqa":
            print(f"  avg max score: {results.get('avg_max_score', 0):.6f}")
            print(f"  avg accuracy: {results.get('accuracy', 0):.3f}")
    
    print("\n" + "=" * 60)

