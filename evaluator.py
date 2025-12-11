"""
Evaluation module - Calculate evaluation metrics for models on various datasets
"""
from typing import List, Dict, Tuple, Optional
import json
import os
from collections import defaultdict
import re


def evaluate_qmsum(samples: List[Dict]) -> Dict[str, float]:
    """
    Evaluate QMSum dataset results
    
    Uses evaluate library's ROUGE evaluator, consistent with official evaluation
    
    Args:
        samples: Sample list with prediction results
        
    Returns:
        Dict[str, float]: Evaluation scores
    """
    import evaluate
    
    print("\nEvaluating QMSum dataset...")
    
    # Load ROUGE evaluator
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    sample_ids = []
    
    for sample in samples:
        pred = sample.get("prediction", "").strip()
        # Use reference field (pre-processed single reference answer)
        ref = sample.get("reference", "")
        if not ref:
            # Fall back to answers field
            ans = sample.get("answers", [])
            if isinstance(ans, list) and len(ans) > 0:
                ref = ans[0].strip()
            else:
                ref = str(ans).strip() if ans else ""
        
        predictions.append(pred)
        references.append(ref)
        sample_ids.append(sample.get("id", len(sample_ids)))
    
    # Calculate aggregated ROUGE scores
    aggregated = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    # Calculate per-sample scores
    per_sample = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False
    )
    
    # Print per-sample ROUGE-L scores
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
    Run BLEURT evaluation in an independent process
    
    Args:
        samples_data: Sample data list
        result_queue: Queue for returning results
        gpu_id: GPU ID to use
    """
    import os
    import numpy as np
    
    # Set CUDA_VISIBLE_DEVICES in subprocess, doesn't affect main process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    import evaluate
    
    print(f"BLEURT running on GPU {gpu_id}")
    print("Loading BLEURT evaluator (bleurt-large-128)...")
    bleurt = evaluate.load('bleurt', 'bleurt-large-128')
    print("BLEURT evaluator loaded")
    
    n = len(samples_data)
    max_score_arr = []
    acc_score_arr = []
    
    for i, sample in enumerate(samples_data):
        pred = sample.get("prediction", "")
        correct_answers = sample.get("correct_answers", [])
        incorrect_answers = sample.get("incorrect_answers", [])
        
        # Ensure they are lists
        if isinstance(correct_answers, str):
            correct_answers = [correct_answers]
        if isinstance(incorrect_answers, str):
            incorrect_answers = [incorrect_answers]
        
        # Calculate BLEURT scores with correct_answers
        if correct_answers:
            predictions_true = [pred] * len(correct_answers)
            score_true = bleurt.compute(predictions=predictions_true, references=correct_answers)['scores']
        else:
            score_true = [0.0]
        
        # Calculate BLEURT scores with incorrect_answers
        if incorrect_answers:
            predictions_false = [pred] * len(incorrect_answers)
            score_false = bleurt.compute(predictions=predictions_false, references=incorrect_answers)['scores']
        else:
            score_false = [0.0]
        
        # Calculate metrics
        max_score = max(score_true)
        acc_score = int(max(score_true) > max(score_false))
        
        max_score_arr.append(max_score)
        acc_score_arr.append(acc_score)
        
        # Print progress
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  Evaluated {i + 1}/{n} samples")
    
    # Calculate averages
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
    
    # Cleanup
    del bleurt
    
    result_queue.put(results)


def evaluate_truthfulqa(samples: List[Dict], bleurt_gpu: int = 7) -> Dict[str, float]:
    """
    Evaluate TruthfulQA dataset results
    
    Uses independent subprocess to run BLEURT, avoiding GPU conflicts with main process
    
    Args:
        samples: Sample list with prediction results
        bleurt_gpu: GPU ID for BLEURT (default 7)
        
    Returns:
        Dict[str, float]: Evaluation scores
    """
    import multiprocessing as mp
    
    print("\nEvaluating TruthfulQA dataset...")
    
    # Prepare sample data (only necessary fields)
    samples_data = []
    for s in samples:
        samples_data.append({
            "prediction": s.get("prediction", ""),
            "correct_answers": s.get("correct_answers", []),
            "incorrect_answers": s.get("incorrect_answers", []),
        })
    
    # Use spawn to create subprocess, ensuring completely independent environment
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    # Run BLEURT evaluation in subprocess
    p = ctx.Process(target=_bleurt_worker, args=(samples_data, result_queue, bleurt_gpu))
    p.start()
    
    # Wait for subprocess to complete
    results = result_queue.get()
    p.join()
    
    print(f"\nTruthfulQA evaluation complete:")
    print(f"  avg max score: {results['avg_max_score']:.6f}")
    print(f"  avg accuracy: {results['accuracy']:.3f}")
    print("  BLEURT subprocess ended, GPU memory released")
    
    return results


def evaluate_dataset(dataset_name: str, samples: List[Dict]) -> Dict[str, float]:
    """
    Select evaluation method based on dataset name
    
    Args:
        dataset_name: Dataset name
        samples: Sample list
        
    Returns:
        Dict[str, float]: Evaluation results
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "qmsum":
        return evaluate_qmsum(samples)
    elif dataset_name == "truthfulqa":
        return evaluate_truthfulqa(samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_results(
    results: Dict,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    temperature: float = 0.0
):
    """
    Save evaluation results
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        model_name: Model name
        dataset_name: Dataset name
        temperature: Inference temperature
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean model name for filename
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")
    
    # Save JSON results (filename includes temperature)
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset_name}_temp{temperature}_results.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")


def save_predictions(
    samples: List[Dict],
    output_dir: str,
    model_name: str,
    dataset_name: str,
    temperature: float = 0.0
):
    """
    Save prediction results
    
    Args:
        samples: Sample list
        output_dir: Output directory
        model_name: Model name
        dataset_name: Dataset name
        temperature: Inference temperature
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(output_dir, f"{model_name_clean}_{dataset_name}_temp{temperature}_predictions.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            # Only save key fields
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
    
    print(f"Predictions saved to: {output_file}")


def print_summary(all_results: Dict[str, Dict]):
    """
    Print evaluation results summary
    
    Args:
        all_results: Evaluation results for all datasets
    """
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{results.get('dataset', dataset_name)}:")
        print(f"  Sample count: {results.get('num_samples', 'N/A')}")
        
        if dataset_name == "qmsum":
            print(f"  ROUGE-L (F1): {results.get('rougeL', 0):.4f}")
            print(f"  ROUGE-1: {results.get('rouge1', 0):.4f}")
            print(f"  ROUGE-2: {results.get('rouge2', 0):.4f}")
            print(f"  ROUGE-Lsum: {results.get('rougeLsum', 0):.4f}")
        
        if dataset_name == "truthfulqa":
            print(f"  avg max score: {results.get('avg_max_score', 0):.6f}")
            print(f"  avg accuracy: {results.get('accuracy', 0):.3f}")
    
    print("\n" + "=" * 60)
