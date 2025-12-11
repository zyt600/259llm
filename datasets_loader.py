"""
Dataset loading module - Load LongBench QMSum and TruthfulQA datasets
"""
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset


def load_qmsum_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load LongBench QMSum dataset
    
    QMSum is a meeting summarization dataset, the task is to generate summaries based on meeting transcripts
    Uses zai-org/LongBench dataset
    
    Args:
        max_samples: Maximum number of samples, None means load all
        
    Returns:
        List[Dict]: Data sample list, each sample contains input, context, answer fields
    """
    print("Loading LongBench QMSum dataset...")
    
    # Use THUDM/LongBench dataset, requires trust_remote_code
    try:
        dataset = load_dataset("THUDM/LongBench", "qmsum", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load from THUDM/LongBench: {e}")
        # Try backup dataset
        dataset = load_dataset("namespace-Pt/LongBench", "qmsum", split="test", trust_remote_code=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        # Get reference answer
        ans = item.get("answers", [])
        if isinstance(ans, list) and len(ans) > 0:
            reference = ans[0].strip()
        else:
            reference = str(ans).strip() if ans else ""
            
        sample = {
            "id": i,
            "input": item.get("input", ""),
            "context": item.get("context", ""),
            "answers": ans,
            "reference": reference,  # Reference answer for evaluation
            "length": item.get("length", 0),
            "dataset": "qmsum",
            "all_classes": item.get("all_classes", None),
        }
        
        # Build complete input prompt
        context = sample["context"]
        query = sample["input"]
        
        sample["full_prompt"] = _build_qmsum_prompt(context, query)
        samples.append(sample)
    
    print(f"QMSum dataset loaded, {len(samples)} samples total")
    return samples


def _build_qmsum_prompt(context: str, query: str) -> str:
    """
    Build QMSum prompt
    
    Args:
        context: Meeting transcript content
        query: Query/question
        
    Returns:
        str: Complete prompt
    """
    prompt = f"""You are a helpful assistant. Please summarize the following meeting transcript based on the query.

Meeting Transcript:
{context}

Query: {query}

Summary:"""
    return prompt


def load_truthfulqa_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load TruthfulQA dataset
    
    TruthfulQA evaluates models' ability to generate truthful answers
    Uses truthfulqa/truthful_qa generation configuration
    
    Args:
        max_samples: Maximum number of samples, None means load all
        
    Returns:
        List[Dict]: Data sample list
    """
    print("Loading TruthfulQA dataset...")
    
    # Use official dataset path
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        sample = {
            "id": i,
            "question": item.get("question", ""),
            "best_answer": item.get("best_answer", ""),
            "correct_answers": item.get("correct_answers", []),
            "incorrect_answers": item.get("incorrect_answers", []),
            "category": item.get("category", ""),
            "source": item.get("source", ""),
            "dataset": "truthfulqa",
        }
        
        sample["full_prompt"] = _build_truthfulqa_prompt(sample["question"])
        samples.append(sample)
    
    print(f"TruthfulQA dataset loaded, {len(samples)} samples total")
    return samples


def _build_truthfulqa_prompt(question: str) -> str:
    """
    Build TruthfulQA prompt
    
    Args:
        question: Question
        
    Returns:
        str: Complete prompt
    """
    prompt = f"""Q: {question}
A:"""
    return prompt


def load_dataset_by_name(
    dataset_name: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load dataset by name
    
    Args:
        dataset_name: Dataset name ('qmsum' or 'truthfulqa')
        max_samples: Maximum number of samples
        
    Returns:
        List[Dict]: Data sample list
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "qmsum":
        return load_qmsum_dataset(max_samples)
    elif dataset_name == "truthfulqa":
        return load_truthfulqa_dataset(max_samples)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets: qmsum, truthfulqa")


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get dataset information
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        Dict: Dataset information
    """
    info = {
        "qmsum": {
            "name": "QMSum",
            "full_name": "Query-based Multi-domain Meeting Summarization",
            "task": "Summarization",
            "metric": "ROUGE-L",
            "description": "Query-based meeting summarization dataset, evaluates model's long text understanding and summarization ability",
        },
        "truthfulqa": {
            "name": "TruthfulQA",
            "full_name": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
            "task": "QA/Truthfulness Evaluation",
            "metric": "Accuracy / BLEURT",
            "description": "Evaluates model's ability to generate truthful, accurate answers, avoiding common human misconceptions",
        }
    }
    
    return info.get(dataset_name.lower(), {"name": dataset_name, "description": "Unknown dataset"})


if __name__ == "__main__":
    # Test dataset loading
    print("Testing QMSum dataset loading (first 5 samples)...")
    qmsum_samples = load_qmsum_dataset(max_samples=5)
    if qmsum_samples:
        print(f"First sample prompt (first 500 chars):\n{qmsum_samples[0]['full_prompt'][:500]}...")
    
    print("\n" + "="*50 + "\n")
    
    print("Testing TruthfulQA dataset loading (first 5 samples)...")
    tqa_samples = load_truthfulqa_dataset(max_samples=5)
    if tqa_samples:
        print(f"First sample:\nQuestion: {tqa_samples[0]['question']}")
        print(f"Best answer: {tqa_samples[0]['best_answer']}")
