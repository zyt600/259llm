"""
数据集加载模块 - 加载LongBench QMSum和TruthfulQA数据集
"""
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset


def load_qmsum_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    """
    加载LongBench的QMSum数据集
    
    QMSum是一个会议摘要数据集，任务是根据会议记录生成摘要
    使用 zai-org/LongBench 数据集
    
    Args:
        max_samples: 最大样本数，None表示加载全部
        
    Returns:
        List[Dict]: 数据样本列表，每个样本包含input, context, answer等字段
    """
    print("正在加载LongBench QMSum数据集...")
    
    # 使用 THUDM/LongBench 数据集，需要trust_remote_code
    try:
        dataset = load_dataset("THUDM/LongBench", "qmsum", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"从THUDM/LongBench加载失败: {e}")
        # 尝试备用数据集
        dataset = load_dataset("namespace-Pt/LongBench", "qmsum", split="test", trust_remote_code=True)
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        # 获取参考答案
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
            "reference": reference,  # 用于评估的参考答案
            "length": item.get("length", 0),
            "dataset": "qmsum",
            "all_classes": item.get("all_classes", None),
        }
        
        # 构建完整的输入prompt
        context = sample["context"]
        query = sample["input"]
        
        sample["full_prompt"] = _build_qmsum_prompt(context, query)
        samples.append(sample)
    
    print(f"QMSum数据集加载完成，共 {len(samples)} 个样本")
    return samples


def _build_qmsum_prompt(context: str, query: str) -> str:
    """
    构建QMSum的prompt
    
    Args:
        context: 会议记录内容
        query: 查询/问题
        
    Returns:
        str: 完整的prompt
    """
    prompt = f"""You are a helpful assistant. Please summarize the following meeting transcript based on the query.

Meeting Transcript:
{context}

Query: {query}

Summary:"""
    return prompt


def load_truthfulqa_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    """
    加载TruthfulQA数据集
    
    TruthfulQA用于评估模型生成真实答案的能力
    使用 truthfulqa/truthful_qa 的 generation 配置
    
    Args:
        max_samples: 最大样本数，None表示加载全部
        
    Returns:
        List[Dict]: 数据样本列表
    """
    print("正在加载TruthfulQA数据集...")
    
    # 使用官方数据集路径
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
    
    print(f"TruthfulQA数据集加载完成，共 {len(samples)} 个样本")
    return samples


def _build_truthfulqa_prompt(question: str) -> str:
    """
    构建TruthfulQA的prompt
    
    Args:
        question: 问题
        
    Returns:
        str: 完整的prompt
    """
    prompt = f"""Q: {question}
A:"""
    return prompt


def load_dataset_by_name(
    dataset_name: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    根据名称加载数据集
    
    Args:
        dataset_name: 数据集名称 ('qmsum' 或 'truthfulqa')
        max_samples: 最大样本数
        
    Returns:
        List[Dict]: 数据样本列表
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "qmsum":
        return load_qmsum_dataset(max_samples)
    elif dataset_name == "truthfulqa":
        return load_truthfulqa_dataset(max_samples)
    else:
        raise ValueError(f"未知的数据集名称: {dataset_name}。支持的数据集: qmsum, truthfulqa")


def get_dataset_info(dataset_name: str) -> Dict:
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        Dict: 数据集信息
    """
    info = {
        "qmsum": {
            "name": "QMSum",
            "full_name": "Query-based Multi-domain Meeting Summarization",
            "task": "摘要生成",
            "metric": "ROUGE-L",
            "description": "基于查询的会议摘要数据集，评估模型在长文本理解和摘要生成方面的能力",
        },
        "truthfulqa": {
            "name": "TruthfulQA",
            "full_name": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
            "task": "问答/真实性评估",
            "metric": "Accuracy / BLEURT",
            "description": "评估模型生成真实、准确答案的能力，避免常见的人类错误信念",
        }
    }
    
    return info.get(dataset_name.lower(), {"name": dataset_name, "description": "未知数据集"})


if __name__ == "__main__":
    # 测试数据集加载
    print("测试加载QMSum数据集（前5个样本）...")
    qmsum_samples = load_qmsum_dataset(max_samples=5)
    if qmsum_samples:
        print(f"第一个样本的prompt前500字符:\n{qmsum_samples[0]['full_prompt'][:500]}...")
    
    print("\n" + "="*50 + "\n")
    
    print("测试加载TruthfulQA数据集（前5个样本）...")
    tqa_samples = load_truthfulqa_dataset(max_samples=5)
    if tqa_samples:
        print(f"第一个样本:\n问题: {tqa_samples[0]['question']}")
        print(f"最佳答案: {tqa_samples[0]['best_answer']}")

