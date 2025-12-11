"""
工具函数模块
"""
import os
import json
import torch
from typing import List, Dict, Optional


def check_gpu_availability() -> Dict:
    """
    检查GPU可用性
    
    Returns:
        Dict: GPU信息
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
    }
    
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        
        for i in range(info["gpu_count"]):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
            
            # 获取GPU内存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory
            info["gpu_memory"].append({
                "total_gb": total_memory / (1024**3),
            })
    
    return info


def print_gpu_info():
    """打印GPU信息"""
    info = check_gpu_availability()
    
    print("\n" + "=" * 50)
    print("GPU信息")
    print("=" * 50)
    
    if not info["cuda_available"]:
        print("CUDA不可用")
        return
    
    print(f"可用GPU数量: {info['gpu_count']}")
    
    for i in range(info["gpu_count"]):
        print(f"\nGPU {i}:")
        print(f"  名称: {info['gpu_names'][i]}")
        print(f"  总内存: {info['gpu_memory'][i]['total_gb']:.2f} GB")
    
    print("=" * 50 + "\n")


def set_random_seed(seed: int = 42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(filepath: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        Dict: JSON数据
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """
    保存JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(filepath: str) -> List[Dict]:
    """
    加载JSONL文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        List[Dict]: 数据列表
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """
    保存JSONL文件
    
    Args:
        data: 要保存的数据列表
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀
        
    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


if __name__ == "__main__":
    # 测试GPU信息
    print_gpu_info()

