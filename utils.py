"""
Utility functions module
"""
import os
import json
import torch
from typing import List, Dict, Optional


def check_gpu_availability() -> Dict:
    """
    Check GPU availability
    
    Returns:
        Dict: GPU information
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
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory
            info["gpu_memory"].append({
                "total_gb": total_memory / (1024**3),
            })
    
    return info


def print_gpu_info():
    """Print GPU information"""
    info = check_gpu_availability()
    
    print("\n" + "=" * 50)
    print("GPU Information")
    print("=" * 50)
    
    if not info["cuda_available"]:
        print("CUDA not available")
        return
    
    print(f"Available GPU count: {info['gpu_count']}")
    
    for i in range(info["gpu_count"]):
        print(f"\nGPU {i}:")
        print(f"  Name: {info['gpu_names'][i]}")
        print(f"  Total memory: {info['gpu_memory'][i]['total_gb']:.2f} GB")
    
    print("=" * 50 + "\n")


def set_random_seed(seed: int = 42):
    """
    Set random seed
    
    Args:
        seed: Random seed
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
    Load JSON file
    
    Args:
        filepath: File path
        
    Returns:
        Dict: JSON data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, filepath: str):
    """
    Save JSON file
    
    Args:
        data: Data to save
        filepath: File path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file
    
    Args:
        filepath: File path
        
    Returns:
        List[Dict]: Data list
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """
    Save JSONL file
    
    Args:
        data: Data list to save
        filepath: File path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text
    
    Args:
        text: Original text
        max_length: Maximum length
        suffix: Truncation suffix
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_time(seconds: float) -> str:
    """
    Format time
    
    Args:
        seconds: Number of seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


if __name__ == "__main__":
    # Test GPU info
    print_gpu_info()
