#!/usr/bin/env python3
"""
合并 LoRA adapter 与基础模型，导出完整模型
"""
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_export(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    合并 LoRA adapter 并导出完整模型
    
    Args:
        base_model_name: 基础模型名称 (如 Qwen/Qwen3-0.6B)
        adapter_path: LoRA adapter 路径
        output_path: 输出路径
        push_to_hub: 是否推送到 HuggingFace Hub
        hub_model_id: HuggingFace Hub 模型 ID
    """
    print(f"\n{'='*60}")
    print("合并 LoRA adapter 并导出模型")
    print(f"{'='*60}")
    print(f"基础模型: {base_model_name}")
    print(f"Adapter路径: {adapter_path}")
    print(f"输出路径: {output_path}")
    print(f"{'='*60}\n")
    
    # 1. 加载基础模型
    print("[1/4] 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  基础模型参数量: {base_model.num_parameters() / 1e6:.2f}M")
    
    # 2. 加载 tokenizer
    print("\n[2/4] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    
    # 3. 加载并合并 LoRA adapter
    print("\n[3/4] 加载并合并 LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print(f"  合并后模型参数量: {model.num_parameters() / 1e6:.2f}M")
    
    # 4. 保存合并后的模型
    print(f"\n[4/4] 保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # 计算模型大小
    total_size = 0
    for f in os.listdir(output_path):
        if f.endswith('.safetensors') or f.endswith('.bin'):
            total_size += os.path.getsize(os.path.join(output_path, f))
    print(f"  模型大小: {total_size / 1e9:.2f} GB")
    
    # 列出保存的文件
    print(f"\n保存的文件:")
    for f in sorted(os.listdir(output_path)):
        fpath = os.path.join(output_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f}: {size / 1e6:.2f} MB")
    
    # 推送到 Hub
    if push_to_hub and hub_model_id:
        print(f"\n推送到 HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("推送完成!")
    
    print(f"\n{'='*60}")
    print("✅ 模型导出完成!")
    print(f"模型路径: {output_path}")
    print(f"{'='*60}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 并导出模型")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="基础模型名称",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./merged_model",
        help="输出路径",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="是否推送到 HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub 模型 ID",
    )
    
    args = parser.parse_args()
    
    merge_and_export(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()

