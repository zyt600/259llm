#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and export complete model
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
    Merge LoRA adapter and export complete model
    
    Args:
        base_model_name: Base model name (e.g. Qwen/Qwen3-0.6B)
        adapter_path: LoRA adapter path
        output_path: Output path
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID
    """
    print(f"\n{'='*60}")
    print("Merge LoRA adapter and export model")
    print(f"{'='*60}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}\n")
    
    # 1. Load base model
    print("[1/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Base model parameters: {base_model.num_parameters() / 1e6:.2f}M")
    
    # 2. Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    
    # 3. Load and merge LoRA adapter
    print("\n[3/4] Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print(f"  Merged model parameters: {model.num_parameters() / 1e6:.2f}M")
    
    # 4. Save merged model
    print(f"\n[4/4] Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Calculate model size
    total_size = 0
    for f in os.listdir(output_path):
        if f.endswith('.safetensors') or f.endswith('.bin'):
            total_size += os.path.getsize(os.path.join(output_path, f))
    print(f"  Model size: {total_size / 1e9:.2f} GB")
    
    # List saved files
    print(f"\nSaved files:")
    for f in sorted(os.listdir(output_path)):
        fpath = os.path.join(output_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f}: {size / 1e6:.2f} MB")
    
    # Push to Hub
    if push_to_hub and hub_model_id:
        print(f"\nPushing to HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print("Push complete!")
    
    print(f"\n{'='*60}")
    print("✅ Model export complete!")
    print(f"Model path: {output_path}")
    print(f"{'='*60}\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./merged_model",
        help="Output path",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID",
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
