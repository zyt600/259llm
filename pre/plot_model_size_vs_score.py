#!/usr/bin/env python3
"""
Plotting script - Model size vs composite score
Y-axis: Composite score (sum of three metrics)
  - QMSum ROUGE-L
  - TruthfulQA Accuracy
  - TruthfulQA Max Score (normalized)
X-axis: Model size (parameters, unit: B)
"""

import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.dirname(__file__)
BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "black_list.txt")
WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "white_list.txt")


def load_list_file(filepath, list_name):
    """Load list file (blacklist or whitelist)"""
    items = []
    if not os.path.exists(filepath):
        return items
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    items.append(line)
    except Exception as e:
        print(f"Warning: Failed to read {list_name} file: {e}")
    
    return items


def load_blacklist():
    """Load blacklist file"""
    return load_list_file(BLACKLIST_FILE, "blacklist")


def load_whitelist():
    """Load whitelist file"""
    return load_list_file(WHITELIST_FILE, "whitelist")


def get_display_name(model_name):
    """Get display name for model"""
    original = model_name.replace("_", "/", 1)
    
    if "/" in original:
        parts = original.split("/")
        model_short = parts[-1]
        
        if len(parts) > 1 and parts[0].lower() == "unsloth":
            return f"unsloth/{model_short}"
        
        return model_short
    
    return model_name


def get_display_name_for_blacklist(model_name):
    """Get display name for blacklist matching"""
    return get_display_name(model_name)


def is_in_list(model_name, name_list):
    """Check if model is in list"""
    display_name = get_display_name_for_blacklist(model_name)
    
    for list_item in name_list:
        item = list_item.strip()
        if not item:
            continue
        if display_name.lower() == item.lower():
            return True
    
    return False


def is_blacklisted(model_name, blacklist):
    """Check if model is blacklisted"""
    return is_in_list(model_name, blacklist)


def is_whitelisted(model_name, whitelist):
    """Check if model is whitelisted"""
    return is_in_list(model_name, whitelist)


def extract_model_size(model_name):
    """
    Extract model size from model name (parameters, unit: B)
    
    Examples:
        Llama-3.2-1B -> 1.0
        Qwen3-1.7B -> 1.7
        Qwen3-0.6B -> 0.6
        Qwen3-4B -> 4.0
    """
    # Replace underscores back to slashes
    original = model_name.replace("_", "/", 1)
    
    # Extract model name part (remove organization name)
    if "/" in original:
        model_short = original.split("/")[-1]
    else:
        model_short = original
    
    # Match common model size formats
    # Format 1: X.XB or XB (e.g. 1.7B, 4B)
    match = re.search(r'(\d+\.?\d*)[Bb]', model_short)
    if match:
        return float(match.group(1))
    
    # Format 2: -X.XB- or -XB- (e.g. -1.7B-)
    match = re.search(r'-(\d+\.?\d*)[Bb]-', model_short)
    if match:
        return float(match.group(1))
    
    # If cannot extract, return None
    return None


def parse_filename(filename):
    """Parse result filename, extract model name, dataset and temperature"""
    base = filename.replace("_results.json", "")
    
    match = re.search(r"_temp([\d.]+)$", base)
    if not match:
        return None, None, None
    
    temp = float(match.group(1))
    base = base[:match.start()]
    
    if base.endswith("_qmsum"):
        dataset = "qmsum"
        model = base[:-6]
    elif base.endswith("_truthfulqa"):
        dataset = "truthfulqa"
        model = base[:-11]
    else:
        return None, None, None
    
    return model, dataset, temp


def load_results(max_temp=1.0, target_temp=None, no_filter=False):
    """
    Load all result files
    
    Args:
        max_temp: Maximum temperature, only load data points with temp <= max_temp
        target_temp: If specified, only load data points with this temperature
        no_filter: If True, disable blacklist/whitelist filtering
    
    Returns: (data, filtered_models)
        data: {
            "qmsum_rougeL": {model: score},
            "truthfulqa_acc": {model: score},
            "truthfulqa_max_score": {model: score}
        }
        filtered_models: set of filtered model names
    """
    data = {
        "qmsum_rougeL": {},
        "truthfulqa_acc": {},
        "truthfulqa_max_score": {}
    }
    
    filtered_models = set()
    
    # Load whitelist and blacklist (reference draw/plot_results.py logic)
    whitelist = load_whitelist()
    blacklist = load_blacklist()
    
    # Whitelist priority: if whitelist has content, only consider whitelist, ignore blacklist
    use_whitelist = len(whitelist) > 0
    
    if not no_filter:
        if use_whitelist:
            print(f"Loaded whitelist: {len(whitelist)} entries (whitelist priority, ignoring blacklist)")
        elif blacklist:
            print(f"Loaded blacklist: {len(blacklist)} entries")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return data, filtered_models
    
    # Collect data for each model at different temperatures
    temp_data = defaultdict(lambda: {
        "qmsum_rougeL": {},
        "truthfulqa_acc": {},
        "truthfulqa_max_score": {}
    })
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith("_results.json"):
            continue
        
        model, dataset, temp = parse_filename(filename)
        if model is None:
            continue
        
        # Filter temperature
        if target_temp is not None:
            if abs(temp - target_temp) > 0.01:  # Allow small floating point error
                continue
        elif temp > max_temp:
            continue
        
        # Whitelist priority (reference draw/plot_results.py logic)
        if not no_filter:
            if use_whitelist:
                # If whitelist exists, only keep models in whitelist
                if not is_whitelisted(model, whitelist):
                    filtered_models.add(model)
                    continue
            else:
                # If no whitelist, use blacklist filter
                if is_blacklisted(model, blacklist):
                    filtered_models.add(model)
                    continue
        
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)
            
            if dataset == "qmsum":
                rougeL = result.get("rougeL", 0)
                temp_data[model]["qmsum_rougeL"][temp] = rougeL
            elif dataset == "truthfulqa":
                acc = result.get("accuracy", 0)
                max_score = result.get("avg_max_score", 0)
                temp_data[model]["truthfulqa_acc"][temp] = acc
                temp_data[model]["truthfulqa_max_score"][temp] = max_score
                
        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")
    
    # For each model, select data at specified temperature, if not specified select lowest temperature
    for model, model_data in temp_data.items():
        if target_temp is not None:
            # Use specified temperature
            target_t = target_temp
        else:
            # Select lowest temperature
            all_temps = set()
            for dataset_data in model_data.values():
                all_temps.update(dataset_data.keys())
            if all_temps:
                target_t = min(all_temps)
            else:
                continue
        
        # Get data at that temperature
        if target_t in model_data["qmsum_rougeL"]:
            data["qmsum_rougeL"][model] = model_data["qmsum_rougeL"][target_t]
        if target_t in model_data["truthfulqa_acc"]:
            data["truthfulqa_acc"][model] = model_data["truthfulqa_acc"][target_t]
        if target_t in model_data["truthfulqa_max_score"]:
            data["truthfulqa_max_score"][model] = model_data["truthfulqa_max_score"][target_t]
    
    return data, filtered_models


def normalize_max_score(max_score):
    """
    Normalize TruthfulQA Max Score
    Since Max Score is typically negative (e.g. -0.8), need to normalize to [0, 1] range
    Assuming range is [-1, 0], normalization formula: (score + 1) / 1
    """
    # Map [-1, 0] range to [0, 1]
    # If score = -1, normalized = 0
    # If score = 0, normalized = 1
    normalized = (max_score + 1.0) / 1.0
    # Limit to [0, 1] range
    return max(0.0, min(1.0, normalized))


def calculate_composite_score(rougeL, accuracy, max_score):
    """
    Calculate composite score (sum of three metrics)
    
    Args:
        rougeL: QMSum ROUGE-L score (0-1)
        accuracy: TruthfulQA Accuracy (0-1)
        max_score: TruthfulQA Max Score (needs normalization)
    
    Returns: Composite score
    """
    # Normalize max_score
    normalized_max_score = normalize_max_score(max_score)
    
    # Sum three metrics
    composite = rougeL + accuracy + normalized_max_score
    
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="Plot model size vs composite score",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Specify temperature, only use data at this temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=None,
        help="Maximum temperature, only use data points with temp <= max_temp (deprecated, use --temp instead)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable blacklist/whitelist filtering, show all models"
    )
    args = parser.parse_args()
    
    # If max_temp is specified, give warning and use temp=0.0
    if args.max_temp is not None:
        print("Warning: --max_temp parameter is deprecated, using --temp=0.0 instead")
        args.temp = 0.0
    
    print("Loading result data...")
    print(f"Using temperature: {args.temp}")
    if args.no_filter:
        print("Blacklist/whitelist filtering disabled, showing all models")
    
    data, filtered_models = load_results(max_temp=args.temp + 0.01, target_temp=args.temp, no_filter=args.no_filter)
    
    # Filter models (if filtering not disabled) - reference draw/plot_results.py logic
    if not args.no_filter:
        whitelist = load_whitelist()
        blacklist = load_blacklist()
        use_whitelist = len(whitelist) > 0
        
        # Whitelist priority: if whitelist has content, only consider whitelist, ignore blacklist
        for key in data:
            if use_whitelist:
                # Use whitelist: only keep models in whitelist
                models_to_remove = [model for model in data[key] if not is_whitelisted(model, whitelist)]
            else:
                # Use blacklist: remove models in blacklist
                models_to_remove = [model for model in data[key] if is_blacklisted(model, blacklist)]
            
            for model in models_to_remove:
                del data[key][model]
                filtered_models.add(model)
    
    if filtered_models and not args.no_filter:
        print("\n" + "="*60)
        whitelist = load_whitelist()
        use_whitelist = len(whitelist) > 0
        if use_whitelist:
            print("Models not in whitelist (filtered):")
        else:
            print("Models filtered by blacklist:")
        print("="*60)
        for model in sorted(filtered_models):
            print(f"  - {get_display_name(model)} ({model})")
        print(f"\nFiltered {len(filtered_models)} models total")
        print("="*60 + "\n")
    elif not args.no_filter:
        whitelist = load_whitelist()
        use_whitelist = len(whitelist) > 0
        if use_whitelist:
            print("\nAll models are in whitelist\n")
        else:
            print("\nNo models filtered by blacklist\n")
    
    # Collect all models with complete data
    models_with_all_data = set()
    for model in data["qmsum_rougeL"].keys():
        if model in data["truthfulqa_acc"] and model in data["truthfulqa_max_score"]:
            models_with_all_data.add(model)
    
    if not models_with_all_data:
        print("Error: No models found with both QMSum and TruthfulQA data")
        return
    
    print(f"Found {len(models_with_all_data)} models with complete data")
    
    # Calculate composite score and model size
    model_sizes = []
    composite_scores = []
    model_names = []
    
    for model in sorted(models_with_all_data):
        rougeL = data["qmsum_rougeL"][model]
        accuracy = data["truthfulqa_acc"][model]
        max_score = data["truthfulqa_max_score"][model]
        
        composite = calculate_composite_score(rougeL, accuracy, max_score)
        size = extract_model_size(model)
        
        if size is not None:
            model_sizes.append(size)
            composite_scores.append(composite)
            model_names.append(get_display_name(model))
        else:
            print(f"Warning: Cannot extract model size: {model}")
    
    if not model_sizes:
        print("Error: No models found with extractable model size")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot scatter
    scatter = ax.scatter(model_sizes, composite_scores, s=100, alpha=0.6, edgecolors='black', linewidths=1)
    
    # Add model name labels
    for i, name in enumerate(model_names):
        ax.annotate(name, 
                   (model_sizes[i], composite_scores[i]),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel("Model Size (Billion Parameters)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Composite Score\n(QMSum ROUGE-L + TruthfulQA Accuracy + Normalized Max Score)", 
                  fontsize=14, fontweight='bold')
    
    # Set title
    ax.set_title(f"Model Size vs Composite Score (Temperature: {args.temp})", fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis range
    ax.set_xlim(left=0)
    if model_sizes:
        ax.set_xlim(right=max(model_sizes) * 1.1)
    
    # Add trend line (optional)
    if len(model_sizes) > 1:
        z = np.polyfit(model_sizes, composite_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(model_sizes), max(model_sizes), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label=f'Trend line (slope={z[0]:.3f})')
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save image
    output_filename = f"model_size_vs_score_temp{args.temp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nImage saved to: {output_path}")
    
    plt.close()
    
    # Print data summary
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"{'Model':<40} {'Size (B)':<12} {'Composite Score':<15} {'ROUGE-L':<10} {'Accuracy':<10} {'Max Score':<10}")
    print("-"*100)
    
    # Sort by model size
    sorted_indices = sorted(range(len(model_sizes)), key=lambda i: model_sizes[i])
    for i in sorted_indices:
        model = model_names[i]
        size = model_sizes[i]
        score = composite_scores[i]
        # Find corresponding original model name
        for orig_model in models_with_all_data:
            if get_display_name(orig_model) == model:
                rougeL = data["qmsum_rougeL"][orig_model]
                accuracy = data["truthfulqa_acc"][orig_model]
                max_score = data["truthfulqa_max_score"][orig_model]
                break
        
        print(f"{model:<40} {size:<12.2f} {score:<15.4f} {rougeL:<10.4f} {accuracy:<10.4f} {max_score:<10.4f}")


if __name__ == "__main__":
    main()
