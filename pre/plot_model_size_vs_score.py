#!/usr/bin/env python3
"""
绘图脚本 - 模型大小 vs 综合分数
纵轴：综合分数（三个指标相加）
  - QMSum ROUGE-L
  - TruthfulQA Accuracy
  - TruthfulQA Max Score（归一化）
横轴：模型大小（参数量，单位：B）
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

# 结果目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.dirname(__file__)
BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "black_list.txt")
WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "white_list.txt")


def load_list_file(filepath, list_name):
    """加载列表文件（黑名单或白名单）"""
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
        print(f"警告: 读取{list_name}文件失败: {e}")
    
    return items


def load_blacklist():
    """加载黑名单文件"""
    return load_list_file(BLACKLIST_FILE, "黑名单")


def load_whitelist():
    """加载白名单文件"""
    return load_list_file(WHITELIST_FILE, "白名单")


def get_display_name(model_name):
    """获取模型的显示名称"""
    original = model_name.replace("_", "/", 1)
    
    if "/" in original:
        parts = original.split("/")
        model_short = parts[-1]
        
        if len(parts) > 1 and parts[0].lower() == "unsloth":
            return f"unsloth/{model_short}"
        
        return model_short
    
    return model_name


def get_display_name_for_blacklist(model_name):
    """获取模型的显示名称（用于黑名单匹配）"""
    return get_display_name(model_name)


def is_in_list(model_name, name_list):
    """检查模型是否在列表中"""
    display_name = get_display_name_for_blacklist(model_name)
    
    for list_item in name_list:
        item = list_item.strip()
        if not item:
            continue
        if display_name.lower() == item.lower():
            return True
    
    return False


def is_blacklisted(model_name, blacklist):
    """检查模型是否在黑名单中"""
    return is_in_list(model_name, blacklist)


def is_whitelisted(model_name, whitelist):
    """检查模型是否在白名单中"""
    return is_in_list(model_name, whitelist)


def extract_model_size(model_name):
    """
    从模型名称中提取模型大小（参数量，单位：B）
    
    示例:
        Llama-3.2-1B -> 1.0
        Qwen3-1.7B -> 1.7
        Qwen3-0.6B -> 0.6
        Qwen3-4B -> 4.0
    """
    # 将下划线替换回斜杠
    original = model_name.replace("_", "/", 1)
    
    # 提取模型名部分（去掉组织名）
    if "/" in original:
        model_short = original.split("/")[-1]
    else:
        model_short = original
    
    # 匹配常见的模型大小格式
    # 格式1: X.XB 或 XB (如 1.7B, 4B)
    match = re.search(r'(\d+\.?\d*)[Bb]', model_short)
    if match:
        return float(match.group(1))
    
    # 格式2: -X.XB- 或 -XB- (如 -1.7B-)
    match = re.search(r'-(\d+\.?\d*)[Bb]-', model_short)
    if match:
        return float(match.group(1))
    
    # 如果无法提取，返回 None
    return None


def parse_filename(filename):
    """解析结果文件名，提取模型名称、数据集和温度"""
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
    加载所有结果文件
    
    参数:
        max_temp: 最高温度，只加载温度 <= max_temp 的数据点
        target_temp: 如果指定，只加载该温度的数据点
        no_filter: 如果为True，不进行黑名单/白名单过滤
    
    返回: (data, filtered_models)
        data: {
            "qmsum_rougeL": {model: score},
            "truthfulqa_acc": {model: score},
            "truthfulqa_max_score": {model: score}
        }
        filtered_models: set 被过滤掉的模型集合
    """
    data = {
        "qmsum_rougeL": {},
        "truthfulqa_acc": {},
        "truthfulqa_max_score": {}
    }
    
    filtered_models = set()
    
    # 加载白名单和黑名单（参考 draw/plot_results.py 的逻辑）
    whitelist = load_whitelist()
    blacklist = load_blacklist()
    
    # 白名单优先：如果白名单有内容，则只考虑白名单，忽略黑名单
    use_whitelist = len(whitelist) > 0
    
    if not no_filter:
        if use_whitelist:
            print(f"已加载白名单: {len(whitelist)} 个条目 (优先使用白名单，忽略黑名单)")
        elif blacklist:
            print(f"已加载黑名单: {len(blacklist)} 个条目")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 结果目录不存在: {RESULTS_DIR}")
        return data, filtered_models
    
    # 收集每个模型在不同温度下的数据
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
        
        # 过滤温度
        if target_temp is not None:
            if abs(temp - target_temp) > 0.01:  # 允许小的浮点误差
                continue
        elif temp > max_temp:
            continue
        
        # 白名单优先（参考 draw/plot_results.py 的逻辑）
        if not no_filter:
            if use_whitelist:
                # 如果有白名单，只保留白名单中的模型
                if not is_whitelisted(model, whitelist):
                    filtered_models.add(model)
                    continue
            else:
                # 如果没有白名单，使用黑名单过滤
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
            print(f"警告: 读取 {filename} 失败: {e}")
    
    # 对于每个模型，选择指定温度的数据，如果没有指定温度则选择温度最低的数据
    for model, model_data in temp_data.items():
        if target_temp is not None:
            # 使用指定温度
            target_t = target_temp
        else:
            # 选择最低温度
            all_temps = set()
            for dataset_data in model_data.values():
                all_temps.update(dataset_data.keys())
            if all_temps:
                target_t = min(all_temps)
            else:
                continue
        
        # 获取该温度下的数据
        if target_t in model_data["qmsum_rougeL"]:
            data["qmsum_rougeL"][model] = model_data["qmsum_rougeL"][target_t]
        if target_t in model_data["truthfulqa_acc"]:
            data["truthfulqa_acc"][model] = model_data["truthfulqa_acc"][target_t]
        if target_t in model_data["truthfulqa_max_score"]:
            data["truthfulqa_max_score"][model] = model_data["truthfulqa_max_score"][target_t]
    
    return data, filtered_models


def normalize_max_score(max_score):
    """
    归一化 TruthfulQA Max Score
    由于 Max Score 通常是负数（如 -0.8），需要归一化到 [0, 1] 范围
    假设范围是 [-1, 0]，归一化公式: (score + 1) / 1
    """
    # 将 [-1, 0] 范围映射到 [0, 1]
    # 如果 score = -1，归一化后 = 0
    # 如果 score = 0，归一化后 = 1
    normalized = (max_score + 1.0) / 1.0
    # 限制在 [0, 1] 范围内
    return max(0.0, min(1.0, normalized))


def calculate_composite_score(rougeL, accuracy, max_score):
    """
    计算综合分数（三个指标相加）
    
    参数:
        rougeL: QMSum ROUGE-L 分数 (0-1)
        accuracy: TruthfulQA Accuracy (0-1)
        max_score: TruthfulQA Max Score (需要归一化)
    
    返回: 综合分数
    """
    # 归一化 max_score
    normalized_max_score = normalize_max_score(max_score)
    
    # 三个指标相加
    composite = rougeL + accuracy + normalized_max_score
    
    return composite


def main():
    parser = argparse.ArgumentParser(
        description="绘制模型大小 vs 综合分数图",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="指定温度，只使用该温度的数据（默认: 0.0）"
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=None,
        help="最高温度，只使用温度 <= max_temp 的数据点（已弃用，使用 --temp 代替）"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="禁用黑名单/白名单过滤，显示所有模型"
    )
    args = parser.parse_args()
    
    # 如果指定了 max_temp，给出警告并使用 temp=0.0
    if args.max_temp is not None:
        print("警告: --max_temp 参数已弃用，使用 --temp=0.0 代替")
        args.temp = 0.0
    
    print("正在加载结果数据...")
    print(f"使用温度: {args.temp}")
    if args.no_filter:
        print("已禁用黑名单/白名单过滤，将显示所有模型")
    
    data, filtered_models = load_results(max_temp=args.temp + 0.01, target_temp=args.temp, no_filter=args.no_filter)
    
    # 过滤模型（如果未禁用过滤）- 参考 draw/plot_results.py 的逻辑
    if not args.no_filter:
        whitelist = load_whitelist()
        blacklist = load_blacklist()
        use_whitelist = len(whitelist) > 0
        
        # 白名单优先：如果白名单有内容，则只考虑白名单，忽略黑名单
        for key in data:
            if use_whitelist:
                # 使用白名单：只保留白名单中的模型
                models_to_remove = [model for model in data[key] if not is_whitelisted(model, whitelist)]
            else:
                # 使用黑名单：移除黑名单中的模型
                models_to_remove = [model for model in data[key] if is_blacklisted(model, blacklist)]
            
            for model in models_to_remove:
                del data[key][model]
                filtered_models.add(model)
    
    if filtered_models and not args.no_filter:
        print("\n" + "="*60)
        whitelist = load_whitelist()
        use_whitelist = len(whitelist) > 0
        if use_whitelist:
            print("不在白名单中的模型（已过滤）:")
        else:
            print("被黑名单过滤掉的模型:")
        print("="*60)
        for model in sorted(filtered_models):
            print(f"  - {get_display_name(model)} ({model})")
        print(f"\n共过滤 {len(filtered_models)} 个模型")
        print("="*60 + "\n")
    elif not args.no_filter:
        whitelist = load_whitelist()
        use_whitelist = len(whitelist) > 0
        if use_whitelist:
            print("\n所有模型都在白名单中\n")
        else:
            print("\n没有模型被黑名单过滤\n")
    
    # 收集所有有完整数据的模型
    models_with_all_data = set()
    for model in data["qmsum_rougeL"].keys():
        if model in data["truthfulqa_acc"] and model in data["truthfulqa_max_score"]:
            models_with_all_data.add(model)
    
    if not models_with_all_data:
        print("错误: 没有找到同时具有 QMSum 和 TruthfulQA 数据的模型")
        return
    
    print(f"找到 {len(models_with_all_data)} 个具有完整数据的模型")
    
    # 计算综合分数和模型大小
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
            print(f"警告: 无法提取模型大小: {model}")
    
    if not model_sizes:
        print("错误: 没有找到可以提取模型大小的模型")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制散点图
    scatter = ax.scatter(model_sizes, composite_scores, s=100, alpha=0.6, edgecolors='black', linewidths=1)
    
    # 添加模型名称标签
    for i, name in enumerate(model_names):
        ax.annotate(name, 
                   (model_sizes[i], composite_scores[i]),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   alpha=0.7)
    
    # 设置坐标轴标签
    ax.set_xlabel("Model Size (Billion Parameters)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Composite Score\n(QMSum ROUGE-L + TruthfulQA Accuracy + Normalized Max Score)", 
                  fontsize=14, fontweight='bold')
    
    # 设置标题
    ax.set_title(f"Model Size vs Composite Score (Temperature: {args.temp})", fontsize=16, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置坐标轴范围
    ax.set_xlim(left=0)
    if model_sizes:
        ax.set_xlim(right=max(model_sizes) * 1.1)
    
    # 添加趋势线（可选）
    if len(model_sizes) > 1:
        z = np.polyfit(model_sizes, composite_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(model_sizes), max(model_sizes), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label=f'Trend line (slope={z[0]:.3f})')
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_filename = f"model_size_vs_score_temp{args.temp}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    plt.close()
    
    # 打印数据摘要
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    print(f"{'Model':<40} {'Size (B)':<12} {'Composite Score':<15} {'ROUGE-L':<10} {'Accuracy':<10} {'Max Score':<10}")
    print("-"*100)
    
    # 按模型大小排序
    sorted_indices = sorted(range(len(model_sizes)), key=lambda i: model_sizes[i])
    for i in sorted_indices:
        model = model_names[i]
        size = model_sizes[i]
        score = composite_scores[i]
        # 找到对应的原始模型名
        for orig_model in models_with_all_data:
            if get_display_name(orig_model) == model:
                rougeL = data["qmsum_rougeL"][orig_model]
                accuracy = data["truthfulqa_acc"][orig_model]
                max_score = data["truthfulqa_max_score"][orig_model]
                break
        
        print(f"{model:<40} {size:<12.2f} {score:<15.4f} {rougeL:<10.4f} {accuracy:<10.4f} {max_score:<10.4f}")


if __name__ == "__main__":
    main()


    # 打印数据摘要
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    print(f"{'Model':<40} {'Size (B)':<12} {'Composite Score':<15} {'ROUGE-L':<10} {'Accuracy':<10} {'Max Score':<10}")
    print("-"*100)
    
    # 按模型大小排序
    sorted_indices = sorted(range(len(model_sizes)), key=lambda i: model_sizes[i])
    for i in sorted_indices:
        model = model_names[i]
        size = model_sizes[i]
        score = composite_scores[i]
        # 找到对应的原始模型名
        for orig_model in models_with_all_data:
            if get_display_name(orig_model) == model:
                rougeL = data["qmsum_rougeL"][orig_model]
                accuracy = data["truthfulqa_acc"][orig_model]
                max_score = data["truthfulqa_max_score"][orig_model]
                break
        
        print(f"{model:<40} {size:<12.2f} {score:<15.4f} {rougeL:<10.4f} {accuracy:<10.4f} {max_score:<10.4f}")


if __name__ == "__main__":
    main()

