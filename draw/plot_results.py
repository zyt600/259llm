#!/usr/bin/env python3
"""
绘图脚本 - 根据实验结果绘制温度-分数曲线
每条线代表一个模型，横轴是温度，纵轴是分数
生成三张图：
  1. QMSum (ROUGE-L)
  2. TruthfulQA (BLEURT Accuracy)
  3. TruthfulQA (BLEURT Max Score)
"""

import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# 结果目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.dirname(__file__)
BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "black_list.txt")
WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "white_list.txt")


def load_list_file(filepath, list_name):
    """
    加载列表文件（黑名单或白名单）
    返回: 列表（去除空行和注释）
    """
    items = []
    if not os.path.exists(filepath):
        return items
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 忽略空行和以#开头的注释行
                if line and not line.startswith("#"):
                    items.append(line)
    except Exception as e:
        print(f"警告: 读取{list_name}文件失败: {e}")
    
    return items


def load_blacklist():
    """
    加载黑名单文件
    返回: 黑名单列表（去除空行和注释）
    """
    return load_list_file(BLACKLIST_FILE, "黑名单")


def load_whitelist():
    """
    加载白名单文件
    返回: 白名单列表（去除空行和注释）
    """
    return load_list_file(WHITELIST_FILE, "白名单")


def get_display_name_for_blacklist(model_name):
    """
    获取模型的显示名称（用于黑名单匹配）
    与图例中显示的名称一致（与 get_display_name 保持一致）
    """
    # 将下划线替换回斜杠，获取原始模型名
    original = model_name.replace("_", "/", 1)  # 只替换第一个
    
    # 如果包含斜杠
    if "/" in original:
        parts = original.split("/")
        model_short = parts[-1]
        
        # 如果是 unsloth 的模型，保留组织名来区分（因为可能有同名模型）
        if len(parts) > 1 and parts[0].lower() == "unsloth":
            return f"unsloth/{model_short}"
        
        # 其他情况只显示模型名
        return model_short
    
    return model_name


def is_in_list(model_name, name_list):
    """
    检查模型是否在列表中（白名单或黑名单）
    
    匹配规则：列表条目必须与图例中显示的名称完全一致（不区分大小写）
    
    示例：
    - 写 "gpt-oss-20b" : 只匹配图例中显示为 "gpt-oss-20b" 的模型
    - 写 "gpt-oss-20b-GGUF" : 只匹配图例中显示为 "gpt-oss-20b-GGUF" 的模型
    
    返回: True 如果在列表中，False 否则
    """
    # 获取显示名称（与图例一致）
    display_name = get_display_name_for_blacklist(model_name)
    
    # 检查显示名称是否在列表中（不区分大小写）
    for list_item in name_list:
        item = list_item.strip()
        if not item:
            continue
        
        # 不区分大小写的精确匹配
        if display_name.lower() == item.lower():
            return True
    
    return False


def is_blacklisted(model_name, blacklist):
    """
    检查模型是否在黑名单中
    
    返回: True 如果在黑名单中，False 否则
    """
    return is_in_list(model_name, blacklist)


def is_whitelisted(model_name, whitelist):
    """
    检查模型是否在白名单中
    
    返回: True 如果在白名单中，False 否则
    """
    return is_in_list(model_name, whitelist)


def parse_filename(filename):
    """
    解析结果文件名，提取模型名称、数据集和温度
    文件名格式: {model_name}_{dataset}_temp{temperature}_results.json
    """
    # 移除 _results.json 后缀
    base = filename.replace("_results.json", "")
    
    # 匹配温度部分
    match = re.search(r"_temp([\d.]+)$", base)
    if not match:
        return None, None, None
    
    temp = float(match.group(1))
    base = base[:match.start()]  # 移除温度部分
    
    # 匹配数据集部分 (qmsum 或 truthfulqa)
    if base.endswith("_qmsum"):
        dataset = "qmsum"
        model = base[:-6]  # 移除 _qmsum
    elif base.endswith("_truthfulqa"):
        dataset = "truthfulqa"
        model = base[:-11]  # 移除 _truthfulqa
    else:
        return None, None, None
    
    return model, dataset, temp


def load_results(max_temp=1.0):
    """
    加载所有结果文件
    参数:
        max_temp: 最高温度，只加载温度 <= max_temp 的数据点（默认: 1.0）
    返回: (data, filtered_models)
        data: {
            "qmsum_rougeL": {model: [(temp, score), ...]},
            "truthfulqa_acc": {model: [(temp, score), ...]},
            "truthfulqa_max_score": {model: [(temp, score), ...]}
        }
        filtered_models: set 被过滤掉的模型集合
    """
    data = {
        "qmsum_rougeL": defaultdict(list),
        "truthfulqa_acc": defaultdict(list),
        "truthfulqa_max_score": defaultdict(list)
    }
    
    # 加载白名单和黑名单
    whitelist = load_whitelist()
    blacklist = load_blacklist()
    
    # 白名单优先：如果白名单有内容，则只考虑白名单，忽略黑名单
    use_whitelist = len(whitelist) > 0
    
    if use_whitelist:
        print(f"已加载白名单: {len(whitelist)} 个条目 (优先使用白名单，忽略黑名单)")
    elif blacklist:
        print(f"已加载黑名单: {len(blacklist)} 个条目")
    
    # 收集被过滤的模型
    filtered_models = set()
    
    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 结果目录不存在: {RESULTS_DIR}")
        return data, filtered_models
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith("_results.json"):
            continue
        
        model, dataset, temp = parse_filename(filename)
        if model is None:
            continue
        
        # 过滤温度：只保留温度 <= max_temp 的数据点
        if temp > max_temp:
            continue
        
        # 白名单优先
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
                # QMSum: 获取 ROUGE-L
                rougeL = result.get("rougeL", 0)
                data["qmsum_rougeL"][model].append((temp, rougeL))
            elif dataset == "truthfulqa":
                # TruthfulQA: 获取 accuracy 和 avg_max_score
                acc = result.get("accuracy", 0)
                max_score = result.get("avg_max_score", 0)
                data["truthfulqa_acc"][model].append((temp, acc))
                data["truthfulqa_max_score"][model].append((temp, max_score))
                
        except Exception as e:
            print(f"警告: 读取 {filename} 失败: {e}")
    
    # 按温度排序，并再次过滤温度（确保所有数据点都 <= max_temp）
    for key in data:
        for model in data[key]:
            # 过滤掉超过 max_temp 的数据点
            data[key][model] = [(t, s) for t, s in data[key][model] if t <= max_temp]
            # 按温度排序
            data[key][model].sort(key=lambda x: x[0])
    
    return data, filtered_models


def get_display_name(model_name):
    """
    获取模型的显示名称（简化）
    对于 unsloth/ 开头的模型，保留组织名以区分相同名称的模型
    """
    # 将下划线替换回斜杠，获取原始模型名
    original = model_name.replace("_", "/", 1)  # 只替换第一个
    
    # 如果包含斜杠
    if "/" in original:
        parts = original.split("/")
        model_short = parts[-1]
        
        # 如果是 unsloth 的模型，保留组织名来区分（因为可能有同名模型）
        if len(parts) > 1 and parts[0].lower() == "unsloth":
            return f"unsloth/{model_short}"
        
        # 其他情况只显示模型名
        return model_short
    
    return model_name


def plot_dataset(data, dataset_name, ax, title, ylabel):
    """
    绘制单个数据集的图
    """
    # 使用 tab20 提供20种颜色，结合不同线型和标记确保每条线都不同
    colors = list(plt.cm.tab20.colors)  # 20种颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '8', 'H']  # 14种标记
    linestyles = ['-', '--', '-.', ':']  # 4种线型
    
    models = sorted(data.keys())
    
    for i, model in enumerate(models):
        points = data[model]
        temps = [p[0] for p in points]
        scores = [p[1] for p in points]
        
        # 确保每条线的组合都不同：颜色循环20，标记循环14，线型循环4
        # 理论上可以区分 20*14*4 = 1120 种不同的线
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[(i // len(colors)) % len(linestyles)]  # 每20个模型换一种线型
        label = get_display_name(model)
        
        ax.plot(temps, scores, 
                color=color, 
                marker=marker, 
                linestyle=linestyle,
                linewidth=2, 
                markersize=6,
                label=label)
    
    ax.set_xlabel("Temperature", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # 设置x轴刻度
    all_temps = set()
    for model in data:
        for t, _ in data[model]:
            all_temps.add(t)
    if all_temps:
        ax.set_xticks(sorted(all_temps))


def rank_models_by_best_combined_score(data):
    """
    枚举所有温度，为每个模型找到综合分数最高的温度组合
    综合分数 = qmsum rougel + truthfulqa accuracy + truthfulqa max bleurt
    
    每个模型只输出一个最佳温度组合
    
    参数:
        data: load_results 返回的数据字典
    
    返回: 
        [(model_name, display_name, best_temp, combined_score, qmsum_score, acc_score, max_score), ...] 
        按综合分数从高到低排序
    """
    results = []
    
    # 获取所有在三个数据集中都有数据的模型
    qmsum_models = set(data["qmsum_rougeL"].keys())
    acc_models = set(data["truthfulqa_acc"].keys())
    max_models = set(data["truthfulqa_max_score"].keys())
    
    # 取三个数据集的交集
    common_models = qmsum_models & acc_models & max_models
    
    for model in common_models:
        # 获取该模型的所有数据点
        qmsum_points = data["qmsum_rougeL"][model]
        acc_points = data["truthfulqa_acc"][model]
        max_points = data["truthfulqa_max_score"][model]
        
        # 转换为字典方便查找 {temp: score}
        qmsum_dict = {t: s for t, s in qmsum_points}
        acc_dict = {t: s for t, s in acc_points}
        max_dict = {t: s for t, s in max_points}
        
        # 获取三个数据集都有的温度
        common_temps = set(qmsum_dict.keys()) & set(acc_dict.keys()) & set(max_dict.keys())
        
        if not common_temps:
            continue
        
        # 枚举所有共同温度，找到综合分数最高的温度
        best_temp = None
        best_combined = float('-inf')
        best_qmsum = None
        best_acc = None
        best_max = None
        
        for temp in common_temps:
            qmsum_score = qmsum_dict[temp]
            acc_score = acc_dict[temp]
            max_score = max_dict[temp]
            combined = qmsum_score + acc_score + max_score
            
            if combined > best_combined:
                best_combined = combined
                best_temp = temp
                best_qmsum = qmsum_score
                best_acc = acc_score
                best_max = max_score
        
        if best_temp is not None:
            display_name = get_display_name(model)
            results.append((model, display_name, best_temp, best_combined, best_qmsum, best_acc, best_max))
    
    # 按综合分数从高到低排序
    results.sort(key=lambda x: x[3], reverse=True)
    
    return results


def print_model_ranking(data):
    """
    打印模型排名表（按综合分数从高到低）
    枚举所有温度，每个模型输出综合分数最高的温度组合
    """
    ranking = rank_models_by_best_combined_score(data)
    
    if not ranking:
        print("\n警告: 没有找到同时具有 QMSum 和 TruthfulQA 数据的模型")
        return
    
    print("\n" + "="*115)
    print("模型综合分数排名（从高到低）- 每个模型选取最佳温度")
    print("综合分数 = QMSum ROUGE-L + TruthfulQA Accuracy + TruthfulQA Max BLEURT")
    print("="*115)
    print(f"{'排名':<5} {'模型名称':<30} {'最佳温度':<10} {'综合分数':<8} {'ROUGE-L':<10} {'Accuracy':<10} {'Max BLEURT':<10}")
    print("-"*115)
    
    for i, (model, display_name, best_temp, combined, qmsum, acc, max_bleurt) in enumerate(ranking, 1):
        print(f"{i:<5} {display_name:<40}  {best_temp:<10.2f} {combined:<12.4f} {qmsum:<10.4f} {acc:<10.4f} {max_bleurt:<10.4f}")
    
    print("="*115)
    print(f"共 {len(ranking)} 个模型参与排名\n")
    
    return ranking


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="绘制实验结果图 - 温度-分数曲线",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=1.0,
        help="最高温度，只显示温度 <= max_temp 的数据点 (默认: 1.0)"
    )
    parser.add_argument(
        "--rank_only",
        action="store_true",
        help="仅输出排名，不绘图"
    )
    args = parser.parse_args()
    
    print("正在加载结果数据...")
    print(f"最高温度限制: {args.max_temp}")
    data, filtered_models = load_results(max_temp=args.max_temp)
    
    # 加载白名单和黑名单
    whitelist = load_whitelist()
    blacklist = load_blacklist()
    use_whitelist = len(whitelist) > 0
    
    # 过滤模型（在数据加载后再次过滤，确保完全清除）
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
    
    # 输出被过滤的模型
    if filtered_models:
        print("\n" + "="*60)
        if use_whitelist:
            print("不在白名单中的模型（已过滤）:")
        else:
            print("被黑名单过滤掉的模型:")
        print("="*60)
        for model in sorted(filtered_models):
            print(f"  - {get_display_name(model)} ({model})")
        print(f"\n共过滤 {len(filtered_models)} 个模型")
        print("="*60 + "\n")
    else:
        if use_whitelist:
            print("\n所有模型都在白名单中\n")
        else:
            print("\n没有模型被黑名单过滤\n")
    
    # 统计信息
    qmsum_models = len(data["qmsum_rougeL"])
    truthfulqa_acc_models = len(data["truthfulqa_acc"])
    truthfulqa_max_models = len(data["truthfulqa_max_score"])
    print(f"QMSum (ROUGE-L): {qmsum_models} 个模型")
    print(f"TruthfulQA (Accuracy): {truthfulqa_acc_models} 个模型")
    print(f"TruthfulQA (Max Score): {truthfulqa_max_models} 个模型")
    
    if qmsum_models == 0 and truthfulqa_acc_models == 0:
        print("错误: 没有找到任何结果数据")
        return
    
    # 创建图形 - 三张子图
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # 图1: QMSum ROUGE-L
    if data["qmsum_rougeL"]:
        plot_dataset(data["qmsum_rougeL"], "qmsum", axes[0], 
                     "QMSum (ROUGE-L)", 
                     "ROUGE-L Score")
    else:
        axes[0].text(0.5, 0.5, "No QMSum data", ha='center', va='center', fontsize=14)
        axes[0].set_title("QMSum (ROUGE-L)", fontsize=14)
    
    # 图2: TruthfulQA BLEURT Accuracy
    if data["truthfulqa_acc"]:
        plot_dataset(data["truthfulqa_acc"], "truthfulqa_acc", axes[1],
                     "TruthfulQA (BLEURT Accuracy)",
                     "Accuracy")
    else:
        axes[1].text(0.5, 0.5, "No TruthfulQA data", ha='center', va='center', fontsize=14)
        axes[1].set_title("TruthfulQA (BLEURT Accuracy)", fontsize=14)
    
    # 图3: TruthfulQA BLEURT Max Score
    if data["truthfulqa_max_score"]:
        plot_dataset(data["truthfulqa_max_score"], "truthfulqa_max", axes[2],
                     "TruthfulQA (BLEURT Max Score)",
                     "Max Score")
    else:
        axes[2].text(0.5, 0.5, "No TruthfulQA data", ha='center', va='center', fontsize=14)
        axes[2].set_title("TruthfulQA (BLEURT Max Score)", fontsize=14)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, "results_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 也保存为PDF（矢量图）
    # output_pdf = os.path.join(OUTPUT_DIR, "results_comparison.pdf")
    # plt.savefig(output_pdf, bbox_inches='tight')
    # print(f"PDF已保存到: {output_pdf}")
    
    plt.close()
    
    # 打印数据摘要
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    
    summary_items = [
        ("QMSum (ROUGE-L)", data["qmsum_rougeL"]),
        ("TruthfulQA (Accuracy)", data["truthfulqa_acc"]),
        ("TruthfulQA (Max Score)", data["truthfulqa_max_score"])
    ]
    
    for dataset_name, dataset_data in summary_items:
        if not dataset_data:
            continue
        print(f"\n{dataset_name}:")
        for model in sorted(dataset_data.keys()):
            points = dataset_data[model]
            temps = [p[0] for p in points]
            scores = [p[1] for p in points]
            print(f"  {get_display_name(model)}:")
            print(f"    温度范围: {min(temps):.2f} - {max(temps):.2f}")
            print(f"    分数范围: {min(scores):.4f} - {max(scores):.4f}")
            print(f"    数据点数: {len(points)}")
    
    # 输出模型综合分数排名
    print_model_ranking(data)


if __name__ == "__main__":
    main()

