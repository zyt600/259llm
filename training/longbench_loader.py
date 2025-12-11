"""
原始数据集训练数据加载器 - 用于SFT训练

从各任务的原始数据集加载 train split，避免使用 LongBench test 数据（会用于评估）

支持的原始数据集:
- NarrativeQA: 阅读理解
- Qasper: 科学论文问答  
- HotpotQA: 多跳问答
- TriviaQA: 常识问答
- SAMSum: 对话摘要
- Multi-News: 多文档摘要
- GovReport: 政府报告摘要

注意: qmsum 不加载，因为要用于测试
"""
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
import random


# 原始数据集配置（用于训练）
# 注意：排除 qmsum，因为要用于测试
ORIGINAL_DATASETS = {
    "narrativeqa": {
        "hf_name": "deepmind/narrativeqa",
        "task": "阅读理解",
        "description": "基于小说和电影剧本的阅读理解",
        "split": "train",
        "context_key": "document.text",  # 嵌套字段
        "question_key": "question.text",
        "answer_key": "answers",  # 列表
    },
    "qasper": {
        "hf_name": "allenai/qasper",
        "task": "科学论文问答",
        "description": "基于科学论文的问答",
        "split": "train",
        "needs_special_processing": True,
    },
    "hotpotqa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "task": "多跳问答",
        "description": "多跳推理问答",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answer",
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc",
        "task": "常识问答",
        "description": "常识问答",
        "split": "train",
        "context_key": "search_results.search_context",
        "question_key": "question",
        "answer_key": "answer.value",
    },
    "samsum": {
        "hf_name": "knkarthick/samsum",
        "task": "对话摘要",
        "description": "对话摘要",
        "split": "train",
        "context_key": "dialogue",
        "question_key": None,  # 无问题，直接摘要
        "answer_key": "summary",
    },
    "multi_news": {
        "hf_name": "multi_news",
        "task": "多文档摘要",
        "description": "多文档新闻摘要",
        "split": "train",
        "context_key": "document",
        "question_key": None,
        "answer_key": "summary",
    },
    "gov_report": {
        "hf_name": "ccdv/govreport-summarization",
        "task": "长文档摘要",
        "description": "政府报告摘要",
        "split": "train",
        "context_key": "report",
        "question_key": None,
        "answer_key": "summary",
    },
    "qmsum": {
        "hf_name": "pszemraj/qmsum-cleaned",
        "task": "会议摘要",
        "description": "会议摘要（只加载train，test用于评估）",
        "split": "train",  # 只加载 train，不加载 test
        "context_key": "input",
        "question_key": None,
        "answer_key": "output",
    },
}

# 额外数据集配置（用于 "all" 模式）
EXTRA_DATASETS = {
    "flan_v2": {
        "hf_name": "Muennighoff/flan",
        "hf_config": "default",
        "task": "指令跟随",
        "description": "Google FLAN v2 指令微调数据集",
        "split": "train",
        "context_key": "inputs",
        "question_key": None,
        "answer_key": "targets",
    },
    # natural_questions 数据集太大（40GB+），已移除
    # "natural_questions": {...},
    "trivia_qa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc",
        "task": "常识问答",
        "description": "TriviaQA 常识问答",
        "split": "train",
        "context_key": "search_results.search_context",
        "question_key": "question",
        "answer_key": "answer.value",
    },
    "hotpot_qa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "task": "多跳问答",
        "description": "HotpotQA 多跳推理问答",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answer",
    },
    "squad_v2": {
        "hf_name": "rajpurkar/squad_v2",
        "task": "阅读理解",
        "description": "SQuAD v2 阅读理解（含无答案问题）",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answers.text",
    },
    "no_hallucination": {
        "hf_name": "LLM-Tuning-Safety/HEx-PHI",
        "task": "抗幻觉",
        "description": "无幻觉数据集，帮助模型避免生成虚假信息",
        "split": "train",
        "context_key": "prompt",
        "question_key": None,
        "answer_key": "response",
    },
    "boolq": {
        "hf_name": "google/boolq",
        "task": "是非问答",
        "description": "BoolQ 是非判断问答",
        "split": "train",
        "context_key": "passage",
        "question_key": "question",
        "answer_key": "answer",  # bool类型，需要转换
    },
}

# 兼容旧代码的 LongBench 子集配置
LONGBENCH_SUBSETS = {
    "narrativeqa": {"task": "单文档QA", "description": "阅读理解"},
    "qasper": {"task": "单文档QA", "description": "科学论文问答"},
    "hotpotqa": {"task": "多文档QA", "description": "多跳问答"},
    "triviaqa": {"task": "Few-shot QA", "description": "常识问答"},
    "samsum": {"task": "摘要", "description": "对话摘要"},
    "multi_news": {"task": "摘要", "description": "多文档摘要"},
    "gov_report": {"task": "摘要", "description": "政府报告摘要"},
    "qmsum": {"task": "摘要", "description": "会议摘要"},
    # 额外数据集
    "flan_v2": {"task": "指令跟随", "description": "FLAN v2 指令微调"},
    "trivia_qa": {"task": "常识问答", "description": "TriviaQA"},
    "hotpot_qa": {"task": "多跳问答", "description": "HotpotQA"},
    "squad_v2": {"task": "阅读理解", "description": "SQuAD v2"},
    "no_hallucination": {"task": "抗幻觉", "description": "无幻觉数据"},
    "boolq": {"task": "是非问答", "description": "BoolQ"},
}

# 获取所有可用于训练的子集名称（基础数据集）
ALL_SUBSETS = list(ORIGINAL_DATASETS.keys())

# 额外数据集名称列表
EXTRA_SUBSET_NAMES = list(EXTRA_DATASETS.keys())

# 所有数据集（包括额外数据集）
ALL_SUBSETS_WITH_EXTRA = ALL_SUBSETS + EXTRA_SUBSET_NAMES


def build_sft_prompt(subset_name: str, item: Dict) -> Tuple[str, str]:
    """
    根据子集类型构建SFT训练的prompt和response
    
    Args:
        subset_name: 子集名称
        item: 数据样本
        
    Returns:
        Tuple[str, str]: (prompt, response)
    """
    context = item.get("context", "")
    input_text = item.get("input", "")
    
    # 获取答案
    answers = item.get("answers", [])
    if isinstance(answers, list) and len(answers) > 0:
        answer = answers[0]
    else:
        answer = str(answers) if answers else ""
    
    # 根据任务类型构建不同的prompt
    task_info = LONGBENCH_SUBSETS.get(subset_name, {})
    task_type = task_info.get("task", "QA")
    
    if task_type == "摘要":
        # 摘要任务
        if "qmsum" in subset_name or "vcsum" in subset_name:
            prompt = f"""请根据以下会议记录，回答指定的问题。

会议记录：
{context}

问题：{input_text}

请提供简洁准确的摘要："""
        elif "gov_report" in subset_name:
            prompt = f"""请为以下政府报告生成摘要。

报告内容：
{context}

请生成摘要："""
        elif "multi_news" in subset_name:
            prompt = f"""请为以下新闻文章生成摘要。

新闻内容：
{context}

请生成摘要："""
        elif "samsum" in subset_name:
            prompt = f"""请为以下对话生成摘要。

对话内容：
{context}

{input_text}

请生成摘要："""
        else:
            prompt = f"""请为以下内容生成摘要。

内容：
{context}

{input_text}

摘要："""
    
    elif task_type in ["单文档QA", "多文档QA", "Few-shot QA"]:
        # 问答任务
        prompt = f"""请根据以下内容回答问题。

内容：
{context}

问题：{input_text}

答案："""
    
    elif task_type == "分类":
        # 分类任务
        all_classes = item.get("all_classes", [])
        if all_classes:
            classes_str = ", ".join(all_classes)
            prompt = f"""请对以下内容进行分类，类别包括：{classes_str}

内容：
{context}

问题：{input_text}

分类结果："""
        else:
            prompt = f"""请对以下内容进行分类。

内容：
{context}

问题：{input_text}

分类结果："""
    
    elif task_type == "代码补全":
        # 代码任务
        prompt = f"""请完成以下代码。

{context}

{input_text}

补全的代码："""
    
    elif task_type == "合成":
        # 合成任务
        prompt = f"""请阅读以下内容并回答问题。

内容：
{context}

问题：{input_text}

答案："""
    
    else:
        # 默认格式
        prompt = f"""请根据以下内容回答问题。

内容：
{context}

问题：{input_text}

答案："""
    
    return prompt, answer


def _get_nested_value(item: Dict, key: str):
    """获取嵌套字典的值，支持点号分隔的路径"""
    if key is None:
        return ""
    keys = key.split(".")
    value = item
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, "")
        elif isinstance(value, list) and len(value) > 0:
            value = value[0].get(k, "") if isinstance(value[0], dict) else ""
        else:
            return ""
    return value if value else ""


def _process_original_dataset_item(subset_name: str, item: Dict, config: Dict) -> Dict:
    """处理原始数据集的单个样本，转换为统一格式"""
    
    # 获取 context
    context = _get_nested_value(item, config.get("context_key"))
    if isinstance(context, list):
        # 有些数据集的 context 是列表
        context = "\n\n".join([str(c) for c in context[:5]])  # 最多取前5个
    
    # 获取 question
    question = _get_nested_value(item, config.get("question_key")) or ""
    
    # 获取 answer
    answer_key = config.get("answer_key")
    answer = _get_nested_value(item, answer_key)
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if isinstance(answer, dict):
        answer = answer.get("value", "") or answer.get("text", "") or str(answer)
    
    return {
        "context": str(context)[:50000],  # 限制长度
        "input": str(question),
        "answers": [str(answer)] if answer else [""],
    }


def load_original_dataset(
    subset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    从原始数据集加载训练数据
    
    Args:
        subset_name: 数据集名称
        max_samples: 最大样本数
        
    Returns:
        List[Dict]: 数据样本列表
    """
    if subset_name not in ORIGINAL_DATASETS:
        print(f"警告: {subset_name} 不在支持的原始数据集列表中，跳过")
        return []
    
    config = ORIGINAL_DATASETS[subset_name]
    print(f"正在加载原始数据集: {subset_name} ({config['hf_name']})...", end=" ")
    
    try:
        # 加载数据集
        hf_config = config.get("hf_config")
        if hf_config:
            dataset = load_dataset(
                config["hf_name"],
                hf_config,
                split=config["split"],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                config["hf_name"],
                split=config["split"],
                trust_remote_code=True
            )
    except Exception as e:
        print(f"加载失败: {e}")
        return []
    
    print(f"[{config['split']}]", end=" ")
    
    samples = []
    total = len(dataset)
    
    # 如果数据量太大，随机采样
    if max_samples and total > max_samples:
        indices = random.sample(range(total), max_samples)
    else:
        indices = range(min(total, max_samples) if max_samples else total)
    
    for i in indices:
        item = dataset[i]
        
        # 转换为统一格式
        processed = _process_original_dataset_item(subset_name, item, config)
        
        # 构建SFT格式
        prompt, response = build_sft_prompt(subset_name, processed)
        
        sample = {
            "id": f"{subset_name}_{i}",
            "subset": subset_name,
            "prompt": prompt,
            "response": response,
            "raw_context": processed["context"][:5000],  # 截断保存
            "raw_input": processed["input"],
            "raw_answers": processed["answers"],
            "full_prompt": prompt,
            "reference": response,
        }
        samples.append(sample)
    
    print(f"共 {len(samples)} 个样本")
    return samples


def load_extra_dataset(
    subset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    加载额外数据集（使用 streaming 模式避免下载完整数据集）
    
    Args:
        subset_name: 数据集名称
        max_samples: 最大样本数
        
    Returns:
        List[Dict]: 数据样本列表
    """
    if subset_name not in EXTRA_DATASETS:
        print(f"警告: {subset_name} 不在额外数据集列表中，跳过")
        return []
    
    config = EXTRA_DATASETS[subset_name]
    print(f"正在加载额外数据集: {subset_name} ({config['hf_name']})...", end=" ")
    
    try:
        # 使用 streaming=True 避免下载完整数据集
        hf_config = config.get("hf_config")
        if hf_config:
            dataset = load_dataset(
                config["hf_name"],
                hf_config,
                split=config["split"],
                trust_remote_code=True,
                streaming=True  # 流式加载
            )
        else:
            dataset = load_dataset(
                config["hf_name"],
                split=config["split"],
                trust_remote_code=True,
                streaming=True  # 流式加载
            )
    except Exception as e:
        print(f"加载失败: {e}")
        return []
    
    print(f"[{config['split']} streaming]", end=" ")
    
    samples = []
    count = 0
    max_to_load = max_samples or 1300  # 默认最多1300条
    
    # 流式迭代数据集
    for i, item in enumerate(dataset):
        if len(samples) >= max_to_load:
            break
        
        # 根据数据集类型处理
        try:
            if subset_name == "flan_v2":
                context = item.get("inputs", "")
                question = ""
                answer = item.get("targets", "")
                prompt = f"""请回答以下问题或完成以下任务：

{context}

回答："""
            
            elif subset_name == "trivia_qa" or subset_name == "triviaqa":
                search_results = item.get("search_results", {})
                contexts = search_results.get("search_context", [])
                context = "\n".join(contexts[:3]) if contexts else ""
                question = item.get("question", "")
                answer_dict = item.get("answer", {})
                answer = answer_dict.get("value", "") if isinstance(answer_dict, dict) else str(answer_dict)
                prompt = f"""请根据以下内容回答问题。

内容：
{context[:5000]}

问题：{question}

答案："""
            
            elif subset_name == "hotpot_qa" or subset_name == "hotpotqa":
                context_data = item.get("context", {})
                if isinstance(context_data, dict):
                    titles = context_data.get("title", [])
                    sentences = context_data.get("sentences", [])
                    context_parts = []
                    for t, s in zip(titles, sentences):
                        context_parts.append(f"{t}: {' '.join(s)}")
                    context = "\n".join(context_parts[:5])
                else:
                    context = str(context_data)[:5000]
                question = item.get("question", "")
                answer = item.get("answer", "")
                prompt = f"""请根据以下内容回答问题。

内容：
{context[:5000]}

问题：{question}

答案："""
            
            elif subset_name == "squad_v2":
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", {})
                answer_texts = answers.get("text", [])
                answer = answer_texts[0] if answer_texts else ""
                if not answer:
                    answer = "无法从给定内容中找到答案。"
                prompt = f"""请根据以下内容回答问题。如果内容中没有答案，请回答"无法从给定内容中找到答案"。

内容：
{context}

问题：{question}

答案："""
            
            elif subset_name == "no_hallucination":
                context = item.get("prompt", "")
                answer = item.get("response", "")
                prompt = f"""{context}

回答："""
            
            elif subset_name == "boolq":
                context = item.get("passage", "")
                question = item.get("question", "")
                answer_bool = item.get("answer", False)
                answer = "是" if answer_bool else "否"
                prompt = f"""请根据以下内容判断问题的答案是"是"还是"否"。

内容：
{context}

问题：{question}

答案（是/否）："""
            
            else:
                # 通用处理
                context = _get_nested_value(item, config.get("context_key")) or ""
                question = _get_nested_value(item, config.get("question_key")) or ""
                answer = _get_nested_value(item, config.get("answer_key")) or ""
                if isinstance(context, list):
                    context = "\n".join([str(c) for c in context[:5]])
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                prompt = f"""请根据以下内容回答问题。

内容：
{str(context)[:5000]}

问题：{question}

答案："""
            
            # 确保有有效内容
            if not answer or not prompt:
                continue
            
            sample = {
                "id": f"{subset_name}_{i}",
                "subset": subset_name,
                "prompt": prompt,
                "response": str(answer),
                "raw_context": str(context)[:5000] if context else "",
                "raw_input": str(question) if question else "",
                "raw_answers": [str(answer)],
                "full_prompt": prompt,
                "reference": str(answer),
            }
            samples.append(sample)
            
        except Exception as e:
            # 跳过处理失败的样本
            continue
    
    print(f"共 {len(samples)} 个样本")
    return samples


def load_longbench_subset(
    subset_name: str,
    max_samples: Optional[int] = None,
    for_training: bool = True,
    prefer_train: bool = True
) -> List[Dict]:
    """
    加载数据集（优先从原始数据集加载train，否则从LongBench加载test）
    
    注意: qmsum 不会被加载（用于测试，避免数据泄露）
    
    Args:
        subset_name: 子集名称
        max_samples: 最大样本数
        for_training: 是否用于训练
        prefer_train: 是否优先加载train split
        
    Returns:
        List[Dict]: 数据样本列表
    """
    # 优先从原始数据集加载（qmsum 只加载 train，不加载 test）
    if subset_name in ORIGINAL_DATASETS and prefer_train:
        return load_original_dataset(subset_name, max_samples)
    
    # 回退到 LongBench
    print(f"正在加载LongBench子集: {subset_name}...", end=" ")
    
    dataset = None
    split_used = None
    
    sources = ["THUDM/LongBench", "namespace-Pt/LongBench"]
    
    for source in sources:
        if dataset is not None:
            break
        try:
            dataset = load_dataset(source, subset_name, split="test", trust_remote_code=True)
            split_used = "test"
            break
        except Exception:
            continue
    
    if dataset is None:
        print(f"加载失败")
        return []
    
    print(f"[{split_used}]", end=" ")
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        # 构建SFT格式的数据
        prompt, response = build_sft_prompt(subset_name, item)
        
        sample = {
            "id": f"{subset_name}_{i}",
            "subset": subset_name,
            "prompt": prompt,
            "response": response,
            "raw_context": item.get("context", ""),
            "raw_input": item.get("input", ""),
            "raw_answers": item.get("answers", []),
            "length": item.get("length", len(prompt)),
        }
        
        # 保留原始数据用于评估
        if for_training:
            sample["full_prompt"] = prompt  # 用于推理
            sample["reference"] = response  # 用于评估
        
        samples.append(sample)
    
    print(f"共 {len(samples)} 个样本")
    return samples


def load_all_longbench_subsets(
    subset_names: Optional[List[str]] = None,
    max_samples_per_subset: Optional[int] = None,
    max_total_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    加载LongBench的多个子集
    
    Args:
        subset_names: 要加载的子集列表，None表示全部
        max_samples_per_subset: 每个子集的最大样本数
        max_total_samples: 总最大样本数
        shuffle: 是否打乱数据
        seed: 随机种子
        
    Returns:
        List[Dict]: 所有数据样本的列表
    """
    if subset_names is None:
        subset_names = ALL_SUBSETS
    
    print(f"\n{'='*60}")
    print(f"加载LongBench数据集")
    print(f"子集数量: {len(subset_names)}")
    print(f"{'='*60}")
    
    all_samples = []
    
    for subset_name in subset_names:
        if subset_name not in LONGBENCH_SUBSETS:
            print(f"警告: 未知子集 {subset_name}，跳过")
            continue
        
        samples = load_longbench_subset(
            subset_name=subset_name,
            max_samples=max_samples_per_subset
        )
        all_samples.extend(samples)
    
    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)
    
    # 限制总样本数
    if max_total_samples is not None and len(all_samples) > max_total_samples:
        all_samples = all_samples[:max_total_samples]
    
    print(f"\n总计加载 {len(all_samples)} 个训练样本")
    
    # 统计各子集样本数
    subset_counts = {}
    for sample in all_samples:
        subset = sample["subset"]
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    print("\n各子集样本数:")
    for subset, count in sorted(subset_counts.items()):
        print(f"  {subset}: {count}")
    
    return all_samples


def create_sft_dataset(
    samples: List[Dict],
    tokenizer,
    max_length: int = 4096,
    truncation_side: str = "left"
) -> Dataset:
    """
    创建用于SFT训练的HuggingFace Dataset
    
    Args:
        samples: 数据样本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        truncation_side: 截断方向
        
    Returns:
        Dataset: HuggingFace Dataset对象
    """
    from datasets import Dataset as HFDataset
    
    # 设置截断方向
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    
    # 处理数据
    processed_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for sample in samples:
        prompt = sample["prompt"]
        response = sample["response"]
        
        # 构建完整文本
        # 使用 chat template 如果可用
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            try:
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except:
                full_text = f"{prompt}\n\n{response}"
        else:
            full_text = f"{prompt}\n\n{response}"
        
        # 分词
        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 计算prompt部分的长度（用于设置labels）
        prompt_tokenized = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        prompt_length = len(prompt_tokenized["input_ids"])
        
        # 创建labels：prompt部分设为-100（不计算loss）
        labels = input_ids.copy()
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100
        
        processed_data["input_ids"].append(input_ids)
        processed_data["attention_mask"].append(attention_mask)
        processed_data["labels"].append(labels)
    
    # 恢复原始设置
    tokenizer.truncation_side = original_truncation_side
    
    # 创建Dataset
    dataset = HFDataset.from_dict(processed_data)
    
    return dataset


def get_subset_info(subset_name: str) -> Dict:
    """
    获取子集信息
    
    Args:
        subset_name: 子集名称
        
    Returns:
        Dict: 子集信息
    """
    return LONGBENCH_SUBSETS.get(subset_name, {
        "task": "未知",
        "description": "未知子集"
    })


def list_all_subsets():
    """
    列出所有可用的子集
    """
    print("\nLongBench 所有可用子集:")
    print("=" * 70)
    
    for subset_name, info in LONGBENCH_SUBSETS.items():
        print(f"\n{subset_name}:")
        print(f"  任务: {info['task']}")
        print(f"  描述: {info['description']}")
        print(f"  长度: {info['max_length']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 测试加载
    print("测试加载LongBench数据集...")
    
    # 列出所有子集
    list_all_subsets()
    
    # 测试加载单个子集
    samples = load_longbench_subset("qmsum", max_samples=3)
    if samples:
        print("\n示例数据:")
        print(f"Prompt:\n{samples[0]['prompt'][:500]}...")
        print(f"\nResponse:\n{samples[0]['response'][:200]}...")
    
    # 测试加载多个子集
    all_samples = load_all_longbench_subsets(
        subset_names=["qmsum", "narrativeqa"],
        max_samples_per_subset=5
    )
    print(f"\n加载了 {len(all_samples)} 个样本")

