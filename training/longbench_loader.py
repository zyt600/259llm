"""
Original dataset training data loader - For SFT training

Load train split from original task datasets, avoiding LongBench test data (used for evaluation)

Supported original datasets:
- NarrativeQA: Reading comprehension
- Qasper: Scientific paper QA  
- HotpotQA: Multi-hop QA
- TriviaQA: Commonsense QA
- SAMSum: Dialogue summarization
- Multi-News: Multi-document summarization
- GovReport: Government report summarization

Note: qmsum not loaded as it's used for testing
"""
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
import random


# Original dataset configuration (for training)
# Note: Exclude qmsum as it's used for testing
ORIGINAL_DATASETS = {
    "narrativeqa": {
        "hf_name": "deepmind/narrativeqa",
        "task": "Reading comprehension",
        "description": "Reading comprehension based on novels and movie scripts",
        "split": "train",
        "context_key": "document.text",  # Nested field
        "question_key": "question.text",
        "answer_key": "answers",  # List
    },
    "qasper": {
        "hf_name": "allenai/qasper",
        "task": "Scientific paper QA",
        "description": "QA based on scientific papers",
        "split": "train",
        "needs_special_processing": True,
    },
    "hotpotqa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "task": "Multi-hop QA",
        "description": "Multi-hop reasoning QA",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answer",
    },
    "triviaqa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc",
        "task": "Commonsense QA",
        "description": "Commonsense QA",
        "split": "train",
        "context_key": "search_results.search_context",
        "question_key": "question",
        "answer_key": "answer.value",
    },
    "samsum": {
        "hf_name": "knkarthick/samsum",
        "task": "Dialogue summarization",
        "description": "Dialogue summarization",
        "split": "train",
        "context_key": "dialogue",
        "question_key": None,  # No question, direct summary
        "answer_key": "summary",
    },
    "multi_news": {
        "hf_name": "multi_news",
        "task": "Multi-doc summarization",
        "description": "Multi-document news summarization",
        "split": "train",
        "context_key": "document",
        "question_key": None,
        "answer_key": "summary",
    },
    "gov_report": {
        "hf_name": "ccdv/govreport-summarization",
        "task": "Long doc summarization",
        "description": "Government report summarization",
        "split": "train",
        "context_key": "report",
        "question_key": None,
        "answer_key": "summary",
    },
    "qmsum": {
        "hf_name": "pszemraj/qmsum-cleaned",
        "task": "Meeting summarization",
        "description": "Meeting summarization (only load train, test for evaluation)",
        "split": "train",  # Only load train, not test
        "context_key": "input",
        "question_key": None,
        "answer_key": "output",
    },
}

# Extra dataset configuration (for "all" mode)
EXTRA_DATASETS = {
    "flan_v2": {
        "hf_name": "Muennighoff/flan",
        "hf_config": "default",
        "task": "Instruction following",
        "description": "Google FLAN v2 instruction tuning dataset",
        "split": "train",
        "context_key": "inputs",
        "question_key": None,
        "answer_key": "targets",
    },
    # natural_questions dataset too large (40GB+), removed
    # "natural_questions": {...},
    "trivia_qa": {
        "hf_name": "trivia_qa",
        "hf_config": "rc",
        "task": "Commonsense QA",
        "description": "TriviaQA commonsense QA",
        "split": "train",
        "context_key": "search_results.search_context",
        "question_key": "question",
        "answer_key": "answer.value",
    },
    "hotpot_qa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "task": "Multi-hop QA",
        "description": "HotpotQA multi-hop reasoning QA",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answer",
    },
    "squad_v2": {
        "hf_name": "rajpurkar/squad_v2",
        "task": "Reading comprehension",
        "description": "SQuAD v2 reading comprehension (with unanswerable questions)",
        "split": "train",
        "context_key": "context",
        "question_key": "question",
        "answer_key": "answers.text",
    },
    "no_hallucination": {
        "hf_name": "LLM-Tuning-Safety/HEx-PHI",
        "task": "Anti-hallucination",
        "description": "No hallucination dataset, helps model avoid generating false information",
        "split": "train",
        "context_key": "prompt",
        "question_key": None,
        "answer_key": "response",
    },
    "boolq": {
        "hf_name": "google/boolq",
        "task": "Boolean QA",
        "description": "BoolQ yes/no judgment QA",
        "split": "train",
        "context_key": "passage",
        "question_key": "question",
        "answer_key": "answer",  # bool type, needs conversion
    },
}

# LongBench subset configuration for backward compatibility
LONGBENCH_SUBSETS = {
    "narrativeqa": {"task": "Single-doc QA", "description": "Reading comprehension"},
    "qasper": {"task": "Single-doc QA", "description": "Scientific paper QA"},
    "hotpotqa": {"task": "Multi-doc QA", "description": "Multi-hop QA"},
    "triviaqa": {"task": "Few-shot QA", "description": "Commonsense QA"},
    "samsum": {"task": "Summarization", "description": "Dialogue summarization"},
    "multi_news": {"task": "Summarization", "description": "Multi-doc summarization"},
    "gov_report": {"task": "Summarization", "description": "Government report summarization"},
    "qmsum": {"task": "Summarization", "description": "Meeting summarization"},
    # Extra datasets
    "flan_v2": {"task": "Instruction following", "description": "FLAN v2 instruction tuning"},
    "trivia_qa": {"task": "Commonsense QA", "description": "TriviaQA"},
    "hotpot_qa": {"task": "Multi-hop QA", "description": "HotpotQA"},
    "squad_v2": {"task": "Reading comprehension", "description": "SQuAD v2"},
    "no_hallucination": {"task": "Anti-hallucination", "description": "No hallucination data"},
    "boolq": {"task": "Boolean QA", "description": "BoolQ"},
}

# Get all subset names available for training (base datasets)
ALL_SUBSETS = list(ORIGINAL_DATASETS.keys())

# Extra dataset names list
EXTRA_SUBSET_NAMES = list(EXTRA_DATASETS.keys())

# All datasets (including extra datasets)
ALL_SUBSETS_WITH_EXTRA = ALL_SUBSETS + EXTRA_SUBSET_NAMES


def build_sft_prompt(subset_name: str, item: Dict) -> Tuple[str, str]:
    """
    Build SFT training prompt and response based on subset type
    
    Args:
        subset_name: Subset name
        item: Data sample
        
    Returns:
        Tuple[str, str]: (prompt, response)
    """
    context = item.get("context", "")
    input_text = item.get("input", "")
    
    # Get answer
    answers = item.get("answers", [])
    if isinstance(answers, list) and len(answers) > 0:
        answer = answers[0]
    else:
        answer = str(answers) if answers else ""
    
    # Build different prompts based on task type
    task_info = LONGBENCH_SUBSETS.get(subset_name, {})
    task_type = task_info.get("task", "QA")
    
    if task_type == "Summarization":
        # Summarization task
        if "qmsum" in subset_name or "vcsum" in subset_name:
            prompt = f"""Please answer the specified question based on the following meeting transcript.

Meeting transcript:
{context}

Question: {input_text}

Please provide a concise and accurate summary:"""
        elif "gov_report" in subset_name:
            prompt = f"""Please generate a summary for the following government report.

Report content:
{context}

Please generate summary:"""
        elif "multi_news" in subset_name:
            prompt = f"""Please generate a summary for the following news articles.

News content:
{context}

Please generate summary:"""
        elif "samsum" in subset_name:
            prompt = f"""Please generate a summary for the following dialogue.

Dialogue content:
{context}

{input_text}

Please generate summary:"""
        else:
            prompt = f"""Please generate a summary for the following content.

Content:
{context}

{input_text}

Summary:"""
    
    elif task_type in ["Single-doc QA", "Multi-doc QA", "Few-shot QA"]:
        # QA task
        prompt = f"""Please answer the question based on the following content.

Content:
{context}

Question: {input_text}

Answer:"""
    
    elif task_type == "Classification":
        # Classification task
        all_classes = item.get("all_classes", [])
        if all_classes:
            classes_str = ", ".join(all_classes)
            prompt = f"""Please classify the following content. Categories include: {classes_str}

Content:
{context}

Question: {input_text}

Classification result:"""
        else:
            prompt = f"""Please classify the following content.

Content:
{context}

Question: {input_text}

Classification result:"""
    
    elif task_type == "Code completion":
        # Code task
        prompt = f"""Please complete the following code.

{context}

{input_text}

Completed code:"""
    
    elif task_type == "Synthesis":
        # Synthesis task
        prompt = f"""Please read the following content and answer the question.

Content:
{context}

Question: {input_text}

Answer:"""
    
    else:
        # Default format
        prompt = f"""Please answer the question based on the following content.

Content:
{context}

Question: {input_text}

Answer:"""
    
    return prompt, answer


def _get_nested_value(item: Dict, key: str):
    """Get nested dictionary value, supports dot-separated paths"""
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
    """Process single sample from original dataset, convert to unified format"""
    
    # Get context
    context = _get_nested_value(item, config.get("context_key"))
    if isinstance(context, list):
        # Some datasets have context as list
        context = "\n\n".join([str(c) for c in context[:5]])  # Take at most first 5
    
    # Get question
    question = _get_nested_value(item, config.get("question_key")) or ""
    
    # Get answer
    answer_key = config.get("answer_key")
    answer = _get_nested_value(item, answer_key)
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    if isinstance(answer, dict):
        answer = answer.get("value", "") or answer.get("text", "") or str(answer)
    
    return {
        "context": str(context)[:50000],  # Limit length
        "input": str(question),
        "answers": [str(answer)] if answer else [""],
    }


def load_original_dataset(
    subset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load training data from original dataset
    
    Args:
        subset_name: Dataset name
        max_samples: Maximum samples
        
    Returns:
        List[Dict]: Data sample list
    """
    if subset_name not in ORIGINAL_DATASETS:
        print(f"Warning: {subset_name} not in supported original dataset list, skipping")
        return []
    
    config = ORIGINAL_DATASETS[subset_name]
    print(f"Loading original dataset: {subset_name} ({config['hf_name']})...", end=" ")
    
    try:
        # Load dataset
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
        print(f"Load failed: {e}")
        return []
    
    print(f"[{config['split']}]", end=" ")
    
    samples = []
    total = len(dataset)
    
    # If dataset too large, random sample
    if max_samples and total > max_samples:
        indices = random.sample(range(total), max_samples)
    else:
        indices = range(min(total, max_samples) if max_samples else total)
    
    for i in indices:
        item = dataset[i]
        
        # Convert to unified format
        processed = _process_original_dataset_item(subset_name, item, config)
        
        # Build SFT format
        prompt, response = build_sft_prompt(subset_name, processed)
        
        sample = {
            "id": f"{subset_name}_{i}",
            "subset": subset_name,
            "prompt": prompt,
            "response": response,
            "raw_context": processed["context"][:5000],  # Truncate for storage
            "raw_input": processed["input"],
            "raw_answers": processed["answers"],
            "full_prompt": prompt,
            "reference": response,
        }
        samples.append(sample)
    
    print(f"{len(samples)} samples total")
    return samples


def load_extra_dataset(
    subset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load extra dataset (using streaming mode to avoid downloading full dataset)
    
    Args:
        subset_name: Dataset name
        max_samples: Maximum samples
        
    Returns:
        List[Dict]: Data sample list
    """
    if subset_name not in EXTRA_DATASETS:
        print(f"Warning: {subset_name} not in extra dataset list, skipping")
        return []
    
    config = EXTRA_DATASETS[subset_name]
    print(f"Loading extra dataset: {subset_name} ({config['hf_name']})...", end=" ")
    
    try:
        # Use streaming=True to avoid downloading full dataset
        hf_config = config.get("hf_config")
        if hf_config:
            dataset = load_dataset(
                config["hf_name"],
                hf_config,
                split=config["split"],
                trust_remote_code=True,
                streaming=True  # Streaming load
            )
        else:
            dataset = load_dataset(
                config["hf_name"],
                split=config["split"],
                trust_remote_code=True,
                streaming=True  # Streaming load
            )
    except Exception as e:
        print(f"Load failed: {e}")
        return []
    
    print(f"[{config['split']} streaming]", end=" ")
    
    samples = []
    count = 0
    max_to_load = max_samples or 1300  # Default max 1300
    
    # Stream iterate dataset
    for i, item in enumerate(dataset):
        if len(samples) >= max_to_load:
            break
        
        # Process based on dataset type
        try:
            if subset_name == "flan_v2":
                context = item.get("inputs", "")
                question = ""
                answer = item.get("targets", "")
                prompt = f"""Please answer the following question or complete the following task:

{context}

Answer:"""
            
            elif subset_name == "trivia_qa" or subset_name == "triviaqa":
                search_results = item.get("search_results", {})
                contexts = search_results.get("search_context", [])
                context = "\n".join(contexts[:3]) if contexts else ""
                question = item.get("question", "")
                answer_dict = item.get("answer", {})
                answer = answer_dict.get("value", "") if isinstance(answer_dict, dict) else str(answer_dict)
                prompt = f"""Please answer the question based on the following content.

Content:
{context[:5000]}

Question: {question}

Answer:"""
            
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
                prompt = f"""Please answer the question based on the following content.

Content:
{context[:5000]}

Question: {question}

Answer:"""
            
            elif subset_name == "squad_v2":
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", {})
                answer_texts = answers.get("text", [])
                answer = answer_texts[0] if answer_texts else ""
                if not answer:
                    answer = "Cannot find answer from the given content."
                prompt = f"""Please answer the question based on the following content. If there's no answer in the content, please answer "Cannot find answer from the given content".

Content:
{context}

Question: {question}

Answer:"""
            
            elif subset_name == "no_hallucination":
                context = item.get("prompt", "")
                answer = item.get("response", "")
                prompt = f"""{context}

Answer:"""
            
            elif subset_name == "boolq":
                context = item.get("passage", "")
                question = item.get("question", "")
                answer_bool = item.get("answer", False)
                answer = "Yes" if answer_bool else "No"
                prompt = f"""Please determine if the answer to the question is "Yes" or "No" based on the following content.

Content:
{context}

Question: {question}

Answer (Yes/No):"""
            
            else:
                # Generic processing
                context = _get_nested_value(item, config.get("context_key")) or ""
                question = _get_nested_value(item, config.get("question_key")) or ""
                answer = _get_nested_value(item, config.get("answer_key")) or ""
                if isinstance(context, list):
                    context = "\n".join([str(c) for c in context[:5]])
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                prompt = f"""Please answer the question based on the following content.

Content:
{str(context)[:5000]}

Question: {question}

Answer:"""
            
            # Ensure valid content
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
            # Skip failed samples
            continue
    
    print(f"{len(samples)} samples total")
    return samples


def load_longbench_subset(
    subset_name: str,
    max_samples: Optional[int] = None,
    for_training: bool = True,
    prefer_train: bool = True
) -> List[Dict]:
    """
    Load dataset (prefer train from original dataset, otherwise test from LongBench)
    
    Note: qmsum won't be loaded (used for testing, avoid data leakage)
    
    Args:
        subset_name: Subset name
        max_samples: Maximum samples
        for_training: Whether for training
        prefer_train: Whether to prefer train split
        
    Returns:
        List[Dict]: Data sample list
    """
    # Prefer loading from original dataset (qmsum only loads train, not test)
    if subset_name in ORIGINAL_DATASETS and prefer_train:
        return load_original_dataset(subset_name, max_samples)
    
    # Fallback to LongBench
    print(f"Loading LongBench subset: {subset_name}...", end=" ")
    
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
        print(f"Load failed")
        return []
    
    print(f"[{split_used}]", end=" ")
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        # Build SFT format data
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
        
        # Keep raw data for evaluation
        if for_training:
            sample["full_prompt"] = prompt  # For inference
            sample["reference"] = response  # For evaluation
        
        samples.append(sample)
    
    print(f"{len(samples)} samples total")
    return samples


def load_all_longbench_subsets(
    subset_names: Optional[List[str]] = None,
    max_samples_per_subset: Optional[int] = None,
    max_total_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Load multiple LongBench subsets
    
    Args:
        subset_names: List of subsets to load, None for all
        max_samples_per_subset: Max samples per subset
        max_total_samples: Max total samples
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        List[Dict]: All data samples list
    """
    if subset_names is None:
        subset_names = ALL_SUBSETS
    
    print(f"\n{'='*60}")
    print(f"Loading LongBench Dataset")
    print(f"Subset count: {len(subset_names)}")
    print(f"{'='*60}")
    
    all_samples = []
    
    for subset_name in subset_names:
        if subset_name not in LONGBENCH_SUBSETS:
            print(f"Warning: Unknown subset {subset_name}, skipping")
            continue
        
        samples = load_longbench_subset(
            subset_name=subset_name,
            max_samples=max_samples_per_subset
        )
        all_samples.extend(samples)
    
    # Shuffle data
    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)
    
    # Limit total samples
    if max_total_samples is not None and len(all_samples) > max_total_samples:
        all_samples = all_samples[:max_total_samples]
    
    print(f"\nTotal loaded {len(all_samples)} training samples")
    
    # Count samples per subset
    subset_counts = {}
    for sample in all_samples:
        subset = sample["subset"]
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    
    print("\nSamples per subset:")
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
    Create HuggingFace Dataset for SFT training
    
    Args:
        samples: Data sample list
        tokenizer: Tokenizer
        max_length: Max sequence length
        truncation_side: Truncation direction
        
    Returns:
        Dataset: HuggingFace Dataset object
    """
    from datasets import Dataset as HFDataset
    
    # Set truncation direction
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    
    # Process data
    processed_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for sample in samples:
        prompt = sample["prompt"]
        response = sample["response"]
        
        # Build full text
        # Use chat template if available
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
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Calculate prompt length (for setting labels)
        prompt_tokenized = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        prompt_length = len(prompt_tokenized["input_ids"])
        
        # Create labels: set prompt part to -100 (don't compute loss)
        labels = input_ids.copy()
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100
        
        processed_data["input_ids"].append(input_ids)
        processed_data["attention_mask"].append(attention_mask)
        processed_data["labels"].append(labels)
    
    # Restore original settings
    tokenizer.truncation_side = original_truncation_side
    
    # Create Dataset
    dataset = HFDataset.from_dict(processed_data)
    
    return dataset


def get_subset_info(subset_name: str) -> Dict:
    """
    Get subset information
    
    Args:
        subset_name: Subset name
        
    Returns:
        Dict: Subset information
    """
    return LONGBENCH_SUBSETS.get(subset_name, {
        "task": "Unknown",
        "description": "Unknown subset"
    })


def list_all_subsets():
    """
    List all available subsets
    """
    print("\nLongBench all available subsets:")
    print("=" * 70)
    
    for subset_name, info in LONGBENCH_SUBSETS.items():
        print(f"\n{subset_name}:")
        print(f"  Task: {info['task']}")
        print(f"  Description: {info['description']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test loading
    print("Testing LongBench dataset loading...")
    
    # List all subsets
    list_all_subsets()
    
    # Test loading single subset
    samples = load_longbench_subset("qmsum", max_samples=3)
    if samples:
        print("\nExample data:")
        print(f"Prompt:\n{samples[0]['prompt'][:500]}...")
        print(f"\nResponse:\n{samples[0]['response'][:200]}...")
    
    # Test loading multiple subsets
    all_samples = load_all_longbench_subsets(
        subset_names=["qmsum", "narrativeqa"],
        max_samples_per_subset=5
    )
    print(f"\nLoaded {len(all_samples)} samples")
