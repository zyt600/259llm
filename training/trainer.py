"""
SFT训练器模块 - 支持定期评估和检查点保存
"""
import os
import sys
import json
import time
import torch
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)

# 添加父目录到path以导入评估模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_combined_score(results: Dict) -> float:
    """
    计算综合分数
    
    综合分数 = TruthfulQA accuracy + QMSum ROUGE-L + TruthfulQA avg_max_score
    
    Args:
        results: 包含 qmsum 和 truthfulqa 评估结果的字典
        
    Returns:
        float: 综合分数
    """
    score = 0.0
    
    # QMSum ROUGE-L
    if "qmsum" in results:
        rouge_l = results["qmsum"].get("rougeL", 0.0)
        score += rouge_l
    
    # TruthfulQA accuracy + avg_max_score
    if "truthfulqa" in results:
        accuracy = results["truthfulqa"].get("accuracy", 0.0)
        avg_max_score = results["truthfulqa"].get("avg_max_score", 0.0)
        score += accuracy + avg_max_score
    
    return score


@dataclass
class EvalResults:
    """评估结果数据类"""
    step: int
    timestamp: str
    results: Dict
    combined_score: float = 0.0
    

class PeriodicEvalCallback(TrainerCallback):
    """
    定期评估回调 - 在指定步数执行全量测试
    """
    
    def __init__(
        self,
        eval_fn: Callable,
        eval_steps: int,
        output_dir: str,
        tokenizer,
    ):
        """
        初始化评估回调
        
        Args:
            eval_fn: 评估函数，接收 (model, tokenizer, step) 参数
            eval_steps: 每隔多少步评估一次
            output_dir: 输出目录
            tokenizer: 分词器
        """
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.tokenizer = tokenizer  # 保存tokenizer引用
        self.eval_history = []
        self.best_score = -float('inf')
        self.best_step = -1
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        """每步结束时检查是否需要评估"""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n{'='*70}")
            print(f"步骤 {state.global_step}: 开始全量评估 (TruthfulQA + QMSum 完整测试集)")
            print(f"{'='*70}")
            
            # 执行评估
            try:
                results, combined_score = self.eval_fn(
                    model=model,
                    tokenizer=self.tokenizer,  # 使用保存的tokenizer
                    step=state.global_step,
                )
                
                # 记录评估结果
                eval_result = EvalResults(
                    step=state.global_step,
                    timestamp=datetime.now().isoformat(),
                    results=results,
                    combined_score=combined_score
                )
                self.eval_history.append(eval_result)
                
                # 每次评估都保存 adapter（按步骤命名）
                step_adapter_dir = os.path.join(self.output_dir, f"adapter_step_{state.global_step}")
                if model is not None:
                    model.save_pretrained(step_adapter_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(step_adapter_dir)
                    print(f"\n💾 Adapter 已保存到: {step_adapter_dir}")
                
                # 检查是否是最佳模型
                if combined_score > self.best_score:
                    self.best_score = combined_score
                    self.best_step = state.global_step
                    print(f"🏆 新的最佳模型！综合分数: {combined_score:.4f}")
                    
                    # 同时保存一份到 best_adapter
                    best_adapter_dir = os.path.join(self.output_dir, "best_adapter")
                    if model is not None:
                        model.save_pretrained(best_adapter_dir)
                        if self.tokenizer is not None:
                            self.tokenizer.save_pretrained(best_adapter_dir)
                
                # 保存评估历史
                self._save_eval_history()
                
                print(f"\n步骤 {state.global_step} 评估完成")
                print(f"综合分数: {combined_score:.4f} (最佳: {self.best_score:.4f} @ step {self.best_step})")
                
            except Exception as e:
                print(f"评估出错: {e}")
                import traceback
                traceback.print_exc()
        
        return control
    
    def _save_eval_history(self):
        """保存评估历史到文件"""
        history_file = os.path.join(self.output_dir, "eval_history.json")
        
        history_data = [
            {
                "step": er.step,
                "timestamp": er.timestamp,
                "combined_score": er.combined_score,
                "results": er.results
            }
            for er in self.eval_history
        ]
        
        # 添加最佳模型信息
        summary = {
            "best_step": self.best_step,
            "best_score": self.best_score,
            "history": history_data
        }
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"评估历史已保存到: {history_file}")


class SFTTrainer:
    """
    SFT训练器类 - 封装完整的训练流程
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        gpu_ids: List[int],
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        max_steps: int = -1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        max_length: int = 4096,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        eval_steps: int = 500,
        save_steps: int = 500,
        logging_steps: int = 10,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bf16: bool = True,
        fp16: bool = False,
        seed: int = 42,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        初始化SFT训练器
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.gpu_ids = gpu_ids
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bf16 = bf16
        self.fp16 = fp16
        self.seed = seed
        self.resume_from_checkpoint = resume_from_checkpoint
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """
        加载模型和分词器
        """
        print(f"\n正在加载模型: {self.model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型到第一个可见GPU (device 0)
        # 注意：由于 CUDA_VISIBLE_DEVICES 已在脚本中设置，device 0 就是指定的第一个 GPU
        device_map = {"": 0}
        
        # 确定数据类型
        if self.bf16:
            torch_dtype = torch.bfloat16
        elif self.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        print(f"  使用纯 LoRA (数据类型: {torch_dtype})")
        
        # 构建模型加载参数（纯 LoRA，不使用量化）
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        # 只有确认 flash_attn 可用时才启用
        if self._check_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  使用 Flash Attention 2")
        else:
            print("  使用默认 Attention 实现")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # 启用gradient checkpointing以节省显存
        self.model.gradient_checkpointing_enable()
        
        # 如果使用LoRA
        if self.use_lora:
            self._apply_lora()
        
        print(f"模型加载完成！")
        print(f"  参数量: {self._count_parameters()}")
        if self.use_lora:
            print(f"  可训练参数量: {self._count_trainable_parameters()}")
    
    def _check_flash_attention(self) -> bool:
        """检查是否支持Flash Attention"""
        try:
            import flash_attn
            import importlib.metadata
            # 确保包元数据存在
            importlib.metadata.version("flash_attn")
            return True
        except (ImportError, Exception):
            return False
    
    def _apply_lora(self):
        """应用纯 LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("请安装peft库: pip install peft")
        
        print(f"\n应用 LoRA 配置:")
        print(f"  r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _count_parameters(self) -> str:
        """统计模型参数量"""
        total = sum(p.numel() for p in self.model.parameters())
        if total >= 1e9:
            return f"{total/1e9:.2f}B"
        elif total >= 1e6:
            return f"{total/1e6:.2f}M"
        else:
            return f"{total/1e3:.2f}K"
    
    def _count_trainable_parameters(self) -> str:
        """统计可训练参数量"""
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if total >= 1e9:
            return f"{total/1e9:.2f}B"
        elif total >= 1e6:
            return f"{total/1e6:.2f}M"
        else:
            return f"{total/1e3:.2f}K"
    
    def create_eval_function(self):
        """
        创建评估函数，用于回调
        
        流程：
        1. 保存 LoRA adapter
        2. 重新加载原始模型（float16）并合并 LoRA
        3. 保存合并后的模型
        4. 使用 SGLang 直接推理
        5. 计算综合分数
        """
        def eval_fn(model, tokenizer, step: int):
            """
            执行全量评估（TruthfulQA + QMSum 完整测试集）
            使用 SGLang 直接推理（不转换 GGUF）
            """
            import gc
            from model_loader import ModelManager
            from datasets_loader import load_dataset_by_name, get_dataset_info
            from inference import InferenceRunner
            from evaluator import evaluate_dataset, save_results, save_predictions, print_summary
            
            # 评估的数据集列表
            eval_datasets = ["qmsum", "truthfulqa"]
            
            all_results = {}
            all_samples = {}
            
            # 临时目录
            temp_adapter_dir = os.path.join(self.output_dir, f"_temp_adapter_step_{step}")
            temp_model_dir = os.path.join(self.output_dir, f"_temp_eval_step_{step}")
            os.makedirs(temp_adapter_dir, exist_ok=True)
            os.makedirs(temp_model_dir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"评估步骤 {step}: 使用 SGLang 推理")
            print(f"{'='*60}")
            
            # 设置模型为评估模式
            model.eval()
            
            # 1. 保存 LoRA adapter
            print(f"\n[1/3] 保存 LoRA adapter...")
            model.save_pretrained(temp_adapter_dir)
            tokenizer.save_pretrained(temp_adapter_dir)
            print(f"  Adapter 已保存到: {temp_adapter_dir}")
            
            # 2. 重新加载原始模型（float16）并合并 LoRA
            print(f"\n[2/3] 加载原始模型并合并 LoRA...")
            try:
                from peft import PeftModel
                
                # 加载原始模型（float16，不量化）
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=True,
                )
                
                # 加载并合并 LoRA
                peft_model = PeftModel.from_pretrained(base_model, temp_adapter_dir)
                merged_model = peft_model.merge_and_unload()
                
                # 保存合并后的模型
                merged_model.save_pretrained(temp_model_dir, safe_serialization=True)
                tokenizer.save_pretrained(temp_model_dir)
                print(f"  合并后模型已保存到: {temp_model_dir}")
                
                # 释放内存
                del base_model, peft_model, merged_model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  合并失败: {e}")
                raise
            
            # 3. 使用 SGLang 直接推理合并后的模型
            print(f"\n[3/3] 使用 SGLang 推理...")
            print(f"  模型路径: {temp_model_dir}")
            print(f"  使用 GPU: {self.gpu_ids}")
            
            try:
                with ModelManager(
                    model_name=temp_model_dir,  # 合并后的 HuggingFace 模型目录
                    gpu_ids=self.gpu_ids,
                    tp_size=len(self.gpu_ids),  # 张量并行
                    mem_fraction=0.7
                ) as model_manager:
                    
                    # 创建推理运行器
                    runner = InferenceRunner(
                        engine=model_manager.engine,
                        temperature=0.0,
                        batch_size=8,
                        backend=model_manager.backend,  # 应该是 "sglang"
                        max_tokens=512,
                        model_path=temp_model_dir,
                        gpu_ids=self.gpu_ids
                    )
                    
                    # 遍历每个数据集进行推理
                    for dataset_name in eval_datasets:
                        print(f"\n{'='*60}")
                        print(f"数据集: {dataset_name.upper()} (完整测试集)")
                        print(f"{'='*60}")
                        
                        # 打印数据集信息
                        dataset_info = get_dataset_info(dataset_name)
                        print(f"任务: {dataset_info.get('task', 'N/A')}")
                        print(f"描述: {dataset_info.get('description', 'N/A')}")
                        
                        # 加载完整数据集
                        samples = load_dataset_by_name(
                            dataset_name=dataset_name,
                            max_samples=None  # 完整测试集
                        )
                        
                        if not samples:
                            print(f"警告: 数据集 {dataset_name} 为空，跳过")
                            continue
                        
                        # 运行推理
                        print(f"\n开始推理，共 {len(samples)} 个样本...")
                        samples = runner.run(samples)
                        
                        # 保存推理结果
                        all_samples[dataset_name] = samples
                
                # 评估阶段
                print(f"\n{'='*60}")
                print("评估阶段")
                print(f"{'='*60}")
                
                for dataset_name, samples in all_samples.items():
                    print(f"\n正在评估 {dataset_name.upper()} 数据集...")
                    
                    # 评估结果
                    results = evaluate_dataset(dataset_name, samples)
                    results["step"] = step
                    results["timestamp"] = datetime.now().isoformat()
                    results["model_name"] = f"{self.model_name}_step{step}"
                    
                    # 保存结果
                    step_output_dir = os.path.join(self.output_dir, f"eval_step_{step}")
                    os.makedirs(step_output_dir, exist_ok=True)
                    
                    save_results(
                        results=results,
                        output_dir=step_output_dir,
                        model_name=f"{self.model_name}_step{step}",
                        dataset_name=dataset_name,
                        temperature=0.0
                    )
                    
                    save_predictions(
                        samples=samples,
                        output_dir=step_output_dir,
                        model_name=f"{self.model_name}_step{step}",
                        dataset_name=dataset_name,
                        temperature=0.0
                    )
                    
                    all_results[dataset_name] = results
                
                # 打印评估摘要
                print_summary(all_results)
                
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(temp_adapter_dir):
                        shutil.rmtree(temp_adapter_dir)
                    if os.path.exists(temp_model_dir):
                        shutil.rmtree(temp_model_dir)
                    print(f"\n已清理临时文件")
                except:
                    pass
            
            # 恢复训练模式
            model.train()
            
            # 计算综合分数
            combined_score = compute_combined_score(all_results)
            
            print(f"\n{'='*60}")
            print("综合评分")
            print(f"{'='*60}")
            print(f"  QMSum ROUGE-L: {all_results.get('qmsum', {}).get('rougeL', 0):.4f}")
            print(f"  TruthfulQA Accuracy: {all_results.get('truthfulqa', {}).get('accuracy', 0):.4f}")
            print(f"  TruthfulQA Avg Max Score: {all_results.get('truthfulqa', {}).get('avg_max_score', 0):.4f}")
            print(f"  综合分数: {combined_score:.4f}")
            print(f"{'='*60}")
            
            return all_results, combined_score
        
        return eval_fn
    
    def train(self, train_dataset):
        """
        执行训练
        
        Args:
            train_dataset: HuggingFace Dataset对象
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # 检查是否有 DeepSpeed 配置
        deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", None)
        
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            max_steps=self.max_steps,  # -1 表示不限制
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            bf16=self.bf16,
            fp16=self.fp16,
            seed=self.seed,
            dataloader_num_workers=0,  # 禁用多进程，避免与BLEURT子进程冲突
            remove_unused_columns=False,
            report_to="none",  # 禁用wandb等
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
            deepspeed=deepspeed_config,  # DeepSpeed 配置
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
        
        # 创建评估回调
        eval_callback = PeriodicEvalCallback(
            eval_fn=self.create_eval_function(),
            eval_steps=self.eval_steps,
            output_dir=self.output_dir,
            tokenizer=self.tokenizer,  # 传入tokenizer
        )
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[eval_callback],
        )
        
        # 开始训练
        print(f"\n{'='*70}")
        print("开始SFT训练")
        print(f"{'='*70}")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"有效批次大小: {self.batch_size * self.gradient_accumulation_steps * len(self.gpu_ids)}")
        print(f"评估间隔: 每 {self.eval_steps} 步 (完整测试集)")
        print(f"评估GPU: {self.gpu_ids}")
        print(f"综合分数 = TruthfulQA_accuracy + QMSum_rougeL + TruthfulQA_avg_max_score")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # 训练
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_model")
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("训练完成！")
        print(f"{'='*70}")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"最终模型保存到: {final_model_path}")
        print(f"评估历史保存到: {os.path.join(self.output_dir, 'eval_history.json')}")
        print(f"最佳模型: step {eval_callback.best_step}, 综合分数 {eval_callback.best_score:.4f}")
        
        # 执行最终评估
        print(f"\n{'='*70}")
        print("执行最终全量评估")
        print(f"{'='*70}")
        
        final_results, final_score = self.create_eval_function()(
            model=self.model,
            tokenizer=self.tokenizer,
            step=-1,  # -1表示最终评估
        )
        
        # 保存最终评估结果
        final_eval_data = {
            "combined_score": final_score,
            "best_step": eval_callback.best_step,
            "best_score": eval_callback.best_score,
            "results": final_results
        }
        
        final_eval_file = os.path.join(self.output_dir, "final_eval_results.json")
        with open(final_eval_file, "w", encoding="utf-8") as f:
            json.dump(final_eval_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n最终评估结果已保存到: {final_eval_file}")
        print(f"最终综合分数: {final_score:.4f}")
        
        return final_results, final_score
    
    def save_checkpoint(self, step: int):
        """保存检查点"""
        if self.trainer is not None:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"检查点已保存到: {checkpoint_dir}")


if __name__ == "__main__":
    # 测试训练器初始化
    trainer = SFTTrainer(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        output_dir="./test_output",
        gpu_ids=[0],
        use_lora=True,
    )
    print("训练器初始化成功！")
