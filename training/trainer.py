"""
SFT Trainer Module - Supports periodic evaluation and checkpoint saving
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

# Add parent directory to path for importing evaluation module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_combined_score(results: Dict) -> float:
    """
    Compute combined score
    
    Combined score = TruthfulQA accuracy + QMSum ROUGE-L + TruthfulQA avg_max_score
    
    Args:
        results: Dictionary containing qmsum and truthfulqa evaluation results
        
    Returns:
        float: Combined score
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
    """Evaluation results dataclass"""
    step: int
    timestamp: str
    results: Dict
    combined_score: float = 0.0
    

class PeriodicEvalCallback(TrainerCallback):
    """
    Periodic evaluation callback - Execute full test at specified steps
    """
    
    def __init__(
        self,
        eval_fn: Callable,
        eval_steps: int,
        output_dir: str,
        tokenizer,
    ):
        """
        Initialize evaluation callback
        
        Args:
            eval_fn: Evaluation function, accepts (model, tokenizer, step) parameters
            eval_steps: Evaluate every N steps
            output_dir: Output directory
            tokenizer: Tokenizer
        """
        self.eval_fn = eval_fn
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.tokenizer = tokenizer  # Save tokenizer reference
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
        """Check if evaluation needed at each step end"""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n{'='*70}")
            print(f"Step {state.global_step}: Starting full evaluation (TruthfulQA + QMSum full test set)")
            print(f"{'='*70}")
            
            # Execute evaluation
            try:
                results, combined_score = self.eval_fn(
                    model=model,
                    tokenizer=self.tokenizer,  # Use saved tokenizer
                    step=state.global_step,
                )
                
                # Record evaluation results
                eval_result = EvalResults(
                    step=state.global_step,
                    timestamp=datetime.now().isoformat(),
                    results=results,
                    combined_score=combined_score
                )
                self.eval_history.append(eval_result)
                
                # Save adapter at each evaluation (named by step)
                step_adapter_dir = os.path.join(self.output_dir, f"adapter_step_{state.global_step}")
                if model is not None:
                    model.save_pretrained(step_adapter_dir)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(step_adapter_dir)
                    print(f"\n💾 Adapter saved to: {step_adapter_dir}")
                
                # Check if best model
                if combined_score > self.best_score:
                    self.best_score = combined_score
                    self.best_step = state.global_step
                    print(f"🏆 New best model! Combined score: {combined_score:.4f}")
                    
                    # Also save to best_adapter
                    best_adapter_dir = os.path.join(self.output_dir, "best_adapter")
                    if model is not None:
                        model.save_pretrained(best_adapter_dir)
                        if self.tokenizer is not None:
                            self.tokenizer.save_pretrained(best_adapter_dir)
                
                # Save evaluation history
                self._save_eval_history()
                
                print(f"\nStep {state.global_step} evaluation complete")
                print(f"Combined score: {combined_score:.4f} (Best: {self.best_score:.4f} @ step {self.best_step})")
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                import traceback
                traceback.print_exc()
        
        return control
    
    def _save_eval_history(self):
        """Save evaluation history to file"""
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
        
        # Add best model info
        summary = {
            "best_step": self.best_step,
            "best_score": self.best_score,
            "history": history_data
        }
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation history saved to: {history_file}")


class SFTTrainer:
    """
    SFT Trainer class - Encapsulates complete training workflow
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
        Initialize SFT trainer
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
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """
        Load model and tokenizer
        """
        print(f"\nLoading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model to first visible GPU (device 0)
        # Note: Since CUDA_VISIBLE_DEVICES is set in script, device 0 is the first specified GPU
        device_map = {"": 0}
        
        # Determine data type
        if self.bf16:
            torch_dtype = torch.bfloat16
        elif self.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        print(f"  Using pure LoRA (dtype: {torch_dtype})")
        
        # Build model loading parameters (pure LoRA, no quantization)
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        # Only enable flash_attn if available
        if self._check_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        else:
            print("  Using default attention implementation")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing to save VRAM
        self.model.gradient_checkpointing_enable()
        
        # If using LoRA
        if self.use_lora:
            self._apply_lora()
        
        print(f"Model loaded!")
        print(f"  Parameters: {self._count_parameters()}")
        if self.use_lora:
            print(f"  Trainable parameters: {self._count_trainable_parameters()}")
    
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is supported"""
        try:
            import flash_attn
            import importlib.metadata
            # Ensure package metadata exists
            importlib.metadata.version("flash_attn")
            return True
        except (ImportError, Exception):
            return False
    
    def _apply_lora(self):
        """Apply pure LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Please install peft library: pip install peft")
        
        print(f"\nApplying LoRA config:")
        print(f"  r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        
        # LoRA configuration
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
        """Count model parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        if total >= 1e9:
            return f"{total/1e9:.2f}B"
        elif total >= 1e6:
            return f"{total/1e6:.2f}M"
        else:
            return f"{total/1e3:.2f}K"
    
    def _count_trainable_parameters(self) -> str:
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if total >= 1e9:
            return f"{total/1e9:.2f}B"
        elif total >= 1e6:
            return f"{total/1e6:.2f}M"
        else:
            return f"{total/1e3:.2f}K"
    
    def create_eval_function(self):
        """
        Create evaluation function for callback
        
        Flow:
        1. Save LoRA adapter
        2. Reload original model (float16) and merge LoRA
        3. Save merged model
        4. Use SGLang for direct inference
        5. Compute combined score
        """
        def eval_fn(model, tokenizer, step: int):
            """
            Execute full evaluation (TruthfulQA + QMSum full test set)
            Use SGLang for direct inference (no GGUF conversion)
            """
            import gc
            from model_loader import ModelManager
            from datasets_loader import load_dataset_by_name, get_dataset_info
            from inference import InferenceRunner
            from evaluator import evaluate_dataset, save_results, save_predictions, print_summary
            
            # Dataset list for evaluation
            eval_datasets = ["qmsum", "truthfulqa"]
            
            all_results = {}
            all_samples = {}
            
            # Temporary directories
            temp_adapter_dir = os.path.join(self.output_dir, f"_temp_adapter_step_{step}")
            temp_model_dir = os.path.join(self.output_dir, f"_temp_eval_step_{step}")
            os.makedirs(temp_adapter_dir, exist_ok=True)
            os.makedirs(temp_model_dir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Evaluation step {step}: Using SGLang inference")
            print(f"{'='*60}")
            
            # Set model to evaluation mode
            model.eval()
            
            # 1. Save LoRA adapter
            print(f"\n[1/3] Saving LoRA adapter...")
            model.save_pretrained(temp_adapter_dir)
            tokenizer.save_pretrained(temp_adapter_dir)
            print(f"  Adapter saved to: {temp_adapter_dir}")
            
            # 2. Reload original model (float16) and merge LoRA
            print(f"\n[2/3] Loading original model and merging LoRA...")
            try:
                from peft import PeftModel
                
                # Load original model (float16, no quantization)
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=True,
                )
                
                # Load and merge LoRA
                peft_model = PeftModel.from_pretrained(base_model, temp_adapter_dir)
                merged_model = peft_model.merge_and_unload()
                
                # Save merged model
                merged_model.save_pretrained(temp_model_dir, safe_serialization=True)
                tokenizer.save_pretrained(temp_model_dir)
                print(f"  Merged model saved to: {temp_model_dir}")
                
                # Release memory
                del base_model, peft_model, merged_model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Merge failed: {e}")
                raise
            
            # 3. Use SGLang for direct inference on merged model
            print(f"\n[3/3] Using SGLang inference...")
            print(f"  Model path: {temp_model_dir}")
            print(f"  Using GPUs: {self.gpu_ids}")
            
            try:
                with ModelManager(
                    model_name=temp_model_dir,  # Merged HuggingFace model directory
                    gpu_ids=self.gpu_ids,
                    tp_size=len(self.gpu_ids),  # Tensor parallel
                    mem_fraction=0.7
                ) as model_manager:
                    
                    # Create inference runner
                    runner = InferenceRunner(
                        engine=model_manager.engine,
                        temperature=0.0,
                        batch_size=8,
                        backend=model_manager.backend,  # Should be "sglang"
                        max_tokens=512,
                        model_path=temp_model_dir,
                        gpu_ids=self.gpu_ids
                    )
                    
                    # Iterate through each dataset for inference
                    for dataset_name in eval_datasets:
                        print(f"\n{'='*60}")
                        print(f"Dataset: {dataset_name.upper()} (full test set)")
                        print(f"{'='*60}")
                        
                        # Print dataset info
                        dataset_info = get_dataset_info(dataset_name)
                        print(f"Task: {dataset_info.get('task', 'N/A')}")
                        print(f"Description: {dataset_info.get('description', 'N/A')}")
                        
                        # Load full dataset
                        samples = load_dataset_by_name(
                            dataset_name=dataset_name,
                            max_samples=None  # Full test set
                        )
                        
                        if not samples:
                            print(f"Warning: Dataset {dataset_name} is empty, skipping")
                            continue
                        
                        # Run inference
                        print(f"\nStarting inference, {len(samples)} samples total...")
                        samples = runner.run(samples)
                        
                        # Save inference results
                        all_samples[dataset_name] = samples
                
                # Evaluation phase
                print(f"\n{'='*60}")
                print("Evaluation Phase")
                print(f"{'='*60}")
                
                for dataset_name, samples in all_samples.items():
                    print(f"\nEvaluating {dataset_name.upper()} dataset...")
                    
                    # Evaluate results
                    results = evaluate_dataset(dataset_name, samples)
                    results["step"] = step
                    results["timestamp"] = datetime.now().isoformat()
                    results["model_name"] = f"{self.model_name}_step{step}"
                    
                    # Save results
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
                
                # Print evaluation summary
                print_summary(all_results)
                
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_adapter_dir):
                        shutil.rmtree(temp_adapter_dir)
                    if os.path.exists(temp_model_dir):
                        shutil.rmtree(temp_model_dir)
                    print(f"\nTemporary files cleaned up")
                except:
                    pass
            
            # Restore training mode
            model.train()
            
            # Compute combined score
            combined_score = compute_combined_score(all_results)
            
            print(f"\n{'='*60}")
            print("Combined Score")
            print(f"{'='*60}")
            print(f"  QMSum ROUGE-L: {all_results.get('qmsum', {}).get('rougeL', 0):.4f}")
            print(f"  TruthfulQA Accuracy: {all_results.get('truthfulqa', {}).get('accuracy', 0):.4f}")
            print(f"  TruthfulQA Avg Max Score: {all_results.get('truthfulqa', {}).get('avg_max_score', 0):.4f}")
            print(f"  Combined score: {combined_score:.4f}")
            print(f"{'='*60}")
            
            return all_results, combined_score
        
        return eval_fn
    
    def train(self, train_dataset):
        """
        Execute training
        
        Args:
            train_dataset: HuggingFace Dataset object
        """
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Check if DeepSpeed config exists
        deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", None)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            max_steps=self.max_steps,  # -1 means unlimited
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
            dataloader_num_workers=0,  # Disable multiprocessing to avoid conflicts with BLEURT subprocess
            remove_unused_columns=False,
            report_to="none",  # Disable wandb etc
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # Gradient clipping to prevent gradient explosion
            deepspeed=deepspeed_config,  # DeepSpeed config
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
        
        # Create evaluation callback
        eval_callback = PeriodicEvalCallback(
            eval_fn=self.create_eval_function(),
            eval_steps=self.eval_steps,
            output_dir=self.output_dir,
            tokenizer=self.tokenizer,  # Pass tokenizer
        )
        
        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[eval_callback],
        )
        
        # Start training
        print(f"\n{'='*70}")
        print("Starting SFT Training")
        print(f"{'='*70}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps * len(self.gpu_ids)}")
        print(f"Eval interval: Every {self.eval_steps} steps (full test set)")
        print(f"Eval GPUs: {self.gpu_ids}")
        print(f"Combined score = TruthfulQA_accuracy + QMSum_rougeL + TruthfulQA_avg_max_score")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Train
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Final model saved to: {final_model_path}")
        print(f"Evaluation history saved to: {os.path.join(self.output_dir, 'eval_history.json')}")
        print(f"Best model: step {eval_callback.best_step}, combined score {eval_callback.best_score:.4f}")
        
        # Execute final evaluation
        print(f"\n{'='*70}")
        print("Executing Final Full Evaluation")
        print(f"{'='*70}")
        
        final_results, final_score = self.create_eval_function()(
            model=self.model,
            tokenizer=self.tokenizer,
            step=-1,  # -1 indicates final evaluation
        )
        
        # Save final evaluation results
        final_eval_data = {
            "combined_score": final_score,
            "best_step": eval_callback.best_step,
            "best_score": eval_callback.best_score,
            "results": final_results
        }
        
        final_eval_file = os.path.join(self.output_dir, "final_eval_results.json")
        with open(final_eval_file, "w", encoding="utf-8") as f:
            json.dump(final_eval_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFinal evaluation results saved to: {final_eval_file}")
        print(f"Final combined score: {final_score:.4f}")
        
        return final_results, final_score
    
    def save_checkpoint(self, step: int):
        """Save checkpoint"""
        if self.trainer is not None:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    # Test trainer initialization
    trainer = SFTTrainer(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        output_dir="./test_output",
        gpu_ids=[0],
        use_lora=True,
    )
    print("Trainer initialized successfully!")
