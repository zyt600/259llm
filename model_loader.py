"""
Model loading module - Load and manage LLMs using SGLang or llama.cpp
"""
import os
from typing import Optional, List


def is_gguf_model(model_name: str) -> bool:
    """
    Check if model is in GGUF format
    
    Args:
        model_name: Model name or path
        
    Returns:
        bool: True if GGUF model
        - Files ending with .gguf are GGUF
        - HuggingFace repo names ending with -GGUF are GGUF
        - Local directories are not GGUF (they are HuggingFace format)
    """
    # If it's a local directory, it's not GGUF (it's HuggingFace format)
    if os.path.isdir(model_name):
        return False
    # Files ending with .gguf are GGUF
    if model_name.lower().endswith('.gguf'):
        return True
    # HuggingFace repo names ending with -GGUF are GGUF
    if model_name.endswith('-GGUF'):
        return True
    return False


def load_model_with_llama_cpp(
    model_path: str,
    verbose: bool = False
):
    """
    Load GGUF model using llama.cpp
    
    Args:
        model_path: GGUF model file path or HuggingFace repo name (ending with GGUF)
        verbose: Whether to print detailed information
        
    Returns:
        Llama: llama-cpp-python model instance
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "Please install llama-cpp-python: pip install llama-cpp-python\n"
            "For GPU support: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
            "Reference: https://github.com/abetlen/llama-cpp-python"
        )
    
    print(f"\nLoading model with llama.cpp: {model_path}")
    print(f"GPU layers: All")
    
    # Check if local file or HuggingFace repo
    # Local path: starts with ./ or /, or file exists
    is_local_path = model_path.startswith('./') or model_path.startswith('/')
    is_local_file = is_local_path or os.path.exists(model_path)
    
    if is_local_file:
        # Local GGUF file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local GGUF file not found: {model_path}")
        print(f"Loading from local file: {model_path}")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=32768,  # Set reasonable context length
            n_batch=5120,  # Increase batch for faster long prompt processing
            verbose=verbose
        )
    else:
        # Download from HuggingFace (repo name ends with GGUF)
        print("Downloading GGUF file from HuggingFace...")
        # Use from_pretrained to load from HuggingFace
        # Auto-select optimal quantization version (Q4_K_M is a good balance)
        model = Llama.from_pretrained(
            repo_id=model_path,
            filename="*Q4_K_M.gguf",  # Default to Q4_K_M quantization
            n_gpu_layers=-1,
            n_ctx=32768,  # Set reasonable context length
            n_batch=5120,  # Increase batch for faster long prompt processing
            verbose=verbose
        )
    
    # Get actual context length
    actual_n_ctx = model.n_ctx()
    print(f"Context length: {actual_n_ctx}")
    print("Model loaded successfully!\n")
    
    return model


def load_model_with_sglang(
    model_name: str,
    gpu_ids: List[int],
    tp_size: int = 1,
    mem_fraction: float = 0.7,
    trust_remote_code: bool = True
):
    """
    Load model using SGLang
    
    Args:
        model_name: Hugging Face model name
        gpu_ids: GPU ID list
        tp_size: Tensor parallel size
        mem_fraction: GPU memory usage fraction
        trust_remote_code: Whether to trust remote code
        
    Returns:
        sglang.Engine: SGLang engine instance
    """
    try:
        import sglang as sgl
    except ImportError:
        raise ImportError(
            "Please install sglang: pip install 'sglang[all]'\n"
            "Reference: https://github.com/sgl-project/sglang"
        )
    
    print(f"\nLoading model: {model_name}")
    print(f"Using GPUs: {gpu_ids}")
    print(f"Tensor parallel size: {tp_size}")
    
    # Create SGLang Engine
    engine = sgl.Engine(
        model_path=model_name,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction,
        trust_remote_code=trust_remote_code,
    )
    
    print("Model loaded successfully!\n")
    
    return engine


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    Load tokenizer
    
    Args:
        model_name: Hugging Face model name
        trust_remote_code: Whether to trust remote code
        
    Returns:
        tokenizer: HuggingFace tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


class ModelManager:
    """
    Model manager class - Unified management of model loading and inference
    Supports SGLang and llama.cpp backends
    llama.cpp supports multi-GPU parallel (each GPU loads a model instance)
    """
    
    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        tp_size: int = 1,
        mem_fraction: float = 0.7
    ):
        """
        Initialize model manager
        
        Args:
            model_name: Hugging Face model name or GGUF file path
            gpu_ids: GPU ID list
            tp_size: Tensor parallel size (SGLang only)
            mem_fraction: GPU memory usage fraction (SGLang only)
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.tp_size = tp_size
        self.mem_fraction = mem_fraction
        
        self.engine = None
        self.tokenizer = None
        self.backend = "llama_cpp" if is_gguf_model(model_name) else "sglang"
        # llama.cpp multi-GPU parallel mode
        self.multi_gpu_parallel = self.backend == "llama_cpp" and len(gpu_ids) > 1
        
    def load(self):
        """Load model and tokenizer"""
        if self.backend == "llama_cpp":
            print(f"Detected GGUF model, using llama.cpp backend")
            if self.multi_gpu_parallel:
                # Multi-GPU parallel mode: don't load model in main process, workers load separately
                print(f"[Multi-GPU parallel mode] Using {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
                print("Model will be loaded by each GPU worker process during inference")
                self.engine = None  # Placeholder, workers will load during inference
            else:
                # Single GPU mode
                self.engine = load_model_with_llama_cpp(
                    model_path=self.model_name
                )
            # llama.cpp doesn't need separate tokenizer loading, it has built-in tokenizer
            self.tokenizer = None
        else:
            print(f"Using SGLang backend")
            self.engine = load_model_with_sglang(
                model_name=self.model_name,
                gpu_ids=self.gpu_ids,
                tp_size=self.tp_size,
                mem_fraction=self.mem_fraction
            )
            self.tokenizer = load_tokenizer(self.model_name)
        
    def shutdown(self):
        """Shutdown model engine"""
        if self.engine is not None:
            if self.backend == "sglang":
                self.engine.shutdown()
                print("SGLang engine shutdown")
            else:
                # llama.cpp automatically releases resources when object is destroyed
                del self.engine
                self.engine = None
                print("llama.cpp model unloaded")
        elif self.multi_gpu_parallel:
            print("Multi-GPU parallel mode: worker processes cleaned up automatically")
            
    def __enter__(self):
        self.load()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
