"""
模型加载模块 - 使用SGLang或llama.cpp加载和管理大模型
"""
import os
from typing import Optional, List


def is_gguf_model(model_name: str) -> bool:
    """
    检查模型是否为GGUF格式
    
    Args:
        model_name: 模型名称或路径
        
    Returns:
        bool: 如果是GGUF模型返回True
        - 以.gguf结尾的文件视为GGUF
        - 以-GGUF结尾的HuggingFace仓库名视为GGUF
        - 本地目录不是GGUF（是HuggingFace格式）
    """
    # 如果是本地目录，则不是GGUF（是HuggingFace格式模型）
    if os.path.isdir(model_name):
        return False
    # 以.gguf结尾的文件是GGUF
    if model_name.lower().endswith('.gguf'):
        return True
    # 以-GGUF结尾的HuggingFace仓库名是GGUF
    if model_name.endswith('-GGUF'):
        return True
    return False


def load_model_with_llama_cpp(
    model_path: str,
    verbose: bool = False
):
    """
    使用llama.cpp加载GGUF模型
    
    Args:
        model_path: GGUF模型文件路径或HuggingFace仓库名(以GGUF结尾)
        verbose: 是否打印详细信息
        
    Returns:
        Llama: llama-cpp-python模型实例
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "请安装llama-cpp-python: pip install llama-cpp-python\n"
            "如需GPU支持，请使用: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
            "详情参考: https://github.com/abetlen/llama-cpp-python"
        )
    
    print(f"\n正在使用llama.cpp加载模型: {model_path}")
    print(f"GPU层数: 全部")
    
    # 判断是本地文件还是HuggingFace仓库
    # 本地路径：以./或/开头，或者文件存在
    is_local_path = model_path.startswith('./') or model_path.startswith('/')
    is_local_file = is_local_path or os.path.exists(model_path)
    
    if is_local_file:
        # 本地GGUF文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"本地GGUF文件不存在: {model_path}")
        print(f"从本地文件加载: {model_path}")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=32768,  # 设置合理的上下文长度
            n_batch=5120,  # 增大batch加速长prompt处理
            verbose=verbose
        )
    else:
        # 从HuggingFace下载 (仓库名以GGUF结尾)
        print("从HuggingFace下载GGUF文件...")
        # 使用from_pretrained从HuggingFace加载
        # 自动选择最优的量化版本（Q4_K_M是一个不错的平衡点）
        model = Llama.from_pretrained(
            repo_id=model_path,
            filename="*Q4_K_M.gguf",  # 默认选择Q4_K_M量化版本
            n_gpu_layers=-1,
            n_ctx=32768,  # 设置合理的上下文长度
            n_batch=5120,  # 增大batch加速长prompt处理
            verbose=verbose
        )
    
    # 获取实际的上下文长度
    actual_n_ctx = model.n_ctx()
    print(f"上下文长度: {actual_n_ctx}")
    print("模型加载完成！\n")
    
    return model


def load_model_with_sglang(
    model_name: str,
    gpu_ids: List[int],
    tp_size: int = 1,
    mem_fraction: float = 0.7,
    trust_remote_code: bool = True
):
    """
    使用SGLang加载模型
    
    Args:
        model_name: Hugging Face模型名称
        gpu_ids: GPU ID列表
        tp_size: 张量并行大小
        mem_fraction: GPU内存使用比例
        trust_remote_code: 是否信任远程代码
        
    Returns:
        sglang.Engine: SGLang引擎实例
    """
    try:
        import sglang as sgl
    except ImportError:
        raise ImportError(
            "请安装sglang: pip install 'sglang[all]'\n"
            "详情参考: https://github.com/sgl-project/sglang"
        )
    
    print(f"\n正在加载模型: {model_name}")
    print(f"使用GPU: {gpu_ids}")
    print(f"张量并行大小: {tp_size}")
    
    # 创建SGLang Engine
    engine = sgl.Engine(
        model_path=model_name,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction,
        trust_remote_code=trust_remote_code,
    )
    
    print("模型加载完成！\n")
    
    return engine


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    加载tokenizer
    
    Args:
        model_name: Hugging Face模型名称
        trust_remote_code: 是否信任远程代码
        
    Returns:
        tokenizer: HuggingFace tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


class ModelManager:
    """
    模型管理器类 - 统一管理模型的加载和推理
    支持SGLang和llama.cpp两种后端
    llama.cpp支持多GPU并行（每个GPU加载一个模型实例）
    """
    
    def __init__(
        self,
        model_name: str,
        gpu_ids: List[int],
        tp_size: int = 1,
        mem_fraction: float = 0.7
    ):
        """
        初始化模型管理器
        
        Args:
            model_name: Hugging Face模型名称或GGUF文件路径
            gpu_ids: GPU ID列表
            tp_size: 张量并行大小 (仅SGLang使用)
            mem_fraction: GPU内存使用比例 (仅SGLang使用)
        """
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.tp_size = tp_size
        self.mem_fraction = mem_fraction
        
        self.engine = None
        self.tokenizer = None
        self.backend = "llama_cpp" if is_gguf_model(model_name) else "sglang"
        # llama.cpp多GPU并行模式
        self.multi_gpu_parallel = self.backend == "llama_cpp" and len(gpu_ids) > 1
        
    def load(self):
        """加载模型和tokenizer"""
        if self.backend == "llama_cpp":
            print(f"检测到GGUF模型，使用llama.cpp后端")
            if self.multi_gpu_parallel:
                # 多GPU并行模式：不在主进程加载模型，由worker进程各自加载
                print(f"[多GPU并行模式] 使用 {len(self.gpu_ids)} 个GPU: {self.gpu_ids}")
                print("模型将在推理时由各个GPU worker进程分别加载")
                self.engine = None  # 占位符，实际推理时worker会加载
            else:
                # 单GPU模式
                self.engine = load_model_with_llama_cpp(
                    model_path=self.model_name
                )
            # llama.cpp不需要单独加载tokenizer，它内置了
            self.tokenizer = None
        else:
            print(f"使用SGLang后端")
            self.engine = load_model_with_sglang(
                model_name=self.model_name,
                gpu_ids=self.gpu_ids,
                tp_size=self.tp_size,
                mem_fraction=self.mem_fraction
            )
            self.tokenizer = load_tokenizer(self.model_name)
        
    def shutdown(self):
        """关闭模型引擎"""
        if self.engine is not None:
            if self.backend == "sglang":
                self.engine.shutdown()
                print("SGLang引擎已关闭")
            else:
                # llama.cpp会在对象销毁时自动释放资源
                del self.engine
                self.engine = None
                print("llama.cpp模型已卸载")
        elif self.multi_gpu_parallel:
            print("多GPU并行模式：worker进程已自动清理")
            
    def __enter__(self):
        self.load()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

