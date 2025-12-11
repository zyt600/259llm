"""
推理模块 - 使用SGLang或llama.cpp进行批量推理
支持多GPU并行推理和Server模式并发推理
"""
from typing import List, Dict, Optional, Tuple
import time
import os
import re
import subprocess
import signal
import requests
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from threading import Thread, Lock
import queue

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

console = Console()


def _send_request_to_server(args):
    """
    发送单个请求到llama.cpp server (OpenAI兼容API)
    
    Args:
        args: (idx, prompt, server_url, temperature, max_tokens)
    
    Returns:
        (idx, output_text, success) - success 表示请求是否成功
    """
    idx, prompt, server_url, temperature, max_tokens = args
    
    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature if temperature > 0 else 0.0,
                "top_p": 1.0 if temperature == 0 else 0.95,
                "stream": False
            },
            timeout=600  # 10分钟超时
        )
        response.raise_for_status()
        result = response.json()
        # OpenAI格式的响应
        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0].get("text", "")
        else:
            text = ""
        return (idx, text, True)
    except Exception as e:
        print(f"\n请求错误 (idx={idx}, server={server_url}): {e}")
        return (idx, "", False)


def batch_inference_llama_cpp_server(
    model_path: str,
    prompts: List[str],
    gpu_ids: List[int],
    temperature: float = 0.0,
    max_tokens: int = 512,
    servers_per_gpu: int = 2,
    show_progress: bool = True
) -> List[str]:
    """
    使用llama.cpp server模式进行并发推理
    在每个GPU上启动多个server实例，实现真正的并行处理
    
    Args:
        model_path: 模型路径或HuggingFace仓库名
        prompts: 输入prompt列表
        gpu_ids: GPU ID列表
        temperature: 采样温度
        max_tokens: 最大生成token数
        servers_per_gpu: 每个GPU启动的server数量（需要足够显存）
        show_progress: 是否显示进度条
        
    Returns:
        List[str]: 生成的文本列表（保持原始顺序）
    """
    from huggingface_hub import hf_hub_download
    import glob
    
    total_samples = len(prompts)
    num_gpus = len(gpu_ids)
    total_servers = num_gpus * servers_per_gpu
    
    print(f"\n[Server模式] 使用 {num_gpus} 个GPU: {gpu_ids}")
    print(f"  每GPU {servers_per_gpu} 个server实例, 总共 {total_servers} 个server")
    
    # 检查是否需要下载模型
    if os.path.exists(model_path):
        gguf_path = model_path
    else:
        # 从HuggingFace下载
        print(f"  从HuggingFace下载模型...")
        
        # 检查是否是包含文件名的路径（如 unsloth/Qwen3-1.7B-Q4_0.gguf）
        # 如果是，尝试从对应的GGUF仓库下载
        if model_path.endswith('.gguf') and '/' in model_path:
            # 提取文件名和可能的仓库名
            # 例如: unsloth/Qwen3-1.7B-Q4_0.gguf
            # 应该映射到: 仓库 unsloth/Qwen3-1.7B-GGUF, 文件 Qwen3-1.7B-Q4_0.gguf
            parts = model_path.rsplit('/', 1)
            if len(parts) == 2:
                org_part = parts[0]  # 如 unsloth
                filename_part = parts[1]   # 如 Qwen3-1.7B-Q4_0.gguf
                
                # 从文件名中提取基础模型名（去掉量化后缀）
                # 例如: 
                #   Qwen3-1.7B-Q4_0.gguf -> Qwen3-1.7B
                #   Qwen3-1.7B-Q8_K_XL.gguf -> Qwen3-1.7B
                #   Qwen3-1.7B-UD-Q8_K_XL.gguf -> Qwen3-1.7B-UD
                # 量化后缀格式: -Q4_0, -Q4_K_M, -Q8_K_XL, -UD-Q8_K_XL 等
                
                # 特殊处理：如果文件名包含 -UD-，先提取 UD 部分
                base = filename_part.replace('.gguf', '')
                if '-UD-' in base:
                    # 例如: Qwen3-1.7B-UD-Q8_K_XL -> Qwen3-1.7B-UD
                    base_model_name = base.rsplit('-UD-', 1)[0] + '-UD'
                else:
                    # 匹配常见的量化后缀模式
                    patterns = [
                        # 匹配 Q8_K_XL, Q4_K_M 等格式（包含多个下划线部分）
                        r'^(.+?)-(Q\d+(_[KM])+(_[A-Z]+)?)\.gguf$',
                        # 匹配 Q4_0, Q8_0 等简单格式
                        r'^(.+?)-(Q\d+_\d+)\.gguf$',
                        # 匹配 Q4_K, Q8_M 等格式
                        r'^(.+?)-(Q\d+_[KM])\.gguf$',
                        # 匹配 Q4, Q8 等格式
                        r'^(.+?)-(Q\d+[KM]?)\.gguf$',
                        # 匹配 BF16, F16, IQ4_XS 等格式
                        r'^(.+?)-(BF16|F16|IQ4_[NX]S?)\.gguf$',
                    ]
                    
                    base_model_name = None
                    for pattern in patterns:
                        match = re.match(pattern, filename_part)
                        if match:
                            base_model_name = match.group(1)
                            break
                    
                    if base_model_name is None:
                        # 如果无法匹配，尝试去掉最后的 -xxx 部分
                        if '-' in base:
                            base_model_name = base.rsplit('-', 1)[0]
                        else:
                            base_model_name = base
                
                # 构建可能的仓库名
                # 对于 UD 版本（如 Qwen3-1.7B-UD），使用去掉 -UD 的基础模型名来构建仓库
                # 因为 UD 版本通常也在同一个 GGUF 仓库中
                repo_base_model = base_model_name
                if base_model_name.endswith('-UD'):
                    repo_base_model = base_model_name[:-3]  # 去掉 -UD
                
                # 策略1: {org}/{base_model}-GGUF (如 unsloth/Qwen3-1.7B-GGUF)
                # 策略2: {org}/{base_model} (如 unsloth/Qwen3-1.7B)
                possible_repos = [
                    f"{org_part}/{repo_base_model}-GGUF",  # unsloth/Qwen3-1.7B-GGUF
                    f"{org_part}/{repo_base_model}",  # unsloth/Qwen3-1.7B
                ]
                
                downloaded = False
                for repo_id in possible_repos:
                    try:
                        # 先尝试下载指定的文件名
                        gguf_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename_part,
                            local_dir="./models_cache"
                        )
                        downloaded = True
                        print(f"  从仓库 {repo_id} 下载文件 {filename_part}")
                        break
                    except Exception as e:
                        # 如果指定文件不存在，尝试下载默认的Q4_K_M版本（仅对-GGUF仓库）
                        if repo_id.endswith('-GGUF'):
                            try:
                                gguf_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename="*Q4_K_M.gguf",
                                    local_dir="./models_cache"
                                )
                                downloaded = True
                                print(f"  从仓库 {repo_id} 下载默认Q4_K_M版本（文件 {filename_part} 不存在）")
                                break
                            except:
                                continue
                        continue
                
                if not downloaded:
                    raise ValueError(f"无法从 {possible_repos} 下载模型文件 {filename_part}")
            else:
                # 如果解析失败，按原逻辑处理
                raise ValueError(f"无法解析模型路径: {model_path}")
        else:
            # 原逻辑：作为HuggingFace仓库名处理
            try:
                # 尝试下载Q4_K_M版本
                gguf_path = hf_hub_download(
                    repo_id=model_path,
                    filename="*Q4_K_M.gguf",
                    local_dir="./models_cache"
                )
            except:
                # 列出仓库中的gguf文件并下载第一个
                from huggingface_hub import list_repo_files
                try:
                    files = list_repo_files(model_path)
                    gguf_files = [f for f in files if f.endswith('.gguf') and 'Q4_K_M' in f]
                    if not gguf_files:
                        gguf_files = [f for f in files if f.endswith('.gguf')]
                    if gguf_files:
                        gguf_path = hf_hub_download(
                            repo_id=model_path,
                            filename=gguf_files[0],
                            local_dir="./models_cache"
                        )
                    else:
                        raise ValueError(f"仓库 {model_path} 中没有找到GGUF文件")
                except Exception as e:
                    raise ValueError(f"无法从HuggingFace下载模型 {model_path}: {e}")
        
        print(f"  模型下载完成: {gguf_path}")
    
    # 启动server进程（每个GPU多个实例）
    servers = []
    server_urls = []
    base_port = 8080
    
    print(f"  启动 {total_servers} 个llama.cpp server...")
    
    server_idx = 0
    for gpu_id in gpu_ids:
        for instance in range(servers_per_gpu):
            port = base_port + server_idx
            server_url = f"http://localhost:{port}"
            server_urls.append(server_url)
            
            # 启动server进程
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # 设置CUDA库路径
            cuda_lib_paths = [
                "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib",
                "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib",
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ]
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths) + ":" + existing_ld_path
            
            cmd = [
                "python", "-m", "llama_cpp.server",
                "--model", gguf_path,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--n_gpu_layers", "-1",
                "--n_ctx", "32768",
                "--n_batch", "2048",
            ]
            
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            servers.append(proc)
            print(f"    GPU {gpu_id} #{instance+1}: server启动中 (port={port}, pid={proc.pid})")
            server_idx += 1
    
    # 等待所有server启动（30B模型需要较长时间）
    print(f"  等待server就绪（大模型加载需要1-2分钟）...")
    time.sleep(30)  # 初始等待30秒
    
    # 检查server是否就绪（使用/v1/models端点）
    for i, server_url in enumerate(server_urls):
        gpu_idx = i // servers_per_gpu
        instance_idx = i % servers_per_gpu
        for retry in range(120):  # 最多等120秒
            try:
                resp = requests.get(f"{server_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    print(f"    GPU {gpu_ids[gpu_idx]} #{instance_idx+1}: server就绪")
                    break
            except:
                pass
            time.sleep(1)
        else:
            print(f"    GPU {gpu_ids[gpu_idx]} #{instance_idx+1}: server启动超时，继续尝试...")
    
    start_time = time.time()
    
    # 动态分配模式：使用队列，空闲server自动取任务
    # 任务格式: (idx, prompt, retry_count)
    task_queue = queue.Queue()
    for idx, prompt in enumerate(prompts):
        task_queue.put((idx, prompt, 0))  # 初始重试次数为0
    
    all_results = []
    completed_count = 0
    failed_count = 0  # 跟踪最终失败的任务数
    lock = Lock()
    results_lock = Lock()
    
    # Server健康状态跟踪
    server_healthy = {url: True for url in server_urls}
    server_healthy_lock = Lock()
    
    MAX_RETRIES = 3  # 最大重试次数
    
    def check_server_health(server_url):
        """检查server是否健康"""
        try:
            resp = requests.get(f"{server_url}/v1/models", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def get_healthy_server():
        """获取一个健康的server URL"""
        with server_healthy_lock:
            for url, healthy in server_healthy.items():
                if healthy:
                    return url
        return None
    
    def worker(server_url, server_id):
        """Worker线程：从队列取任务，处理完再取下一个，失败时重试"""
        nonlocal completed_count, failed_count
        current_server = server_url
        consecutive_failures = 0
        
        while True:
            try:
                idx, prompt, retry_count = task_queue.get_nowait()
            except queue.Empty:
                break
            
            # 如果当前server连续失败多次，检查健康状态
            if consecutive_failures >= 2:
                if not check_server_health(current_server):
                    with server_healthy_lock:
                        server_healthy[current_server] = False
                    print(f"\n  Server {current_server} 标记为不健康")
                    # 尝试切换到其他健康的server
                    new_server = get_healthy_server()
                    if new_server and new_server != current_server:
                        print(f"  Worker {server_id} 切换到 {new_server}")
                        current_server = new_server
                        consecutive_failures = 0
            
            # 发送请求
            idx, text, success = _send_request_to_server((idx, prompt, current_server, temperature, max_tokens))
            
            if success:
                # 成功：记录结果
                with results_lock:
                    all_results.append((idx, text))
                with lock:
                    completed_count += 1
                consecutive_failures = 0
            else:
                # 失败：检查重试
                consecutive_failures += 1
                if retry_count < MAX_RETRIES:
                    # 还有重试机会，放回队列
                    task_queue.put((idx, prompt, retry_count + 1))
                    print(f"\n  任务 {idx} 重试 ({retry_count + 1}/{MAX_RETRIES})")
                else:
                    # 超过最大重试次数，记录空结果
                    with results_lock:
                        all_results.append((idx, ""))
                    with lock:
                        completed_count += 1
                        failed_count += 1
                    print(f"\n  任务 {idx} 达到最大重试次数，标记为失败")
            
            task_queue.task_done()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
        ) as progress:
            task = progress.add_task(
                f"[cyan]推理进度 ({total_servers}个server并行)", 
                total=total_samples
            )
            
            # 启动worker线程（每个server一个）
            threads = []
            for i, server_url in enumerate(server_urls):
                t = Thread(target=worker, args=(server_url, i))
                t.start()
                threads.append(t)
            
            # 更新进度直到完成
            while completed_count < total_samples:
                progress.update(task, completed=completed_count)
                time.sleep(0.5)
            
            progress.update(task, completed=total_samples)
            
            # 等待所有线程完成
            for t in threads:
                t.join()
    
    finally:
        # 关闭所有server
        print(f"\n  关闭server...")
        for proc in servers:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
        time.sleep(1)
        for proc in servers:
            try:
                proc.kill()
            except:
                pass
    
    # 按原始索引排序，恢复顺序
    all_results.sort(key=lambda x: x[0])
    outputs = [text for idx, text in all_results]
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ Server模式推理完成！[/bold green]")
    console.print(f"  总样本数: [cyan]{total_samples}[/cyan]")
    console.print(f"  使用GPU数: [cyan]{num_gpus}[/cyan]")
    console.print(f"  每GPU server数: [cyan]{servers_per_gpu}[/cyan]")
    console.print(f"  总server数: [cyan]{total_servers}[/cyan]")
    console.print(f"  总耗时: [cyan]{elapsed_time:.2f}[/cyan] 秒")
    console.print(f"  平均速度: [cyan]{total_samples/elapsed_time:.2f}[/cyan] 样本/秒")
    if failed_count > 0:
        console.print(f"  [bold red]失败任务数: {failed_count}[/bold red] (已达最大重试次数)")
    
    return outputs


def _worker_inference_llama_cpp(
    args: Tuple[int, str, List[Tuple[int, str]], float, int, any]
) -> List[Tuple[int, str]]:
    """
    单个GPU上的worker进程，执行llama.cpp推理
    
    Args:
        args: (gpu_id, model_path, indexed_prompts, temperature, max_tokens, progress_queue)
              indexed_prompts: [(original_index, prompt), ...]
              progress_queue: 进度队列，用于报告进度
    
    Returns:
        List[(original_index, output_text), ...]
    """
    gpu_id, model_path, indexed_prompts, temperature, max_tokens, progress_queue = args
    
    # 设置当前进程只使用指定的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python未安装")
    
    # 判断是本地文件还是HuggingFace仓库
    is_local_file = os.path.exists(model_path) or model_path.endswith('.gguf')
    
    if is_local_file and os.path.exists(model_path):
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=32768,
            n_batch=2048,
            verbose=False
        )
    else:
        model = Llama.from_pretrained(
            repo_id=model_path,
            filename="*Q4_K_M.gguf",
            n_gpu_layers=-1,
            n_ctx=32768,
            n_batch=2048,
            verbose=False
        )
    
    results = []
    actual_max_tokens = max_tokens if max_tokens > 0 else 512
    
    for idx, prompt in indexed_prompts:
        output = model(
            prompt,
            max_tokens=actual_max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=1.0 if temperature == 0 else 0.95,
            top_k=1 if temperature == 0 else 40,
            echo=False
        )
        
        if isinstance(output, dict) and 'choices' in output:
            text = output['choices'][0].get('text', '')
        else:
            text = str(output)
        
        results.append((idx, text))
        
        # 报告进度
        if progress_queue is not None:
            progress_queue.put((gpu_id, 1))
    
    # 清理模型
    del model
    
    return results


def _progress_listener(progress_queue, total_samples, num_gpus, gpu_ids):
    """
    进度监听线程，从队列读取进度并更新显示
    """
    completed = {gpu_id: 0 for gpu_id in gpu_ids}
    total_completed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(f"[cyan]推理进度 ({num_gpus}GPU并行)", total=total_samples)
        
        while total_completed < total_samples:
            try:
                gpu_id, count = progress_queue.get(timeout=1)
                completed[gpu_id] += count
                total_completed = sum(completed.values())
                progress.update(task, completed=total_completed)
            except:
                # 超时，继续等待
                pass
    
    return total_completed


def batch_inference_llama_cpp_parallel(
    model_path: str,
    prompts: List[str],
    gpu_ids: List[int],
    temperature: float = 0.0,
    max_tokens: int = 512,
    show_progress: bool = True,
    use_server_mode: bool = True
) -> List[str]:
    """
    使用多GPU并行进行llama.cpp推理
    
    Args:
        model_path: 模型路径或HuggingFace仓库名
        prompts: 输入prompt列表
        gpu_ids: GPU ID列表
        temperature: 采样温度
        max_tokens: 最大生成token数
        show_progress: 是否显示进度条
        use_server_mode: 是否使用server模式（推荐，更快）
        
    Returns:
        List[str]: 生成的文本列表（保持原始顺序）
    """
    # 默认使用server模式，更快
    if use_server_mode:
        return batch_inference_llama_cpp_server(
            model_path=model_path,
            prompts=prompts,
            gpu_ids=gpu_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            servers_per_gpu=2,  # 每个GPU启动2个server实例
            show_progress=show_progress
        )
    
    # 备用：多进程模式
    total_samples = len(prompts)
    num_gpus = len(gpu_ids)
    
    print(f"\n[多GPU并行] 使用 {num_gpus} 个GPU: {gpu_ids}")
    
    # 为每个prompt添加索引，以便后续恢复顺序
    indexed_prompts = list(enumerate(prompts))
    
    # 将prompts均匀分配到各个GPU
    chunks = [[] for _ in range(num_gpus)]
    for i, (idx, prompt) in enumerate(indexed_prompts):
        chunks[i % num_gpus].append((idx, prompt))
    
    # 打印分配情况
    for i, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(chunks[i])} 个样本")
    
    start_time = time.time()
    
    # 使用spawn方式创建进程，避免CUDA上下文问题
    ctx = mp.get_context('spawn')
    
    # 创建进度队列
    progress_queue = ctx.Manager().Queue()
    
    # 准备worker参数（添加进度队列）
    worker_args = [
        (gpu_ids[i], model_path, chunks[i], temperature, max_tokens, progress_queue)
        for i in range(num_gpus)
    ]
    
    # 使用进程池并行执行
    all_results = []
    
    # 启动进度监听线程
    progress_thread = Thread(
        target=_progress_listener,
        args=(progress_queue, total_samples, num_gpus, gpu_ids),
        daemon=True
    )
    progress_thread.start()
    
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        futures = {executor.submit(_worker_inference_llama_cpp, args): i 
                   for i, args in enumerate(worker_args)}
        
        for future in as_completed(futures):
            gpu_idx = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"\n  GPU {gpu_ids[gpu_idx]} 错误: {e}")
                raise
    
    # 等待进度线程结束
    progress_thread.join(timeout=2)
    
    # 按原始索引排序，恢复顺序
    all_results.sort(key=lambda x: x[0])
    outputs = [text for idx, text in all_results]
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ 多GPU并行推理完成！[/bold green]")
    console.print(f"  总样本数: [cyan]{total_samples}[/cyan]")
    console.print(f"  使用GPU数: [cyan]{num_gpus}[/cyan]")
    console.print(f"  总耗时: [cyan]{elapsed_time:.2f}[/cyan] 秒")
    console.print(f"  平均速度: [cyan]{total_samples/elapsed_time:.2f}[/cyan] 样本/秒")
    
    return outputs


def batch_inference_sglang(
    engine,
    prompts: List[str],
    temperature: float = 0.0,
    batch_size: int = 8,
    show_progress: bool = True
) -> List[str]:
    """
    使用SGLang引擎进行批量推理
    
    Args:
        engine: SGLang引擎实例
        prompts: 输入prompt列表
        temperature: 采样温度
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        
    Returns:
        List[str]: 生成的文本列表
    """
    import sglang as sgl
    
    all_outputs = []
    total_samples = len(prompts)
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    # 设置采样参数（不限制max_new_tokens）
    sampling_params = {
        "temperature": temperature,
    }
    
    if temperature == 0:
        sampling_params["top_p"] = 1.0
        sampling_params["top_k"] = 1
    
    start_time = time.time()
    
    # 使用rich进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        
        task = progress.add_task("[cyan]推理进度 (SGLang)", total=total_samples)
        
        for i in range(0, total_samples, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            current_batch_size = len(batch_prompts)
            
            # 使用SGLang进行批量推理
            outputs = engine.generate(
                batch_prompts,
                sampling_params=sampling_params
            )
            
            # 提取生成的文本
            for output in outputs:
                if hasattr(output, 'text'):
                    all_outputs.append(output.text)
                elif isinstance(output, dict):
                    all_outputs.append(output.get('text', str(output)))
                else:
                    all_outputs.append(str(output))
            
            # 更新进度
            progress.update(task, advance=current_batch_size)
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ 推理完成！[/bold green]")
    console.print(f"  总样本数: [cyan]{total_samples}[/cyan]")
    console.print(f"  总耗时: [cyan]{elapsed_time:.2f}[/cyan] 秒")
    console.print(f"  平均速度: [cyan]{total_samples/elapsed_time:.2f}[/cyan] 样本/秒")
    
    return all_outputs


def batch_inference_llama_cpp(
    model,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    show_progress: bool = True
) -> List[str]:
    """
    使用llama.cpp进行批量推理
    
    Args:
        model: llama-cpp-python模型实例
        prompts: 输入prompt列表
        temperature: 采样温度
        max_tokens: 最大生成token数
        show_progress: 是否显示进度条
        
    Returns:
        List[str]: 生成的文本列表
    """
    all_outputs = []
    total_samples = len(prompts)
    
    start_time = time.time()
    
    # 使用rich进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        
        task = progress.add_task("[cyan]推理进度 (llama.cpp)", total=total_samples)
        
        for prompt in prompts:
            # 使用llama.cpp进行单个推理
            # max_tokens: 0表示不限制，但这里设置合理默认值避免无限生成
            actual_max_tokens = max_tokens if max_tokens > 0 else 512
            output = model(
                prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                top_p=1.0 if temperature == 0 else 0.95,
                top_k=1 if temperature == 0 else 40,
                echo=False  # 不返回prompt
            )
            
            # 提取生成的文本
            if isinstance(output, dict) and 'choices' in output:
                text = output['choices'][0].get('text', '')
            else:
                text = str(output)
            
            all_outputs.append(text)
            
            # 更新进度
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ 推理完成！[/bold green]")
    console.print(f"  总样本数: [cyan]{total_samples}[/cyan]")
    console.print(f"  总耗时: [cyan]{elapsed_time:.2f}[/cyan] 秒")
    console.print(f"  平均速度: [cyan]{total_samples/elapsed_time:.2f}[/cyan] 样本/秒")
    
    return all_outputs


def batch_inference(
    engine,
    prompts: List[str],
    temperature: float = 0.0,
    batch_size: int = 8,
    show_progress: bool = True,
    backend: str = "sglang",
    max_tokens: int = 2048,
    model_path: str = None,
    gpu_ids: List[int] = None
) -> List[str]:
    """
    统一的批量推理接口，根据后端类型选择对应的推理方法
    
    Args:
        engine: 模型引擎实例 (SGLang Engine 或 llama-cpp-python Llama)
        prompts: 输入prompt列表
        temperature: 采样温度
        batch_size: 批处理大小 (仅SGLang使用)
        show_progress: 是否显示进度条
        backend: 后端类型 ("sglang" 或 "llama_cpp")
        max_tokens: 最大生成token数 (仅llama.cpp使用)
        model_path: 模型路径 (多GPU并行时使用)
        gpu_ids: GPU ID列表 (多GPU并行时使用)
        
    Returns:
        List[str]: 生成的文本列表
    """
    if backend == "llama_cpp":
        # 如果指定了多个GPU，使用并行推理
        if gpu_ids and len(gpu_ids) > 1 and model_path:
            return batch_inference_llama_cpp_parallel(
                model_path=model_path,
                prompts=prompts,
                gpu_ids=gpu_ids,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress
            )
        else:
            return batch_inference_llama_cpp(
                model=engine,
                prompts=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress
            )
    else:
        return batch_inference_sglang(
            engine=engine,
            prompts=prompts,
            temperature=temperature,
            batch_size=batch_size,
            show_progress=show_progress
        )


def run_inference_on_dataset(
    engine,
    samples: List[Dict],
    temperature: float = 0.0,
    batch_size: int = 8,
    prompt_key: str = "full_prompt",
    backend: str = "sglang",
    max_tokens: int = 2048,
    model_path: str = None,
    gpu_ids: List[int] = None
) -> List[Dict]:
    """
    在数据集上运行推理
    
    Args:
        engine: 模型引擎实例
        samples: 数据样本列表
        temperature: 采样温度
        batch_size: 批处理大小
        prompt_key: prompt在样本中的键名
        backend: 后端类型 ("sglang" 或 "llama_cpp")
        max_tokens: 最大生成token数 (仅llama.cpp使用)
        model_path: 模型路径 (多GPU并行时使用)
        gpu_ids: GPU ID列表 (多GPU并行时使用)
        
    Returns:
        List[Dict]: 带有预测结果的样本列表
    """
    # 提取所有prompt
    prompts = [sample[prompt_key] for sample in samples]
    
    # 批量推理
    outputs = batch_inference(
        engine=engine,
        prompts=prompts,
        temperature=temperature,
        batch_size=batch_size,
        backend=backend,
        max_tokens=max_tokens,
        model_path=model_path,
        gpu_ids=gpu_ids
    )
    
    # 将预测结果添加到样本中
    for sample, output in zip(samples, outputs):
        sample["prediction"] = output.strip()
    
    return samples


class InferenceRunner:
    """
    推理运行器类 - 支持SGLang和llama.cpp后端，支持多GPU并行
    """
    
    def __init__(
        self,
        engine,
        temperature: float = 0.0,
        batch_size: int = 8,
        backend: str = "sglang",
        max_tokens: int = 2048,
        model_path: str = None,
        gpu_ids: List[int] = None
    ):
        """
        初始化推理运行器
        
        Args:
            engine: 模型引擎实例
            temperature: 采样温度
            batch_size: 批处理大小
            backend: 后端类型 ("sglang" 或 "llama_cpp")
            max_tokens: 最大生成token数 (仅llama.cpp使用)
            model_path: 模型路径 (多GPU并行时使用)
            gpu_ids: GPU ID列表 (多GPU并行时使用)
        """
        self.engine = engine
        self.temperature = temperature
        self.batch_size = batch_size
        self.backend = backend
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        
    def run(self, samples: List[Dict], prompt_key: str = "full_prompt") -> List[Dict]:
        """
        运行推理
        
        Args:
            samples: 数据样本列表
            prompt_key: prompt键名
            
        Returns:
            List[Dict]: 带预测结果的样本列表
        """
        return run_inference_on_dataset(
            engine=self.engine,
            samples=samples,
            temperature=self.temperature,
            batch_size=self.batch_size,
            prompt_key=prompt_key,
            backend=self.backend,
            max_tokens=self.max_tokens,
            model_path=self.model_path,
            gpu_ids=self.gpu_ids
        )


def batch_inference(
    engine,
    prompts: List[str],
    temperature: float = 0.0,
    batch_size: int = 8,
    show_progress: bool = True,
    backend: str = "sglang",
    max_tokens: int = 2048,
    model_path: str = None,
    gpu_ids: List[int] = None
) -> List[str]:
    """
    统一的批量推理接口，根据后端类型选择对应的推理方法
    
    Args:
        engine: 模型引擎实例 (SGLang Engine 或 llama-cpp-python Llama)
        prompts: 输入prompt列表
        temperature: 采样温度
        batch_size: 批处理大小 (仅SGLang使用)
        show_progress: 是否显示进度条
        backend: 后端类型 ("sglang" 或 "llama_cpp")
        max_tokens: 最大生成token数 (仅llama.cpp使用)
        model_path: 模型路径 (多GPU并行时使用)
        gpu_ids: GPU ID列表 (多GPU并行时使用)
        
    Returns:
        List[str]: 生成的文本列表
    """
    if backend == "llama_cpp":
        # 如果指定了多个GPU，使用并行推理
        if gpu_ids and len(gpu_ids) > 1 and model_path:
            return batch_inference_llama_cpp_parallel(
                model_path=model_path,
                prompts=prompts,
                gpu_ids=gpu_ids,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress
            )
        else:
            return batch_inference_llama_cpp(
                model=engine,
                prompts=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                show_progress=show_progress
            )
    else:
        return batch_inference_sglang(
            engine=engine,
            prompts=prompts,
            temperature=temperature,
            batch_size=batch_size,
            show_progress=show_progress
        )


def run_inference_on_dataset(
    engine,
    samples: List[Dict],
    temperature: float = 0.0,
    batch_size: int = 8,
    prompt_key: str = "full_prompt",
    backend: str = "sglang",
    max_tokens: int = 2048,
    model_path: str = None,
    gpu_ids: List[int] = None
) -> List[Dict]:
    """
    在数据集上运行推理
    
    Args:
        engine: 模型引擎实例
        samples: 数据样本列表
        temperature: 采样温度
        batch_size: 批处理大小
        prompt_key: prompt在样本中的键名
        backend: 后端类型 ("sglang" 或 "llama_cpp")
        max_tokens: 最大生成token数 (仅llama.cpp使用)
        model_path: 模型路径 (多GPU并行时使用)
        gpu_ids: GPU ID列表 (多GPU并行时使用)
        
    Returns:
        List[Dict]: 带有预测结果的样本列表
    """
    # 提取所有prompt
    prompts = [sample[prompt_key] for sample in samples]
    
    # 批量推理
    outputs = batch_inference(
        engine=engine,
        prompts=prompts,
        temperature=temperature,
        batch_size=batch_size,
        backend=backend,
        max_tokens=max_tokens,
        model_path=model_path,
        gpu_ids=gpu_ids
    )
    
    # 将预测结果添加到样本中
    for sample, output in zip(samples, outputs):
        sample["prediction"] = output.strip()
    
    return samples


class InferenceRunner:
    """
    推理运行器类 - 支持SGLang和llama.cpp后端，支持多GPU并行
    """
    
    def __init__(
        self,
        engine,
        temperature: float = 0.0,
        batch_size: int = 8,
        backend: str = "sglang",
        max_tokens: int = 2048,
        model_path: str = None,
        gpu_ids: List[int] = None
    ):
        """
        初始化推理运行器
        
        Args:
            engine: 模型引擎实例
            temperature: 采样温度
            batch_size: 批处理大小
            backend: 后端类型 ("sglang" 或 "llama_cpp")
            max_tokens: 最大生成token数 (仅llama.cpp使用)
            model_path: 模型路径 (多GPU并行时使用)
            gpu_ids: GPU ID列表 (多GPU并行时使用)
        """
        self.engine = engine
        self.temperature = temperature
        self.batch_size = batch_size
        self.backend = backend
        self.max_tokens = max_tokens
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        
    def run(self, samples: List[Dict], prompt_key: str = "full_prompt") -> List[Dict]:
        """
        运行推理
        
        Args:
            samples: 数据样本列表
            prompt_key: prompt键名
            
        Returns:
            List[Dict]: 带预测结果的样本列表
        """
        return run_inference_on_dataset(
            engine=self.engine,
            samples=samples,
            temperature=self.temperature,
            batch_size=self.batch_size,
            prompt_key=prompt_key,
            backend=self.backend,
            max_tokens=self.max_tokens,
            model_path=self.model_path,
            gpu_ids=self.gpu_ids
        )

