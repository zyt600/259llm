"""
Inference module - Batch inference using SGLang or llama.cpp
Supports multi-GPU parallel inference and Server mode concurrent inference
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
    Send single request to llama.cpp server (OpenAI compatible API)
    
    Args:
        args: (idx, prompt, server_url, temperature, max_tokens)
    
    Returns:
        (idx, output_text, success) - success indicates if request was successful
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
            timeout=600  # 10 minute timeout
        )
        response.raise_for_status()
        result = response.json()
        # OpenAI format response
        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0].get("text", "")
        else:
            text = ""
        return (idx, text, True)
    except Exception as e:
        print(f"\nRequest error (idx={idx}, server={server_url}): {e}")
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
    Concurrent inference using llama.cpp server mode
    Launches multiple server instances on each GPU for true parallel processing
    
    Args:
        model_path: Model path or HuggingFace repo name
        prompts: Input prompt list
        gpu_ids: GPU ID list
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        servers_per_gpu: Number of servers per GPU (requires sufficient VRAM)
        show_progress: Whether to show progress bar
        
    Returns:
        List[str]: Generated text list (maintains original order)
    """
    from huggingface_hub import hf_hub_download
    import glob
    
    total_samples = len(prompts)
    num_gpus = len(gpu_ids)
    total_servers = num_gpus * servers_per_gpu
    
    print(f"\n[Server mode] Using {num_gpus} GPUs: {gpu_ids}")
    print(f"  {servers_per_gpu} server instances per GPU, {total_servers} servers total")
    
    # Check if model needs to be downloaded
    if os.path.exists(model_path):
        gguf_path = model_path
    else:
        # Download from HuggingFace
        print(f"  Downloading model from HuggingFace...")
        
        # Check if path contains filename (e.g. unsloth/Qwen3-1.7B-Q4_0.gguf)
        # If so, try to download from corresponding GGUF repo
        if model_path.endswith('.gguf') and '/' in model_path:
            # Extract filename and possible repo name
            # e.g.: unsloth/Qwen3-1.7B-Q4_0.gguf
            # Should map to: repo unsloth/Qwen3-1.7B-GGUF, file Qwen3-1.7B-Q4_0.gguf
            parts = model_path.rsplit('/', 1)
            if len(parts) == 2:
                org_part = parts[0]  # e.g. unsloth
                filename_part = parts[1]   # e.g. Qwen3-1.7B-Q4_0.gguf
                
                # Extract base model name from filename (remove quantization suffix)
                # e.g.: 
                #   Qwen3-1.7B-Q4_0.gguf -> Qwen3-1.7B
                #   Qwen3-1.7B-Q8_K_XL.gguf -> Qwen3-1.7B
                #   Qwen3-1.7B-UD-Q8_K_XL.gguf -> Qwen3-1.7B-UD
                # Quantization suffix format: -Q4_0, -Q4_K_M, -Q8_K_XL, -UD-Q8_K_XL etc.
                
                # Special handling: if filename contains -UD-, extract UD part first
                base = filename_part.replace('.gguf', '')
                if '-UD-' in base:
                    # e.g.: Qwen3-1.7B-UD-Q8_K_XL -> Qwen3-1.7B-UD
                    base_model_name = base.rsplit('-UD-', 1)[0] + '-UD'
                else:
                    # Match common quantization suffix patterns
                    patterns = [
                        # Match Q8_K_XL, Q4_K_M etc. format (with multiple underscore parts)
                        r'^(.+?)-(Q\d+(_[KM])+(_[A-Z]+)?)\.gguf$',
                        # Match Q4_0, Q8_0 etc. simple format
                        r'^(.+?)-(Q\d+_\d+)\.gguf$',
                        # Match Q4_K, Q8_M etc. format
                        r'^(.+?)-(Q\d+_[KM])\.gguf$',
                        # Match Q4, Q8 etc. format
                        r'^(.+?)-(Q\d+[KM]?)\.gguf$',
                        # Match BF16, F16, IQ4_XS etc. format
                        r'^(.+?)-(BF16|F16|IQ4_[NX]S?)\.gguf$',
                    ]
                    
                    base_model_name = None
                    for pattern in patterns:
                        match = re.match(pattern, filename_part)
                        if match:
                            base_model_name = match.group(1)
                            break
                    
                    if base_model_name is None:
                        # If no pattern matches, try removing last -xxx part
                        if '-' in base:
                            base_model_name = base.rsplit('-', 1)[0]
                        else:
                            base_model_name = base
                
                # Build possible repo names
                # For UD versions (e.g. Qwen3-1.7B-UD), use base model name without -UD to build repo
                # Because UD versions are usually in the same GGUF repo
                repo_base_model = base_model_name
                if base_model_name.endswith('-UD'):
                    repo_base_model = base_model_name[:-3]  # Remove -UD
                
                # Strategy 1: {org}/{base_model}-GGUF (e.g. unsloth/Qwen3-1.7B-GGUF)
                # Strategy 2: {org}/{base_model} (e.g. unsloth/Qwen3-1.7B)
                possible_repos = [
                    f"{org_part}/{repo_base_model}-GGUF",  # unsloth/Qwen3-1.7B-GGUF
                    f"{org_part}/{repo_base_model}",  # unsloth/Qwen3-1.7B
                ]
                
                downloaded = False
                for repo_id in possible_repos:
                    try:
                        # First try downloading specified filename
                        gguf_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename_part,
                            local_dir="./models_cache"
                        )
                        downloaded = True
                        print(f"  Downloaded file {filename_part} from repo {repo_id}")
                        break
                    except Exception as e:
                        # If specified file doesn't exist, try downloading default Q4_K_M version (only for -GGUF repos)
                        if repo_id.endswith('-GGUF'):
                            try:
                                gguf_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename="*Q4_K_M.gguf",
                                    local_dir="./models_cache"
                                )
                                downloaded = True
                                print(f"  Downloaded default Q4_K_M version from repo {repo_id} (file {filename_part} not found)")
                                break
                            except:
                                continue
                        continue
                
                if not downloaded:
                    raise ValueError(f"Cannot download model file {filename_part} from {possible_repos}")
            else:
                # If parsing fails, use original logic
                raise ValueError(f"Cannot parse model path: {model_path}")
        else:
            # Original logic: treat as HuggingFace repo name
            try:
                # Try downloading Q4_K_M version
                gguf_path = hf_hub_download(
                    repo_id=model_path,
                    filename="*Q4_K_M.gguf",
                    local_dir="./models_cache"
                )
            except:
                # List gguf files in repo and download first one
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
                        raise ValueError(f"No GGUF files found in repo {model_path}")
                except Exception as e:
                    raise ValueError(f"Cannot download model {model_path} from HuggingFace: {e}")
        
        print(f"  Model download complete: {gguf_path}")
    
    # Start server processes (multiple instances per GPU)
    servers = []
    server_urls = []
    base_port = 8080
    
    print(f"  Starting {total_servers} llama.cpp servers...")
    
    server_idx = 0
    for gpu_id in gpu_ids:
        for instance in range(servers_per_gpu):
            port = base_port + server_idx
            server_url = f"http://localhost:{port}"
            server_urls.append(server_url)
            
            # Start server process
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Set CUDA library paths
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
            print(f"    GPU {gpu_id} #{instance+1}: server starting (port={port}, pid={proc.pid})")
            server_idx += 1
    
    # Wait for all servers to start (large models need longer time)
    print(f"  Waiting for servers to be ready (large models may take 1-2 minutes)...")
    time.sleep(30)  # Initial wait 30 seconds
    
    # Check if servers are ready (using /v1/models endpoint)
    for i, server_url in enumerate(server_urls):
        gpu_idx = i // servers_per_gpu
        instance_idx = i % servers_per_gpu
        for retry in range(120):  # Wait up to 120 seconds
            try:
                resp = requests.get(f"{server_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    print(f"    GPU {gpu_ids[gpu_idx]} #{instance_idx+1}: server ready")
                    break
            except:
                pass
            time.sleep(1)
        else:
            print(f"    GPU {gpu_ids[gpu_idx]} #{instance_idx+1}: server start timeout, continuing...")
    
    start_time = time.time()
    
    # Dynamic allocation mode: use queue, idle servers automatically take tasks
    # Task format: (idx, prompt, retry_count)
    task_queue = queue.Queue()
    for idx, prompt in enumerate(prompts):
        task_queue.put((idx, prompt, 0))  # Initial retry count is 0
    
    all_results = []
    completed_count = 0
    failed_count = 0  # Track final failed task count
    lock = Lock()
    results_lock = Lock()
    
    # Server health tracking
    server_healthy = {url: True for url in server_urls}
    server_healthy_lock = Lock()
    
    MAX_RETRIES = 3  # Maximum retry count
    
    def check_server_health(server_url):
        """Check if server is healthy"""
        try:
            resp = requests.get(f"{server_url}/v1/models", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def get_healthy_server():
        """Get a healthy server URL"""
        with server_healthy_lock:
            for url, healthy in server_healthy.items():
                if healthy:
                    return url
        return None
    
    def worker(server_url, server_id):
        """Worker thread: take task from queue, process it, then take next one, retry on failure"""
        nonlocal completed_count, failed_count
        current_server = server_url
        consecutive_failures = 0
        
        while True:
            try:
                idx, prompt, retry_count = task_queue.get_nowait()
            except queue.Empty:
                break
            
            # If current server has consecutive failures, check health
            if consecutive_failures >= 2:
                if not check_server_health(current_server):
                    with server_healthy_lock:
                        server_healthy[current_server] = False
                    print(f"\n  Server {current_server} marked as unhealthy")
                    # Try switching to another healthy server
                    new_server = get_healthy_server()
                    if new_server and new_server != current_server:
                        print(f"  Worker {server_id} switching to {new_server}")
                        current_server = new_server
                        consecutive_failures = 0
            
            # Send request
            idx, text, success = _send_request_to_server((idx, prompt, current_server, temperature, max_tokens))
            
            if success:
                # Success: record result
                with results_lock:
                    all_results.append((idx, text))
                with lock:
                    completed_count += 1
                consecutive_failures = 0
            else:
                # Failure: check retry
                consecutive_failures += 1
                if retry_count < MAX_RETRIES:
                    # Still have retry chances, put back in queue
                    task_queue.put((idx, prompt, retry_count + 1))
                    print(f"\n  Task {idx} retrying ({retry_count + 1}/{MAX_RETRIES})")
                else:
                    # Exceeded max retries, record empty result
                    with results_lock:
                        all_results.append((idx, ""))
                    with lock:
                        completed_count += 1
                        failed_count += 1
                    print(f"\n  Task {idx} reached max retries, marked as failed")
            
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
                f"[cyan]Inference progress ({total_servers} servers parallel)", 
                total=total_samples
            )
            
            # Start worker threads (one per server)
            threads = []
            for i, server_url in enumerate(server_urls):
                t = Thread(target=worker, args=(server_url, i))
                t.start()
                threads.append(t)
            
            # Update progress until complete
            while completed_count < total_samples:
                progress.update(task, completed=completed_count)
                time.sleep(0.5)
            
            progress.update(task, completed=total_samples)
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
    
    finally:
        # Shutdown all servers
        print(f"\n  Shutting down servers...")
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
    
    # Sort by original index, restore order
    all_results.sort(key=lambda x: x[0])
    outputs = [text for idx, text in all_results]
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ Server mode inference complete![/bold green]")
    console.print(f"  Total samples: [cyan]{total_samples}[/cyan]")
    console.print(f"  GPUs used: [cyan]{num_gpus}[/cyan]")
    console.print(f"  Servers per GPU: [cyan]{servers_per_gpu}[/cyan]")
    console.print(f"  Total servers: [cyan]{total_servers}[/cyan]")
    console.print(f"  Total time: [cyan]{elapsed_time:.2f}[/cyan] seconds")
    console.print(f"  Average speed: [cyan]{total_samples/elapsed_time:.2f}[/cyan] samples/sec")
    if failed_count > 0:
        console.print(f"  [bold red]Failed tasks: {failed_count}[/bold red] (reached max retries)")
    
    return outputs


def _worker_inference_llama_cpp(
    args: Tuple[int, str, List[Tuple[int, str]], float, int, any]
) -> List[Tuple[int, str]]:
    """
    Worker process on single GPU, performs llama.cpp inference
    
    Args:
        args: (gpu_id, model_path, indexed_prompts, temperature, max_tokens, progress_queue)
              indexed_prompts: [(original_index, prompt), ...]
              progress_queue: Progress queue for reporting progress
    
    Returns:
        List[(original_index, output_text), ...]
    """
    gpu_id, model_path, indexed_prompts, temperature, max_tokens, progress_queue = args
    
    # Set current process to only use specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python not installed")
    
    # Check if local file or HuggingFace repo
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
        
        # Report progress
        if progress_queue is not None:
            progress_queue.put((gpu_id, 1))
    
    # Cleanup model
    del model
    
    return results


def _progress_listener(progress_queue, total_samples, num_gpus, gpu_ids):
    """
    Progress listener thread, reads progress from queue and updates display
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
        task = progress.add_task(f"[cyan]Inference progress ({num_gpus}GPU parallel)", total=total_samples)
        
        while total_completed < total_samples:
            try:
                gpu_id, count = progress_queue.get(timeout=1)
                completed[gpu_id] += count
                total_completed = sum(completed.values())
                progress.update(task, completed=total_completed)
            except:
                # Timeout, continue waiting
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
    Multi-GPU parallel llama.cpp inference
    
    Args:
        model_path: Model path or HuggingFace repo name
        prompts: Input prompt list
        gpu_ids: GPU ID list
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar
        use_server_mode: Whether to use server mode (recommended, faster)
        
    Returns:
        List[str]: Generated text list (maintains original order)
    """
    # Default to server mode, faster
    if use_server_mode:
        return batch_inference_llama_cpp_server(
            model_path=model_path,
            prompts=prompts,
            gpu_ids=gpu_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            servers_per_gpu=2,  # 2 server instances per GPU
            show_progress=show_progress
        )
    
    # Fallback: multi-process mode
    total_samples = len(prompts)
    num_gpus = len(gpu_ids)
    
    print(f"\n[Multi-GPU parallel] Using {num_gpus} GPUs: {gpu_ids}")
    
    # Add index to each prompt for later order restoration
    indexed_prompts = list(enumerate(prompts))
    
    # Distribute prompts evenly to GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, (idx, prompt) in enumerate(indexed_prompts):
        chunks[i % num_gpus].append((idx, prompt))
    
    # Print distribution
    for i, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(chunks[i])} samples")
    
    start_time = time.time()
    
    # Use spawn to create processes, avoid CUDA context issues
    ctx = mp.get_context('spawn')
    
    # Create progress queue
    progress_queue = ctx.Manager().Queue()
    
    # Prepare worker arguments (add progress queue)
    worker_args = [
        (gpu_ids[i], model_path, chunks[i], temperature, max_tokens, progress_queue)
        for i in range(num_gpus)
    ]
    
    # Use process pool for parallel execution
    all_results = []
    
    # Start progress listener thread
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
                print(f"\n  GPU {gpu_ids[gpu_idx]} error: {e}")
                raise
    
    # Wait for progress thread to finish
    progress_thread.join(timeout=2)
    
    # Sort by original index, restore order
    all_results.sort(key=lambda x: x[0])
    outputs = [text for idx, text in all_results]
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ Multi-GPU parallel inference complete![/bold green]")
    console.print(f"  Total samples: [cyan]{total_samples}[/cyan]")
    console.print(f"  GPUs used: [cyan]{num_gpus}[/cyan]")
    console.print(f"  Total time: [cyan]{elapsed_time:.2f}[/cyan] seconds")
    console.print(f"  Average speed: [cyan]{total_samples/elapsed_time:.2f}[/cyan] samples/sec")
    
    return outputs


def batch_inference_sglang(
    engine,
    prompts: List[str],
    temperature: float = 0.0,
    batch_size: int = 8,
    show_progress: bool = True
) -> List[str]:
    """
    Batch inference using SGLang engine
    
    Args:
        engine: SGLang engine instance
        prompts: Input prompt list
        temperature: Sampling temperature
        batch_size: Batch size
        show_progress: Whether to show progress bar
        
    Returns:
        List[str]: Generated text list
    """
    import sglang as sgl
    
    all_outputs = []
    total_samples = len(prompts)
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    # Set sampling parameters (don't limit max_new_tokens)
    sampling_params = {
        "temperature": temperature,
    }
    
    if temperature == 0:
        sampling_params["top_p"] = 1.0
        sampling_params["top_k"] = 1
    
    start_time = time.time()
    
    # Use rich progress bar
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
        
        task = progress.add_task("[cyan]Inference progress (SGLang)", total=total_samples)
        
        for i in range(0, total_samples, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            current_batch_size = len(batch_prompts)
            
            # Batch inference using SGLang
            outputs = engine.generate(
                batch_prompts,
                sampling_params=sampling_params
            )
            
            # Extract generated text
            for output in outputs:
                if hasattr(output, 'text'):
                    all_outputs.append(output.text)
                elif isinstance(output, dict):
                    all_outputs.append(output.get('text', str(output)))
                else:
                    all_outputs.append(str(output))
            
            # Update progress
            progress.update(task, advance=current_batch_size)
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ Inference complete![/bold green]")
    console.print(f"  Total samples: [cyan]{total_samples}[/cyan]")
    console.print(f"  Total time: [cyan]{elapsed_time:.2f}[/cyan] seconds")
    console.print(f"  Average speed: [cyan]{total_samples/elapsed_time:.2f}[/cyan] samples/sec")
    
    return all_outputs


def batch_inference_llama_cpp(
    model,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    show_progress: bool = True
) -> List[str]:
    """
    Batch inference using llama.cpp
    
    Args:
        model: llama-cpp-python model instance
        prompts: Input prompt list
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        show_progress: Whether to show progress bar
        
    Returns:
        List[str]: Generated text list
    """
    all_outputs = []
    total_samples = len(prompts)
    
    start_time = time.time()
    
    # Use rich progress bar
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
        
        task = progress.add_task("[cyan]Inference progress (llama.cpp)", total=total_samples)
        
        for prompt in prompts:
            # Single inference using llama.cpp
            # max_tokens: 0 means unlimited, but set reasonable default to avoid infinite generation
            actual_max_tokens = max_tokens if max_tokens > 0 else 512
            output = model(
                prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                top_p=1.0 if temperature == 0 else 0.95,
                top_k=1 if temperature == 0 else 40,
                echo=False  # Don't return prompt
            )
            
            # Extract generated text
            if isinstance(output, dict) and 'choices' in output:
                text = output['choices'][0].get('text', '')
            else:
                text = str(output)
            
            all_outputs.append(text)
            
            # Update progress
            progress.update(task, advance=1)
    
    elapsed_time = time.time() - start_time
    
    console.print(f"\n[bold green]✓ Inference complete![/bold green]")
    console.print(f"  Total samples: [cyan]{total_samples}[/cyan]")
    console.print(f"  Total time: [cyan]{elapsed_time:.2f}[/cyan] seconds")
    console.print(f"  Average speed: [cyan]{total_samples/elapsed_time:.2f}[/cyan] samples/sec")
    
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
    Unified batch inference interface, selects inference method based on backend type
    
    Args:
        engine: Model engine instance (SGLang Engine or llama-cpp-python Llama)
        prompts: Input prompt list
        temperature: Sampling temperature
        batch_size: Batch size (SGLang only)
        show_progress: Whether to show progress bar
        backend: Backend type ("sglang" or "llama_cpp")
        max_tokens: Maximum tokens to generate (llama.cpp only)
        model_path: Model path (for multi-GPU parallel)
        gpu_ids: GPU ID list (for multi-GPU parallel)
        
    Returns:
        List[str]: Generated text list
    """
    if backend == "llama_cpp":
        # If multiple GPUs specified, use parallel inference
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
    Run inference on dataset
    
    Args:
        engine: Model engine instance
        samples: Data sample list
        temperature: Sampling temperature
        batch_size: Batch size
        prompt_key: Key name for prompt in samples
        backend: Backend type ("sglang" or "llama_cpp")
        max_tokens: Maximum tokens to generate (llama.cpp only)
        model_path: Model path (for multi-GPU parallel)
        gpu_ids: GPU ID list (for multi-GPU parallel)
        
    Returns:
        List[Dict]: Sample list with prediction results
    """
    # Extract all prompts
    prompts = [sample[prompt_key] for sample in samples]
    
    # Batch inference
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
    
    # Add predictions to samples
    for sample, output in zip(samples, outputs):
        sample["prediction"] = output.strip()
    
    return samples


class InferenceRunner:
    """
    Inference runner class - Supports SGLang and llama.cpp backends, supports multi-GPU parallel
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
        Initialize inference runner
        
        Args:
            engine: Model engine instance
            temperature: Sampling temperature
            batch_size: Batch size
            backend: Backend type ("sglang" or "llama_cpp")
            max_tokens: Maximum tokens to generate (llama.cpp only)
            model_path: Model path (for multi-GPU parallel)
            gpu_ids: GPU ID list (for multi-GPU parallel)
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
        Run inference
        
        Args:
            samples: Data sample list
            prompt_key: Prompt key name
            
        Returns:
            List[Dict]: Sample list with prediction results
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
