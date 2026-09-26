"""
GPU Configuration for HARPO

Optimized settings for different GPU configurations.
Auto-detects available GPUs and configures optimal settings.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class GPUConfig:
    """GPU-specific configuration"""
    # Device settings
    device: str = "cuda"
    num_gpus: int = 1
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Memory settings
    total_vram_gb: float = 0.0
    per_gpu_vram_gb: float = 0.0
    
    # Training optimization
    use_bf16: bool = False
    use_fp16: bool = False
    use_tf32: bool = True  # A100 optimization
    
    # Batch size settings (will be computed)
    optimal_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 16
    
    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Multi-GPU settings
    use_data_parallel: bool = False
    use_distributed: bool = False  # For DDP (more complex setup)
    
    # Memory optimization
    gradient_checkpointing: bool = False
    empty_cache_freq: int = 0  # Steps between cache clearing (0 = disabled)


def detect_gpu_config() -> GPUConfig:
    """
    Auto-detect GPU configuration and return optimal settings.
    
    Returns:
        GPUConfig with optimal settings for detected hardware
    """
    config = GPUConfig()
    
    if not torch.cuda.is_available():
        print("No CUDA GPU detected, using CPU")
        config.device = "cpu"
        config.num_gpus = 0
        config.optimal_batch_size = 1
        config.gradient_accumulation_steps = 16
        config.num_workers = 0
        config.pin_memory = False
        return config
    
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    config.num_gpus = num_gpus
    config.gpu_ids = list(range(num_gpus))
    
    print(f"\n{'='*60}")
    print("GPU CONFIGURATION")
    print(f"{'='*60}")
    print(f"Detected {num_gpus} GPU(s)")
    
    # Get VRAM info for each GPU
    total_vram = 0
    min_vram = float('inf')
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        total_vram += vram_gb
        min_vram = min(min_vram, vram_gb)
        print(f"  GPU {i}: {props.name} - {vram_gb:.1f} GB VRAM")
    
    config.total_vram_gb = total_vram
    config.per_gpu_vram_gb = min_vram
    
    # Determine GPU tier and optimal settings
    if min_vram >= 70:  # A100 80GB class
        print(f"\nâœ“ Detected HIGH-END GPUs (A100 80GB class)")
        config = _configure_high_end(config, num_gpus, min_vram)
    elif min_vram >= 35:  # A100 40GB, A6000 class
        print(f"\nâœ“ Detected MID-HIGH GPUs (A100 40GB / A6000 class)")
        config = _configure_mid_high(config, num_gpus, min_vram)
    elif min_vram >= 20:  # RTX 3090, 4090 class
        print(f"\nâœ“ Detected MID GPUs (RTX 3090/4090 class)")
        config = _configure_mid(config, num_gpus, min_vram)
    elif min_vram >= 8:  # RTX 3080, 4080 class
        print(f"\nâœ“ Detected CONSUMER GPUs (RTX 3080/4080 class)")
        config = _configure_consumer(config, num_gpus, min_vram)
    else:  # Low VRAM
        print(f"\nâš  Detected LOW VRAM GPUs")
        config = _configure_low_vram(config, num_gpus, min_vram)
    
    # Multi-GPU settings
    if num_gpus > 1:
        config.use_data_parallel = True
        print(f"\nâœ“ Multi-GPU: DataParallel enabled across {num_gpus} GPUs")
    
    # Check for Ampere+ architecture (A100, RTX 30xx, 40xx)
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        if props.major >= 8:  # Ampere or newer
            config.use_bf16 = True
            config.use_tf32 = True
            break
    
    # Print final config
    print(f"\nOptimal Settings:")
    print(f"  Batch size per GPU: {config.optimal_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  BF16 training: {config.use_bf16}")
    print(f"  TF32 enabled: {config.use_tf32}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"  DataLoader workers: {config.num_workers}")
    print(f"{'='*60}\n")
    
    return config


def _configure_high_end(config: GPUConfig, num_gpus: int, vram_gb: float) -> GPUConfig:
    """Configure for A100 80GB class GPUs
    
    OPTIMIZED FOR: DeepSeek-R1-Distill-Qwen-7B (7B params)
    
    With max_seq_length=512 (reduced from 768):
    - Memory per sample reduced by ~33%
    - Can fit larger batches
    
    User data analysis showed:
    - 95th percentile: 355 tokens
    - 99th percentile: 471 tokens  
    - Only 0.5% truncated at 512
    """
    # OPTIMIZED: Increased batch size from 10 to 14
    # Shorter sequences (512 vs 768) = ~33% less memory per sample
    config.optimal_batch_size = 14  # OPTIMIZED: Increased for shorter sequences
    config.gradient_accumulation_steps = 4
    config.effective_batch_size = config.optimal_batch_size * num_gpus * config.gradient_accumulation_steps
    
    config.use_bf16 = True
    config.use_tf32 = True
    config.gradient_checkpointing = True
    
    config.num_workers = 8
    config.pin_memory = True
    config.prefetch_factor = 4
    
    return config


def _configure_mid_high(config: GPUConfig, num_gpus: int, vram_gb: float) -> GPUConfig:
    """Configure for A100 40GB / A6000 class GPUs"""
    config.optimal_batch_size = 8
    config.gradient_accumulation_steps = 4
    config.effective_batch_size = config.optimal_batch_size * num_gpus * config.gradient_accumulation_steps
    
    config.use_bf16 = True
    config.use_tf32 = True
    config.gradient_checkpointing = False
    
    config.num_workers = 8
    config.pin_memory = True
    config.prefetch_factor = 4
    
    return config


def _configure_mid(config: GPUConfig, num_gpus: int, vram_gb: float) -> GPUConfig:
    """Configure for RTX 3090/4090 class GPUs"""
    config.optimal_batch_size = 4
    config.gradient_accumulation_steps = 8
    config.effective_batch_size = config.optimal_batch_size * num_gpus * config.gradient_accumulation_steps
    
    config.use_bf16 = True
    config.use_tf32 = True
    config.gradient_checkpointing = True
    
    config.num_workers = 4
    config.pin_memory = True
    config.prefetch_factor = 2
    
    return config


def _configure_consumer(config: GPUConfig, num_gpus: int, vram_gb: float) -> GPUConfig:
    """Configure for RTX 3080/4080 class GPUs"""
    config.optimal_batch_size = 2
    config.gradient_accumulation_steps = 8
    config.effective_batch_size = config.optimal_batch_size * num_gpus * config.gradient_accumulation_steps
    
    config.use_fp16 = True
    config.use_tf32 = True
    config.gradient_checkpointing = True
    
    config.num_workers = 4
    config.pin_memory = True
    config.prefetch_factor = 2
    config.empty_cache_freq = 50
    
    return config


def _configure_low_vram(config: GPUConfig, num_gpus: int, vram_gb: float) -> GPUConfig:
    """Configure for low VRAM GPUs"""
    config.optimal_batch_size = 1
    config.gradient_accumulation_steps = 16
    config.effective_batch_size = config.optimal_batch_size * num_gpus * config.gradient_accumulation_steps
    
    config.use_fp16 = True
    config.gradient_checkpointing = True
    
    config.num_workers = 2
    config.pin_memory = True
    config.prefetch_factor = 2
    config.empty_cache_freq = 20
    
    return config


def setup_gpu_environment(config: GPUConfig) -> None:
    """
    Setup GPU environment based on config.
    
    Call this before training starts.
    """
    if config.device == "cpu":
        return
    
    # Enable TF32 for Ampere GPUs (significant speedup)
    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set CUDA device order for consistent GPU ordering
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Optimize CUDA memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    print("GPU environment configured")


def wrap_model_for_multi_gpu(model: torch.nn.Module, config: GPUConfig) -> torch.nn.Module:
    """
    Wrap model for multi-GPU training if needed.
    
    Args:
        model: The PyTorch model
        config: GPU configuration
        
    Returns:
        Model wrapped with DataParallel if multiple GPUs available
    """
    if config.num_gpus <= 1 or not config.use_data_parallel:
        return model
    
    print(f"Wrapping model with DataParallel across GPUs: {config.gpu_ids}")
    
    # DataParallel automatically handles splitting batches across GPUs
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    
    return model


def get_model_for_saving(model: torch.nn.Module) -> torch.nn.Module:
    """
    Get the underlying model for saving (unwrap DataParallel if needed).
    
    Args:
        model: Potentially wrapped model
        
    Returns:
        Underlying model module
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def create_optimized_dataloader(
    dataset,
    config: GPUConfig,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader based on GPU config.
    
    Args:
        dataset: PyTorch Dataset
        config: GPU configuration
        batch_size: Override batch size (default: use config.optimal_batch_size)
        shuffle: Whether to shuffle data
        drop_last: Whether to drop incomplete last batch
        
    Returns:
        Optimized DataLoader
    """
    if batch_size is None:
        batch_size = config.optimal_batch_size
    
    # Scale batch size for multi-GPU
    if config.use_data_parallel and config.num_gpus > 1:
        # DataParallel splits the batch across GPUs
        # So we multiply batch size by num_gpus for effective batch
        batch_size = batch_size * config.num_gpus
    
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": config.pin_memory,
    }
    
    # Only use workers on GPU (CPU training should use 0 workers)
    if config.device != "cpu" and config.num_workers > 0:
        loader_kwargs["num_workers"] = config.num_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    else:
        loader_kwargs["num_workers"] = 0
    
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def get_amp_context(config: GPUConfig):
    """
    Get automatic mixed precision context manager.
    
    Args:
        config: GPU configuration
        
    Returns:
        AMP autocast context manager
    """
    if config.use_bf16:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif config.use_fp16:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        # Return a no-op context manager
        return torch.cuda.amp.autocast(enabled=False)


def get_grad_scaler(config: GPUConfig) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Get gradient scaler for mixed precision training.
    
    Note: GradScaler is only needed for FP16, not BF16.
    
    Args:
        config: GPU configuration
        
    Returns:
        GradScaler if FP16, None otherwise
    """
    if config.use_fp16:
        return torch.cuda.amp.GradScaler()
    return None


# Convenience function to print memory usage
def print_gpu_memory_usage():
    """Print current GPU memory usage for all GPUs"""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.1f}GB total")


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()