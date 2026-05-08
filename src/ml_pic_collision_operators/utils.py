import os
import importlib
import numpy as np
import torch
import torch.distributed as dist
from typing import Any, Type


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def class_from_str(class_name: str, module_name: str | None = None) -> Type[Any]:
    """Load a Python class from string name

    Args:
        class_name: class name with optional module path
        module_name: module name where class is located, if None, will be
            inferred from class_name

    Returns:
        class: loaded class type, object could now be instantiated with class(args)
    """
    if module_name is None:
        module_name = ".".join(class_name.split(".")[:-1])
        class_name = class_name.split(".")[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def rank_print(*args):
    """Print only from the current rank in distributed environment"""
    rank = dist.get_rank()
    print(f"[Rank {rank}]", *args)


def root_print(*args):
    """Print only from rank 0 in distributed environment"""
    if is_distributed():
        rank = dist.get_rank()
        if rank == 0:
            print(*args)
    else:
        print(*args)


def is_distributed():
    """Check if running in distributed environment"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed(force_not_ddp: bool = False) -> tuple[int, int, int, int | str]:
    """Setup distributed environment if in distributed setting."""
    cuda_available = torch.cuda.is_available()
    backend = "nccl" if cuda_available else "gloo"
    if is_distributed() and not force_not_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = local_rank if torch.cuda.is_available() else "cpu"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if cuda_available:
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, device
    else:
        device = 0 if cuda_available else "cpu"
        if cuda_available:
            torch.cuda.set_device(0)
        return 0, 0, 1, device


def cleanup_ddp(force_not_ddp: bool = False):
    """Cleanup distributed environment"""
    if is_distributed() and not force_not_ddp:
        dist.destroy_process_group()
