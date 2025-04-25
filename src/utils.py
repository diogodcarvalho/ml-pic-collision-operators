import os
import importlib
import sys
import types

import torch
import torch.distributed as dist


def str_to_class(field):
    try:
        identifier = getattr(sys.modules[__name__], field)
    except AttributeError:
        raise NameError("%s doesn't exist." % field)
    if isinstance(identifier, (types.ClassType, types.TypeType)):
        return identifier
    raise TypeError("%s is not a class." % field)


def class_from_str(class_name):
    module_name = ".".join(class_name.split(".")[:-1])
    class_name = class_name.split(".")[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def rank_print(*args):
    rank = dist.get_rank()
    print(f"[Rank {rank}]", *args)


def root_print(*args):
    if is_distributed():
        rank = dist.get_rank()
        if rank == 0:
            print(*args)
    else:
        print(*args)


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    if is_distributed():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        torch.cuda.set_device(0)
        return 0, 0, 1


def cleanup_ddp():
    dist.destroy_process_group()
