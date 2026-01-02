"""
mmdet 호환성 모듈 - mmdet 의존성 없이 핵심 기능 구현
"""
import torch
import torch.nn as nn
import random
import numpy as np
import os


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    
    Args:
        func: Function to apply.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        tuple: Results from applying func to each argument.
    """
    pfunc = lambda *args, **kwargs: func(*args, **kwargs)
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """Reduce mean across all processes if distributed training is enabled.
    
    Args:
        tensor: Input tensor.
    
    Returns:
        Tensor: Reduced tensor.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid function.
    
    Args:
        x: Input tensor.
        eps: Small value to avoid numerical instability.
    
    Returns:
        Tensor: Inverse sigmoid of x.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to set deterministic options for CUDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def to_tensor(data):
    """Convert data to tensor.
    
    Args:
        data: Input data (numpy array, list, etc.)
    
    Returns:
        Tensor: Converted tensor.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    else:
        return torch.tensor(data)


def replace_ImageToTensor(pipelines):
    """Replace ImageToTensor to DefaultFormatBundle in pipeline.
    
    Args:
        pipelines: List of pipeline configs.
    
    Returns:
        List: Modified pipeline configs.
    """
    pipelines = pipelines.copy()
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'ImageToTensor':
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines


# ==================== Model Builder Registry ====================
from .registry import Registry, build_from_cfg

# mmdet.models.builder의 레지스트리들을 대체
BACKBONES = Registry('backbone')
HEADS = Registry('head')

