"""
seed.py — Reproducibility

Sets random seeds for Python, NumPy, PyTorch (CPU + CUDA) and
configures CuDNN for deterministic behaviour.

Usage:
    from utils.seed import set_seed
    set_seed(42)

Note:
    CUDA determinism may reduce GPU throughput slightly.
    Set deterministic=False to disable if training is too slow.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed:          integer seed value
        deterministic: if True, force CuDNN deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        os.environ['PYTHONHASHSEED']       = str(seed)
