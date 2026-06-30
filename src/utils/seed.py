"""
Reproducibility helpers.

The paper enforces full reproducibility by fixing the global random seed
to 42 across NumPy, PyTorch and CUDA (Sec. III-F). This module centralises
that logic so every script/notebook gets identical behaviour.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) for reproducible runs.

    Parameters
    ----------
    seed:
        Global random seed (default 42, matching the paper).
    deterministic:
        If True, also forces deterministic cuDNN kernels. This can slow
        down training slightly but guarantees bit-for-bit reproducibility
        on the same hardware/driver combination.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
