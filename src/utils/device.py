"""Device resolution helper (CPU / CUDA)."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device.

    Parameters
    ----------
    preference:
        "auto" picks CUDA if available, otherwise CPU. "cuda" or "cpu"
        force that device explicitly (raises if CUDA is unavailable and
        forced).
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(preference)
