"""
Model complexity profiling (Table 6): trainable parameter counts,
on-disk checkpoint size, and per-sample inference latency.
"""

from __future__ import annotations

import io
import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """On-disk size (MB) of the model's state_dict, computed in-memory."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def measure_inference_latency_ms(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """Average per-sample inference latency in milliseconds.

    `sample_input` should have batch size 1, shape (1, T, F).
    """
    model = model.to(device).eval()
    sample_input = sample_input.to(device)

    for _ in range(n_warmup):
        model(sample_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1000.0
