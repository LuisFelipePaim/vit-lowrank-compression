"""Parametros, tamanho em disco e latencia."""
from __future__ import annotations

import io
import time

import numpy as np
import torch
import torch.nn as nn


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def disk_size_mb(model: nn.Module) -> float:
    """Tamanho do state_dict serializado (torch.save), em MB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.inference_mode()
def latency_ms(model: nn.Module, device: torch.device, input_size, batch_size: int = 1,
               warmup: int = 10, reps: int = 50, dtype=torch.float32, threads: int | None = None) -> dict:
    """Tempo por forward (ms). O modelo ja deve estar em `device`/`dtype`.

    GPU: torch.cuda.synchronize() antes de ligar e antes de parar o cronometro,
    pois as chamadas CUDA sao assincronas. Warmup descartado (cudnn, alocador, caches).
    """
    model.eval()
    cuda = device.type == "cuda"
    old_threads = torch.get_num_threads()
    if not cuda and threads:
        torch.set_num_threads(threads)
    x = torch.randn(batch_size, *input_size, device=device, dtype=dtype)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    if not cuda and threads:
        torch.set_num_threads(old_threads)
    t = np.asarray(times)
    return {"mean": float(t.mean()), "std": float(t.std(ddof=1)), "median": float(np.median(t)),
            "img_per_s": float(batch_size / (np.median(t) / 1e3))}
