from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def _param_groups(model: nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(("pos_embed", "cls_token", "dist_token")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train_model(model: nn.Module, loader, tcfg: dict, device: torch.device, desc: str = "train") -> dict:
    """AdamW + warmup linear + cosseno, AMP e gradient accumulation.
    Usa o checkpoint da ULTIMA epoca (nenhuma selecao de checkpoint na validacao)."""
    model.to(device).train()
    accum = int(tcfg.get("accum_steps", 1))
    opt = torch.optim.AdamW(_param_groups(model, tcfg["weight_decay"]), lr=tcfg["lr"])
    total = math.ceil(len(loader) / accum) * tcfg["epochs"]
    warm = max(1, int(tcfg.get("warmup_frac", 0.0) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0, last = time.perf_counter(), float("nan")
    epoch_metrics = []
    for ep in range(tcfg["epochs"]):
        ep_start, examples = time.perf_counter(), 0
        run, n = 0.0, 0
        bar = tqdm(loader, desc=f"{desc} ep{ep + 1}/{tcfg['epochs']}", leave=False)
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(bar):
            examples += len(y)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss nao finita em {desc} (ep {ep + 1}, it {i})")
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run, n = run + loss.item(), n + 1
            bar.set_postfix(loss=f"{run / n:.3f}", refresh=False)
        last = run / max(1, n)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - ep_start
        epoch_metrics.append({"epoch": ep + 1, "examples": examples, "seconds": seconds,
                              "images_per_second": examples / seconds})
        print(f"[{desc}] epoch={ep + 1} examples={examples} seconds={seconds:.3f} "
              f"images_per_second={examples / seconds:.3f}", flush=True)
    model.eval()
    return {"train_loss": last, "train_seconds": time.perf_counter() - t0,
            "epoch_metrics": epoch_metrics,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1e6 if device.type == "cuda" else None}
