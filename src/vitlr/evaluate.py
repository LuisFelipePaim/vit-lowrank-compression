from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, dtype=torch.float32, desc: str = "eval") -> dict:
    """Top-1/top-5 e loss. Sem autocast: fp32 e avaliado em fp32 de verdade, para
    que a comparacao com fp16 faca sentido. Devolve acertos por exemplo (bootstrap)."""
    model.eval()
    c1, c5, loss, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device, dtype=dtype, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        top = logits.topk(min(5, logits.shape[1]), dim=1).indices
        c1.append((top[:, 0] == y).cpu().numpy())
        c5.append((top == y[:, None]).any(1).cpu().numpy())
        n += y.numel()
    c1, c5 = np.concatenate(c1), np.concatenate(c5)
    return {"top1": 100 * c1.mean(), "top5": 100 * c5.mean(), "loss": loss / n, "n": n, "correct1": c1}
