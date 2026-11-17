"""Contador de FLOPs por hooks (convencao fvcore/ptflops: 1 multiply-accumulate = 1 FLOP).

Conta o que e EXECUTADO no forward: cada chamada de nn.Linear / nn.Conv2d e os dois
produtos matriciais da atencao (QK^T e attn.V). Uma FactorizedLinear dispara dois
hooks de Linear, logo custa r*(d_in+d_out) por token - se alguem reconstruisse a
matriz densa no forward, o contador (e o teste) acusariam. LayerNorm, GELU, softmax
e somas sao ignorados (< 1% em um ViT).
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def count_flops(model: nn.Module, input_size=(3, 224, 224)) -> dict:
    stats = {"linear": 0, "conv": 0, "attn_matmul": 0, "linear_calls": 0}
    hooks = []

    def lin_hook(m, inp, out):
        stats["linear"] += out.numel() * m.in_features
        stats["linear_calls"] += 1

    def conv_hook(m, inp, out):
        stats["conv"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def attn_hook(m, inp, out):
        b, n, c = inp[0].shape
        stats["attn_matmul"] += 2 * b * n * n * c

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif hasattr(m, "qkv") and hasattr(m, "num_heads"):
            hooks.append(m.register_forward_hook(attn_hook))

    p = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        model(torch.zeros(1, *input_size, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    stats["total"] = stats["linear"] + stats["conv"] + stats["attn_matmul"]
    return stats
