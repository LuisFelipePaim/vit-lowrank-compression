from __future__ import annotations

import copy

import torch
import torch.nn as nn


def to_fp16(model: nn.Module) -> nn.Module:
    """Copia em meia precisao (uso pratico: GPU)."""
    return copy.deepcopy(model).half().eval()


def to_int8_dynamic(model: nn.Module) -> nn.Module:
    """Quantizacao dinamica INT8 das nn.Linear (pesos int8, ativacoes quantizadas
    em tempo de execucao). Somente CPU. Tambem atinge as duas Linear de cada
    FactorizedLinear."""
    from torch.ao.quantization import quantize_dynamic

    m = copy.deepcopy(model).float().cpu().eval()
    return quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
