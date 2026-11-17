from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Le um YAML; a chave opcional `_base_` aponta para outro YAML que e herdado."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def thr_tag(thr: float) -> str:
    return f"{thr:.2f}"
