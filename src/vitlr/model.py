from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from .compress import apply_ranks


def create_model(mcfg: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    pre = mcfg.get("pretrained", True) if pretrained is None else pretrained
    return timm.create_model(mcfg["name"], pretrained=pre, num_classes=num_classes, **(mcfg.get("kwargs") or {}))


def data_config(model: nn.Module) -> dict:
    """input_size, mean, std, crop_pct e interpolacao esperados pelo checkpoint pre-treinado."""
    dc = dict(timm.data.resolve_model_data_config(model))
    img = getattr(getattr(model, "patch_embed", None), "img_size", None)
    if img is not None:  # respeita img_size passado em kwargs (o pretrained_cfg diria 224)
        dc["input_size"] = (dc["input_size"][0], *img)
    return dc


def save_checkpoint(path, model: nn.Module, mcfg: dict, num_classes: int, ranks: dict | None = None, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": mcfg, "num_classes": num_classes,
                "ranks": ranks, **extra}, path)


def build_from_meta(mcfg: dict, num_classes: int, ranks: dict | None) -> nn.Module:
    model = create_model(mcfg, num_classes, pretrained=False)
    if ranks:
        apply_ranks(model, ranks)
    return model


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_from_meta(ck["model_cfg"], ck["num_classes"], ck.get("ranks"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
