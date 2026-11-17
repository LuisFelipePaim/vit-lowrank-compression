"""Dados: Food-101 (HF `ethz/food101`) com split oficial + validacao estratificada
tirada do treino. O split oficial de TESTE e o split `validation` do repo no HF.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .stats import stratified_split, subsample_per_class


class HFImageDataset(Dataset):
    """Embrulha um datasets.Dataset com colunas `image` (PIL) e `label`."""

    def __init__(self, hf_ds, indices, transform):
        self.ds, self.indices, self.transform = hf_ds, np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.ds[int(self.indices[i])]
        return self.transform(ex["image"].convert("RGB")), int(ex["label"])


class TensorImageDataset(Dataset):
    def __init__(self, x, y, indices, flip=False):
        self.x, self.y, self.indices, self.flip = x, y, np.asarray(indices), flip

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        img = self.x[j]
        if self.flip and torch.rand(()) < 0.5:
            img = img.flip(-1)
        return img, int(self.y[j])


class OriginalImageDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.ds, self.indices, self.transform = dataset, np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.ds[int(self.indices[i])]
        return self.transform(image.convert("RGB")), int(label)


def build_transforms(data_cfg: dict, train: bool):
    from torchvision import transforms as T

    size = data_cfg["input_size"][-1]
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR}.get(
        data_cfg.get("interpolation", "bicubic"), T.InterpolationMode.BICUBIC)
    norm = [T.ToTensor(), T.Normalize(data_cfg["mean"], data_cfg["std"])]
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.35, 1.0), interpolation=interp),
                          T.RandomHorizontalFlip(), *norm])
    resize = int(round(size / data_cfg.get("crop_pct", 0.875)))
    return T.Compose([T.Resize(resize, interpolation=interp), T.CenterCrop(size), *norm])


def _fake(n_per_class, num_classes, size, seed, proto_seed=0):
    g = torch.Generator().manual_seed(proto_seed)
    protos = torch.nn.functional.interpolate(torch.randn(num_classes, 3, 4, 4, generator=g), size=size, mode="bilinear")
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(num_classes).repeat_interleave(n_per_class)
    x = protos[y] + 1.5 * torch.randn(len(y), 3, size, size, generator=g)
    return x, y


class DataBundle:
    """Guarda os indices de treino/validacao e cria DataLoaders. O teste so e
    carregado quando `loader("test", ...)` e chamado (scripts/04_final_test.py)."""

    def __init__(self, cfg: dict, data_cfg: dict):
        self.cfg, self.dcfg, self.data_cfg = cfg, cfg["data"], data_cfg
        d = self.dcfg
        self.fake = d["name"] == "fake"
        self.original = d.get("source") == "original" and not self.fake
        if self.fake:
            self.class_names = [str(i) for i in range(d["num_classes"])]
            self._x, self._y = _fake(d["fake_per_class"], d["num_classes"], data_cfg["input_size"][-1], seed=1)
            self.labels = self._y.numpy()
        elif self.original:
            from torchvision.datasets import Food101

            self.root = os.environ[d.get("root_env", "FOOD101_ROOT")]
            self._train = Food101(self.root, split="train", download=False)
            self.labels = np.asarray(self._train._labels)
            self.class_names = self._train.classes
            assert len(self.class_names) == d["num_classes"], "numero de classes inesperado"
        else:
            from datasets import load_dataset

            if d.get("train_files"):
                # Local official shards may be downloaded independently and hash-verified.
                self._train = load_dataset("parquet", data_files={"train": d["train_files"]},
                                           split="train", cache_dir=d.get("cache_dir"))
            else:
                self._train = load_dataset(d["hf_id"], split="train", cache_dir=d.get("cache_dir"))
            self.labels = self._train.select_columns(["label"]).to_pandas()["label"].to_numpy()
            self.class_names = self._train.features["label"].names
            assert len(np.unique(self.labels)) == d["num_classes"], "numero de classes inesperado"
        self.train_idx, self.val_idx = stratified_split(
            self.labels, d["val_per_class"], d["split_seed"], d.get("train_per_class"))
        assert not set(self.train_idx) & set(self.val_idx)
        self._test = None

    def n_train(self, per_class=None, seed=0):
        return len(subsample_per_class(self.train_idx, self.labels, per_class, seed))

    def loader(self, split: str, batch_size: int, train: bool = False, seed: int = 0, per_class: int | None = None):
        d = self.dcfg
        if split == "train":
            idx = subsample_per_class(self.train_idx, self.labels, per_class, seed)
        elif split == "val":
            idx = self.val_idx
        elif split == "test":
            idx = None
        else:
            raise ValueError(split)

        if self.fake:
            if split == "test":
                x, y = _fake(d["fake_test_per_class"], d["num_classes"], self.data_cfg["input_size"][-1], seed=2)
                ds = TensorImageDataset(x, y, np.arange(len(y)))
            else:
                ds = TensorImageDataset(self._x, self._y, idx, flip=train)
        else:
            tf = build_transforms(self.data_cfg, train)
            if self.original:
                if split == "test":
                    if self._test is None:
                        from torchvision.datasets import Food101
                        self._test = Food101(self.root, split="test", download=False)
                    ds = OriginalImageDataset(self._test, np.arange(len(self._test)), tf)
                else:
                    ds = OriginalImageDataset(self._train, idx, tf)
            elif split == "test":
                if self._test is None:
                    from datasets import load_dataset

                    self._test = load_dataset(d["hf_id"], split="validation", cache_dir=d.get("cache_dir"))
                ds = HFImageDataset(self._test, np.arange(len(self._test)), tf)
            else:
                ds = HFImageDataset(self._train, idx, tf)

        nw = d.get("num_workers", 0)
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=nw,
                          pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0, generator=g)
