"""Bootstrap pareado e agregacao por seed."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(correct_model, correct_base, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """IC percentil para delta de acuracia (modelo - baseline), em pontos percentuais.

    Pareado: os MESMOS exemplos sao reamostrados para os dois modelos, entao a
    correlacao entre acertos (alta) reduz a largura do IC. Entradas: vetores 0/1
    de shape (n,) ou (n_seeds, n); com varias seeds, tira-se a media por exemplo
    antes de reamostrar (IC sobre o delta medio entre seeds, condicional as seeds).
    """
    d = np.asarray(correct_model, dtype=np.float64) - np.asarray(correct_base, dtype=np.float64)
    if d.ndim == 2:
        d = d.mean(0)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, int(2e7 // n)))
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        means[s:e] = d[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"delta": 100 * float(d.mean()), "lo": 100 * float(lo), "hi": 100 * float(hi)}


def stratified_split(labels, val_per_class: int, seed: int, train_per_class: int | None = None):
    """Separa `val_per_class` exemplos de cada classe para validacao; opcionalmente
    limita o treino a `train_per_class` por classe. Deterministico em `seed`."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        va.append(idx[:val_per_class])
        rest = idx[val_per_class:]
        tr.append(rest if train_per_class is None else rest[:train_per_class])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def subsample_per_class(indices, labels, per_class: int | None, seed: int):
    """Subconjunto estratificado de `indices` (labels = rotulos do dataset inteiro)."""
    if per_class is None:
        return np.asarray(indices)
    indices, labels = np.asarray(indices), np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[indices]):
        idx = indices[labels[indices] == c]
        keep.append(rng.permutation(idx)[:per_class])
    return np.sort(np.concatenate(keep))
