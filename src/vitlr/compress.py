"""Compressao de baixo posto: Linear(d_in, d_out) -> Linear(d_in, r) -> Linear(r, d_out).

W (d_out x d_in) = U S V^T. Mantemos os r maiores valores singulares e dividimos
sqrt(S) entre os dois fatores (melhor condicionamento para o fine-tuning):
    first.weight  = sqrt(S_r) V_r^T     (r x d_in, sem bias)
    second.weight = U_r sqrt(S_r)       (d_out x r, com o bias original)
O forward executa as DUAS multiplicacoes; a matriz densa nunca e reconstruida.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    @torch.no_grad()
    def reconstruct(self) -> torch.Tensor:
        """Somente para testes/analise. Nunca chamado no forward."""
        return self.second.weight @ self.first.weight

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, rank: int, svd=None) -> "FactorizedLinear":
        U, S, Vh = svd if svd is not None else svd_of(linear)
        fl = cls(linear.in_features, linear.out_features, rank, bias=linear.bias is not None)
        sq = S[:rank].sqrt()
        w = linear.weight
        fl.first.weight.copy_((sq[:, None] * Vh[:rank]).to(w.dtype))
        fl.second.weight.copy_((U[:, :rank] * sq[None, :]).to(w.dtype))
        if linear.bias is not None:
            fl.second.bias.copy_(linear.bias)
        return fl.to(w.device)


@torch.no_grad()
def svd_of(linear: nn.Linear):
    """SVD em float64 na CPU (as matrizes sao pequenas; evita ruido numerico)."""
    return torch.linalg.svd(linear.weight.detach().double().cpu(), full_matrices=False)


def rank_for_energy(S: torch.Tensor, threshold: float) -> int:
    """Menor r tal que sum_{i<=r} s_i^2 / sum_i s_i^2 >= threshold."""
    energy = torch.cumsum(S.double() ** 2, 0)
    energy = energy / energy[-1]
    r = int(torch.searchsorted(energy, torch.tensor(threshold - 1e-12, dtype=energy.dtype)).item()) + 1
    return max(1, min(r, S.numel()))


def reduces_params(d_in: int, d_out: int, rank: int) -> bool:
    return rank * (d_in + d_out) < d_in * d_out


def target_linears(model: nn.Module, targets: list[str]) -> list[tuple[str, nn.Linear]]:
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name.startswith("blocks.") and any(name.endswith(t) for t in targets)
    ]


def _set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


@torch.no_grad()
def compress_model(model: nn.Module, threshold: float, targets: list[str], skip_if_not_smaller: bool = True) -> dict:
    """Fatora, in-place, as camadas-alvo. Retorna {nome: info} com o rank de cada camada.

    `rank` = rank efetivamente usado (0 = camada mantida densa porque o rank pedido
    pelo limiar nao reduziria parametros); `rank_energy` = rank pedido pelo limiar.
    """
    info = {}
    for name, lin in target_linears(model, targets):
        U, S, Vh = svd_of(lin)
        r = rank_for_energy(S, threshold)
        ok = reduces_params(lin.in_features, lin.out_features, r) or not skip_if_not_smaller
        if ok:
            _set_module(model, name, FactorizedLinear.from_linear(lin, r, (U, S, Vh)))
        tot = float((S**2).sum())
        info[name] = {
            "rank": r if ok else 0,
            "rank_energy": r,
            "full_rank": int(S.numel()),
            "energy_kept": float((S[:r] ** 2).sum() / tot) if ok else 1.0,
            "rel_fro_err": float(((S[r:] ** 2).sum() / tot).sqrt()) if ok else 0.0,
        }
    return info


def apply_ranks(model: nn.Module, ranks: dict[str, int]) -> nn.Module:
    """Recria a ESTRUTURA fatorada (pesos nao inicializados) para carregar um state_dict."""
    for name, r in ranks.items():
        if r > 0:
            lin = model.get_submodule(name)
            _set_module(model, name, FactorizedLinear(lin.in_features, lin.out_features, r, lin.bias is not None))
    return model


def ranks_of(info: dict) -> dict[str, int]:
    return {k: v["rank"] for k, v in info.items()}
