import copy

import pytest
import torch
import torch.nn as nn

from vitlr.benchmark import param_count
from vitlr.compress import (FactorizedLinear, apply_ranks, compress_model, rank_for_energy, ranks_of,
                            reduces_params, svd_of, target_linears)
from vitlr.flops import count_flops
from vitlr.model import create_model

TARGETS = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
TINY = {"name": "deit_tiny_patch16_224", "pretrained": False,
        "kwargs": {"img_size": 32, "patch_size": 8, "embed_dim": 64, "depth": 2, "num_heads": 4}}


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return create_model(TINY, 7).eval()


def test_rank_for_energy_known_spectrum():
    S = torch.tensor([3.0, 2.0, 1.0, 1.0, 1.0])  # energias 9,4,1,1,1 / 16
    assert rank_for_energy(S, 0.5) == 1          # 9/16 = 0.5625
    assert rank_for_energy(S, 0.5625) == 1       # limiar exatamente atingido
    assert rank_for_energy(S, 0.6) == 2          # 13/16
    assert rank_for_energy(S, 0.9) == 4          # 15/16 = 0.9375
    assert rank_for_energy(S, 1.0) == 5


def test_rank_monotonic_in_threshold():
    S = torch.linalg.svdvals(torch.randn(48, 96))
    ranks = [rank_for_energy(S, t) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)]
    assert ranks == sorted(ranks) and ranks[0] >= 1 and ranks[-1] <= 48


@pytest.mark.parametrize("d_in,d_out", [(32, 96), (96, 32), (48, 48)])
def test_full_rank_is_exact(d_in, d_out):
    lin = nn.Linear(d_in, d_out)
    fl = FactorizedLinear.from_linear(lin, min(d_in, d_out))
    assert torch.allclose(fl.reconstruct(), lin.weight, atol=1e-5)
    x = torch.randn(3, 5, d_in)
    assert torch.allclose(fl(x), lin(x), atol=1e-4)


@pytest.mark.parametrize("rank", [1, 4, 17])
def test_reconstruction_error_is_eckart_young(rank):
    lin = nn.Linear(40, 64)
    _, S, _ = svd_of(lin)
    fl = FactorizedLinear.from_linear(lin, rank)
    err = torch.linalg.matrix_norm(lin.weight.double() - fl.reconstruct().double())
    assert err.item() == pytest.approx(S[rank:].pow(2).sum().sqrt().item(), rel=1e-4)


def test_param_count_and_shapes_and_bias():
    lin = nn.Linear(40, 64)
    fl = FactorizedLinear.from_linear(lin, 6)
    assert param_count(fl) == 6 * (40 + 64) + 64
    assert fl.first.weight.shape == (6, 40) and fl.second.weight.shape == (64, 6)
    assert fl.first.bias is None and torch.equal(fl.second.bias, lin.bias)
    for shape in [(40,), (2, 40), (2, 9, 40)]:
        assert fl(torch.randn(*shape)).shape == lin(torch.randn(*shape)).shape
    nb = FactorizedLinear.from_linear(nn.Linear(40, 64, bias=False), 6)
    assert nb.second.bias is None and param_count(nb) == 6 * 104


def test_no_dense_matrix_and_two_ops_in_forward():
    """A camada fatorada nao guarda nem reconstroi W: sao duas Linear executadas em sequencia."""
    d_in, d_out, r, tokens = 40, 64, 6, 9
    fl = FactorizedLinear.from_linear(nn.Linear(d_in, d_out), r)
    assert all(tuple(p.shape) != (d_out, d_in) for p in fl.parameters())
    calls = []
    hooks = [m.register_forward_hook(lambda m, i, o: calls.append((m.in_features, m.out_features)))
             for m in fl.modules() if isinstance(m, nn.Linear)]
    fl(torch.randn(1, tokens, d_in))
    [h.remove() for h in hooks]
    assert calls == [(d_in, r), (r, d_out)]


def test_reduces_params_break_even():
    assert reduces_params(384, 384, 191) and not reduces_params(384, 384, 192)
    assert reduces_params(384, 1536, 307) and not reduces_params(384, 1536, 308)


def test_compress_model_reduces_params_and_flops(tiny):
    base_p, base_f = param_count(tiny), count_flops(tiny, (3, 32, 32))
    m = copy.deepcopy(tiny)
    info = compress_model(m, 0.6, TARGETS)
    assert len(info) == 2 * 4 == len(target_linears(tiny, TARGETS))
    assert any(v["rank"] > 0 for v in info.values())
    f = count_flops(m, (3, 32, 32))
    assert param_count(m) < base_p and f["total"] < base_f["total"]
    # parametros e FLOPs batem com a formula r*(d_in+d_out)
    tokens, dp, df = 17, 0, 0
    for name, v in info.items():
        if v["rank"] > 0:
            lin = tiny.get_submodule(name)
            saved = lin.in_features * lin.out_features - v["rank"] * (lin.in_features + lin.out_features)
            dp, df = dp + saved, df + saved * tokens
    assert base_p - param_count(m) == dp
    assert base_f["total"] - f["total"] == df
    assert f["linear_calls"] == base_f["linear_calls"] + sum(v["rank"] > 0 for v in info.values())
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 7)


def test_skip_layers_that_would_grow(tiny):
    m = copy.deepcopy(tiny)
    info = compress_model(m, 0.9999, TARGETS, skip_if_not_smaller=True)
    assert all(v["rank"] == 0 for v in info.values())      # pesos aleatorios: rank ~cheio
    assert param_count(m) == param_count(tiny)
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(m(x), tiny(x))


def test_untruncated_compression_preserves_outputs(tiny):
    m = copy.deepcopy(tiny)
    info = compress_model(m, 1.0, TARGETS, skip_if_not_smaller=False)
    assert all(v["rank"] == v["full_rank"] for v in info.values())
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(m(x), tiny(x), atol=1e-4)


def test_checkpoint_roundtrip(tiny, tmp_path):
    from vitlr.model import load_checkpoint, save_checkpoint

    m = copy.deepcopy(tiny)
    info = compress_model(m, 0.7, TARGETS)
    save_checkpoint(tmp_path / "m.pt", m, TINY, 7, ranks=ranks_of(info))
    m2, ck = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(m2(x), m(x)) and ck["ranks"] == ranks_of(info)
    assert param_count(apply_ranks(create_model(TINY, 7), ranks_of(info))) == param_count(m)
