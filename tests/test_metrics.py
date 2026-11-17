import numpy as np
import pytest
import torch

from vitlr.benchmark import disk_size_mb, latency_ms, param_count
from vitlr.flops import count_flops
from vitlr.model import create_model
from vitlr.quantize import to_fp16, to_int8_dynamic
from vitlr.stats import paired_bootstrap_ci, stratified_split, subsample_per_class


def test_flops_deit_small_matches_literature():
    m = create_model({"name": "deit_small_patch16_224", "pretrained": False}, 1000)
    f = count_flops(m)
    assert param_count(m) == 22_050_664
    assert 4.2e9 < f["total"] < 4.7e9          # DeiT-S: ~4.6 GFLOPs (Touvron et al., 2021)
    # conferencia analitica: 12 blocos, 197 tokens, d=384
    n, d = 197, 384
    per_block = n * (d * 3 * d + d * d + d * 4 * d + 4 * d * d) + 2 * n * n * d
    assert f["total"] == 12 * per_block + 196 * 3 * 16 * 16 * d + d * 1000


def test_bootstrap_identical_and_known_delta():
    rng = np.random.default_rng(0)
    base = rng.random(4000) < 0.8
    ci = paired_bootstrap_ci(base, base, 500)
    assert ci["delta"] == ci["lo"] == ci["hi"] == 0.0
    worse = base & (rng.random(4000) > 0.1)      # perde ~8 pp
    ci = paired_bootstrap_ci(worse, base, 2000)
    assert ci["lo"] < ci["delta"] < ci["hi"] < 0 and ci["lo"] < -8 < ci["hi"]
    pooled = paired_bootstrap_ci(np.stack([worse, worse]), np.stack([base, base]), 2000)
    assert pooled["delta"] == pytest.approx(ci["delta"])


def test_paired_ci_is_tighter_than_unpaired():
    rng = np.random.default_rng(1)
    base = rng.random(5000) < 0.85
    model = base.copy()
    flip = rng.random(5000) < 0.02
    model[flip] = ~model[flip]
    paired = paired_bootstrap_ci(model, base, 2000)
    unpaired = paired_bootstrap_ci(rng.permutation(model), base, 2000)
    assert paired["hi"] - paired["lo"] < 0.6 * (unpaired["hi"] - unpaired["lo"])


def test_stratified_split():
    labels = np.repeat(np.arange(10), 50)
    tr, va = stratified_split(labels, 5, seed=3, train_per_class=20)
    assert len(va) == 50 and len(tr) == 200 and not set(tr) & set(va)
    assert (np.bincount(labels[va]) == 5).all() and (np.bincount(labels[tr]) == 20).all()
    assert np.array_equal(stratified_split(labels, 5, seed=3, train_per_class=20)[1], va)
    sub = subsample_per_class(tr, labels, 7, seed=0)
    assert (np.bincount(labels[sub]) == 7).all() and set(sub) <= set(tr)


def test_quantization_and_latency():
    m = create_model({"name": "deit_tiny_patch16_224", "pretrained": False,
                      "kwargs": {"img_size": 32, "patch_size": 8, "embed_dim": 64, "depth": 2, "num_heads": 4}}, 5).eval()
    q, h = to_int8_dynamic(m), to_fp16(m)
    x = torch.randn(2, 3, 32, 32)
    assert q(x).shape == (2, 5) and torch.isfinite(q(x)).all()
    assert disk_size_mb(q) < 0.5 * disk_size_mb(m)
    assert disk_size_mb(h) < 0.6 * disk_size_mb(m)
    assert next(m.parameters()).dtype == torch.float32    # original intacto
    r = latency_ms(m, torch.device("cpu"), (3, 32, 32), 1, warmup=1, reps=3)
    assert r["mean"] > 0 and r["img_per_s"] > 0


def test_hf_image_dataset_and_transforms():
    datasets = pytest.importorskip("datasets")
    from PIL import Image

    from vitlr.data import HFImageDataset, build_transforms

    imgs = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (300, 400, 3), dtype=np.uint8)) for i in range(4)]
    ds = datasets.Dataset.from_dict({"image": imgs, "label": [0, 1, 0, 1]}).cast_column("image", datasets.Image())
    dc = {"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
          "crop_pct": 0.9, "interpolation": "bicubic"}
    for train in (True, False):
        x, y = HFImageDataset(ds, [1, 3], build_transforms(dc, train))[0]
        assert x.shape == (3, 224, 224) and y == 1
