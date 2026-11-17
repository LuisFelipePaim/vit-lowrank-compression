import numpy as np
from PIL import Image
import datasets

from vitlr.data import DataBundle


def test_local_parquet_training_shards_preserve_labels_and_split(tmp_path):
    images = [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(6)]
    ds = datasets.Dataset.from_dict({"image": images, "label": [0, 0, 0, 1, 1, 1]})
    ds = ds.cast_column("image", datasets.Image()).cast_column("label", datasets.ClassLabel(names=["a", "b"]))
    ds.to_parquet(tmp_path / "train.parquet")
    cfg = {"data": {"name": "food101", "train_files": str(tmp_path / "train.parquet"),
                    "cache_dir": str(tmp_path / "cache"), "num_classes": 2, "split_seed": 1234,
                    "val_per_class": 1, "num_workers": 0}}
    dc = {"input_size": (3, 32, 32), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0}
    bundle = DataBundle(cfg, dc)
    assert len(bundle.train_idx) == 4 and len(bundle.val_idx) == 2
    assert not set(bundle.train_idx) & set(bundle.val_idx)
    assert bundle._train.features["label"].names == ["a", "b"]
    x, y = next(iter(bundle.loader("val", 2)))
    assert x.shape == (2, 3, 32, 32) and sorted(y.tolist()) == [0, 1]
    assert bundle._test is None


def test_original_training_source_preserves_class_order_and_never_opens_test(tmp_path, monkeypatch):
    import json
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    metadata = {name: [f"{name}/{i}" for i in range(3)] for name in ("b", "a")}
    (root / "meta/train.json").write_text(json.dumps(metadata))
    # test.json is deliberately absent: construction/validation must not open it.
    for paths in metadata.values():
        for path in paths:
            p = root / "images" / (path + ".jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32)).save(p)
    monkeypatch.setenv("FOOD101_ROOT", str(tmp_path))
    cfg = {"data": {"name": "food101", "source": "original", "num_classes": 2,
                    "split_seed": 1234, "val_per_class": 1, "num_workers": 0}}
    dc = {"input_size": (3, 32, 32), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0}
    bundle = DataBundle(cfg, dc)
    assert bundle.class_names == ["a", "b"]
    assert bundle.labels.tolist() == [1, 1, 1, 0, 0, 0]
    x, y = next(iter(bundle.loader("val", 2)))
    assert x.shape == (2, 3, 32, 32) and sorted(y.tolist()) == [0, 1]
    assert not set(bundle.train_idx) & set(bundle.val_idx)
    assert bundle._test is None
