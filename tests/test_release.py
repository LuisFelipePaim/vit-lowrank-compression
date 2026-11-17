import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vitlr.pipeline import FIELDS, _append, environment_info
from vitlr.utils import load_config


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/06_publish_hf.py"
spec = importlib.util.spec_from_file_location("exporter", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.mark.parametrize("existing", [False, True])
def test_dry_run_leaves_destination_untouched(tmp_path, existing):
    out = tmp_path / "export"
    if existing:
        out.mkdir()
        (out / "keep.txt").write_text("keep")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--config", "configs/smoke.yaml",
                             "--repo-id", "test/test", "--export-dir", str(out), "--dry-run"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["eligible_dataset"] is False
    assert out.exists() == existing
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_synthetic_export_is_refused_without_touching_files(tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--config", "configs/smoke.yaml",
                             "--repo-id", "test/test", "--export-dir", str(out), "--local-only"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "test-only" in result.stderr
    assert sentinel.read_text() == "keep"
    assert list(out.iterdir()) == [sentinel]


def test_existing_destination_refused_before_checkpoint_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", "configs/full.yaml", "--repo-id",
                                     "test/test", "--export-dir", str(tmp_path), "--local-only"])
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.main()
    assert sentinel.read_text() == "keep"


def test_card_metadata_uses_training_config_and_rejects_mismatch():
    cfg = load_config("configs/full.yaml")
    ck = {"experiment_config": copy.deepcopy(cfg), "model_cfg": cfg["models"]["baseline"],
          "num_classes": 101, "seed": 42}
    meta = exporter.export_metadata(cfg, ck, {"input_size": (3, 224, 224)}, "baseline", 42, None)
    card = exporter.model_card(meta, "CSV TABLE", "test/model", "PENDING")
    assert 'ethz/food101' in card and 'Classes: 101' in card and '[3, 224, 224]' in card
    assert 'base_model: "timm/deit_small_patch16_224.fb_in1k"' in card
    ck["purpose"] = "feasibility_only"
    with pytest.raises(ValueError, match="Feasibility-only"):
        exporter.export_metadata(cfg, ck, {}, "baseline", 42, None)
    del ck["purpose"]
    ck["experiment_config"]["data"]["name"] = "fake"
    with pytest.raises(ValueError, match="differs"):
        exporter.export_metadata(cfg, ck, {}, "baseline", 42, None)
    with pytest.raises(ValueError, match="lacks"):
        exporter.export_metadata(cfg, {}, {}, "baseline", 42, None)


def test_csv_environment_and_legacy_schema_protection(tmp_path):
    cfg = {"experiment": {"results_dir": str(tmp_path)}}
    info = environment_info()
    assert all(info[k] for k in ("python_version", "torch_version", "cuda_version", "driver_version"))
    _append(cfg, {"run_id": "test", **info})
    import csv
    with (tmp_path / "runs.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        row = next(reader)
        assert all(row[k] == v for k, v in info.items())
    (tmp_path / "runs.csv").write_text("run_id,torch_version\nlegacy,unknown\n")
    before = (tmp_path / "runs.csv").read_bytes()
    with pytest.raises(ValueError, match="schema"):
        _append(cfg, {"run_id": "new"})
    assert (tmp_path / "runs.csv").read_bytes() == before
