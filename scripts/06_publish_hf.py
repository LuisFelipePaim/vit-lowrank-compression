"""Prepare local model packages; remote publication is suspended.

--dry-run is strictly read-only. --local-only requires a new destination.
Synthetic checkpoints cannot be exported or published.
"""
import sys

# Keep even Python's import cache read-only for dry-run invocations.
sys.dont_write_bytecode = True

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from safetensors.torch import save_file

from vitlr.model import data_config, load_checkpoint
from vitlr.pipeline import ckpt_path
from vitlr.report import fmt_table
from vitlr.utils import load_config

LOADER = '''"""Loader autossuficiente (depende so de torch, timm, safetensors, huggingface_hub).

    from vitlr_loader import load
    model, cfg = load("USUARIO/REPO")     # ou um diretorio local
"""
import json
import os

import timm
import torch.nn as nn
from safetensors.torch import load_file


class FactorizedLinear(nn.Module):
    """Linear(d_in, d_out) de baixo posto: Linear(d_in, r) -> Linear(r, d_out)."""

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.second(self.first(x))


def load(repo_or_dir):
    if not os.path.isdir(repo_or_dir):
        from huggingface_hub import snapshot_download
        repo_or_dir = snapshot_download(repo_or_dir)
    with open(os.path.join(repo_or_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=cfg["num_classes"], **cfg["model_kwargs"])
    for name, r in (cfg.get("ranks") or {}).items():
        if r > 0:
            parent, _, child = name.rpartition(".")
            lin = model.get_submodule(name)
            setattr(model.get_submodule(parent), child,
                    FactorizedLinear(lin.in_features, lin.out_features, r, lin.bias is not None))
    model.load_state_dict(load_file(os.path.join(repo_or_dir, "model.safetensors")))
    return model.eval(), cfg
'''

def export_metadata(cfg, ck, dc, variant, seed, threshold):
    """Fail closed when the effective training configuration cannot be verified."""
    saved = ck.get("experiment_config")
    if saved is None:
        raise ValueError("Checkpoint lacks experiment_config; legacy artifacts cannot be released.")
    if ck.get("purpose") == "feasibility_only":
        raise ValueError("Feasibility-only checkpoints cannot be exported for release.")
    role = "control" if variant == "control" else "baseline"
    if saved["data"] != cfg["data"] or saved["models"][role] != cfg["models"][role]:
        raise ValueError("Requested config differs from the checkpoint training configuration.")
    if ck["model_cfg"] != saved["models"][role] or ck["num_classes"] != saved["data"]["num_classes"]:
        raise ValueError("Checkpoint architecture/classes disagree with its training configuration.")
    if ck.get("seed") != seed or (variant == "compressed" and ck.get("threshold") != threshold):
        raise ValueError("Checkpoint seed/threshold does not match the requested artifact.")
    if saved["data"]["name"] != "food101":
        raise ValueError("Synthetic/unsupported datasets are test-only; export and publication refused.")
    return {"architecture": ck["model_cfg"]["name"],
            "model_kwargs": ck["model_cfg"].get("kwargs") or {},
            "num_classes": ck["num_classes"], "ranks": ck.get("ranks"),
            "energy_threshold": threshold, "seed": seed,
            "dataset": saved["data"], "pretrained": saved["models"][role].get("pretrained", True),
            "data_config": {k: list(v) if isinstance(v, tuple) else v for k, v in dc.items()},
            "experiment_config": saved}


def model_card(meta, table, repo, github):
    # Weight license is deliberately pending until the licensing review.
    dataset = meta["dataset"]["hf_id"]
    base = ("base_model: " + json.dumps("timm/" + meta["architecture"]) + "\n") if meta["pretrained"] else ""
    return ("---\npipeline_tag: image-classification\nlibrary_name: timm\n"
            + "datasets: [" + json.dumps(dataset) + "]\n" + base
            + "tags: [vision-transformer, svd, model-compression]\n---\n\n"
            + "# " + repo.split("/")[-1] + "\n\n"
            + f"Dataset: `{dataset}`. Classes: {meta['num_classes']}. "
            + f"Input: `{meta['data_config']['input_size']}`. "
            + f"Initialized from pretrained weights: {meta['pretrained']}.\n\n"
            + f"Architecture: `{meta['architecture']}`. SVD threshold: {meta['energy_threshold']}. "
            + f"Checkpoint seed: {meta['seed']}.\n\n"
            + "Weight license: PENDING verification; this local package is not release-ready.\n\n"
            + "## Results\n\n" + table + "\n\n"
            + "Results aggregate the seeds in the CSVs; the checkpoint uses the seed above.\n\n"
            + "## Local loading\n\n```python\nfrom vitlr_loader import load\n"
            + "model, config = load(LOCAL_MODEL_DIRECTORY)\n```\n\n"
            + f"Source: {github}\n\nAuthor: Luis Felipe Garcia de Souza Paim, UFU.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--variant", choices=["compressed", "baseline", "control"], default="compressed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--github", default="PENDING")
    ap.add_argument("--export-dir", default="hf_export")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Read-only plan; creates, deletes and uploads nothing")
    ap.add_argument("--local-only", action="store_true", help="Build a new local directory without uploading")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.export_dir)
    # This branch precedes all model loading, dataset access and filesystem writes.
    if a.dry_run:
        print(json.dumps({"action": "read-only plan", "destination": str(out), "repo_id": a.repo_id,
                          "dataset": cfg["data"], "model": cfg["models"]["control" if a.variant == "control" else "baseline"],
                          "seed": a.seed, "destination_exists": out.exists(),
                          "eligible_dataset": cfg["data"]["name"] == "food101"}, indent=2))
        return
    if cfg["data"]["name"] != "food101":
        raise ValueError("Synthetic/unsupported datasets are test-only; export and publication refused.")
    if not a.local_only:
        raise ValueError("Publication is suspended pending experiments and weight-license verification. Use --local-only.")
    if out.exists():
        raise FileExistsError(f"Destination already exists; nothing overwritten: {out}")
    res = Path(cfg["experiment"]["results_dir"])
    thr = json.loads((res / "selection.json").read_text(encoding="utf-8"))["threshold"] if a.variant == "compressed" else None
    method = {"compressed": "svd"}.get(a.variant, a.variant)
    model, ck = load_checkpoint(ckpt_path(cfg, method, a.seed, thr))
    meta = export_metadata(cfg, ck, data_config(model), a.variant, a.seed, thr)
    labels = ck.get("label_names")
    if not labels or len(labels) != meta["num_classes"]:
        raise ValueError("Dataset labels do not match the checkpoint classes.")
    meta["label_names"] = labels
    table = fmt_table(pd.read_csv(res / "summary_test.csv"))
    card = model_card(meta, table, a.repo_id, a.github)
    # Exclusive creation: even an empty existing destination is never overwritten.
    out.mkdir(parents=True, exist_ok=False)
    (out / "results").mkdir()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "vitlr_loader.py").write_text(LOADER, encoding="utf-8")
    (out / "README.md").write_text(card, encoding="utf-8")
    for f in list(res.glob("*.csv")) + list(res.glob("*.json")) + list((res / "figures").glob("*.png")):
        shutil.copy(f, out / "results" / f.name)
    print(f"Local package created at {out.resolve()}; license review and release checks remain pending.")


if __name__ == "__main__":
    main()
