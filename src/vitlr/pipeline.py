"""Cola: avalia + mede + calcula o IC pareado + grava UMA linha por execucao no CSV."""
from __future__ import annotations

import copy
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .benchmark import disk_size_mb, latency_ms, param_count
from .evaluate import evaluate
from .flops import count_flops
from .stats import paired_bootstrap_ci
from .utils import thr_tag

FIELDS = [
    "run_id", "timestamp", "split", "seed", "method", "model", "threshold", "finetuned", "precision",
    "top1", "top5", "loss", "n_eval",
    "params", "size_mb", "flops_g",
    "lat_gpu_bs1_ms", "lat_gpu_bs1_std", "gpu_throughput_img_s", "gpu_throughput_bs",
    "lat_cpu_bs1_ms", "lat_cpu_bs1_std", "cpu_threads",
    "delta_top1_pp", "ci95_lo", "ci95_hi",
    "n_factorized", "n_targets", "mean_rank", "ranks_json",
    "train_epochs", "train_examples", "train_seconds", "device_name", "torch_version",
    "python_version", "cuda_version", "driver_version",
]


def environment_info() -> dict:
    """Runtime versions, without inferring historical or unavailable values."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
        driver = ";".join(dict.fromkeys(result.stdout.strip().splitlines())) or "unknown"
    except (OSError, subprocess.SubprocessError):
        driver = "unknown"
    return {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda or "none (CPU build)", "driver_version": driver}


def run_id(method, seed, split, precision="fp32", threshold=None, finetuned=None) -> str:
    mid = f"-thr{thr_tag(threshold)}-ft{int(finetuned)}" if method == "svd" else ""
    return f"{method}-s{seed}{mid}-{precision}-{split}"


def ckpt_path(cfg, method, seed, threshold=None) -> Path:
    out = Path(cfg["experiment"]["out_dir"])
    if method == "svd":
        return out / "svd" / f"seed{seed}" / f"thr{thr_tag(threshold)}_ft.pt"
    return out / method / f"seed{seed}.pt"


def csv_path(cfg) -> Path:
    return Path(cfg["experiment"]["results_dir"]) / "runs.csv"


def done_ids(cfg) -> set:
    p = csv_path(cfg)
    if not p.exists():
        return set()
    with p.open(newline="", encoding="utf-8") as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def _append(cfg, row: dict) -> None:
    p = csv_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    if not new:
        with p.open(newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != FIELDS:
                raise ValueError(f"CSV schema differs: {p}. Use a new results directory; preserve legacy CSVs.")
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def preds_path(cfg, rid: str) -> Path:
    return Path(cfg["experiment"]["results_dir"]) / "preds" / f"{rid}.npy"


def profile_model(model, cfg, device, input_size, precision="fp32", ref: dict | None = None) -> dict:
    """Parametros, tamanho, FLOPs e latencia. Para fp16/int8, parametros e FLOPs
    vem de `ref` (o modelo fp32 de origem): quantizar nao muda a contagem de operacoes."""
    b = cfg["bench"]
    out = {"size_mb": disk_size_mb(model), "cpu_threads": b["cpu"].get("threads") or torch.get_num_threads()}
    if precision == "fp32":
        out["params"] = param_count(model)
        out["flops_g"] = count_flops(model, input_size)["total"] / 1e9
    else:
        out["params"], out["flops_g"] = ref["params"], ref["flops_g"]

    if precision in ("fp32", "fp16") and device.type == "cuda":
        dt = torch.float16 if precision == "fp16" else torch.float32
        bss = sorted(b["gpu"]["batch_sizes"])
        r1 = latency_ms(model, device, input_size, 1, b["gpu"]["warmup"], b["gpu"]["reps"], dt)
        rn = latency_ms(model, device, input_size, bss[-1], b["gpu"]["warmup"], b["gpu"]["reps"], dt)
        out.update(lat_gpu_bs1_ms=r1["mean"], lat_gpu_bs1_std=r1["std"],
                   gpu_throughput_img_s=rn["img_per_s"], gpu_throughput_bs=bss[-1])
    if precision in ("fp32", "int8"):
        cpu_model = model if precision == "int8" else copy.deepcopy(model).cpu()
        r = latency_ms(cpu_model, torch.device("cpu"), input_size, b["cpu"]["batch_size"],
                       b["cpu"]["warmup"], b["cpu"]["reps"], threads=b["cpu"].get("threads"))
        out.update(lat_cpu_bs1_ms=r["mean"], lat_cpu_bs1_std=r["std"])
    return out


def eval_and_log(cfg, bundle, model, *, split, seed, method, model_name, device, input_size,
                 precision="fp32", threshold=None, finetuned=None, comp_info=None, ref=None, train_info=None) -> dict:
    """Avalia `model` em `split`, mede custo e anexa uma linha a results/runs.csv."""
    rid = run_id(method, seed, split, precision, threshold, finetuned)
    if precision == "int8":
        dev, dt, bs = torch.device("cpu"), torch.float32, cfg["eval"]["cpu_batch_size"]
    else:
        dev, dt, bs = device, (torch.float16 if precision == "fp16" else torch.float32), cfg["eval"]["batch_size"]
    ev = evaluate(model, bundle.loader(split, bs), dev, dt, desc=rid)
    prof = profile_model(model, cfg, device, input_size, precision, ref)

    pp = preds_path(cfg, rid)
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.save(pp, ev["correct1"])

    row = {"run_id": rid, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": split, "seed": seed,
           "method": method, "model": model_name, "precision": precision,
           "threshold": "" if threshold is None else threshold, "finetuned": "" if finetuned is None else int(finetuned),
           "top1": ev["top1"], "top5": ev["top5"], "loss": ev["loss"], "n_eval": ev["n"], **prof,
           "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
           **(train_info or {}), **environment_info()}

    base = preds_path(cfg, run_id("baseline", seed, split))
    if rid != base.stem and base.exists():
        ci = paired_bootstrap_ci(ev["correct1"], np.load(base), cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
        row.update(delta_top1_pp=ci["delta"], ci95_lo=ci["lo"], ci95_hi=ci["hi"])
    if comp_info is not None:
        ranks = {k: v["rank"] for k, v in comp_info.items()}
        used = [r for r in ranks.values() if r > 0]
        row.update(n_factorized=len(used), n_targets=len(ranks),
                   mean_rank=float(np.mean(used)) if used else 0.0, ranks_json=json.dumps(ranks))
    _append(cfg, row)
    extra = f" | delta {row['delta_top1_pp']:+.2f} pp [{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}]" if "ci95_lo" in row else ""
    print(f"[{rid}] top1 {row['top1']:.2f} top5 {row['top5']:.2f} | params {row['params'] / 1e6:.2f}M "
          f"| {row['flops_g']:.3f} GFLOPs | {row['size_mb']:.1f} MB{extra}", flush=True)
    return row
