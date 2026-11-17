"""Measure one baseline epoch on full Food-101, without reading official test.

python scripts/00_feasibility.py --config configs/feasibility.yaml
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

# Inputs have been downloaded and verified before this measurement starts.
os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from vitlr.benchmark import latency_ms
from vitlr.compress import compress_model
from vitlr.data import DataBundle
from vitlr.evaluate import evaluate
from vitlr.model import create_model, data_config, save_checkpoint
from vitlr.pipeline import environment_info
from vitlr.quantize import to_int8_dynamic
from vitlr.train import train_model
from vitlr.utils import load_config, set_seed


def monitor_gpu(path, stop, env):
    """Whole-device memory includes display/context/other applications."""
    with path.open("x", encoding="utf-8", newline="") as f:
        fields = ["memory_used_mib", "utilization_percent", "temperature_c", "power_w"]
        writer = csv.DictWriter(f, fieldnames=["timestamp", *fields, *env])
        writer.writeheader()
        while not stop.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                values = [v.strip() for v in result.stdout.strip().split(",")]
                writer.writerow({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 **dict(zip(fields, values)), **env})
                f.flush()
            stop.wait(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feasibility.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for feasibility; refusing a CPU fallback.")
    if cfg["data"]["name"] != "food101" or cfg["data"].get("train_per_class") is not None:
        raise ValueError("Feasibility requires the full Food-101 training pool.")
    if cfg["train"]["epochs"] != 1 or cfg["experiment"]["seeds"] != [42]:
        raise ValueError("Feasibility must run exactly one epoch with seed 42.")
    out = Path(cfg["experiment"]["results_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "effective_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    env = environment_info()
    device = torch.device("cuda")
    set_seed(42)
    torch.set_num_threads(4)
    started = time.perf_counter()
    mcfg = cfg["models"]["baseline"]
    model = create_model(mcfg, cfg["data"]["num_classes"])
    dc = data_config(model)
    bundle = DataBundle(cfg, dc)
    if len(bundle.train_idx) != 68175 or len(bundle.val_idx) != 7575:
        raise ValueError("Unexpected full-dataset split sizes.")
    print(f"CUDA={env} GPU={torch.cuda.get_device_name(0)} train={len(bundle.train_idx)} "
          f"val={len(bundle.val_idx)} input={dc['input_size']}", flush=True)
    loader = bundle.loader("train", cfg["train"]["batch_size"], train=True, seed=42)
    setup_seconds = time.perf_counter() - started
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(out / "gpu_telemetry.csv", stop, env), daemon=True)
    monitor.start()
    try:
        info = train_model(model, loader, cfg["train"], device, desc="feasibility baseline s42")
    finally:
        stop.set()
        monitor.join(timeout=15)
    del loader  # Release persistent training workers before validation workers start.
    epoch = info.pop("epoch_metrics")[0]
    row = {"purpose": "feasibility_only", "seed": 42, "model": mcfg["name"],
           "base_model_revision": json.loads(Path("audit/base_model_manifest.json").read_text(encoding="utf-8"))["revision"],
           "dataset_revision": cfg["data"].get("archive_md5", cfg["data"].get("hf_revision", "unknown")),
           "dataset": cfg["data"]["hf_id"], "train_pool": len(bundle.train_idx),
           "validation_pool": len(bundle.val_idx), "batch_size": cfg["train"]["batch_size"],
           "accum_steps": cfg["train"]["accum_steps"], "num_workers": cfg["data"]["num_workers"],
           "cpu_threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0),
           "setup_seconds": setup_seconds, **env, **epoch, **info}
    # Persist the training measurement before optional validation/checkpoint IO.
    with (out / "feasibility.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    save_checkpoint(Path(cfg["experiment"]["out_dir"]) / "baseline_seed42_epoch1.pt",
                    model, mcfg, cfg["data"]["num_classes"], seed=42,
                    experiment_config=cfg, label_names=bundle.class_names, purpose="feasibility_only")
    torch.cuda.synchronize()
    start = time.perf_counter()
    ev = evaluate(model, bundle.loader("val", cfg["eval"]["batch_size"]), device,
                  desc="feasibility validation (no test)")
    torch.cuda.synchronize()
    val_seconds = time.perf_counter() - start
    row.update(validation_seconds=val_seconds, validation_images_per_second=ev["n"] / val_seconds,
               validation_top1=ev["top1"], validation_top5=ev["top5"])
    with (out / "validation.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2), flush=True)
    # Short inference probes support a budget estimate; no test examples are used.
    probes = {"purpose": "feasibility_only", **env}
    for bs in (1, 32):
        r = latency_ms(model, device, dc["input_size"], bs, warmup=5, reps=10)
        probes[f"gpu_fp32_bs{bs}_ms"] = r["mean"]
    # CPU SVD timing only; no candidate accuracy or rank selection is performed.
    dense_cpu = copy.deepcopy(model).cpu()
    start = time.perf_counter()
    compress_model(dense_cpu, cfg["compress"]["thresholds"][0], cfg["compress"]["targets"],
                   cfg["compress"]["skip_if_not_smaller"])
    probes["cpu_svd_seconds"] = time.perf_counter() - start
    del dense_cpu
    quant = to_int8_dynamic(model)
    for bs in (1, 32):
        r = latency_ms(quant, torch.device("cpu"), dc["input_size"], bs,
                       warmup=1, reps=3, threads=4)
        probes[f"cpu_int8_bs{bs}_ms"] = r["mean"]
    with (out / "inference_probes.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(probes))
        writer.writeheader()
        writer.writerow(probes)
    print(json.dumps(probes, indent=2), flush=True)


if __name__ == "__main__":
    main()
