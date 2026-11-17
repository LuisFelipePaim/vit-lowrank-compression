"""Estimate the three-seed matrix from measured feasibility CSVs.

Scenarios are planning assumptions, not confidence intervals or measured model
speedups. No accuracy values are used to choose a seed or threshold.
"""
import csv
import json
from pathlib import Path

from vitlr.utils import load_config


def first(path):
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def main():
    cfg = load_config("configs/full.yaml")
    out = Path("results_feasibility")
    training = first(out / "feasibility.csv")
    validation = first(out / "validation.csv")
    probes = first(out / "inference_probes.csv")
    t = float(training["seconds"])
    v = float(validation["validation_images_per_second"])
    seeds, thresholds = len(cfg["experiment"]["seeds"]), len(cfg["compress"]["thresholds"])
    baseline_epochs = seeds * cfg["train"]["epochs"]
    control_epochs = baseline_epochs
    ft_epochs = seeds * thresholds * cfg["finetune"]["epochs"]
    val_passes = seeds * (2 + 2 * thresholds)
    gpu_test_passes, cpu_test_passes = seeds * 5, seeds * 2
    gpu_profiles = val_passes + gpu_test_passes
    cpu_profiles = val_passes + seeds * 5
    inference_seconds = (val_passes * 7575 + gpu_test_passes * 25250) / v
    bench = cfg["bench"]
    gpu_bench = gpu_profiles * (bench["gpu"]["warmup"] + bench["gpu"]["reps"]) * (
        float(probes["gpu_fp32_bs1_ms"]) + float(probes["gpu_fp32_bs32_ms"])) / 1000
    cpu_seconds = cpu_test_passes * 25250 * float(probes["cpu_int8_bs32_ms"]) / 32000
    cpu_seconds += cpu_profiles * (bench["cpu"]["warmup"] + bench["cpu"]["reps"]) * float(probes["cpu_int8_bs1_ms"]) / 1000
    svd_seconds = seeds * thresholds * float(probes["cpu_svd_seconds"])
    rows = []
    for name, ft_factor, tiny_factor, inference_factor in (
        ("optimistic_assumption", 0.5, 0.25, 0.5),
        ("conservative_equal_training_cost", 1.0, 1.0, 1.0),
        ("stress_assumption", 1.5, 1.0, 1.5),
    ):
        train_seconds = (baseline_epochs + ft_epochs * ft_factor + control_epochs * tiny_factor) * t
        gpu_seconds = train_seconds + inference_factor * (inference_seconds + gpu_bench)
        rows.append({"scenario": name, "baseline_epochs": baseline_epochs, "svd_ft_epochs": ft_epochs,
                     **{k: training[k] for k in ("python_version", "torch_version", "cuda_version", "driver_version")},
                     "control_epochs": control_epochs, "validation_passes": val_passes,
                     "gpu_test_passes": gpu_test_passes, "cpu_int8_test_passes": cpu_test_passes,
                     "gpu_hours": gpu_seconds / 3600,
                     "cpu_int8_and_bench_hours_proxy": cpu_seconds / 3600,
                     "cpu_svd_hours": svd_seconds / 3600,
                     "serial_hours_excluding_unmeasured_overheads": (gpu_seconds + cpu_seconds + svd_seconds) / 3600,
                     "serial_hours_with_20pct_planning_margin": 1.2 * (gpu_seconds + cpu_seconds + svd_seconds) / 3600})
    with (out / "budget_estimate.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print("Unmeasured: full control/SVD epoch cost, fp16 inference speed, dataset decoding for CPU INT8, "
          "checkpoint IO, bootstrap, plotting, thermal drift and competing applications. "
          "CPU batch-1 timing uses INT8 as a proxy for mixed CPU benchmarks. "
          "These scenarios are assumptions, not bounds or confidence intervals.")


if __name__ == "__main__":
    main()
