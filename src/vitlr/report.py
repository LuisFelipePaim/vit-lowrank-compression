"""Agregacao: media +- desvio padrao entre seeds e IC pareado agregado."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import csv_path, preds_path, run_id
from .stats import paired_bootstrap_ci

KEYS = ["method", "model", "threshold", "finetuned", "precision"]
METRICS = ["top1", "top5", "params", "size_mb", "flops_g", "lat_gpu_bs1_ms", "gpu_throughput_img_s",
           "lat_cpu_bs1_ms", "mean_rank", "n_factorized"]


def load_runs(cfg) -> pd.DataFrame:
    return pd.read_csv(csv_path(cfg))


def summarize(cfg, split: str) -> pd.DataFrame:
    """Uma linha por configuracao: media e desvio padrao (ddof=1) entre seeds, mais o
    delta de top-1 contra o baseline com IC 95% por bootstrap pareado, agregando as seeds."""
    df = load_runs(cfg)
    df = df[df["split"] == split]
    rows = []
    for key, g in df.groupby(KEYS, dropna=False, sort=False):
        row = dict(zip(KEYS, key))
        row["n_seeds"] = g["seed"].nunique()
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
        if not (row["method"] == "baseline" and row["precision"] == "fp32"):
            a, b = [], []
            for _, r in g.iterrows():
                pa, pb = preds_path(cfg, r["run_id"]), preds_path(cfg, run_id("baseline", int(r["seed"]), split))
                if pa.exists() and pb.exists():
                    a.append(np.load(pa)), b.append(np.load(pb))
            if a:
                ci = paired_bootstrap_ci(np.stack(a), np.stack(b), cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"])
                row.update(delta_top1_pp=ci["delta"], ci95_lo=ci["lo"], ci95_hi=ci["hi"])
        rows.append(row)
    out = pd.DataFrame(rows)
    path = Path(cfg["experiment"]["results_dir"]) / f"summary_{split}.csv"
    out.to_csv(path, index=False)
    return out


def fmt_table(s: pd.DataFrame) -> str:
    """Tabela markdown compacta (media +- dp)."""
    def pm(r, m, f="{:.2f}"):
        a, b = r.get(f"{m}_mean"), r.get(f"{m}_std")
        if pd.isna(a):
            return "-"
        return f.format(a) + ("" if pd.isna(b) else " ± " + f.format(b))

    lines = ["| metodo | limiar | FT | precisao | top-1 (%) | top-5 (%) | Δ top-1 pp [IC 95%] | params (M) | disco (MB) | GFLOPs | GPU bs1 (ms) | CPU bs1 (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        d = "-" if pd.isna(r.get("delta_top1_pp")) else f"{r['delta_top1_pp']:+.2f} [{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]"
        thr = "-" if pd.isna(r["threshold"]) else f"{r['threshold']:.2f}"
        ft = "-" if pd.isna(r["finetuned"]) else ("sim" if r["finetuned"] else "nao")
        pM = "-" if pd.isna(r["params_mean"]) else f"{r['params_mean'] / 1e6:.2f}"
        lines.append(f"| {r['method']} | {thr} | {ft} | {r['precision']} | {pm(r, 'top1')} | {pm(r, 'top5')} | {d} | {pM} | "
                     f"{pm(r, 'size_mb', '{:.1f}')} | {pm(r, 'flops_g', '{:.3f}')} | {pm(r, 'lat_gpu_bs1_ms')} | {pm(r, 'lat_cpu_bs1_ms')} |")
    return "\n".join(lines)
