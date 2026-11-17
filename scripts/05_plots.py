"""Graficos de Pareto a partir dos CSVs (validacao; media entre seeds, barras = dp).

    python scripts/05_plots.py --config configs/full.yaml
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vitlr.report import summarize
from vitlr.utils import load_config

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # cor segue a entidade; marcador diferente = identidade nunca depende so da cor
    "svd_ft": ("#2a78d6", "o", "SVD + fine-tuning"),
    "svd_noft": ("#eb6834", "s", "SVD sem fine-tuning"),
    "control": ("#1baf7a", "^", "Controle (DeiT-Ti)"),
    "baseline": (INK, "D", "Baseline (DeiT-S)"),
}


def pareto(points):
    """Pontos nao dominados (menor x, maior y)."""
    best, out = -1e9, []
    for x, y in sorted(points):
        if y > best:
            out.append((x, y))
            best = y
    return out


def plot(s, xcol, xlabel, scale, path, title):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    groups = {"svd_ft": s[(s.method == "svd") & (s.finetuned == 1)].sort_values("threshold"),
              "svd_noft": s[(s.method == "svd") & (s.finetuned == 0)].sort_values("threshold"),
              "control": s[s.method == "control"], "baseline": s[s.method == "baseline"]}
    pts = []
    for key, g in groups.items():
        g = g.dropna(subset=[f"{xcol}_mean"])
        if g.empty:
            continue
        color, marker, label = SERIES[key]
        x, y = g[f"{xcol}_mean"] * scale, g["top1_mean"]
        ax.errorbar(x, y, yerr=g["top1_std"].fillna(0), color=color, marker=marker, ms=7, lw=2 if key.startswith("svd") else 0,
                    elinewidth=1, capsize=2, mec=SURFACE, mew=1.5, label=label, zorder=3)
        pts += list(zip(x, y))
        if key == "svd_ft":
            for xi, yi, t in zip(x, y, g["threshold"]):
                ax.annotate(f"{t:g}", (xi, yi), textcoords="offset points", xytext=(7, -11), ha="left", fontsize=8, color=INK2)
    if pts:
        fx, fy = zip(*pareto(pts))
        ax.step(fx, fy, where="post", color=INK2, lw=1, ls=(0, (4, 3)), zorder=2, label="Fronteira de Pareto")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("Top-1 na validação (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.01, 0.01, "Rótulos = limiar de energia. Barras = desvio padrão entre seeds.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    res = Path(cfg["experiment"]["results_dir"])
    p = res / "summary_val.csv"
    s = pd.read_csv(p) if p.exists() else summarize(cfg, "val")
    s = s[s.precision == "fp32"]
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plot(s, "params", "Parâmetros (milhões)", 1e-6, figs / "pareto_params.png", "Acurácia × parâmetros")
    plot(s, "flops_g", "GFLOPs por imagem", 1, figs / "pareto_flops.png", "Acurácia × FLOPs")
    plot(s, "lat_cpu_bs1_ms", "Latência em CPU, batch 1 (ms)", 1, figs / "pareto_latency_cpu.png", "Acurácia × latência (CPU)")
    if s["lat_gpu_bs1_ms_mean"].notna().any():
        plot(s, "lat_gpu_bs1_ms", "Latência em GPU, batch 1 (ms)", 1, figs / "pareto_latency_gpu.png", "Acurácia × latência (GPU)")


if __name__ == "__main__":
    main()
