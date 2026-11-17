"""Agrega a validacao (media +- dp entre seeds) e escolhe o limiar por uma regra
PRE-DECLARADA (configs: select.max_drop_pp): entre os modelos comprimidos com
fine-tuning, o de menos parametros cuja queda media de top-1 contra o baseline
nao passa de max_drop_pp. Se nenhum satisfaz, o de maior top-1 medio.

    python scripts/03_select.py --config configs/full.yaml
"""
import argparse
import json
from pathlib import Path

from vitlr.report import fmt_table, summarize
from vitlr.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    s = summarize(cfg, "val")
    print(fmt_table(s))
    base = s[(s.method == "baseline") & (s.precision == "fp32")].iloc[0]
    cand = s[(s.method == "svd") & (s.finetuned == 1) & (s.precision == "fp32")].copy()
    cand["drop"] = base.top1_mean - cand.top1_mean
    ok = cand[(cand["drop"] <= cfg["select"]["max_drop_pp"]) & (cand.params_mean < base.params_mean)]
    fallback = ok.empty
    best = (cand.sort_values("top1_mean", ascending=False) if fallback else ok.sort_values("params_mean")).iloc[0]
    sel = {"threshold": float(best.threshold), "fallback_used": bool(fallback),
           "rule": f"menor params com queda media <= {cfg['select']['max_drop_pp']} pp na validacao (com FT)",
           "val_top1_mean": float(best.top1_mean), "val_drop_pp": float(best["drop"]),
           "params_ratio": float(best.params_mean / base.params_mean)}
    out = Path(cfg["experiment"]["results_dir"]) / "selection.json"
    out.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"\nLimiar selecionado: {sel['threshold']} (queda {sel['val_drop_pp']:.2f} pp, "
          f"{100 * sel['params_ratio']:.1f}% dos parametros)" + (" [FALLBACK: nenhum limiar cumpriu a regra]" if fallback else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
