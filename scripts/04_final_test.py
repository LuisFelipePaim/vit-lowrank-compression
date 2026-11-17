"""Avaliacao final no TESTE oficial - roda UMA vez. Para cada seed:
baseline e modelo comprimido selecionado em fp32, fp16 (GPU) e INT8 dinamico (CPU),
mais o controle (DeiT-Ti) em fp32. Depois grava um lock; rodar de novo exige --force.

    python scripts/04_final_test.py --config configs/full.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

from vitlr.data import DataBundle
from vitlr.model import data_config, load_checkpoint
from vitlr.pipeline import ckpt_path, done_ids, eval_and_log, run_id
from vitlr.quantize import to_fp16, to_int8_dynamic
from vitlr.report import fmt_table, summarize
from vitlr.utils import get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="ignora o lock (invalida o protocolo de uso unico)")
    a = ap.parse_args()
    cfg, device = load_config(a.config), get_device()
    res = Path(cfg["experiment"]["results_dir"])
    lock = res / "TEST_LOCK.json"
    if lock.exists() and not a.force:
        sys.exit(f"O teste ja foi usado ({lock}). Nao reavalie apos ver o resultado.")
    thr = json.loads((res / "selection.json").read_text(encoding="utf-8"))["threshold"]
    bundle = None

    for seed in cfg["experiment"]["seeds"]:
        variants = [("baseline", None), ("svd", thr)]
        if ckpt_path(cfg, "control", seed).exists():
            variants.append(("control", None))
        else:
            print(f"[aviso] sem checkpoint de controle para a seed {seed}")
        for method, t in variants:
            model, ck = load_checkpoint(ckpt_path(cfg, method, seed, t), device)
            dcfg = data_config(model)
            bundle = bundle or DataBundle(cfg, dcfg)
            kw = dict(split="test", seed=seed, method=method, model_name=ck["model_cfg"]["name"], device=device,
                      input_size=dcfg["input_size"], threshold=t, finetuned=1 if method == "svd" else None,
                      comp_info={k: {"rank": r} for k, r in ck["ranks"].items()} if ck.get("ranks") else None)
            precisions = ["fp32"] if method == "control" else ["fp32", "fp16", "int8"]
            ref = None
            for prec in precisions:
                if run_id(method, seed, "test", prec, t, kw["finetuned"]) in done_ids(cfg):
                    continue
                if prec == "fp16" and device.type != "cuda":
                    print("[aviso] fp16 exige GPU; pulando")
                    continue
                if prec == "fp32":
                    ref = eval_and_log(cfg, bundle, model, **kw)
                else:
                    if ref is None:  # retomada: fp32 ja estava no CSV
                        from vitlr.benchmark import param_count
                        from vitlr.flops import count_flops
                        ref = {"params": param_count(model), "flops_g": count_flops(model, dcfg["input_size"])["total"] / 1e9}
                    q = to_fp16(model) if prec == "fp16" else to_int8_dynamic(model)
                    eval_and_log(cfg, bundle, q, precision=prec, ref=ref, **kw)

    print(fmt_table(summarize(cfg, "test")))
    lock.write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "threshold": thr}), encoding="utf-8")


if __name__ == "__main__":
    main()
