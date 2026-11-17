"""Varredura de limiares de energia: para cada seed e limiar, comprime o baseline
por SVD truncado, avalia SEM fine-tuning, faz fine-tuning curto e avalia de novo.
Tudo na VALIDACAO.

    python scripts/02_sweep.py --config configs/full.yaml
"""
import argparse
import copy

from vitlr.compress import compress_model, ranks_of
from vitlr.data import DataBundle
from vitlr.model import data_config, load_checkpoint, save_checkpoint
from vitlr.pipeline import ckpt_path, done_ids, eval_and_log, run_id
from vitlr.train import train_model
from vitlr.utils import get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--thresholds", type=float, nargs="*")
    a = ap.parse_args()
    cfg, device = load_config(a.config), get_device()
    mcfg, nc, cc = cfg["models"]["baseline"], cfg["data"]["num_classes"], cfg["compress"]
    ft = {**cfg["train"], **cfg["finetune"]}
    bundle = None

    for seed in a.seeds or cfg["experiment"]["seeds"]:
        base, _ = load_checkpoint(ckpt_path(cfg, "baseline", seed), device)
        dcfg = data_config(base)
        bundle = bundle or DataBundle(cfg, dcfg)
        common = dict(split="val", seed=seed, method="svd", model_name=mcfg["name"], device=device,
                      input_size=dcfg["input_size"])
        for thr in a.thresholds or cc["thresholds"]:
            done, ck = done_ids(cfg), ckpt_path(cfg, "svd", seed, thr)
            id0, id1 = (run_id("svd", seed, "val", "fp32", thr, f) for f in (0, 1))
            if id0 in done and id1 in done and ck.exists():
                print(f"[skip] seed {seed} limiar {thr}")
                continue
            model = copy.deepcopy(base)
            info = compress_model(model, thr, cc["targets"], cc["skip_if_not_smaller"])
            if id0 not in done:
                eval_and_log(cfg, bundle, model, threshold=thr, finetuned=0, comp_info=info, **common)
            set_seed(seed * 1000 + int(round(thr * 100)))
            loader = bundle.loader("train", ft["batch_size"], train=True, seed=seed, per_class=ft.get("train_per_class"))
            train_examples = len(loader.dataset)
            tinfo = train_model(model, loader, ft, device, desc=f"ft s{seed} thr{thr}")
            del loader  # Avoid retaining an idle worker pool during validation.
            save_checkpoint(ck, model, mcfg, nc, ranks=ranks_of(info), seed=seed, threshold=thr,
                            experiment_config=cfg, label_names=bundle.class_names)
            eval_and_log(cfg, bundle, model, threshold=thr, finetuned=1, comp_info=info,
                         train_info={"train_epochs": ft["epochs"], "train_examples": train_examples,
                                     "train_seconds": round(tinfo["train_seconds"], 1)}, **common)


if __name__ == "__main__":
    main()
