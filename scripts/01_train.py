"""Fine-tuning do baseline (DeiT-S) ou do controle (DeiT-Ti) no treino; avalia na VALIDACAO.

    python scripts/01_train.py --config configs/full.yaml --role baseline
    python scripts/01_train.py --config configs/full.yaml --role control
"""
import argparse

from vitlr.data import DataBundle
from vitlr.model import create_model, data_config, save_checkpoint
from vitlr.pipeline import ckpt_path, done_ids, eval_and_log, run_id
from vitlr.train import train_model
from vitlr.utils import get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role", choices=["baseline", "control"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    cfg, device = load_config(a.config), get_device()
    mcfg, nc = cfg["models"][a.role], cfg["data"]["num_classes"]
    dcfg = data_config(create_model(mcfg, nc, pretrained=False))
    bundle = DataBundle(cfg, dcfg)
    print(f"device={device} | treino={len(bundle.train_idx)} | validacao={len(bundle.val_idx)} | modelo={mcfg['name']}")

    for seed in a.seeds or cfg["experiment"]["seeds"]:
        ck = ckpt_path(cfg, a.role, seed)
        if ck.exists() and run_id(a.role, seed, "val") in done_ids(cfg):
            print(f"[skip] {a.role} seed {seed} ja concluido")
            continue
        set_seed(seed)
        model = create_model(mcfg, nc)
        t = cfg["train"]
        info = train_model(model, bundle.loader("train", t["batch_size"], train=True, seed=seed), t, device,
                           desc=f"{a.role} s{seed}")
        save_checkpoint(ck, model, mcfg, nc, seed=seed, experiment_config=cfg,
                        label_names=bundle.class_names)
        eval_and_log(cfg, bundle, model, split="val", seed=seed, method=a.role, model_name=mcfg["name"],
                     device=device, input_size=dcfg["input_size"],
                     train_info={"train_epochs": t["epochs"], "train_examples": len(bundle.train_idx),
                                 "train_seconds": round(info["train_seconds"], 1)})


if __name__ == "__main__":
    main()
