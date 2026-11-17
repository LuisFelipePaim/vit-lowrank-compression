import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from vitlr.train import train_model


def test_training_measurements_count_processed_examples():
    # A non-divisible dataset verifies that throughput uses actual processed
    # examples, rather than dataset size when drop_last=True.
    ds = TensorDataset(torch.randn(11, 3), torch.randint(0, 2, (11,)))
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    model = torch.nn.Linear(3, 2)
    info = train_model(model, loader, {"epochs": 1, "lr": 0.001, "weight_decay": 0.0,
                                     "accum_steps": 2}, torch.device("cpu"))
    epoch = info["epoch_metrics"][0]
    assert epoch["examples"] == 8 and epoch["seconds"] > 0
    assert epoch["images_per_second"] > 0 and math.isfinite(info["train_loss"])
    assert info["train_seconds"] >= epoch["seconds"]
    assert info["peak_allocated_mb"] is None
