# Food-101 feasibility - 2026-11-17

Release and the full experiment matrix remain suspended pending approval.
No commit, remote creation, push or Hub upload was performed.

## Measured run

One baseline epoch, seed 42, original official Food-101 archive. The archive MD5
matches torchvision 0.20.1's expected value; the extracted training metadata has
101 classes, 750 examples per class, and no missing training image files.
The original dataset was read in place. No images were copied into release packages.
Class IDs follow sorted class names; image order follows official `meta/train.json`.
The fixed validation split uses seed 1234 and 75 images per class.
The official test was not evaluated.

| Measurement | Observed value |
|---|---:|
| Training pool | 68175 |
| Validation pool | 7575 |
| Images actually processed | 68160 |
| Epoch seconds, including worker startup | 531.917131 |
| Training images/s | 128.140261 |
| Peak PyTorch allocated GiB | 1.453593 |
| Peak PyTorch reserved GiB | 1.535156 |
| Sampled whole-device peak MiB | 2981 |
| Validation seconds, including loader startup | 185.876262 |
| Validation images/s | 40.752918 |

The final partial training batch was dropped (`drop_last=True`): 15 of the selected
68,175 images were not processed in this epoch. This run uses a one-epoch schedule
and must not be resumed as the five-epoch final baseline. Its checkpoint is marked
`feasibility_only` and the exporter refuses it.

The validation CSV includes one-epoch accuracy for provenance. It is not the final
study result and was not used to select seeds, thresholds or checkpoints.

## Environment and checks

Python 3.12.7; PyTorch 2.5.1+cu121; CUDA runtime
12.1; NVIDIA driver 591.59. Hardware details:
[`cuda_environment.json`](cuda_environment.json).
An isolated environment was installed from `requirements-cuda.txt`; `pip check`
reported no broken requirements. The full unit suite passed: 30 tests in 70.22 s.
The additional feasibility-export guard passed its targeted regression test.
The CUDA synthetic pipeline completed training, SVD, selection, fp16, INT8 and plots.

The desktop remained in use and RAM pressure was high. The validation measurement
retained idle training workers while creating validation workers. After the run,
explicit worker cleanup was added before validation in the feasibility and sweep
scripts. The budget retains the measured validation rate and assumes no improvement
from that cleanup. The revised sweep passed a fresh CUDA end-to-end smoke run (baseline, control,
SVD, selection, fp16/INT8 final synthetic evaluation and all four plots), using
`audit/feasibility/cleanup_smoke.yaml`.

## Budget for the full matrix

Three seeds [42, 0, 1]; seven thresholds; 5 baseline epochs and 5 control epochs
per seed; 1 fine-tuning epoch per threshold/seed. Total: 15 baseline epochs,
21 SVD fine-tuning epochs and 15 control epochs (51 training epochs).
Validation has 48 passes. Official testing is planned for 15 GPU passes and
6 CPU INT8 passes. No official-test predictions were generated for this estimate.

| Scenario | GPU-stage hours | Serial hours | Serial hours + 20% planning margin |
|---|---:|---:|---:|
| optimistic_assumption | 7.101394 | 8.811050 | 10.573260 |
| conservative_equal_training_cost | 13.094627 | 14.804284 | 17.765140 |
| stress_assumption | 17.425619 | 19.135276 | 22.962331 |

These are planning scenarios, not confidence intervals or guaranteed bounds.
The conservative scenario prices control and SVD training epochs at one baseline
epoch each. Stress prices SVD fine-tuning at 1.5 baseline epochs, control at one,
and GPU inference/benchmark work at 1.5 times the baseline-based estimate.
GPU-stage hours include data loading/worker startup; they are not pure CUDA-kernel time.
CPU INT8 evaluation uses a short batch-32 synthetic-input timing probe. CPU batch-1
benchmark cost uses the INT8 baseline as a proxy for the mixed model benchmarks.
SVD cost comes from one CPU decomposition timing probe, without candidate evaluation.

Unmeasured overheads include complete control/SVD epoch costs, fp16 speed,
CPU image decoding, bootstrap, checkpoint IO, plots and workload/thermal drift.
The 20% allowance is an explicit planning assumption, not an observed result.
Both conservative and stress GPU estimates are below 48 hours, so no experimental
cuts are proposed. The author must approve the matrix before it starts.

## Evidence and reproduction

Raw measurements and the effective run config are preserved under
[`feasibility/`](feasibility/). The local checkpoint stays ignored under
`runs_feasibility/`; no weights or dataset images are included in this report.
The scripts refuse existing output directories/files instead of overwriting evidence.

```powershell
$env:PYTHONUTF8 = "1"
# Set FOOD101_ROOT to the parent of the extracted food-101 directory.
python scripts/00_download_base_model.py
python scripts/00_feasibility.py --config configs/feasibility.yaml
python scripts/00_estimate_budget.py
```

Re-running requires new output paths/configs because this run's evidence is preserved.
The interrupted HF parquet downloads are ignored and preserved; none were deleted.
No other project's trained checkpoint was loaded.

Sources for the input checks:
[torchvision Food101 source and archive checksum](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/food101.html),
[official PyTorch version combinations](https://pytorch.org/get-started/previous-versions/).
Weight and dataset license review remains pending in the licensing stage.
