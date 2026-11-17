# Release preparation — 2026-11-17

Historical snapshot before CUDA feasibility. Current measured status is in
[FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md).

Release is suspended until Food-101 experiments and weight-license review finish.
No remote repository, upload or commit is authorized in this preparation step.

## Decisions

- Code license selected by the author: Apache-2.0. License file and external-license
  review remain part of the licensing stage; no weight license is assumed.
- Author: Luis Felipe Garcia de Souza Paim, UFU.
- Release checkpoint seed: 42, chosen before viewing Food-101 results.
- Full experiment seeds: [42, 0, 1], retaining three seeds in total.
- Release checkpoints: baseline and SVD after fine-tuning, threshold chosen on validation.
- Repository names retain user/HF namespace placeholders until supplied.

## Implemented before feasibility experiments

- Export dry-run only reads configuration and prints a plan. It does not load
  checkpoints, fetch data, create directories, overwrite files, or upload.
- Local export requires `--local-only` and an absent destination. Existing
  destinations are refused, including empty directories. No recursive deletion.
- Remote publication is disabled while release and licensing are pending.
- Synthetic and unsupported datasets cannot be exported or published.
- New checkpoints preserve the effective merged experiment config. Export checks
  it against the requested dataset/model, checkpoint classes, seed and threshold.
  Legacy checkpoints without provenance are refused.
- Cards derive dataset, class count, input resolution, architecture and pretrained
  initialization from verified metadata. Weight licensing is explicitly pending.
  Complete release cards and tested inference examples remain a later stage.
- CSV rows now include Python, PyTorch, CUDA build version and NVIDIA driver.
  Unavailable driver values are `unknown`; CPU builds are explicit. Existing
  CSVs are preserved; appending a different schema is refused.
- Ignore rules cover environments, datasets, caches, checkpoints, generated
  runs/results/exports, notebook checkpoints, credentials and OS files.

## Environment and validation

Validated existing CPU environment: Python 3.12.7, PyTorch 2.15.0+cpu,
torchvision 0.29.2+cpu, no CUDA runtime in this PyTorch build.
NVIDIA driver observed: 591.59; GPU: RTX 3050 6GB Laptop GPU.
`requirements.txt` pins this CPU environment, including transitive packages.
A GPU environment must be selected, pinned and validated before feasibility.
This is not a clean-install validation; that remains in the reproducibility stage.

Validation command: `audit/.venv/Scripts/python.exe -m pytest -q`.
Result: 27 passed in 30.03 seconds (21 existing + 6 release regression cases).
`python -m pip check`: No broken requirements found.
Regression coverage: read-only dry-run with absent/present destinations,
synthetic refusal, existing destination preservation, config provenance, card
metadata, CSV environment persistence and legacy schema protection.

## Scan and preserved artifacts

Before Git initialization, a pattern scan of 52 project text files found no
secret candidates, emails or machine-specific absolute paths. It covered source,
configs, docs, CSVs, JSON and the existing export loader/card. It excluded the
dependency virtual environment, bytecode/cache directories and binary payloads.
Patterns covered common HF/GitHub/AWS/OpenAI token forms, private-key headers,
literal credential assignments, emails and user/home absolute paths. This is
pattern-based screening, not a guarantee that no credential can exist.

Machine-specific paths exist in ignored virtual-environment metadata, including
`audit/.venv/pyvenv.cfg` and its editable-install `.pth` file. These are not
release source files. No original Git history exists to inspect.

No files were deleted or moved. `audit/export/` is an ignored, preserved,
**invalid legacy synthetic export**: its Food-101 claims and license metadata
must not be used. Smoke results/checkpoints are preserved and ignored.
No non-environment file above 10 MiB was identified in the initial size check.

## Next gate

Review this change before any experiment. Then validate a CUDA environment,
run one baseline epoch on full Food-101 training data, measure throughput,
peak VRAM and epoch time, estimate the complete three-seed matrix and request
approval before proceeding with that matrix. Do not touch official test data
for feasibility or use feasibility results to select the publication seed.
