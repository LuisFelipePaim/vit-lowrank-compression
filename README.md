# Compressão de Vision Transformers por baixo posto × quantização (Food-101)

**Pergunta.** Em um ViT, quanto de redução de parâmetros, FLOPs e latência a decomposição de
baixo posto (SVD truncado nas camadas lineares) entrega, a que custo de acurácia — e como ela se
compara e se combina com quantização (fp16, INT8 dinâmica) e com simplesmente escolher uma
arquitetura menor?

> **Release suspended:** the full Food-101 matrix and final release checkpoints are pending.
> A one-epoch feasibility run is complete; see [the measured report](audit/FEASIBILITY_REPORT.md).
> Legacy `results_smoke/` and `audit/export/` artifacts are synthetic tests.
> Author: Luis Felipe Garcia de Souza Paim, UFU. Code license selected: Apache-2.0;
> weight licensing remains pending. Publication seed: **42**, chosen before results.

## Método

| Peça | Escolha |
|---|---|
| Baseline | `deit_small_patch16_224.fb_in1k` (timm, 22 M parâmetros), fine-tuning no Food-101 |
| Compressão | cada `Linear(d_in, d_out)` de `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` (48 camadas) vira `Linear(d_in, r) → Linear(r, d_out)` por SVD truncado |
| Rank | menor `r` com `Σ_{i≤r} σ_i² / Σ σ_i² ≥ limiar`; limiares {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} |
| Fine-tuning curto | 1 época após a decomposição, mesma receita do baseline |
| Quantização | fp16 (GPU) e INT8 dinâmica das `nn.Linear` (CPU), no baseline e no ponto comprimido selecionado |
| Controle | `deit_tiny_patch16_224.fb_in1k` (5,7 M) com o mesmo fine-tuning |

DeiT-S e DeiT-Ti têm o mesmo pré-treino (ImageNet-1k, mesma receita), então o controle isola o
efeito "arquitetura menor" do efeito "pré-treino diferente".

### Decisões que afetam a leitura dos resultados

- **Validação.** O Food-101 oficial só tem treino (75.750) e teste (25.250). A validação
  (75 imagens/classe = 7.575) é separada do treino oficial, estratificada, com seed fixa
  (`data.split_seed`) — o split não muda entre seeds de treino. O treino efetivo tem 68.175 imagens.
- **Camadas que não compensam ficam densas.** Fatorar só reduz parâmetros se
  `r·(d_in+d_out) < d_in·d_out` (ex.: `r < 192` para 384→384). Acima disso a camada é mantida
  densa e registrada com rank 0 em `ranks_json`. Em limiares altos, espere poucas camadas fatoradas.
- **O que as seeds variam.** Inicialização da cabeça, ordem dos dados e augmentations do
  fine-tuning. Cada seed tem seu baseline; a compressão da seed *s* parte do baseline da seed *s*.
- **Sem seleção de checkpoint.** Usa-se sempre a última época; hiperparâmetros fixados a priori.
- **"Treinado direto"** no controle significa fine-tuning do DeiT-Ti pré-treinado, não treino do zero.
- **FLOPs** = multiply-accumulates (convenção fvcore) de Linear, Conv e dos dois produtos da
  atenção, contados por hooks no forward real. O contador reproduz a fórmula analítica do DeiT-S
  (4,60 G) em `tests/test_metrics.py`.
- **Latência** depende da máquina: laptop na tomada, nada mais usando a GPU. FLOPs menores não
  garantem latência menor — em batch 1 na GPU, duas matmuls pequenas podem custar o mesmo que uma grande.

## Protocolo experimental

1. 3 seeds; tudo é reportado como média ± desvio padrão (ddof=1).
2. A varredura de limiares é avaliada **só na validação**.
3. O limiar é escolhido por regra pré-declarada (`select.max_drop_pp`): o modelo com menos
   parâmetros cuja queda média de top-1 (com fine-tuning) não passa de 1 pp. Se nenhum cumpre,
   o de maior top-1 — e isso fica marcado em `selection.json`.
4. O teste oficial é avaliado **uma vez** por `04_final_test.py`, que grava `TEST_LOCK.json` e se
   recusa a rodar de novo sem `--force`.
5. Δ top-1 contra o baseline com IC 95% por **bootstrap pareado** (reamostra os mesmos exemplos
   para os dois modelos; 10.000 réplicas), por seed e agregado entre seeds.
6. `results/runs.csv`: uma linha por execução, com seed, limiar, ranks por camada (JSON) e todas
   as métricas. Acertos por exemplo ficam em `results/preds/`.

## Como rodar (Windows, RTX 3050 6 GB)

```powershell
$env:PYTHONUTF8 = "1"  # Unicode tables on Windows, including redirected output
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
pip install -e . --no-deps --no-build-isolation
pytest
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The isolated CUDA environment uses Python 3.12.7, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121 and NVIDIA driver 591.59. Unit tests and the synthetic
CUDA pipeline passed; the full-pool Food-101 feasibility measurement is complete.
See `audit/FEASIBILITY_REPORT.md` for measurements and the matrix budget.
`requirements.txt` retains the separately validated CPU environment. Do not
install both requirement files into the same environment.

Feasibility only (one baseline epoch, full training pool after the fixed validation split):

```powershell
python scripts/00_download_base_model.py
$env:FOOD101_ROOT = "<local-parent-of-food-101>"
python scripts/00_feasibility.py --config configs/feasibility.yaml
```

These commands do not evaluate the official test split or run the full matrix.
The default source is the original Food-101 archive, accessed through torchvision.
Set `FOOD101_ROOT` to the parent of its extracted `food-101` directory; no machine path
is stored in configs. The archive MD5 is recorded in `configs/feasibility.yaml`.
Classes are sorted by name, and examples follow the official `meta/train.json` order.
`audit/original_food101_validation.json` records the local integrity/count checks.
The optional HF parquet downloader and SHA-256 manifest remain available for
`data.source: hf` with `data.train_files: data/food101_train/train-*.parquet`;
no images or weights are committed.
The full matrix requires the author's approval after reviewing the measured budget.

Smoke test (dados sintéticos, ~1 min, sem download) — rode antes de tudo:

```powershell
$C = "configs/smoke.yaml"
python scripts/01_train.py --config $C --role baseline
python scripts/01_train.py --config $C --role control
python scripts/02_sweep.py --config $C
python scripts/03_select.py --config $C
python scripts/04_final_test.py --config $C
python scripts/05_plots.py --config $C
```

Depois repita a mesma sequência com `configs/quick.yaml` (subconjunto estratificado declarado,
150 imagens/classe — valida GPU, AMP e fp16 em cerca de uma hora) e, por fim, com
`configs/full.yaml`. Todos os scripts são retomáveis: execuções já presentes no CSV são puladas.

**Measured planning budget:** the one-epoch feasibility run projects 13.09 GPU-stage
hours under equal training costs, or 17.43 hours under the stress assumptions.
Serial totals including CPU work and a 20% planning allowance are 17.77 and 22.96 hours.
These are scenarios, not confidence intervals or guaranteed bounds. The complete
matrix has 51 training epochs (15 baseline + 21 SVD fine-tuning + 15 control).
See [the report and assumptions](audit/FEASIBILITY_REPORT.md) and
[the derived budget CSV](audit/feasibility/budget_estimate.csv). The matrix has not run
and still requires approval.

Memória: batch 32 com AMP e `accum_steps: 2` (batch efetivo 64). Se faltar VRAM, use
`batch_size: 16` e `accum_steps: 4`. No Windows, se o DataLoader travar, teste `num_workers: 0`.

## Publicação no Hugging Face

```powershell
huggingface-cli login
python scripts/06_publish_hf.py --config configs/full.yaml --repo-id SEU_USUARIO/deit-small-food101-svd --github https://github.com/SEU_USUARIO/vit-lowrank-compression --dry-run
```

`--dry-run` is read-only: it creates, deletes and uploads nothing.
Remote publication is suspended. `--local-only` builds a **new** local directory
only after validating checkpoint provenance. Synthetic exports are refused.

## Resultados

_A preencher com a saída de `03_select.py` (validação) e `04_final_test.py` (teste)._
Figuras em `results/figures/`: `pareto_params.png`, `pareto_flops.png`,
`pareto_latency_gpu.png`, `pareto_latency_cpu.png`.

## Estrutura

```
src/vitlr/
  data.py        Food-101 (HF), split estratificado, transforms
  model.py       criação via timm, checkpoints (com ranks)
  compress.py    FactorizedLinear, rank por energia, compress_model
  train.py       AdamW + cosseno, AMP, gradient accumulation
  evaluate.py    top-1/top-5, acertos por exemplo
  flops.py       contador por hooks
  benchmark.py   parâmetros, tamanho em disco, latência
  quantize.py    fp16, INT8 dinâmica
  stats.py       bootstrap pareado, splits estratificados
  pipeline.py    uma linha por execução no CSV
  report.py      média ± dp, IC agregado, tabela
configs/         full.yaml, quick.yaml, smoke.yaml
scripts/         01_train, 02_sweep, 03_select, 04_final_test, 05_plots, 06_publish_hf
tests/           decomposição, FLOPs, bootstrap, quantização, dados
```

## Limitações

- SVD dos pesos ignora as ativações; variantes cientes dos dados (ex.: SVD ponderado pela
  covariância das entradas) tendem a perder menos acurácia sem fine-tuning.
- Um limiar global de energia; não há busca de rank por camada.
- INT8 dinâmica não acelera GPU; fp16 não é medido em CPU.
- Uma tarefa (Food-101), uma família de modelos (DeiT) e um único laptop para latência.
- As imagens do Food-101 vêm do Foodspotting; a licença do dataset no HF consta como desconhecida.
