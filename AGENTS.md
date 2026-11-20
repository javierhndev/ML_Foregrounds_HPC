# AGENTS.md

## What this repo is

Research code for 4x super-resolution of Planck dust polarization (`superres_dust`
package). There is **no packaging** (`setup.py` / `pyproject.toml` absent) and
**no tests, lint, typecheck, or CI**. Training runs through the minimal CLI
`superres_dust/train.py`, which reads `configs/runs/*.yaml` and assembles `Model`
(LightningModule), `SRDDataModule`, `MapLogger`, and the model/loss YAML files.
End-to-end entry points are `tutorials/test_superres_model.ipynb` and the smoke test
below.

## Setup and imports

- Environment: `conda env create -f environment.yml && conda activate pytorch-nightly`
  (deps come from the `pytorch-nightly` channel).
- Intra-package imports are non-relative (`from datasets import SRDDataset`,
  `from metrics import VGGLoss`), so `superres_dust/` itself must be on `sys.path`.
  Use e.g. `PYTHONPATH="$PWD:$PWD/superres_dust" python ...`; local `datasets/` and
  `metrics/` also shadow same-named site-packages, so keep them first.
- No test suite. Dependency-free smoke check: `python -m compileall superres_dust tutorials`.

## Data pipeline gotchas

- Data is not committed (`data/` and `*.ckpt` are gitignored). `SRDDataModule` reads
  pickles named `resol*` from `dataset.dir` and parses the resolution from the first
  two digits after `resol` (`resol15...` -> 15). Samples are `(7, 80, 80)` arrays,
  channels `tau, q_hi, u_hi, q_lr, u_lr, q_hr, u_hr`.
- `prepare_data()` normalizes, applies D4 augmentation, randomly splits 80/10/10, and
  caches `train_subset.pt` / `val_subset.pt` / `test_subset.pt` in `dataset.dir`.
  Subsequent runs load that cache and ignore code changes — delete those three files
  after changing `transforms/normalize.py`, `transforms/augment.py`, split logic, or
  the raw pickles.
- Normalization bounds in `transforms/normalize.py` are hard-coded; update them for
  new data.
- Resolution embedding indexes `nn.Embedding(4)` via `resol.int() // 5 - 3`, so only
  15/20/25/30 arcmin work. `MapLogger` also hard-codes `{15, 20, 25, 30}` and the
  5-input + resolution token layout (6 tokens).
- `use_resolution_embedding: false` is currently broken: `Linear` receives `(N,)`
  instead of `(N, 1)`.

## Configs and runtime

- `configs/runs/*.yaml`: `trainer.accelerator` is `mps` (macOS) — change to `gpu` or
  `cpu` on Linux/HPC; `dataset.dir` and `trainer.checkpoint_path` are empty. W&B
  defaults to online.
- `model.name` selects `configs/model/<name>.yaml` (`main` vs `attention`). Loss
  weights in `configs/loss_functions.yaml` must sum to `(0, 1]`; `get_loss` enforces it.
- Known bug: `get_extra_metrics*` (VIF, FSIM, GMSD, MS-GMSD, HaarPSI, MDSI, VGG loss)
  fail because `CustomMetric.update` writes `self.metric` / `self.total` while the
  registered states are `accumulated_metric` / `sample_count`. Standard torchmetrics
  metrics and the MAE loss work.
- The tutorial notebook assumes an undefined `WDIR` variable and a hard-coded
  checkpoint/run path; set those before executing.

## Running the smoke test

`configs/runs/local_smoke.yaml` runs one CPU epoch with `overfit_batches: 1` and
offline W&B. Set `dataset.dir` to the directory containing the cached
`train/val/test_subset.pt` splits (currently `/home/javierhn/datasets/ml_foregrounds`).

`train.py` resolves `../configs/...` relative to the current directory, so run it
from `superres_dust/`:

```bash
conda run --no-capture-output -n pytorch-nightly python train.py train local_smoke.yaml
```

Offline W&B logs go to `superres_dust/wandb/offline-run-*/`; the checkpoint is saved
under `.../files/checkpoints/`.
