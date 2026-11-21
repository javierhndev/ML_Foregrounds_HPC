# AGENTS.md

## What this repo is

Research code for 4x super-resolution of Planck dust polarization (`superres_dust`
package). There is **no packaging** (`setup.py` / `pyproject.toml` absent) and
**no tests, lint, typecheck, or CI**. Training runs through the minimal CLI
`superres_dust/train.py`, which reads `configs/runs/*.yaml` and assembles `Model`
(LightningModule), `SRDDataModule`, `MapLogger`, and the model/loss YAML files.
End-to-end entry points are `tutorials/test_superres_model.ipynb`,
`superres_dust/predict.py` (single-patch checkpoint inference), and the smoke test
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
- `Normalize.normalize_image`, `denormalize_image`, and `__call__` currently crash with
  `'float' object has no attribute 'item'` / `'float'` because the hard-coded bounds are
  Python floats. Cached-split training never calls them, so this only appears when
  normalizing fresh data. `predict.py` denormalizes Q/U by multiplying by `QU_BOUND`
  directly.
- Resolution embedding indexes `nn.Embedding(4)` via `resol.int() // 5 - 3`, so only
  15/20/25/30 arcmin work. `MapLogger` also hard-codes `{15, 20, 25, 30}` and the
  5-input + resolution token layout (6 tokens).
- `use_resolution_embedding: false` is currently broken: `Linear` receives `(N,)`
  instead of `(N, 1)`.

## Configs and runtime

- `configs/runs/*.yaml`: `trainer.accelerator` is `mps` (macOS) — change to `gpu` or
  `cpu` on Linux/HPC; `dataset.dir` and `trainer.checkpoint_path` are empty. W&B
  defaults to online.
- `configs/runs/local_gpu_1epoch.yaml` is the full-data GPU sanity run: 1 epoch,
  `overfit_batches: 0`, `debug: False`, 4 workers, offline W&B. It reuses the cached
  splits, so no cache deletion is needed.
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

## Running inference

`superres_dust/predict.py` runs one cached `val` or `test` patch through a checkpoint
and writes `<out-dir>/<tag>_<split>_<index>_prediction.{npz,png}` (Q maps top row, U
maps bottom row). It reads the architecture config and `total_steps` from the
checkpoint's `hyper_parameters`, loads only the requested split directly, and avoids
the 4.2 GB `train_subset.pt`. Run from `superres_dust/`:

```bash
conda run --no-capture-output -n pytorch-nightly python predict.py \
  --checkpoint /home/javierhn/datasets/ml_foregrounds/epoch_00209-val_loss_0.00366.ckpt \
  --data-dir /home/javierhn/datasets/ml_foregrounds \
  --split test --index 0 --tag pretrained
```

Other useful flags: `--device auto|cpu|cuda`, `--out-dir` (default
`logs/inference`). Offline W&B checkpoint filenames contain `:`, so quote checkpoint
paths in the shell.

## Measured timings

`configs/runs/local_gpu_1epoch.yaml` on the RTX 4070 Laptop GPU (8 GB): 23,576 train
steps at ~50 it/s (7:51), then 2,944 validation steps at ~187 it/s (15 s) — **8:03
total wall clock**, peak RSS ~10.6 GiB, final `val_loss` 0.00592.

```bash
/usr/bin/time -v conda run --no-capture-output -n pytorch-nightly \
  python train.py train local_gpu_1epoch.yaml
```

Run from `superres_dust/`; redirect output to a file because the Lightning progress bar
emits one update per step.
