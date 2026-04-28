# UQ Training Engine

UQ Training Engine is a training-focused project for urban perception modeling.
It learns category-specific preferences from pairwise image comparisons (`left` vs `right`)
using a Vision Transformer (ViT) multi-head model.

The pipeline supports:

- single-run training from CLI,
- hyperparameter optimization with Optuna,
- global target regularization based on TrueSkill-derived scores,
- reproducible experiment outputs and calibration artifacts.

## What This Project Learns

For each image, the model predicts scores for 6 perception categories:

- `safer`
- `wealthier`
- `more beautiful`
- `livelier`
- `less depressing`
- `less boring`

The training objective combines:

- pairwise margin ranking loss (learns which image should rank higher),
- auxiliary Smooth L1 regression to TrueSkill targets (stabilizes and anchors learning).

## Quick Start

Install dependencies:

```bash
poetry install
```

Run a training job:

```bash
python -m uq_training_engine train \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --votes-csv data/votes.csv \
  --images-root data/images \
  --output-dir outputs \
  --device auto
```

Run Optuna hyperparameter search:

```bash
python -m uq_training_engine optuna \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --votes-csv data/votes.csv \
  --images-root data/images \
  --output-dir outputs_optuna \
  --n-trials 20 \
  --device auto
```

## Input Data

The CLI expects three CSV files:

- `train.csv` - training pairs,
- `val.csv` - validation pairs,
- `votes.csv` - full vote set used to build TrueSkill maps.

Required columns:

- `study_question`
- `left`
- `right`
- `choice` (`left` or `right`)

Images are loaded from `--images-root` and resolved recursively (`.jpg`, `.jpeg`, `.png`).

## Outputs and Metrics

Training produces:

- model checkpoints (`model_final_ep{N}.pt`),
- per-epoch isotonic calibrators,
- calibration metadata.

Validation reports include:

- pairwise accuracy per category,
- Spearman rho vs TrueSkill per category,
- mean accuracy and mean rho.

Optuna optimizes:

`(mean_acc + mean_rho) / 2`

## Local Docs Preview

To preview or build this documentation site:

```bash
mkdocs serve
mkdocs build
```

## Notes

- Required paths (`--train-csv`, `--val-csv`, `--votes-csv`, `--images-root`) are validated before run.
- If a required path is missing, CLI exits with code `2`.
- Output directories are created automatically.
