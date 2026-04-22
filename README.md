# UQ Training Engine

> Training repository for urban perception modeling that uses a ViT
> multi-head architecture to learn category-specific preferences from
> pairwise comparisons, supports TrueSkill-derived global targets, and
> provides a reproducible CLI workflow for single-run training,
> hyperparameter optimization with Optuna, and structured experiment
> outputs.

![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-ViT%20Multi--Head-6f42c1)
![CLI](https://img.shields.io/badge/interface-CLI-0A7EA4)

## Quick Start

``` bash
poetry install
python -m uq_training_engine train --train-csv data/train.csv --val-csv data/val.csv --votes-csv data/votes.csv --images-root data/images --output-dir outputs --device auto
```

## Overview

This project learns urban perception preferences from pairwise image
comparisons (`left` vs `right`).

The model predicts 6 category scores per image:

-   `safer`
-   `wealthier`
-   `more beautiful`
-   `livelier`
-   `less depressing`
-   `less boring`

Training objective is a weighted combination of two complementary
components:

-   **Pairwise margin ranking loss** - learns which image should rank
    higher for the active category in each training pair.
-   **Auxiliary Smooth L1 regression to TrueSkill targets** - anchors
    predictions to global, vote-aggregated quality scores and stabilizes
    optimization across categories.

Hyperparameter search is supported via **Optuna (TPE sampler)**.

## Training Strategy

The current training setup is designed for robust preference learning on
noisy pairwise labels:

-   **Backbone + head with separate learning rates**\
    `AdamW` uses two parameter groups: one LR for the ViT backbone
    (`--lr-backbone`) and one for the prediction head (`--lr-head`).\
    A shared `OneCycleLR` scheduler updates both groups through
    training, while preserving different LR scales. This keeps large
    pretrained features stable and lets the small task-specific head
    adapt faster.

-   **TrueSkill-regularized ranking objective**\
    Training combines pairwise ranking with auxiliary regression to
    per-category TrueSkill maps built from the full vote set
    (`--votes-csv`).\
    `--lambda-bt` controls the loss balance between pairwise ranking and
    auxiliary regression.

-   **TrueSkill-based training pair construction**\
    Training/validation pairs are prepared from per-category TrueSkill
    rankings rather than raw single-vote pairs.\
    Candidate images are selected from high-confidence ratings (low
    TrueSkill `sigma`), and pair winners are assigned by relative
    TrueSkill rank (`mu`).\
    This reduces noise from individual votes during data construction.\
    Separately, TrueSkill maps are also used as auxiliary regression
    targets during model optimization.

-   **Unfrozen backbone by default**\
    The default config trains with an unfrozen backbone
    (`--freeze-backbone` is optional).\
    This allows the pretrained ViT representation to adapt to the urban
    perception domain instead of relying only on a fixed feature
    extractor.

-   **Augmentation choices for urban perception**\
    The training pipeline uses geometric augmentation
    (e.g. `RandomHorizontalFlip`, `RandomResizedCrop`) without color
    augmentation, to avoid altering visual cues that may directly affect
    human perception labels.

-   **Why DINOv2 ViT backbone**\
    The default `vit_base_patch14_dinov2.lvd142m` backbone is chosen for
    strong transferable visual representations and good spatial/scene
    understanding (depth perception), which are important for perception
    tasks such as safety, beauty, and liveliness ranking.

## Feature Highlights

-   `timm` Vision Transformer backbone (default:
    `vit_base_patch14_dinov2.lvd142m`)
-   6-head prediction architecture (`N x 6` outputs)
-   Automatic Mixed Precision (AMP)
-   Built-in evaluation:
    -   pairwise accuracy
    -   Spearman rho vs TrueSkill
-   Isotonic calibration export per epoch (standard training mode)
-   Simple CLI with `train` and `optuna` commands

## Requirements

  -----------------------------------------------------------------------
  Component                           Version
  ----------------------------------- -----------------------------------
  Python                              `>=3.12,<3.15`

  Core libs                           `torch`, `torchvision`, `timm`,
                                      `numpy`, `pandas`, `scipy`,
                                      `scikit-learn`, `optuna`,
                                      `trueskill`, `pillow`
  -----------------------------------------------------------------------

Install dependencies:

``` bash
# Poetry (recommended)
poetry install

# or pip
pip install -r requirements.txt
```

Development tools:

``` bash
pip install -r requirements-dev.txt
```

## Input Data

The CLI expects 3 CSV files:

-   `train.csv` - training pairs
-   `val.csv` - validation pairs
-   `votes.csv` - all votes set used to fit global TrueSkill maps

You can provide prefiltered `train.csv` / `val.csv` built from a
TrueSkill ranking pipeline; the trainer will use them as-is.

Required CSV columns:

  Column             Description
  ------------------ ----------------------------
  `study_question`   one of the 6 categories
  `left`             left image filename/id
  `right`            right image filename/id
  `choice`           winner (`left` or `right`)

Images are provided with `--images-root` and searched recursively
(`.jpg`, `.jpeg`, `.png`).\
If `final_photo_dataset` exists under this root, it is preferred
automatically.

## CLI Usage

Use either:

-   `python -m uq_training_engine ...`
-   `uq-train ...` (if installed as script)

### Train (single run)

``` bash
python -m uq_training_engine train \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --votes-csv data/votes.csv \
  --images-root data/images \
  --output-dir outputs \
  --epochs 2 \
  --batch-size 32 \
  --lr-backbone 2e-6 \
  --lr-head 5e-5 \
  --lambda-bt 0.6 \
  --device auto
```

### Kaggle notebook (single copy-paste cell)

``` python
# Paste this into one Kaggle notebook cell and run.
# It installs dependencies, downloads a runnable script from GitHub, imports core modules,
# and launches a short training run.

!pip -q install uq-training-engine
!wget -q -O sample_run.py https://raw.githubusercontent.com/UrbanQuality-AI/uq-training-engine/main/src/uq_training_engine/examples/sample_run.py

import os
import subprocess
import pandas as pd
import torch

from uq_training_engine import Config, fit_trueskill_large, run_training, set_seed
from uq_training_engine.logging_config import configure_logging

INPUT_DIR = "/kaggle/input/placepulse-project"
OUTPUT_DIR = "/kaggle/working/output"
IMAGES_DIR = f"{INPUT_DIR}/images"

configure_logging()

train_df = pd.read_csv(f"{INPUT_DIR}/train.csv").dropna(subset=["study_question", "left", "right", "choice"])
val_df = pd.read_csv(f"{INPUT_DIR}/val.csv").dropna(subset=["study_question", "left", "right", "choice"])
votes_df = pd.read_csv(f"{INPUT_DIR}/all_votes.csv").dropna(subset=["study_question", "left", "right", "choice"])

categories = ["safer", "wealthier", "more beautiful", "livelier", "less depressing", "less boring"]
ts_maps = {
    cat: fit_trueskill_large(votes_df[votes_df["study_question"] == cat])
    for cat in categories
}

cfg = Config(
    model_name="vit_base_patch14_dinov2.lvd142m",
    images_root=IMAGES_DIR,
    output_dir=OUTPUT_DIR,
    epochs=2,
    batch_size=32,
    lr_backbone=2e-6,
    lr_head=5e-5,
    lambda_bt=0.6,
)

set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score = run_training(cfg, train_df, val_df, ts_maps, device)
print("Training finished. Best score:", score)
```

Kaggle note: - input data should come from `/kaggle/input/...`
(read-only) - write outputs to `/kaggle/working/...`

Common flags:

-   `--freeze-backbone` freeze ViT backbone
-   `--no-amp` disable mixed precision
-   `--num-workers` set DataLoader workers
-   `--weight-decay`, `--seed`, `--image-size`, `--model-name`

### Optuna (hyperparameter search)

``` bash
python -m uq_training_engine optuna \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --votes-csv data/votes.csv \
  --images-root data/images \
  --output-dir outputs_optuna \
  --n-trials 20 \
  --device auto
```

Persist/resume study:

``` bash
python -m uq_training_engine optuna \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --votes-csv data/votes.csv \
  --images-root data/images \
  --output-dir outputs_optuna \
  --n-trials 20 \
  --storage sqlite:///optuna.db \
  --study-name uq_training_engine \
  --load-if-exists
```

## Output Artifacts

### `train`

Saved under `OUTPUT_DIR/final/`:

-   `model_final_ep{N}.pt` - model checkpoint after epoch `N`
-   `calibrators_epoch_{N}/`
-   `calibrator_<category>.joblib` - isotonic calibration models
-   `calibrators_meta.json` - metadata (`y_min`, `y_max`, file names)

### `optuna`

Each trial writes to `OUTPUT_DIR/trial_<trial_number>/`.

## Metrics

Validation reports:

-   pairwise accuracy per category
-   Spearman rho per category
-   mean accuracy and mean rho

Optuna target score:

`(mean_acc + mean_rho) / 2`

## Project Layout

``` text
src/uq_training_engine/
  __main__.py              # CLI entry (train / optuna)
  config.py                # training configuration
  data/
    dataset.py             # PlacePulse dataset
    trueskill.py           # TrueSkill fitting
  models/
    vit_multihead.py       # ViT + 6-head predictor
  training/
    train.py               # training loop
    evaluation.py          # metrics
    calibration.py         # isotonic calibrators
    objective.py           # Optuna objective
```

## Notes

> \[!IMPORTANT\] `--train-csv`, `--val-csv`, `--votes-csv`, and
> `--images-root` are validated before execution.\
> If any required path is missing, CLI exits with code `2`.

> The output directory is created automatically.
