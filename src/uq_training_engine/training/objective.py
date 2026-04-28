import logging
from collections.abc import Mapping

import optuna
import pandas as pd
import torch

from uq_training_engine.config import Config
from uq_training_engine.training.train import run_training

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.trial.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ts_maps: Mapping[str, Mapping[str, float]],
    device: torch.device,
    base_cfg: Config,
) -> float:
    """
    Optuna objective: sample backbone/head learning rates and ``lambda_bt``, then train one run.

    Hyperparameters are logged; training uses fixed batch size 32 and one epoch per trial
    (see implementation).

    :param trial: Optuna trial (``suggest_float`` for ``lr_backbone``, ``lr_head``, ``lambda_bt``).
    :param train_df: Training pairs DataFrame.
    :param val_df: Validation pairs DataFrame.
    :param ts_maps: Per-category TrueSkill maps.
    :param device: Torch device.
    :param base_cfg: ``Config`` carrying ``model_name``, ``image_size``, ``images_root``, ``output_dir``, ``seed``.
    :return: Best validation composite score from ``run_training`` (maximize in study).

    Example::
        In: objective(trial, train_df, val_df, ts_maps, device, base_cfg)
        Out: 0.38  # float returned to Optuna
    """
    cfg = Config(
        model_name=base_cfg.model_name,
        image_size=base_cfg.image_size,
        images_root=base_cfg.images_root,
        output_dir=base_cfg.output_dir,
        lr_backbone=trial.suggest_float("lr_backbone", 5e-7, 5e-6, log=True),
        lr_head=trial.suggest_float("lr_head", 1e-5, 2e-4, log=True),
        lambda_bt=trial.suggest_float("lambda_bt", 0.4, 0.8),
        batch_size=32,
        epochs=1,  # short runs keep hyperparameter search tractable; tune n_trials instead.
    )
    logger.info(
        "Optuna trial %d: starting training (lr_backbone=%.2e, lr_head=%.2e, lambda_bt=%.4f)",
        trial.number,
        cfg.lr_backbone,
        cfg.lr_head,
        cfg.lambda_bt,
    )
    return run_training(cfg, train_df, val_df, ts_maps, device, trial=trial)
