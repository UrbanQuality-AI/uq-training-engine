import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import timm
import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader
from torchvision import transforms

from uq_training_engine.config import Config
from uq_training_engine.data.dataset import PlacePulse
from uq_training_engine.models.vit_multihead import ViTMultiHead
from uq_training_engine.training.calibration import calibrate_model
from uq_training_engine.training.evaluation import evaluate_model
from uq_training_engine.training.losses import smooth_l1_masked
from uq_training_engine.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

BatchType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]


def setup_training_env(config: Config, trial: optuna.trial.Trial | None = None) -> Path:
    """
    Configure logging, seeds, and create the output directory for checkpoints.

    :param config: Training configuration object.
    :param trial: Optional Optuna trial for unique directory naming.
    :return: Path object pointing to the output directory.

    Example::
        In: setup_training_env(cfg, trial)
        Out: PosixPath('outputs/trial_0')
    """
    root = logging.getLogger()
    if not root.handlers:
        from uq_training_engine.logging_config import configure_logging

        configure_logging()

    set_seed(config.seed)

    trial_id = trial.number if trial is not None else None
    out_dir = Path(config.output_dir)

    if trial_id is not None:
        out_dir = out_dir / f"trial_{trial_id}"
    else:
        out_dir = out_dir / "final"

    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def get_data_transforms(
    config: Config, model_mean: list[float] | tuple[float, ...], model_std: list[float] | tuple[float, ...]
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation torchvision transformations.

    :param config: Training configuration containing image size.
    :param model_mean: Normalization mean from the model config.
    :param model_std: Normalization standard deviation from the model config.
    :return: Tuple of (train_transforms, val_transforms).

    Example::
        In: get_data_transforms(cfg, [0.5], [0.5])
        Out: (Compose(...), Compose(...))
    """
    train_tf = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(model_mean, model_std),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(model_mean, model_std),
        ]
    )

    return train_tf, val_tf


def prepare_dataloaders(
    config: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ts_maps: Mapping[str, Mapping[str, float]],
    train_tf: transforms.Compose,
    val_tf: transforms.Compose,
) -> tuple[DataLoader[BatchType], DataLoader[BatchType]]:
    """
    Initialize PlacePulse datasets and wrap them in DataLoaders.

    :param config: Configuration for batch size and workers.
    :param train_df: Training DataFrame.
    :param val_df: Validation DataFrame.
    :param ts_maps: TrueSkill maps.
    :param train_tf: Training transformations.
    :param val_tf: Validation transformations.
    :return: Tuple of (train_loader, val_loader).

    Example::
        In: prepare_dataloaders(cfg, df1, df2, maps, t1, t2)
        Out: (DataLoader(...), DataLoader(...))
    """
    img_root = Path(config.images_root)

    train_loader = DataLoader(
        PlacePulse(train_df, img_root, train_tf, ts_maps),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        PlacePulse(val_df, img_root, val_tf, ts_maps),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader


def initialize_model(config: Config, device: torch.device) -> tuple[ViTMultiHead, dict[str, Any]]:
    """
    Instantiate the Multi-Head ViT and resolve its data configuration.

    :param config: Config containing model name and freeze settings.
    :param device: Torch device to move the model to.
    :return: Tuple of (model, timm_data_config).

    Example::
        In: initialize_model(cfg, torch.device("cuda"))
        Out: (ViTMultiHead(...), {'mean': [...], 'std': [...]})
    """
    model = ViTMultiHead(config.model_name, freeze=config.freeze_backbone).to(device)
    m_cfg = timm.data.resolve_data_config({}, model=model.backbone)
    return model, m_cfg


def setup_optimization(
    config: Config, model: torch.nn.Module, steps_per_epoch: int
) -> tuple[torch.optim.AdamW, torch.cuda.amp.GradScaler, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Initialize the optimizer, AMP scaler, and OneCycleLR scheduler.

    :param config: Training hyperparameters.
    :param model: The model to optimize.
    :param steps_per_epoch: Number of batches per epoch for the scheduler.
    :return: Tuple of (optimizer, scaler, scheduler).

    Example::
        In: setup_optimization(cfg, model, 100)
        Out: (AdamW(...), GradScaler(...), OneCycleLR(...))
    """
    optimizer = torch.optim.AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, model.backbone.parameters()), "lr": config.lr_backbone},
            {"params": filter(lambda p: p.requires_grad, model.head.parameters()), "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.lr_backbone, config.lr_head],
        total_steps=steps_per_epoch * config.epochs,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    return optimizer, scaler, scheduler


def train_one_step(
    model: torch.nn.Module,
    batch: BatchType,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Config,
    device: torch.device,
) -> float:
    """
    Perform a single forward and backward pass for one batch.

    :param model: The multi-head model.
    :param batch: Tuple of data from the DataLoader.
    :param optimizer: Torch optimizer.
    :param scaler: AMP GradScaler.
    :param scheduler: Learning rate scheduler.
    :param config: Training configuration for loss weighting and AMP.
    :param device: Torch device.
    :return: The scalar loss value for the step.

    Example::
        In: train_one_step(model, batch, opt, scaler, sched, cfg, device)
        Out: 0.456
    """
    img_l, img_r, cat_idx, y_rank, l_aux, r_aux, _, _ = batch

    # Move data to device
    img_l, img_r, cat_idx, y_rank, l_aux, r_aux = [t.to(device) for t in [img_l, img_r, cat_idx, y_rank, l_aux, r_aux]]

    optimizer.zero_grad(set_to_none=True)

    # Forward pass with AMP
    with torch.cuda.amp.autocast(enabled=config.amp):
        sl, sr = model(img_l), model(img_r)
        idx_col = cat_idx.view(-1, 1).long()

        # Pairwise loss on the head column for this row's study_question
        s_l = torch.gather(sl, 1, idx_col).squeeze(1)
        s_r = torch.gather(sr, 1, idx_col).squeeze(1)

        loss_pair = torch_f.margin_ranking_loss(s_l, s_r, y_rank, margin=0.2)
        loss_aux = 0.5 * (smooth_l1_masked(sl, l_aux) + smooth_l1_masked(sr, r_aux))
        loss = config.lambda_bt * loss_pair + (1.0 - config.lambda_bt) * loss_aux

    # Backward and optimize
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return loss.item()


def run_training(
    config: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ts_maps: Mapping[str, Mapping[str, float]],
    device: torch.device,
    trial: optuna.trial.Trial | None = None,
) -> float:
    """
    Train the multi-head ViT on pairwise data, validate periodically, optionally report to Optuna.

    Writes checkpoints under ``output_dir/final`` or ``output_dir/trial_<id>``; when not in an Optuna
    trial, saves per-epoch weights and runs calibration on the validation loader.

    :param config: Training hyperparameters and paths (``Config``).
    :param train_df: Training pairs DataFrame.
    :param val_df: Validation pairs DataFrame.
    :param ts_maps: Per-category TrueSkill maps for auxiliary regression targets.
    :param device: Torch device for model and batches.
    :param trial: Optional Optuna trial for intermediate reporting and pruning.
    :return: Best validation score ``(mean_acc + mean_rho) / 2`` seen during training.

    Example::
        In: run_training(cfg, train_df, val_df, ts_maps, torch.device("cuda"))
        Out: 0.42  # example composite score
    """
    # --- Setup ---
    out_dir = setup_training_env(config, trial)
    model, m_cfg = initialize_model(config, device)

    # --- Data ---
    train_tf, val_tf = get_data_transforms(config, m_cfg["mean"], m_cfg["std"])
    train_loader, val_loader = prepare_dataloaders(config, train_df, val_df, ts_maps, train_tf, val_tf)

    # --- Optimization ---
    optimizer, scaler, scheduler = setup_optimization(config, model, len(train_loader))

    log_interval = max(1, len(train_loader) // 100)
    val_interval = max(1, len(train_loader) // 4)
    best_val_score = 0.0

    logger.info(
        "Training on %s | Backbone Max LR: %.1e | Head Max LR: %.1e", device, config.lr_backbone, config.lr_head
    )

    # --- Main Training Loop ---
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        logger.info("--- Epoch %d/%d ---", epoch + 1, config.epochs)

        for i, batch in enumerate(train_loader):
            step_loss = train_one_step(model, batch, optimizer, scaler, scheduler, config, device)
            epoch_loss += step_loss

            # Logging
            if (i + 1) % log_interval == 0:
                logger.info(
                    "Step [%4d/%d] | %3.0f%% | Loss: %.4f | Head LR: %.2e",
                    i + 1,
                    len(train_loader),
                    (i + 1) / len(train_loader) * 100,
                    epoch_loss / (i + 1),
                    optimizer.param_groups[-1]["lr"],
                )

            # Periodic Validation
            if (i + 1) % val_interval == 0 or (i + 1) == len(train_loader):
                logger.info("--- Validation at %.0f%% ---", (i + 1) / len(train_loader) * 100)

                acc, rho = evaluate_model(model, val_loader, device)
                score = (acc + rho) / 2
                best_val_score = max(best_val_score, score)

                # Optuna reporting and pruning
                if trial is not None:
                    report_step = epoch * 4 + (i // val_interval)
                    trial.report(score, report_step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        # End of epoch: persist weights and calibrate (skip if Optuna trial)
        if trial is None:
            torch.save(model.state_dict(), out_dir / f"model_final_ep{epoch + 1}.pt")
            calibrate_model(model, val_loader, device, out_dir, epoch + 1)

    return best_val_score
