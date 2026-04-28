import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collect_data_for_calibration(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform inference on the provided loader to gather all model predictions and targets.

    :param model: The model used for inference.
    :param loader: DataLoader providing PlacePulse batches.
    :param device: Torch device for computation.
    :return: Tuple of (all_preds, all_targets) as stacked NumPy arrays.

    Example::
        In: collect_data_for_calibration(model, loader, device)
        Out: (np.array([[0.1, ...]]), np.array([[2.5, ...]]))
    """
    all_preds, all_targets = [], []

    with torch.no_grad():
        # Collect all (pred row, target row) pairs from batches into two big lists.
        for img_l, img_r, _, _, l_aux, r_aux, _, _ in loader:
            pred_l = model(img_l.to(device)).cpu().numpy()
            pred_r = model(img_r.to(device)).cpu().numpy()

            # Stack left and right as independent (image, target) rows for isotonic fitting.
            all_preds.extend([pred_l, pred_r])
            all_targets.extend([l_aux.numpy(), r_aux.numpy()])

    return np.vstack(all_preds), np.vstack(all_targets)


def fit_and_save_category(x: np.ndarray, y: np.ndarray, cat_name: str, calib_dir: Path) -> dict[str, Any] | None:
    """
    Fit an IsotonicRegressor for a single category, normalize targets, and save the model.

    :param x: Array of model predictions for the category.
    :param y: Array of TrueSkill targets for the category.
    :param cat_name: Name of the category (e.g., "safer").
    :param calib_dir: Path to the directory where the .joblib file will be saved.
    :return: Dictionary with metadata (file, y_min, y_max) or None if fitting is skipped.

    Example::
        In: fit_and_save_category(preds, targets, "safer", Path("out/"))
        Out: {"file": "calibrator_safer.joblib", "y_min": -2.0, "y_max": 2.0}
    """
    mask = np.isfinite(y)

    # One isotonic mapping per output dimension when enough finite targets exist.
    if mask.sum() > 10:
        y_min, y_max = float(y[mask].min()), float(y[mask].max())

        # Per-category min-max to 0..100 so isotonic inputs are comparable across runs.
        y_norm = 100 * (y[mask] - y_min) / (y_max - y_min + 1e-8)
        iso_reg = IsotonicRegression(out_of_bounds="clip").fit(x[mask], y_norm)

        save_name = f"calibrator_{cat_name.replace(' ', '_')}.joblib"
        joblib.dump(iso_reg, calib_dir / save_name)

        return {"file": str(save_name), "y_min": y_min, "y_max": y_max}

    return None


def calibrate_model(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, out_dir: Path | str, epoch: int
) -> None:
    """
    Fit per-category isotonic regressors on model scores vs. TrueSkill targets and persist them.

    Writes ``calibrators_epoch_<epoch>/`` with ``.joblib`` files and ``calibrators_meta.json``.

    :param model: Trained ``ViTMultiHead`` in eval mode for forward passes.
    :param loader: Validation ``DataLoader`` yielding PlacePulse batches.
    :param device: Torch device for inference.
    :param out_dir: Base directory (e.g. ``.../final``); calibrator subfolder is created inside.
    :param epoch: Epoch index used in the output subdirectory name.
    :return: None.

    Example::
        In: calibrate_model(model, val_loader, device, Path("out/final"), 2)
        Out: creates out/final/calibrators_epoch_2/ with joblib + json
    """
    logger.info("--- Calibration phase (epoch %d) ---", epoch)
    model.eval()

    categories = ["safer", "wealthier", "more beautiful", "livelier", "less depressing", "less boring"]

    # --- Data Collection ---
    all_preds, all_targets = collect_data_for_calibration(model, loader, device)

    # --- Setup Output Directory ---
    calib_dir = Path(out_dir) / f"calibrators_epoch_{epoch}"
    calib_dir.mkdir(exist_ok=True, parents=True)
    meta = {}

    # --- Category Loop ---
    for cat_id, cat in enumerate(categories):
        res = fit_and_save_category(all_preds[:, cat_id], all_targets[:, cat_id], cat, calib_dir)
        if res:
            meta[cat] = res

    # --- Metadata Persistence ---
    with open(calib_dir / "calibrators_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    logger.info("Calibration finished; wrote metadata to %s", calib_dir)
