import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def init_eval_storage() -> tuple[list[str], dict[int, dict[str, Any]]]:
    """
    Initialize storage structures and names for category-wise evaluation.

    :return: Tuple containing a list of category names and a dictionary for metrics storage.

    Example::
        In: init_eval_storage()
        Out: (["safer", ...], {0: {"correct": 0, ...}, ...})
    """
    cat_names = ["safer", "wealthier", "more beautiful", "livelier", "less depressing", "less boring"]
    per_cat = {c: {"correct": 0, "total": 0, "preds": {}, "gts": {}} for c in range(6)}
    return cat_names, per_cat


def update_batch_metrics(
    per_cat: dict[int, dict[str, Any]],
    scores_l: np.ndarray,
    scores_r: np.ndarray,
    l_aux: np.ndarray,
    r_aux: np.ndarray,
    cat_idx: np.ndarray,
    y_rank: np.ndarray,
    l_ids: Sequence[str],
    r_ids: Sequence[str],
) -> None:
    """
    Update pairwise counters and score dictionaries for Spearman correlation from a batch.

    :param per_cat: The metric storage dictionary to update.
    :param scores_l: Predicted scores for the left images (N, 6).
    :param scores_r: Predicted scores for the right images (N, 6).
    :param l_aux: Auxiliary TrueSkill targets for the left images.
    :param r_aux: Auxiliary TrueSkill targets for the right images.
    :param cat_idx: Indices of the active category for each pair.
    :param y_rank: Ground truth ranking labels (-1, 0, 1).
    :param l_ids: Unique identifiers for left images.
    :param r_ids: Unique identifiers for right images.
    :return: None (updates per_cat in-place).

    Example::
        In: update_batch_metrics(per_cat, s_l, s_r, a_l, a_r, c_idx, y, id_l, id_r)
        Out: None
    """
    # Within-batch loop: pairwise accuracy for the active question; fill six Spearman buckets.
    for batch_id in range(len(l_ids)):
        c = cat_idx[batch_id]
        l_id, r_id = l_ids[batch_id], r_ids[batch_id]

        # Update pairwise accuracy for the specific question asked in this pair
        diff = scores_l[batch_id, c] - scores_r[batch_id, c]
        if np.sign(diff) == np.sign(y_rank[batch_id]):
            per_cat[c]["correct"] += 1
        per_cat[c]["total"] += 1

        # Six heads: record pred vs. auxiliary TS for each category when the target is not NaN.
        for c_aux in range(6):
            if not np.isnan(l_aux[batch_id, c_aux]):
                per_cat[c_aux]["preds"][l_id] = scores_l[batch_id, c_aux]
                per_cat[c_aux]["gts"][l_id] = l_aux[batch_id, c_aux]
            if not np.isnan(r_aux[batch_id, c_aux]):
                per_cat[c_aux]["preds"][r_id] = scores_r[batch_id, c_aux]
                per_cat[c_aux]["gts"][r_id] = r_aux[batch_id, c_aux]


def compute_final_metrics(per_cat: dict[int, dict[str, Any]]) -> tuple[list[float], list[float]]:
    """
    Aggregate accumulated predictions into final accuracy and Spearman rho per category.

    :param per_cat: The metric storage dictionary populated during evaluation.
    :return: Tuple of (list of accuracies, list of rhos).

    Example::
        In: compute_final_metrics(per_cat)
        Out: ([0.7, 0.65, ...], [0.4, 0.5, ...])
    """
    all_accs, all_rhos = [], []

    for c in range(6):
        # Calculate pairwise accuracy
        acc = per_cat[c]["correct"] / (per_cat[c]["total"] + 1e-8)

        # Calculate Spearman Rho if enough data points exist
        y_pred, y_target = list(per_cat[c]["preds"].values()), list(per_cat[c]["gts"].values())
        rho = 0.0
        if len(y_target) > 5:
            rho, _ = spearmanr(y_pred, y_target)
            if np.isnan(rho):
                rho = 0.0

        all_accs.append(acc)
        all_rhos.append(rho)

    return all_accs, all_rhos


def log_evaluation_table(
    cat_names: list[str], all_accs: list[float], all_rhos: list[float], mean_acc: float, mean_rho: float
) -> None:
    """
    Print a formatted summary table of the evaluation results.

    :param cat_names: List of category labels.
    :param all_accs: List of accuracies per category.
    :param all_rhos: List of Spearman rhos per category.
    :param mean_acc: Average accuracy across all categories.
    :param mean_rho: Average Spearman rho across all categories.
    :return: None

    Example::
        In: log_evaluation_table(names, accs, rhos, 0.6, 0.4)
        Out: None
    """
    logger.info("\n%s", "=" * 65)
    logger.info("%-20s | %-15s | %-15s", "Category", "Pairwise Acc", "Spearman Rho")
    logger.info("-" * 65)

    for c in range(6):
        logger.info("%-20s | %-15.4f | %-15.4f", cat_names[c], all_accs[c], all_rhos[c])

    logger.info("-" * 65)
    logger.info("%-20s | %-15.4f | %-15.4f", "AVERAGE", mean_acc, mean_rho)
    logger.info("%s\n", "=" * 65)


def evaluate_model(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader[Any], device: torch.device
) -> tuple[float, float]:
    """
    Compute mean pairwise accuracy and mean Spearman correlation across six categories.

    Pairwise accuracy uses the active category head; Spearman aggregates per-image predictions
    vs. auxiliary TrueSkill scores where targets are finite.

    :param model: Model with ``forward`` returning ``(N, 6)`` scores.
    :param loader: ``DataLoader`` over PlacePulse (same layout as training).
    :param device: Torch device.
    :return: Tuple ``(mean_acc, mean_rho)`` averaged over the six categories.

    Example::
        In: evaluate_model(model, val_loader, device)
        Out: (0.61, 0.45)  # example mean accuracy and mean Spearman rho
    """
    model.eval()
    cat_names, per_cat = init_eval_storage()

    n_steps = len(loader)
    eval_log_step = max(1, n_steps // 4)

    logger.info("Starting evaluation (%d batches).", n_steps)

    with torch.no_grad():
        # Batches: score both images, then update per-category counters and per-image pred/gt maps.
        for i, (img_l, img_r, cat_idx, y_rank, l_aux, r_aux, l_ids, r_ids) in enumerate(loader):
            img_l, img_r, cat_idx, y_rank = img_l.to(device), img_r.to(device), cat_idx.to(device), y_rank.to(device)

            with torch.amp.autocast("cuda", enabled=True):
                scores_l, scores_r = model(img_l), model(img_r)

            # Convert tensors to numpy for metric processing
            scores_l_np, scores_r_np = scores_l.cpu().numpy(), scores_r.cpu().numpy()
            l_aux_np, r_aux_np = l_aux.numpy(), r_aux.numpy()
            cat_np, y_np = cat_idx.cpu().numpy(), y_rank.cpu().numpy()

            # Update metrics using helper
            update_batch_metrics(per_cat, scores_l_np, scores_r_np, l_aux_np, r_aux_np, cat_np, y_np, l_ids, r_ids)

            # Rough progress through the loader (~4 logs per eval).
            if (i + 1) % eval_log_step == 0:
                logger.info("[Eval] Step %d/%d", i + 1, n_steps)

    # Collapse accumulated data into final metrics
    all_accs, all_rhos = compute_final_metrics(per_cat)
    mean_acc, mean_rho = float(np.mean(all_accs)), float(np.mean(all_rhos))

    # Print results summary
    log_evaluation_table(cat_names, all_accs, all_rhos, mean_acc, mean_rho)

    return mean_acc, mean_rho
