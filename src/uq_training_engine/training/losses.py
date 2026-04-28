import torch
import torch.nn.functional as torch_f


def smooth_l1_masked(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 loss between predictions and targets, averaging only over finite target positions.

    Non-finite targets are masked out; if none are finite, returns a zero scalar on ``pred``'s device.

    :param pred: Tensor of predictions (same shape as ``tgt``).
    :param tgt: Tensor with possible ``NaN``/``inf`` for missing auxiliary targets.
    :return: Scalar mean smooth L1 over valid elements.

    Example::
        In: smooth_l1_masked(scores, aux_targets)
        Out: tensor(0.15)  # example scalar loss
    """
    mask = torch.isfinite(tgt).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    # nan_to_num only affects masked-out positions; masked loss still uses finite targets only.
    return torch_f.smooth_l1_loss(pred * mask, torch.nan_to_num(tgt) * mask, reduction="sum") / mask.sum()
