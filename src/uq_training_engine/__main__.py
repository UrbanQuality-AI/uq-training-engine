"""
CLI entry: `python -m uq_training_engine train ...` or `... optuna ...`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import optuna
import pandas as pd
import torch

from uq_training_engine.config import Config
from uq_training_engine.data import fit_trueskill_large
from uq_training_engine.logging_config import configure_logging
from uq_training_engine.training import objective, optuna_logging_callback, run_training
from uq_training_engine.utils import set_seed

logger = logging.getLogger(__name__)
_CFG_DEFAULTS = Config()

CATEGORIES = (
    "safer",
    "wealthier",
    "more beautiful",
    "livelier",
    "less depressing",
    "less boring",
)


def _resolve_device(name: str | None) -> torch.device:
    """
    Map a device name (or ``auto``) to a ``torch.device``.

    :param name: ``"auto"``, ``None``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    :return: Resolved PyTorch device (CUDA when available for ``auto``).

    Example::
        In: _resolve_device("auto")
        Out: device(type='cuda', index=0)  # or cpu if no GPU
    """
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_pairs_csv(path: Path) -> pd.DataFrame:
    """
    Load a pairwise-comparison CSV and drop rows missing required columns.

    :param path: Path to the CSV file.
    :return: DataFrame with at least ``study_question``, ``left``, ``right``, ``choice``.

    Example::
        In: _load_pairs_csv(Path("data/train.csv"))
        Out: DataFrame with NaN rows in key columns removed
    """
    df = pd.read_csv(path).dropna(subset=["study_question", "left", "right", "choice"])
    return df


def _build_ts_maps(votes_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Fit one TrueSkill rating map per study question (category) from votes.

    Uses the module-level ``CATEGORIES`` tuple and ``fit_trueskill_large`` on each slice.

    :param votes_df: Pairwise votes CSV (same schema as train/val pairs).
    :return: Mapping ``category_name -> {image_id_stem: trueskill_mu}``.

    Example::
        In: _build_ts_maps(votes_df)
        Out: {"safer": {"img123": 24.5, ...}, "wealthier": {...}, ...}
    """
    # One independent TS fit per category (only rows with that study_question).
    return {cat: fit_trueskill_large(votes_df[votes_df["study_question"] == cat]) for cat in CATEGORIES}


def _add_data_args(p: argparse.ArgumentParser) -> None:
    """
    Register shared CLI arguments for data paths, output, model, and logging.

    Used by both ``train`` and ``optuna`` subparsers.

    :param p: Argument parser to extend.
    :return: None (mutates ``p``).

    Example::
        In: p = argparse.ArgumentParser(); _add_data_args(p)
        Out: parser now accepts --train-csv, --val-csv, --votes-csv, ...
    """
    p.add_argument("--train-csv", type=Path, required=True, help="Train pairs CSV.")
    p.add_argument("--val-csv", type=Path, required=True, help="Validation pairs CSV.")
    p.add_argument(
        "--votes-csv",
        type=Path,
        required=True,
        help="CSV used to fit global TrueSkill maps (e.g. votes_clean2.csv).",
    )
    p.add_argument("--images-root", type=Path, required=True, help="Root folder for images.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (trial subfolders for Optuna, final/ for train).",
    )
    p.add_argument("--model-name", type=str, default=_CFG_DEFAULTS.model_name)
    p.add_argument("--image-size", type=int, default=_CFG_DEFAULTS.image_size)
    p.add_argument("--seed", type=int, default=_CFG_DEFAULTS.seed)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device: "auto", "cpu", "cuda", "cuda:0", ...',
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )


def _cmd_train(ns: argparse.Namespace) -> int:
    """
    Execute the ``train`` subcommand: load CSVs, fit TrueSkill maps, run one training job.

    :param ns: Parsed namespace from the ``train`` subparser (paths, hyperparameters, device).
    :return: Exit code ``0`` on success.

    Example::
        In: _cmd_train(parser.parse_args(["train", "--train-csv", "...", ...]))
        Out: 0
    """
    train_df = _load_pairs_csv(ns.train_csv)
    val_df = _load_pairs_csv(ns.val_csv)
    votes_df = _load_pairs_csv(ns.votes_csv)
    logger.info("Loaded rows: train=%d val=%d votes=%d", len(train_df), len(val_df), len(votes_df))

    ts_maps = _build_ts_maps(votes_df)
    device = _resolve_device(ns.device)
    cfg = Config(
        model_name=ns.model_name,
        image_size=ns.image_size,
        batch_size=ns.batch_size,
        lr_backbone=ns.lr_backbone,
        lr_head=ns.lr_head,
        epochs=ns.epochs,
        num_workers=ns.num_workers,
        amp=not ns.no_amp,
        lambda_bt=ns.lambda_bt,
        weight_decay=ns.weight_decay,
        freeze_backbone=ns.freeze_backbone,
        seed=ns.seed,
        images_root=str(ns.images_root.resolve()),
        output_dir=str(ns.output_dir.resolve()),
    )
    set_seed(cfg.seed)
    run_training(cfg, train_df, val_df, ts_maps, device)
    return 0


def _cmd_optuna(ns: argparse.Namespace) -> int:
    """
    Execute the ``optuna`` subcommand: TPE search over learning rates and ``lambda_bt``.

    :param ns: Parsed namespace from the ``optuna`` subparser (``--n-trials``, optional ``--storage``, etc.).
    :return: ``2`` if arguments are invalid; ``0`` after optimization completes.

    Example::
        In: _cmd_optuna(ns)  # after successful parse
        Out: 0
    """
    if ns.load_if_exists and not ns.storage:
        logger.error("--load-if-exists only applies when --storage is set.")
        return 2
    train_df = _load_pairs_csv(ns.train_csv)
    val_df = _load_pairs_csv(ns.val_csv)
    votes_df = _load_pairs_csv(ns.votes_csv)
    logger.info("Loaded rows: train=%d val=%d votes=%d", len(train_df), len(val_df), len(votes_df))

    ts_maps = _build_ts_maps(votes_df)
    device = _resolve_device(ns.device)
    base_cfg = Config(
        model_name=ns.model_name,
        image_size=ns.image_size,
        images_root=str(ns.images_root.resolve()),
        output_dir=str(ns.output_dir.resolve()),
        seed=ns.seed,
    )

    sampler = optuna.samplers.TPESampler(seed=ns.seed)
    # Persistent storage resumes studies; default is in-memory only (lost after exit).
    if ns.storage:
        study = optuna.create_study(
            study_name=ns.study_name or "uq_training_engine",
            storage=ns.storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=ns.load_if_exists,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run n_trials independent trainings; each trial suggests lr_* and lambda_bt then calls objective.
    study.optimize(
        lambda trial: objective(trial, train_df, val_df, ts_maps, device, base_cfg),
        n_trials=ns.n_trials,
        callbacks=[optuna_logging_callback],
    )
    logger.info("Optuna finished. Best value: %s | params: %s", study.best_value, study.best_params)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser with ``train`` and ``optuna`` subcommands.

    :return: Configured :class:`argparse.ArgumentParser`.

    Example::
        In: build_parser().parse_args(["train", "--help"])
        Out: help text for train (SystemExit in normal argparse flow)
    """
    parser = argparse.ArgumentParser(
        prog="uq-training-engine",
        description="Train or hyperparameter-search the urban quality preference model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Single training run with fixed hyperparameters.")
    _add_data_args(p_train)
    p_train.add_argument("--epochs", type=int, default=_CFG_DEFAULTS.epochs)
    p_train.add_argument("--batch-size", type=int, default=_CFG_DEFAULTS.batch_size)
    p_train.add_argument("--lr-backbone", type=float, default=_CFG_DEFAULTS.lr_backbone)
    p_train.add_argument("--lr-head", type=float, default=_CFG_DEFAULTS.lr_head)
    p_train.add_argument("--lambda-bt", type=float, default=_CFG_DEFAULTS.lambda_bt)
    p_train.add_argument("--weight-decay", type=float, default=_CFG_DEFAULTS.weight_decay)
    p_train.add_argument("--num-workers", type=int, default=_CFG_DEFAULTS.num_workers)
    p_train.add_argument("--freeze-backbone", action="store_true", help="Freeze ViT backbone.")
    p_train.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p_train.set_defaults(func=_cmd_train)

    p_opt = sub.add_parser("optuna", help="Optuna hyperparameter search (TPE).")
    _add_data_args(p_opt)
    p_opt.add_argument("--n-trials", type=int, required=True)
    p_opt.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (only used with --storage).",
    )
    p_opt.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db (default: in-memory).",
    )
    p_opt.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Resume or extend an existing study when using --storage.",
    )
    p_opt.set_defaults(func=_cmd_optuna)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point: configure logging, validate input paths, run the selected subcommand.

    :param argv: Argument list (default: ``sys.argv``).
    :return: Subcommand exit code, or ``2`` if a required path is missing.

    Example::
        In: main(["train", "--train-csv", "t.csv", "--val-csv", "v.csv", "--votes-csv", "votes.csv", "--images-root", "img", "--output-dir", "out"])
        Out: 0 or 2 depending on whether paths exist
    """
    args = build_parser().parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level))
    # Fail fast: require train/val/votes CSVs and images root; exit on first missing path.
    for label, path in (
        ("--train-csv", args.train_csv),
        ("--val-csv", args.val_csv),
        ("--votes-csv", args.votes_csv),
        ("--images-root", args.images_root),
    ):
        if not path.exists():
            logger.error("%s does not exist: %s", label, path)
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
