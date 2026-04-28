from dataclasses import dataclass


@dataclass
class Config:
    """
    Hyperparameters, paths, and toggles for one training or Optuna trial.

    :ivar model_name: timm backbone identifier.
    :ivar image_size: Input resize / crop side length in pixels.
    :ivar batch_size: DataLoader batch size.
    :ivar lr_backbone: AdamW learning rate for trainable backbone parameters.
    :ivar lr_head: AdamW learning rate for the MLP head.
    :ivar epochs: Number of training epochs.
    :ivar num_workers: DataLoader worker processes.
    :ivar amp: Whether to use automatic mixed precision (CUDA).
    :ivar lambda_bt: Weight for pairwise ranking loss vs. auxiliary Smooth L1 (``1 - lambda_bt`` on aux).
    :ivar weight_decay: AdamW weight decay.
    :ivar freeze_backbone: If True, backbone parameters are frozen.
    :ivar seed: RNG seed (also used where Optuna passes base config).
    :ivar images_root: Root directory for training images (string path).
    :ivar output_dir: Base directory for checkpoints and calibration output (string path).

    Example::
        In: Config(model_name="vit_base_patch14_dinov2.lvd142m", images_root="/data/img", output_dir="/out")
        Out: Config instance with defaults for lr, epochs, batch_size, etc.
    """

    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    image_size: int = 294
    batch_size: int = 32
    lr_backbone: float = 2e-6
    lr_head: float = 5e-5
    epochs: int = 2
    num_workers: int = 2
    amp: bool = True
    lambda_bt: float = 0.6
    weight_decay: float = 0.1
    freeze_backbone: bool = False
    seed: int = 42
    images_root: str = ""
    output_dir: str = ""
