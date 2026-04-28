import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set Python, NumPy, and PyTorch (CPU/CUDA) seeds and enable deterministic cuDNN.

    :param seed: Integer seed (default 42).
    :return: None.

    Example::
        In: set_seed(123)
        Out: random/numpy/torch state fixed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducible convs on GPU; can reduce throughput vs. benchmark kernels.
    torch.backends.cudnn.deterministic = True
