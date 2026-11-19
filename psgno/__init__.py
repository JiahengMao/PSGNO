"""PSGNO: Physics-Informed Spectrally Gated Neural Operator."""

from .data_utils import (
    Normalizer,
    SlidingWindowDataset,
    load_trajectories,
    split_trajectories,
    split_train_val_test,
    window_count,
)
from .losses import (
    consistency_loss,
    denoising_loss,
    physics_loss,
    prediction_loss,
)
from .model import PSGNO, LoRALinear
from .noise import structured_noise, white_noise
from .physics import (
    cfdbench_residual,
    cns_residual,
    dr_residual,
    get_residual,
    ins_residual,
    register,
    swe_residual,
)

__all__ = [
    "PSGNO",
    "LoRALinear",
    "Normalizer",
    "SlidingWindowDataset",
    "load_trajectories",
    "split_trajectories",
    "split_train_val_test",
    "window_count",
    "structured_noise",
    "white_noise",
    "swe_residual",
    "dr_residual",
    "ins_residual",
    "cns_residual",
    "cfdbench_residual",
    "get_residual",
    "register",
    "denoising_loss",
    "prediction_loss",
    "physics_loss",
    "consistency_loss",
]
