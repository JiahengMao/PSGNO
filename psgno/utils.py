"""Shared helpers: configuration, seeding, checkpoints and metrics."""

import random
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .model import PSGNO
from .physics import get_residual


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_model(config: Dict[str, Any]) -> PSGNO:
    model_cfg = dict(config["model"])
    residual_name = model_cfg.pop("residual", "ins")
    residual_params = model_cfg.pop("residual_params", None) or {}
    return PSGNO(
        residual_fn=get_residual(residual_name, **residual_params), **model_cfg
    )


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error, averaged over the batch."""
    diff = (pred - target).reshape(pred.shape[0], -1).norm(dim=1)
    ref = target.reshape(target.shape[0], -1).norm(dim=1)
    return (diff / (ref + 1e-8)).mean()


def save_checkpoint(path: str, model: PSGNO, optimizer=None, **extra) -> None:
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    payload.update(extra)
    torch.save(payload, path)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )
