"""Dataset utilities: normalization, sliding windows and domain splits.

A trajectory array is expected to have shape ``(N, T, H, W, C)``.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def window_count(length: int, t_in: int, horizon: int, stride: int = 1) -> int:
    """Number of valid windows in one trajectory.

    A window consists of ``t_in`` input frames followed by ``horizon`` target
    frames, both fully inside the trajectory:
        n = floor((T - t_in - horizon) / stride) + 1, clipped at zero.
    """
    return max(0, (length - t_in - horizon) // stride + 1)


class Normalizer:
    """Per-channel standardization computed over real frames only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, data: np.ndarray) -> "Normalizer":
        flat = data.reshape(-1, data.shape[-1])
        return cls(flat.mean(axis=0), flat.std(axis=0) + 1e-8)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "Normalizer":
        return cls(state["mean"], state["std"])


class SlidingWindowDataset(Dataset):
    """Sliding windows ``(x, y)`` from a trajectory array.

    ``x`` has shape ``(t_in, H, W, C)`` and ``y`` has shape
    ``(horizon, H, W, C)``.  Extra physical quantities stored with the data,
    such as the body force of the incompressible case, are passed as ``aux``
    and returned unchanged alongside the window.
    """

    def __init__(
        self,
        data: np.ndarray,
        t_in: int,
        horizon: int,
        stride: int = 1,
        aux: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.data = data
        self.t_in = t_in
        self.horizon = horizon
        self.aux = aux or {}
        self.index = []
        for traj in range(data.shape[0]):
            n = window_count(data.shape[1], t_in, horizon, stride)
            self.index.extend((traj, start * stride) for start in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        traj, start = self.index[idx]
        x = self.data[traj, start : start + self.t_in]
        y = self.data[traj, start + self.t_in : start + self.t_in + self.horizon]
        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).float()
        if not self.aux:
            return x, y
        aux = {}
        for key, value in self.aux.items():
            if value.ndim == 5:                     # (N, T, H, W, C)
                tile = value[traj, start : start + self.t_in]
            elif value.ndim == 4:                   # (N, H, W, C), constant in time
                tile = np.repeat(value[traj][None], self.t_in, axis=0)
            else:
                raise ValueError(
                    f"auxiliary '{key}' must be (N, T, H, W, C) or (N, H, W, C), "
                    f"got shape {value.shape}"
                )
            aux[key] = torch.from_numpy(tile.copy()).float()
        return x, y, aux


def split_trajectories(
    data: np.ndarray, source_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic source/target split along the trajectory axis.

    The two domains are strictly disjoint; the split point is
    ``round(N * source_ratio)``.
    """
    cut = int(round(data.shape[0] * source_ratio))
    return data[:cut], data[cut:]


def split_train_val_test(
    n: int, ratios: Sequence[float] = (0.8, 0.1, 0.1)
) -> Tuple[slice, slice, slice]:
    """Contiguous 8:1:1 split along the first axis."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n),
    )


def load_auxiliary(path: str, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load the extra physical quantities stored next to the trajectories.

    Equations that contain an external term need it at residual-evaluation
    time.  The incompressible case (Appendix A.3) stores the body force ``f``
    with the dataset, because it is re-sampled per trajectory; the shallow
    water case stores the bathymetry ``b``.  Both live in the same ``.npz``
    archive as the trajectories, either as ``(N, T, H, W, C)`` or as
    ``(N, H, W, C)`` when the quantity is constant in time.
    """
    if not keys:
        return {}
    if not path.endswith(".npz"):
        raise ValueError(
            f"auxiliary keys {list(keys)} need an .npz archive, got '{path}'"
        )
    archive = np.load(path)
    missing = [key for key in keys if key not in archive]
    if missing:
        raise KeyError(
            f"keys {missing} not found in {path}; available: {list(archive.keys())}"
        )
    return {key: archive[key].astype(np.float32) for key in keys}


def load_trajectories(path: str) -> np.ndarray:
    """Load a ``(N, T, H, W, C)`` array from ``.npy`` or an ``.npz`` archive."""
    if path.endswith(".npz"):
        archive = np.load(path)
        key = "data" if "data" in archive else list(archive.keys())[0]
        return archive[key].astype(np.float32)
    return np.load(path).astype(np.float32)
