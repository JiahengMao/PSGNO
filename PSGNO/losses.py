"""Training objectives of the two stages, Sec. 3.6 of the paper.

All losses follow the notation of the paper: ``|| . ||^2`` is the squared
Frobenius norm of one sample, and ``E[.]`` is the average over the batch.
"""

from typing import Callable, Optional, Tuple

import torch


def _sq_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample squared norm ``|| x_i ||^2``, shape ``(B,)``."""
    return x.reshape(x.shape[0], -1).pow(2).sum(dim=1)


def denoising_loss(eta_pred: torch.Tensor, eta_true: torch.Tensor) -> torch.Tensor:
    """Eq. (12): ``E || D_phi(u~) - eta ||^2``."""
    return _sq_norm(eta_pred - eta_true).mean()


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. (13): ``E sum_delta || u_{t+delta} - P^(delta)(u^) ||^2``."""
    return _sq_norm(pred - target).mean()


def physics_loss(
    residual_fn: Callable[..., torch.Tensor], pred: torch.Tensor, aux=None
) -> torch.Tensor:
    """Eq. (18): ``E || R(u^_{t+1:t+H}) ||^2``."""
    return _sq_norm(residual_fn(pred, aux)).mean()


def _roll(
    predict_fn: Callable[..., torch.Tensor],
    window: torch.Tensor,
    steps: int,
    aux=None,
) -> torch.Tensor:
    """Roll the predictor forward ``steps`` frames and return that frame.

    The predictor emits ``H`` frames per pass, so a rollout of ``steps``
    frames needs ``ceil(steps / H)`` passes, i.e. at most two passes when
    ``H >= 2`` and ``steps <= 2 H``, as stated in Sec. 3.6.
    """
    current = window
    remaining = steps
    out = None
    while remaining > 0:
        pred = predict_fn(current, aux)
        hop = pred.shape[1]
        if remaining <= hop:
            out = pred[:, remaining - 1 : remaining]
            break
        current = torch.cat([current[:, hop:], pred], dim=1)
        remaining -= hop
    return out


def consistency_loss(
    predict_fn: Callable[..., torch.Tensor],
    history: torch.Tensor,
    t_in: int,
    horizon: int = 1,
    h_c: Optional[int] = None,
    aux=None,
) -> Tuple[torch.Tensor, int]:
    """Eq. (17): multi-path autoregressive consistency.

    For every target step ``t + h`` and every pair ``i < j < h``, the target
    frame is reached from the history window ending at ``t - h + i`` and at
    ``t - h + j``; the two predictions of that frame are penalized to agree.

    Parameters
    ----------
    history : ``(B, T_buf, H, W, C)``
        Buffer long enough to host every sub-window, i.e. ``T_buf >= t_in``
        (recommended ``T_buf = t_in + h_c``).
    t_in : int
        Input window length.
    horizon : int
        Per-pass output length ``H``.
    h_c : int, optional
        Consistency horizon; defaults to ``min(t_in - 1, 5)`` for ``H = 1``
        and to ``H`` otherwise, as in Eq. (17).

    Returns
    -------
    (loss, number_of_passes)
    """
    buffer_len = history.shape[1]
    if h_c is None:
        h_c = min(t_in - 1, 5) if horizon == 1 else horizon

    total = history.new_zeros(history.shape[0])
    passes = 0

    for h in range(2, h_c + 1):
        predictions = {}
        for i in range(h):
            end = buffer_len - 1 - h + i
            start = end - t_in + 1
            if start < 0:
                continue
            window = history[:, start : end + 1]
            predictions[i] = _roll(predict_fn, window, steps=2 * h - i, aux=aux)
            passes += 1
        keys = sorted(predictions)
        for a_idx, i in enumerate(keys):
            for j in keys[a_idx + 1 :]:
                total = total + _sq_norm(predictions[i] - predictions[j])

    return total.mean(), passes
