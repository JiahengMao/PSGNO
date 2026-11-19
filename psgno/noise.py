"""Structured measurement-noise model, Sec. 3.3 of the paper.

    eta = lambda * I(x) * eta_t(x)                                   (Eq. 1)

with a Gaussian random field carrying spatial and temporal correlation,
a region-adaptive intensity field and a multi-scale amplitude.
"""

from typing import Tuple

import numpy as np
import torch


def _spatial_grf(
    batch: int,
    steps: int,
    channels: int,
    height: int,
    width: int,
    length_low: float,
    length_high: float,
    device: torch.device,
) -> torch.Tensor:
    """Spatially correlated random field, Eq. (2)."""
    lengths = np.random.uniform(length_low / height, length_high / height, size=batch)
    length_t = torch.as_tensor(lengths, dtype=torch.float32, device=device).view(batch, 1, 1)

    fh = torch.fft.fftfreq(height, d=1.0 / height, device=device)
    fw = torch.fft.fftfreq(width, d=1.0 / width, device=device)
    kh, kw = torch.meshgrid(fh, fw, indexing="ij")
    k_sq = kh ** 2 + kw ** 2

    # Gaussian smoothing kernel in the spectral domain.
    filt = torch.exp(-(np.pi ** 2) * (length_t ** 2) * k_sq.unsqueeze(0))  # (B, H, W)

    white = torch.randn(batch, steps, channels, height, width, device=device)
    white = white.reshape(batch * steps * channels, height, width)
    filt = filt.unsqueeze(1).unsqueeze(1).expand(-1, steps, channels, -1, -1)
    filt = filt.reshape(batch * steps * channels, height, width)

    field = torch.fft.ifft2(torch.fft.fft2(white) * filt).real
    return field.reshape(batch, steps, channels, height, width)


def _ar1(field: torch.Tensor, rho: float) -> torch.Tensor:
    """First-order autoregressive correlation along time, Eq. (3)."""
    out = torch.empty_like(field)
    out[:, 0] = field[:, 0]
    for t in range(1, field.shape[1]):
        out[:, t] = rho * out[:, t - 1] + np.sqrt(1.0 - rho ** 2) * field[:, t]
    return out


def _intensity(clean: torch.Tensor, gamma: float) -> torch.Tensor:
    """Region-adaptive intensity field ``I(x) = 1 + gamma |grad u| / max|grad u|``, Eq. (4).

    The gradient magnitude is the Frobenius norm over all state channels,
    so the operator also applies to scalar fields (``C = 1``).
    """
    grad_sq = torch.zeros_like(clean[..., 0])
    for channel in range(clean.shape[-1]):
        field = clean[..., channel]

        d_dx = torch.zeros_like(field)
        d_dy = torch.zeros_like(field)
        d_dx[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2.0
        d_dy[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / 2.0

        grad_sq = grad_sq + d_dx ** 2 + d_dy ** 2

    grad_mag = torch.sqrt(grad_sq)
    grad_mean = grad_mag.mean(dim=1)
    grad_max = grad_mean.amax(dim=(1, 2), keepdim=True)

    intensity = 1.0 + gamma * grad_mean / (grad_max + 1e-8)
    return intensity.unsqueeze(1).unsqueeze(-1)                     # (B, 1, H, W, 1)


def structured_noise(
    clean: torch.Tensor,
    sigma_mean: float = -2.5,
    sigma_std: float = 0.5,
    length_low: float = 2.0,
    length_high: float = 16.0,
    gamma: float = 2.0,
    rho: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inject structured noise into a clean batch.

    Parameters
    ----------
    clean : ``(B, T, H, W, C)``
        Clean trajectories; the last axis holds the state channels.

    Returns
    -------
    noisy, noise, intensity
        ``noisy`` and ``noise`` are ``(B, T, H, W, C)``;
        ``intensity`` is ``(B, 1, H, W, 1)``.
    """
    if clean.dim() != 5:
        raise ValueError("clean must be (B, T, H, W, C)")

    batch, steps, height, width, channels = clean.shape
    device = clean.device

    # Eq. (2): spatially correlated field.
    eta_s = _spatial_grf(
        batch, steps, channels, height, width, length_low, length_high, device
    )
    eta_s = eta_s.permute(0, 1, 3, 4, 2)                            # (B, T, H, W, C)

    # Eq. (3): temporal correlation, normalized per trajectory.
    eta_t = _ar1(eta_s, rho)
    eta_t = eta_t / (eta_t.reshape(batch, -1).std(dim=1).view(batch, 1, 1, 1, 1) + 1e-8)

    # Eq. (4): region-adaptive intensity.
    intensity = _intensity(clean, gamma)

    # Multi-scale amplitude lambda ~ LogNormal(mu, s^2).
    sigma = torch.exp(
        torch.randn(batch, 1, 1, 1, 1, device=device) * sigma_std + sigma_mean
    )

    noise = sigma * intensity * eta_t
    return clean + noise, noise, intensity


def white_noise(clean: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    """Homogeneous Gaussian noise, used for the ablation in Sec. 4.6."""
    noise = sigma * torch.randn_like(clean)
    return clean + noise, noise, torch.ones_like(clean[..., :1])
