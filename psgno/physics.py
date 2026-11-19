"""Governing-equation residuals used by PISG (Sec. 3.5) and the loss L_p (Eq. 18).

Every operator accepts a spatiotemporal field of shape ``(B, T, H, W, C)`` and
returns the residual ``(B, T, H, W, R)``.  All equations implemented here
contain a time derivative (Eqs. 19, 21, 22, 23 and 25), hence ``T >= 2`` is
required; the predicted horizon ``H = 3`` satisfies this.

Spatial derivatives are second-order central differences with one-sided stencils
at the boundary.  The time derivative uses a forward difference inside the
window and a backward difference for the last frame, so that every frame
receives its own residual and therefore its own PISG gate.

Residuals follow Appendices A and B:

===============  ==========================================================
``swe``          2D shallow water, Eqs. (19), variables ``(h, u, v)``
``dr``           FitzHugh--Nagumo diffusion--reaction, Eq. (21), ``(u, v)``
``ins``          incompressible Navier--Stokes, Eq. (22): vorticity
                 transport + continuity, ``(u_x, u_y)``
``cns``          compressible Navier--Stokes, Eq. (23): mass, momentum,
                 energy, ``(rho, v_x, v_y, p)``
``cfdbench``     pressure-free vorticity transport, Eq. (25) + continuity;
                 ``two_phase=True`` drops the interfacial band for the
                 two-phase tube and dam cases, as in Appendix B.1
===============  ==========================================================
"""

from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

ResidualFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Finite-difference operators on ``(..., T, H, W)``
# ---------------------------------------------------------------------------
def _ddx(field: torch.Tensor) -> torch.Tensor:
    """First derivative along ``x`` (last axis)."""
    grad = torch.zeros_like(field)
    grad[..., 1:-1] = (field[..., 2:] - field[..., :-2]) * 0.5
    grad[..., 0] = field[..., 1] - field[..., 0]
    grad[..., -1] = field[..., -1] - field[..., -2]
    return grad


def _ddy(field: torch.Tensor) -> torch.Tensor:
    """First derivative along ``y`` (second-to-last axis)."""
    grad = torch.zeros_like(field)
    grad[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) * 0.5
    grad[..., 0, :] = field[..., 1, :] - field[..., 0, :]
    grad[..., -1, :] = field[..., -1, :] - field[..., -2, :]
    return grad


def _ddt(field: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """First derivative along time (axis ``-3``)."""
    if field.shape[-3] < 2:
        raise ValueError(
            "time derivatives need at least two frames, got T="
            f"{field.shape[-3]}"
        )
    grad = torch.empty_like(field)
    grad[..., :-1, :, :] = (field[..., 1:, :, :] - field[..., :-1, :, :]) / dt
    grad[..., -1, :, :] = (field[..., -1, :, :] - field[..., -2, :, :]) / dt
    return grad


def _laplacian(field: torch.Tensor) -> torch.Tensor:
    return _ddx(_ddx(field)) + _ddy(_ddy(field))


# ---------------------------------------------------------------------------
# Shallow water, Eq. (19)
# ---------------------------------------------------------------------------
def swe_residual(
    field: torch.Tensor,
    aux=None,
    gr: float = 1.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Continuity and the two momentum equations of the shallow-water system.

    ``field[..., 0:3] = (h, u, v)``.  The bathymetry ``b`` stored with the data
    is read from ``aux["bathymetry"]`` as an ``(H, W)`` profile; it vanishes
    for the flat-bed radial dam-break setup.
    """
    bathymetry = (aux or {}).get("bathymetry")
    h, u, v = field[..., 0], field[..., 1], field[..., 2]
    hu, hv = h * u, h * v

    r_mass = _ddt(h, dt) + _ddx(hu) + _ddy(hv)
    r_mom_x = _ddt(hu, dt) + _ddx(u * u * h + 0.5 * gr * h * h) + _ddy(u * v * h)
    r_mom_y = _ddt(hv, dt) + _ddy(v * v * h + 0.5 * gr * h * h) + _ddx(u * v * h)

    if bathymetry is not None:
        r_mom_x = r_mom_x + gr * h * _ddx(bathymetry)
        r_mom_y = r_mom_y + gr * h * _ddy(bathymetry)

    return torch.stack([r_mass, r_mom_x, r_mom_y], dim=-1)


# ---------------------------------------------------------------------------
# Diffusion--reaction, Eq. (21)
# ---------------------------------------------------------------------------
def dr_residual(
    field: torch.Tensor,
    aux=None,
    du: float = 1e-3,
    dv: float = 5e-3,
    k: float = 5e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    """FitzHugh--Nagumo activator/inhibitor residual ``(R_u, R_v)``."""
    u, v = field[..., 0], field[..., 1]
    r_u = _ddt(u, dt) - du * _laplacian(u) - (u - u ** 3 - k - v)
    r_v = _ddt(v, dt) - dv * _laplacian(v) - (u - v)
    return torch.stack([r_u, r_v], dim=-1)


# ---------------------------------------------------------------------------
# Incompressible Navier--Stokes, Eq. (22)
# ---------------------------------------------------------------------------
def vorticity(field: torch.Tensor) -> torch.Tensor:
    """``omega = d_x u_y - d_y u_x``."""
    return _ddx(field[..., 1]) - _ddy(field[..., 0])


def ins_residual(
    field: torch.Tensor,
    aux=None,
    nu: float = 0.01,
    dt: float = 1.0,
    density: float = 1.0,
) -> torch.Tensor:
    """Pressure-free vorticity transport combined with continuity.

    Taking the curl of the momentum equation eliminates ``grad p``, leaving
    ``d_t w + u . grad w - nu lap w - rho^-1 (curl f)_z = 0``; the continuity
    residual ``div u = 0`` is appended.  The forcing stored with the data is
    read from ``aux["forcing"]`` as a ``(B, T, H, W, 2)`` field.
    """
    forcing = (aux or {}).get("forcing")
    if forcing is None:
        raise ValueError(
            "ins_residual needs the body force stored with the dataset; "
            "pass it as aux={'forcing': f}."
        )
    if forcing.dim() == field.dim() and forcing.shape[-4] != field.shape[-4]:
        # The forcing is constant in time, so one frame broadcasts over the
        # whole window (the horizon H may differ from T_in).
        forcing = forcing[..., :1, :, :, :]
    u, v = field[..., 0], field[..., 1]
    omega = vorticity(field)

    r_omega = _ddt(omega, dt) + u * _ddx(omega) + v * _ddy(omega)
    r_omega = r_omega - nu * _laplacian(omega)
    curl_f = _ddx(forcing[..., 1]) - _ddy(forcing[..., 0])
    if curl_f.dim() == omega.dim() - 1:
        # A per-trajectory force without a time axis broadcasts over the window.
        curl_f = curl_f.unsqueeze(-3)
    r_omega = r_omega - curl_f / density

    r_cont = _ddx(u) + _ddy(v)
    return torch.stack([r_omega, r_cont], dim=-1)


# ---------------------------------------------------------------------------
# Compressible Navier--Stokes, Eq. (23)
# ---------------------------------------------------------------------------
def cns_residual(
    field: torch.Tensor,
    aux=None,
    gamma: float = 5.0 / 3.0,
    eta: float = 1e-4,
    zeta: float = 1e-4,
    dt: float = 1.0,
) -> torch.Tensor:
    """Mass, momentum and energy residuals.

    ``field[..., 0:4] = (rho, v_x, v_y, p)``; ``eta`` and ``zeta`` are the
    shear and bulk viscosities.
    """
    rho, vx, vy, p = field[..., 0], field[..., 1], field[..., 2], field[..., 3]
    div = _ddx(vx) + _ddy(vy)

    # Mass conservation.
    r_mass = _ddt(rho, dt) + _ddx(rho * vx) + _ddy(rho * vy)

    # Momentum balance, non-conservative form of Eq. (23).
    acc_x = _ddt(vx, dt) + vx * _ddx(vx) + vy * _ddy(vx)
    acc_y = _ddt(vy, dt) + vx * _ddx(vy) + vy * _ddy(vy)
    visc_x = eta * _laplacian(vx) + (zeta + eta / 3.0) * _ddx(div)
    visc_y = eta * _laplacian(vy) + (zeta + eta / 3.0) * _ddy(div)
    r_mom_x = rho * acc_x + _ddx(p) - visc_x
    r_mom_y = rho * acc_y + _ddy(p) - visc_y

    # Energy conservation.
    energy = p / (gamma - 1.0) + 0.5 * rho * (vx ** 2 + vy ** 2)
    s_xx = 2.0 * eta * _ddx(vx) + (zeta - 2.0 * eta / 3.0) * div
    s_yy = 2.0 * eta * _ddy(vy) + (zeta - 2.0 * eta / 3.0) * div
    s_xy = eta * (_ddx(vy) + _ddy(vx))
    work_x = vx * s_xx + vy * s_xy
    work_y = vx * s_xy + vy * s_yy
    flux_x = (energy + p) * vx - work_x
    flux_y = (energy + p) * vy - work_y
    r_energy = _ddt(energy, dt) + _ddx(flux_x) + _ddy(flux_y)

    return torch.stack([r_mass, r_mom_x, r_mom_y, r_energy], dim=-1)


# ---------------------------------------------------------------------------
# CFDBench, Eq. (25)
# ---------------------------------------------------------------------------
def interface_band(
    field: torch.Tensor, quantile: float = 0.95, dilate: int = 2
) -> torch.Tensor:
    """Locate the phase interface from the velocity field.

    The released CFDBench arrays carry only ``(u, v)`` and no phase indicator,
    so the interface is inferred from the local speed ``|u|``: a phase jump
    produces a much larger speed gradient than the smooth single-phase motion.
    Cells whose speed-gradient magnitude lies in the top ``1 - quantile``
    fraction of the sample, dilated by ``dilate`` cells, are treated as the
    interfacial band.

    Returns a boolean tensor of shape ``(B, T, H, W)`` that is ``True`` on the
    interface.
    """
    speed = torch.sqrt(field[..., 0] ** 2 + field[..., 1] ** 2 + 1e-12)
    gx, gy = _ddx(speed), _ddy(speed)
    grad = torch.sqrt(gx ** 2 + gy ** 2)
    threshold = torch.quantile(
        grad.reshape(grad.shape[0], -1), quantile, dim=1
    ).reshape(-1, 1, 1, 1)
    band = grad >= threshold

    if dilate > 0:
        kernel = 2 * dilate + 1
        band = F.max_pool2d(
            band.float().reshape(-1, 1, *band.shape[-2:]),
            kernel, stride=1, padding=dilate,
        ).reshape(band.shape) > 0
    return band


def cfdbench_residual(
    field: torch.Tensor,
    aux=None,
    nu: float = 0.01,
    dt: float = 1.0,
    two_phase: bool = False,
    interface_quantile: float = 0.95,
    interface_dilate: int = 2,
) -> torch.Tensor:
    """Vorticity transport plus continuity, for a pressure-free dataset.

    For the two-phase tube and dam cases the baroclinic torque and the surface
    tension of Eq. (24) are not present in the released vorticity form, so the
    residual is evaluated only over the single-phase regions away from the
    interface (Appendix B.1).  Setting ``two_phase=True`` removes the
    interfacial band returned by :func:`interface_band`; alternatively a mask
    stored with the data may be supplied as ``aux["mask"]`` with shape
    ``(B, T, H, W, 1)`` or ``(H, W, 1)``.
    """
    u, v = field[..., 0], field[..., 1]
    omega = vorticity(field)
    r_omega = _ddt(omega, dt) + u * _ddx(omega) + v * _ddy(omega)
    r_omega = r_omega - nu * _laplacian(omega)
    r_cont = _ddx(u) + _ddy(v)

    residual = torch.stack([r_omega, r_cont], dim=-1)

    mask = (aux or {}).get("mask")
    if mask is not None and mask.dim() == field.dim() and mask.shape[-4] != field.shape[-4]:
        mask = mask[..., :1, :, :, :]
    if mask is None and two_phase:
        mask = (~interface_band(field, interface_quantile, interface_dilate))
        mask = mask.unsqueeze(-1).to(residual.dtype)
    if mask is not None:
        residual = residual * mask
    return residual


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PDE_RESIDUALS: Dict[str, ResidualFn] = {
    "swe": swe_residual,
    "dr": dr_residual,
    "ins": ins_residual,
    "cns": cns_residual,
    "cfdbench": cfdbench_residual,
}

# Older name kept so that existing configs keep working.
PDE_RESIDUALS["incompressible"] = ins_residual


def register(name: str, fn: ResidualFn) -> None:
    """Register a custom residual operator under ``name``."""
    PDE_RESIDUALS[name] = fn


def get_residual(name: str, **params) -> ResidualFn:
    """Look up a residual operator, optionally binding equation parameters.

    Example
    -------
    >>> get_residual("ins", nu=5e-3)
    """
    if name not in PDE_RESIDUALS:
        raise KeyError(
            f"Unknown residual '{name}'. Available: {sorted(PDE_RESIDUALS)}. "
            "Use psgno.physics.register to add a custom operator."
        )
    fn = PDE_RESIDUALS[name]
    return partial(fn, **params) if params else fn
