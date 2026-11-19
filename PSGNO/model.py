"""Unified PSGNO model.

Reference implementation of

    "PSGNO: A Physics-Informed Spectrally Gated Neural Operator
     for Noise-Robust PDE Forecasting"

Components
----------
LoRALinear                low-rank adapter, Eq. (15)
SpectralConv2d            PISG-gated Fourier layer, Eqs. (7), (9)-(11)
FNOEncoder                Fourier spatial encoder, Eq. (7)
UNetDenoiser              U-Net noise estimator, Eqs. (5)-(6)
SpatioTemporalTransformer causal temporal + spatial attention, Sec. 3.4
PSGNO                     cascaded denoiser + predictor

Tensor layout
-------------
All public models use channels-last fields of shape ``(B, T, H, W, C)``
and per-frame features of shape ``(BT, H, W, C)``.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import ins_residual


# ---------------------------------------------------------------------------
# Low-rank adapter
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adapter used during fine-tuning, Eq. (15).

    ``W = W_0 + A B`` with ``A in R^{D x r}`` and ``B in R^{r x D}``.
    ``B`` is initialized to zero so that the adapter contributes nothing
    at the beginning of fine-tuning.
    """

    def __init__(self, dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ B^T @ A^T == x @ (A B)^T
        return F.linear(F.linear(x, self.B), self.A) * self.scaling


# ---------------------------------------------------------------------------
# Spectral convolution with PISG
# ---------------------------------------------------------------------------
def _complex_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Complex multiplication of two real-valued ``(..., 2)`` tensors."""
    xr, xi = x[..., 0], x[..., 1]
    wr, wi = w[..., 0], w[..., 1]
    real = torch.einsum("bimn,iomn->bomn", xr, wr) - torch.einsum("bimn,iomn->bomn", xi, wi)
    imag = torch.einsum("bimn,iomn->bomn", xr, wi) + torch.einsum("bimn,iomn->bomn", xi, wr)
    return torch.stack([real, imag], dim=-1)


class SpectralConv2d(nn.Module):
    """Fourier layer with an optional Physics-Informed Spectral Gate.

    Parameters
    ----------
    in_channels, out_channels : int
        Feature channel counts of the spectral kernel.
    modes1, modes2 : int
        Number of retained low-frequency modes along each spatial axis.
    phys_proj : nn.Module, optional
        Maps the field used for residual evaluation to physical channels.
        ``nn.Identity`` for the first Fourier layer (the raw field),
        ``nn.Linear(width, C)`` for the deeper layers (``W_phys``).
        ``None`` disables PISG.
    residual_fn : callable, optional
        ``residual_fn(u) -> (..., H, W)`` or ``(..., H, W, R)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        phys_proj: Optional[nn.Module] = None,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.use_gate = phys_proj is not None
        self.phys_proj = phys_proj
        self.residual_fn = residual_fn or ins_residual

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2)
        )
        if self.use_gate:
            # Eq. (11): a single learnable scalar per Fourier layer.
            self.alpha = nn.Parameter(torch.tensor(0.1))

    # -- Eq. (9)-(11) -------------------------------------------------------
    def _gate(self, u: torch.Tensor, m1: int, m2: int, aux=None) -> torch.Tensor:
        """Per-frequency gate ``g(k)`` of shape ``(B, T, m1, m2)``.

        The residual is evaluated on the whole window ``(B, T, H, W, C)`` so
        that equations containing ``d/dt`` (all of Appendix A and B) can be
        differentiated; the resulting gate is then applied frame by frame,
        because the spectral convolution itself is per-frame (Sec. 3.4).
        """
        batch, steps, h, w, _ = u.shape
        phys = self.phys_proj(u.float())
        res = self.residual_fn(phys, aux)
        if res.dim() == phys.dim() - 1:
            res = res.unsqueeze(-1)

        res = res.reshape(batch * steps, h, w, -1).permute(0, 3, 1, 2).float()
        res_ft = torch.fft.rfft2(res, norm="ortho").abs()
        # |F(R)| for a vector-valued residual: L2 norm over its components.
        magnitude = torch.sqrt((res_ft ** 2).sum(dim=1))           # (BT, H, W//2+1)
        r = magnitude[:, :m1, :m2]                                 # Eq. (9)

        # Batch-level min-max over all batch elements and all time steps, Eq. (10).
        r_min = r.amin()
        r_max = r.amax()
        r_tilde = (r - r_min) / (r_max - r_min + 1e-6)

        gate = 1.0 / (1.0 + self.alpha.abs() * r_tilde)            # Eq. (11)
        return gate.reshape(batch, steps, m1, m2)

    def forward(self, z: torch.Tensor, u_for_gate: torch.Tensor, aux=None) -> torch.Tensor:
        batch, steps, h, w, _ = z.shape
        z = z.reshape(batch * steps, h, w, -1)
        z_ft = torch.fft.rfft2(z.permute(0, 3, 1, 2).float(), norm="ortho")
        z_ft = torch.view_as_real(z_ft)                            # (BT, C, H, W//2+1, 2)

        # A small grid may not support every configured mode.
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, z_ft.shape[-2])

        buf = torch.zeros(
            batch * steps, self.out_channels, h, w // 2 + 1, 2,
            device=z.device, dtype=torch.float32,
        )
        buf[:, :, :m1, :m2] = _complex_mul(
            z_ft[:, :, :m1, :m2], self.weight.float()[:, :, :m1, :m2]
        )
        if self.use_gate:
            gate = self._gate(u_for_gate, m1, m2, aux).reshape(batch * steps, m1, m2)
            buf[:, :, :m1, :m2] *= gate[:, None, :, :, None]

        out = torch.fft.irfft2(torch.view_as_complex(buf), s=(h, w), norm="ortho")
        out = out.permute(0, 2, 3, 1)
        return out.reshape(batch, steps, h, w, -1)


# ---------------------------------------------------------------------------
# Fourier encoder
# ---------------------------------------------------------------------------
class FNOEncoder(nn.Module):
    """FNO spatial encoder, Eq. (7). Input ``(B, T, H, W, C)`` -> ``(B, T, H, W, width)``.

    Each frame is processed independently, as stated in Sec. 3.4, but the
    sequence is kept intact in the leading axes so that PISG can evaluate
    residuals that involve ``d/dt``.
    """

    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_gate: bool = True,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.convs = nn.ModuleList()
        for n in range(n_layers):
            if not use_gate:
                phys_proj = None
            elif n == 0:
                phys_proj = nn.Identity()          # u_hat^(1) = u_hat
            else:
                phys_proj = nn.Linear(width, in_channels)   # W_phys^(n)
            self.convs.append(
                SpectralConv2d(width, width, modes1, modes2, phys_proj, residual_fn)
            )
        self.ws = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, width)

    @staticmethod
    def _grid(h: int, w: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([gy, gx], dim=-1)

    def forward(self, x: torch.Tensor, aux=None) -> torch.Tensor:
        batch, steps, h, w, _ = x.shape
        grid = self._grid(h, w, x.device).view(1, 1, h, w, 2).expand(batch, steps, -1, -1, -1)
        z = self.fc0(torch.cat([x, grid], dim=-1))
        for n, (conv, w_point) in enumerate(zip(self.convs, self.ws)):
            # Layer 1 gates on the physical field; deeper layers on the
            # output features of the preceding layer.
            u_for_gate = x if n == 0 else z
            z = F.gelu(conv(z, u_for_gate, aux) + w_point(z))
        return self.fc1(z)


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDenoiser(nn.Module):
    """U-Net denoiser, Eqs. (5)-(6).

    Input ``(N, C, H, W)`` noisy frame -> estimated noise ``(N, C, H, W)``.
    ``H`` and ``W`` must be divisible by 4.
    """

    def __init__(self, channels: int, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = _ConvBlock(channels, b)
        self.enc2 = _ConvBlock(b, 2 * b)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(2 * b, 4 * b)
        self.up1 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec1 = _ConvBlock(4 * b, 2 * b)
        self.up2 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec2 = _ConvBlock(2 * b, b)
        self.head = nn.Conv2d(b, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.bottleneck(self.pool(e2))
        d = self.dec1(torch.cat([self.up1(d), e2], dim=1))
        d = self.dec2(torch.cat([self.up2(d), e1], dim=1))
        return self.head(d)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Causal temporal self-attention applied independently at each position."""

    def __init__(self, dim: int, heads: int, max_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("mask", mask, persistent=False)
        self.lora_q: Optional[LoRALinear] = None
        self.lora_k: Optional[LoRALinear] = None
        self.lora_v: Optional[LoRALinear] = None

    def attach_lora(self, rank: int, alpha: float) -> None:
        if self.lora_q is not None:
            return
        self.lora_q = LoRALinear(self.dim, rank, alpha)
        self.lora_k = LoRALinear(self.dim, rank, alpha)
        self.lora_v = LoRALinear(self.dim, rank, alpha)

    def _project(self, x: torch.Tensor, linear: nn.Linear,
                 lora: Optional[LoRALinear]) -> torch.Tensor:
        out = linear(x)
        return out if lora is None else out + lora(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = x.shape
        flat = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, d)

        q = self._project(flat, self.Wq, self.lora_q)
        k = self._project(flat, self.Wk, self.lora_k)
        v = self._project(flat, self.Wv, self.lora_v)

        q = q.view(b * h * w, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b * h * w, t, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b * h * w, t, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(
            self.mask[:t, :t].unsqueeze(0).unsqueeze(0), float("-inf")
        )
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.permute(0, 2, 1, 3).reshape(b * h * w, t, d)
        out = self.drop(self.Wo(out))
        out = out.reshape(b, h, w, t, d).permute(0, 3, 1, 2, 4)
        return self.norm(x + out)


class SpatialAttention(nn.Module):
    """Spatial self-attention applied independently at each time step."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.lora_q: Optional[LoRALinear] = None
        self.lora_k: Optional[LoRALinear] = None
        self.lora_v: Optional[LoRALinear] = None

    def attach_lora(self, rank: int, alpha: float) -> None:
        if self.lora_q is not None:
            return
        self.lora_q = LoRALinear(self.dim, rank, alpha)
        self.lora_k = LoRALinear(self.dim, rank, alpha)
        self.lora_v = LoRALinear(self.dim, rank, alpha)

    def _project(self, x: torch.Tensor, linear: nn.Linear,
                 lora: Optional[LoRALinear]) -> torch.Tensor:
        out = linear(x)
        return out if lora is None else out + lora(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = x.shape
        flat = x.reshape(b * t, h * w, d)

        q = self._project(flat, self.Wq, self.lora_q)
        k = self._project(flat, self.Wk, self.lora_k)
        v = self._project(flat, self.Wv, self.lora_v)

        q = q.view(b * t, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b * t, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b * t, h * w, self.heads, self.head_dim).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b * t, h * w, d)
        out = self.drop(self.Wo(out))
        out = out.reshape(b, t, h, w, d)
        return self.norm(x + out)


class STTransformerBlock(nn.Module):
    """Temporal attention -> spatial attention -> feed-forward, each residual."""

    def __init__(self, dim: int, heads: int, max_len: int = 256,
                 dropout: float = 0.0, ffn_ratio: int = 4):
        super().__init__()
        self.temp_attn = TemporalAttention(dim, heads, max_len, dropout)
        self.spat_attn = SpatialAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temp_attn(x)
        x = self.spat_attn(x)
        return self.norm_ffn(x + self.ffn(x))


class SpatioTemporalTransformer(nn.Module):
    """Predictor head: encoded features ``(B,T,H,W,width)`` -> ``(B,H_out,H,W,C)``."""

    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        d_model: int = 64,
        st_layers: int = 2,
        heads: int = 4,
        horizon: int = 1,
        max_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.horizon = horizon
        self.proj_in = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [STTransformerBlock(d_model, heads, max_len, dropout) for _ in range(st_layers)]
        )
        self.proj_out = nn.Linear(d_model, out_channels * horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, _, h, w, _ = z.shape
        z = self.proj_in(z)
        for block in self.blocks:
            z = block(z)
        last = z[:, -1]                                     # (B, H, W, d)
        out = self.proj_out(last)                           # (B, H, W, C*H_out)
        out = out.reshape(b, h, w, self.horizon, self.out_channels)
        return out.permute(0, 3, 1, 2, 4)                   # (B, H_out, H, W, C)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class PSGNO(nn.Module):
    """Cascaded denoiser + FNO-Transformer predictor.

    Forward
    -------
    ``x_noisy`` ``(B, T_in, H, W, C)``
        -> ``x_den = x_noisy - D_phi(x_noisy)``
        -> ``pred = P_theta(x_den)`` ``(B, H_out, H, W, C)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        fno_layers: int = 4,
        d_model: int = 64,
        st_layers: int = 2,
        heads: int = 4,
        horizon: int = 1,
        unet_base: int = 32,
        max_len: int = 256,
        dropout: float = 0.0,
        use_pisg: bool = True,
        residual_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.horizon = horizon

        self.denoiser = UNetDenoiser(in_channels, unet_base)
        self.encoder = FNOEncoder(
            in_channels, width, modes1, modes2, fno_layers,
            residual_fn=residual_fn, use_gate=use_pisg,
        )
        self.predictor = SpatioTemporalTransformer(
            width, self.out_channels, d_model, st_layers, heads,
            horizon, max_len, dropout,
        )

    # -- inference ----------------------------------------------------------
    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, H, W, C)`` -> estimated noise ``(B, T, H, W, C)``."""
        b, t, h, w, c = x.shape
        x2d = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        eta = self.denoiser(x2d).permute(0, 2, 3, 1)
        return eta.reshape(b, t, h, w, c)

    def predict(self, u: torch.Tensor, aux=None) -> torch.Tensor:
        """Denoised history ``(B, T, H, W, C)`` -> ``(B, H_out, H, W, C)``."""
        return self.predictor(self.encoder(u, aux))

    def forward(self, x_noisy: torch.Tensor, aux=None):
        x_den = x_noisy - self.denoise(x_noisy)
        return self.predict(x_den, aux), x_den

    # -- fine-tuning --------------------------------------------------------
    def enable_finetune(self, lora_rank: int = 4, lora_alpha: float = 1.0) -> int:
        """Freeze the backbone and expose the three trainable groups.

        Trainable: PISG gating strengths ``alpha^(n)``, physical projections
        ``W_phys^(n)``, and LoRA adapters on the Transformer Q/K/V projections.
        """
        for param in self.parameters():
            param.requires_grad_(False)

        for block in self.predictor.blocks:
            block.temp_attn.attach_lora(lora_rank, lora_alpha)
            block.spat_attn.attach_lora(lora_rank, lora_alpha)

        for conv in self.encoder.convs:
            if not conv.use_gate:
                continue
            conv.alpha.requires_grad_(True)
            if isinstance(conv.phys_proj, nn.Linear):
                for param in conv.phys_proj.parameters():
                    param.requires_grad_(True)

        return self.count_trainable()

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
