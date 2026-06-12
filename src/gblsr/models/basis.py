"""Fixed patch grid, truncated local Fourier basis (cosine fallback), and adaptivity head.

Convention for local patch coordinates
--------------------------------------
Each patch has local coordinates (u, v) in [-1, 1]^2 evaluated on a P x P grid.
Basis is separable (tensor product of 1D modes). Modes per axis are indexed
k = 0, 1, ..., p_max - 1.

Truncated local Fourier basis (primary):
    phi_0(u) = 1
    phi_{2j-1}(u) = cos(j * pi * u * s_e)    for j = 1, 2, ...
    phi_{2j}(u)   = sin(j * pi * u * s_e)    for j = 1, 2, ...
filled in (cos, sin) order until exactly p_max modes per axis exist;
for even p_max the last mode is an unpaired cos at j = p_max / 2
(p_max = 16: constant, cos/sin pairs j = 1..7, then cos(8 pi u s_e)),
where s_e is a per-patch learnable bandwidth scalar (shared across u, v axes
for simplicity). The 2D basis is the outer product phi_i(u) * phi_j(v).

Local cosine basis (fallback):
    phi_k(u) = cos(k * pi * s_e * (u + 1) / 2), k = 0, ..., p_max - 1
    (standard DCT-II-like, separable; obtained by remapping u in [-1, 1]
    to [0, 1] via (u + 1)/2 before the cosine, with bandwidth s_e
    multiplying the angular argument)

Adaptivity
----------
Per patch we emit:
    s_e        : learnable bandwidth (log-space, bounded via sigmoid)
    p_soft     : continuous effective order in [0, p_max], reported at
                 discrete buckets {4, 8, 16} for visualization. Implemented
                 as a smooth cutoff on modes:
                     w(k_i, k_j) = sigmoid((p_soft - max(k_i, k_j)) * sharpness)
                 Mode indices satisfy k_i, k_j in {0, ..., p_max - 1}, so
                 max(k_i, k_j) = max(|k_i|, |k_j|).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class BasisConfig:
    patch_size: int
    p_max: int = 16  # maximum modes per axis
    family: Literal["fourier", "cosine"] = "fourier"
    s_e_range: tuple[float, float] = (0.25, 2.0)
    p_soft_range: tuple[float, float] = (1.0, 16.0)
    cutoff_sharpness: float = 4.0


def make_patch_coords(patch_size: int, device: torch.device | None = None) -> torch.Tensor:
    """Return (P, P, 2) grid of local patch coordinates in [-1, 1]^2."""
    P = patch_size
    u = torch.linspace(-1.0, 1.0, P, device=device)
    v = torch.linspace(-1.0, 1.0, P, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack([uu, vv], dim=-1)


def image_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, nH, nW, C, P, P).  Non-overlapping."""
    B, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"image {H}x{W} not divisible by patch {P}"
    nH, nW = H // P, W // P
    x = x.reshape(B, C, nH, P, nW, P)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous()


def patches_to_image(p: torch.Tensor) -> torch.Tensor:
    """(B, nH, nW, C, P, P) -> (B, C, H, W)."""
    B, nH, nW, C, P, _ = p.shape
    return p.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, C, nH * P, nW * P)


def build_1d_basis(
    coord: torch.Tensor,
    p_max: int,
    s_e: torch.Tensor,
    family: str,
) -> torch.Tensor:
    """Evaluate 1D basis at a 1D coord grid.

    coord: (P,) coordinates in [-1, 1].
    s_e  : (..., 1) per-patch bandwidth, broadcast.
    returns: (..., P, p_max) basis values.
    """
    P = coord.shape[0]
    coord = coord.view(*((1,) * (s_e.ndim - 1)), P)
    s_e_b = s_e
    if family == "fourier":
        ang0 = 0.0 * coord * s_e_b
        basis = [torch.ones_like(ang0)]
        j = 1
        while len(basis) < p_max:
            ang = j * math.pi * coord * s_e_b
            if len(basis) < p_max:
                basis.append(torch.cos(ang))
            if len(basis) < p_max:
                basis.append(torch.sin(ang))
            j += 1
        return torch.stack(basis, dim=-1)
    elif family == "cosine":
        ks = torch.arange(p_max, device=coord.device, dtype=coord.dtype)
        ks = ks.view(*((1,) * coord.ndim), p_max)
        arg = ks * math.pi * ((coord.unsqueeze(-1) + 1.0) * 0.5) * s_e_b.unsqueeze(-1)
        return torch.cos(arg)
    else:
        raise ValueError(f"unknown family: {family}")


def soft_order_mask(p_soft: torch.Tensor, p_max: int, sharpness: float) -> torch.Tensor:
    """Return per-mode weights w(k_i, k_j) = sigmoid((p_soft - max(k_i, k_j)) * sharpness).

    p_soft: (..., 1) per patch in [0, p_max].
    returns: (..., p_max, p_max) 2D mask.
    """
    device = p_soft.device
    ks = torch.arange(p_max, device=device, dtype=p_soft.dtype)
    k_i = ks.view(p_max, 1)
    k_j = ks.view(1, p_max)
    k_max = torch.maximum(k_i, k_j)
    mask = torch.sigmoid((p_soft.unsqueeze(-1) - k_max) * sharpness)
    return mask.squeeze(-3)


class AdaptivityHead(nn.Module):
    """Predict (s_e, p_soft) per patch from per-patch features.

    Bandwidth parameterization modes:
      - "local_linear"   : s_e = s_lo + (s_hi - s_lo) * sigmoid(raw)
                           (zero-init head -> s_e = arithmetic midpoint).
      - "local_logspace" : s_e = exp(log(s_lo) + (log(s_hi) - log(s_lo)) * sigmoid(raw))
                           (geometric interpolation; zero-init head -> s_e = sqrt(s_lo*s_hi),
                            the geometric midpoint). Lets multiplicative per-patch
                            variation emerge more easily from the midpoint basin.

    Order parameterization is linear-sigmoid in both modes.
    """

    def __init__(
        self,
        d_feat: int,
        s_e_range: tuple[float, float],
        p_soft_range: tuple[float, float],
        bandwidth_mode: str = "local_linear",
    ):
        super().__init__()
        if bandwidth_mode not in ("local_linear", "local_logspace"):
            raise ValueError(
                f"AdaptivityHead: unknown bandwidth_mode={bandwidth_mode!r}; "
                "expected one of 'local_linear', 'local_logspace'"
            )
        self.proj = nn.Linear(d_feat, 2)
        self.s_e_range = s_e_range
        self.p_soft_range = p_soft_range
        self.bandwidth_mode = bandwidth_mode
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feat: (..., d_feat) -> (s_e, p_soft) each (..., 1)."""
        out = self.proj(feat)
        s_e_raw, p_soft_raw = out[..., 0:1], out[..., 1:2]
        s_lo, s_hi = self.s_e_range
        p_lo, p_hi = self.p_soft_range
        if self.bandwidth_mode == "local_linear":
            s_e = s_lo + (s_hi - s_lo) * torch.sigmoid(s_e_raw)
        else:  # "local_logspace"
            log_lo = math.log(s_lo)
            log_hi = math.log(s_hi)
            s_e = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(s_e_raw))
        p_soft = p_lo + (p_hi - p_lo) * torch.sigmoid(p_soft_raw)
        return s_e, p_soft


def reconstruct_patch(
    coeffs: torch.Tensor,
    s_e: torch.Tensor,
    p_soft: torch.Tensor,
    coords_1d: torch.Tensor,
    cfg: BasisConfig,
) -> torch.Tensor:
    """Reconstruct (B, nH, nW, C, P, P) image patches from per-patch coefficients.

    Uses the soft order mask (continuous cutoff) to realize the discrete-order adaptivity.
    """
    B, nH, nW, C, pu, pv = coeffs.shape
    assert pu == cfg.p_max and pv == cfg.p_max, (pu, pv, cfg.p_max)

    phi_u = build_1d_basis(coords_1d, cfg.p_max, s_e, cfg.family)
    phi_v = build_1d_basis(coords_1d, cfg.p_max, s_e, cfg.family)

    mask = soft_order_mask(p_soft, cfg.p_max, cfg.cutoff_sharpness)
    coeffs_m = coeffs * mask.unsqueeze(-3)

    recon = torch.einsum("bnmcij,bnmui,bnmvj->bnmcuv", coeffs_m, phi_u, phi_v)
    return recon


def global_fourier_features(
    coords: torch.Tensor,
    B_freq: torch.Tensor,
) -> torch.Tensor:
    """Return (H, W, 2 * n_freq) of [cos(2*pi*B*x), sin(2*pi*B*x)] features."""
    proj = 2.0 * math.pi * coords @ B_freq.t()
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
