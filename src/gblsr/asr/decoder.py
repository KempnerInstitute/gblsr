"""GB-LSR-Scalar-ASR decoder.

Arbitrary-scale super-resolution adaptation of the GB-LSR-Scalar
native-reconstruction decoder, exposing the LIIF / LTE continuous-query
interface so the same encoder feature map can be decoded at any output
resolution.

Core mechanism. Each feature-map cell carries a ``3 * p_max * p_max`` block
of Fourier coefficients. A query point selects the owning cell via
``grid_sample(..., mode='nearest')``, computes its relative coordinate
within the cell (LIIF convention, scaled to approximately ``[-1, 1]``),
evaluates the 2D truncated Fourier basis at that point with a single
global trainable bandwidth ``s_e``, and contracts the basis against the
cell's coefficients. A 4-corner local ensemble is applied for
LIIF / LTE-compatible behaviour (essentially free in parameters).

Design notes:
  * Single global trainable scalar bandwidth, parameterised as
    ``s_e = softplus(log_s_e)`` for strict positivity; ``log_s_e`` is
    initialised at ``softplus_inv(bandwidth_init)`` so that ``s_e`` equals
    ``bandwidth_init`` exactly at step 0 (default 1.0). Note that the naive
    ``log_s_e = log(1.0) = 0`` would instead give
    ``softplus(0) = ln 2 = 0.693``; the exact softplus inverse is used so
    the initial bandwidth matches ``bandwidth_init`` verbatim.
  * Fourier basis with ``p_max`` modes per axis; the 2D basis is the tensor
    product of two 1D bases (``p_max * p_max`` basis functions per output
    channel). The 1D mode ordering matches :mod:`gblsr.models.basis`
    (the constant, then cosine / sine pairs of increasing frequency,
    ending in an unpaired cosine for even ``p_max``).
  * ``cell`` is accepted in the forward signature (unified with LIIF and
    LTE) but not used inside the decoder: scale adaptation happens via the
    density of the high-resolution query grid, so the native-reconstruction
    decoder's cell-free behaviour is preserved while respecting the shared
    API.
  * The basis-element footprint is the high-resolution region owned by one
    low-resolution feature cell at the evaluation scale
    (``HR_size / feat_size``); it inherits whatever footprint the chosen
    encoder provides.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord import make_coord


def _fourier_basis_1d(
    coord: torch.Tensor, p_max: int, s_e: torch.Tensor
) -> torch.Tensor:
    """Evaluate the 1D truncated Fourier basis pointwise.

    Ordering matches :mod:`gblsr.models.basis` (``family='fourier'``)::

        [1, cos(pi*c*s), sin(pi*c*s), cos(2*pi*c*s), sin(2*pi*c*s),
         ..., up to ``p_max`` terms]

    coord:  (...,) arbitrary shape in the normalised local coordinate.
    s_e:    scalar-compatible bandwidth; broadcasts with ``coord``.
    returns: (..., p_max) basis values.
    """
    basis: list[torch.Tensor] = [torch.ones_like(coord)]
    j = 1
    while len(basis) < p_max:
        ang = j * math.pi * coord * s_e
        if len(basis) < p_max:
            basis.append(torch.cos(ang))
        if len(basis) < p_max:
            basis.append(torch.sin(ang))
        j += 1
    return torch.stack(basis, dim=-1)


class GBLSRScalarASRDecoder(nn.Module):
    """GB-LSR-Scalar decoder adapted to the LIIF / LTE arbitrary-scale interface.

    Args:
        in_dim: encoder output channel count (feeds the coefficient
            projection).
        p_max: per-axis basis truncation (default 16).
        bandwidth_init: initial value of ``s_e`` (default 1.0).
        local_ensemble: enable the 4-corner LIIF local ensemble (default
            True for LIIF / LTE compatibility).
    """

    def __init__(
        self,
        in_dim: int,
        p_max: int = 16,
        bandwidth_init: float = 1.0,
        local_ensemble: bool = True,
    ):
        super().__init__()
        if p_max < 1:
            raise ValueError(f"p_max must be >= 1 (got {p_max})")
        if bandwidth_init <= 0:
            raise ValueError(
                f"bandwidth_init must be positive for softplus_inv (got {bandwidth_init})"
            )
        self.in_dim = in_dim
        self.p_max = p_max
        self.local_ensemble = local_ensemble

        # Per-feature-cell coefficient projection. A 1x1 conv is equivalent
        # to the native decoder's Linear (which permutes channels to last
        # and applies a Linear to the per-cell feature vector).
        # Output: 3 * p_max * p_max.
        self.coeff_proj = nn.Conv2d(
            in_dim, 3 * p_max * p_max, kernel_size=1, bias=True
        )
        # Init: weight ~ N(0, 0.01), bias = 0.
        nn.init.normal_(self.coeff_proj.weight, std=0.01)
        nn.init.zeros_(self.coeff_proj.bias)

        # Single global trainable scalar bandwidth. softplus_inv(x) = log(exp(x) - 1).
        init_raw = math.log(math.expm1(bandwidth_init))
        self.log_s_e = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def bandwidth(self) -> torch.Tensor:
        """Return the current positive scalar bandwidth ``s_e``."""
        return F.softplus(self.log_s_e)

    def forward(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Query RGB at arbitrary coordinates.

        feat:  (B, D, H, W) encoder feature map.
        coord: (B, Q, 2) query coords in [-1, 1], (y, x) order.
        cell:  (B, Q, 2) cell sizes (accepted; not used internally).
        returns: (B, Q, 3) RGB in the training range ([0, 1]).
        """
        del cell  # scalar bandwidth + dense HR grid carries the scale info
        B, _, H, W = feat.shape
        P = self.p_max
        s_e = self.bandwidth()

        coef = self.coeff_proj(feat)  # (B, 3*P*P, H, W)

        if self.local_ensemble:
            vx_list = [-1, 1]
            vy_list = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_list, vy_list, eps_shift = [0], [0], 0

        rx = 2.0 / H / 2.0
        ry = 2.0 / W / 2.0

        feat_coord = (
            make_coord((H, W), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(B, 2, H, W)
        )

        preds: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []
        for vx in vx_list:
            for vy in vy_list:
                coord_ = coord.clone()
                coord_[:, :, 0] = coord_[:, :, 0] + vx * rx + eps_shift
                coord_[:, :, 1] = coord_[:, :, 1] + vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                grid = coord_.flip(-1).unsqueeze(1)  # (B, 1, Q, 2), (x, y)
                q_coef = F.grid_sample(
                    coef, grid, mode="nearest", align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)  # (B, Q, 3*P*P)
                q_fcoord = F.grid_sample(
                    feat_coord, grid, mode="nearest", align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)  # (B, Q, 2)

                rel_coord = coord - q_fcoord
                rel_coord[:, :, 0] *= H  # ~[-1, 1] within owning cell
                rel_coord[:, :, 1] *= W

                Bq = rel_coord.shape[:2]
                phi_y = _fourier_basis_1d(rel_coord[..., 0], P, s_e)  # (B, Q, P)
                phi_x = _fourier_basis_1d(rel_coord[..., 1], P, s_e)  # (B, Q, P)

                q_coef_rgb = q_coef.view(*Bq, 3, P, P)  # (B, Q, 3, P, P)

                # rgb[b,q,c] = sum_{i,j} coef[b,q,c,i,j] * phi_y[b,q,i] * phi_x[b,q,j]
                pred = torch.einsum(
                    "bqcij,bqi,bqj->bqc", q_coef_rgb, phi_y, phi_x
                )
                preds.append(pred)
                areas.append(
                    torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]) + 1e-9
                )

        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        total = sum(areas)
        return sum(
            pred * (area / total).unsqueeze(-1) for pred, area in zip(preds, areas)
        )
