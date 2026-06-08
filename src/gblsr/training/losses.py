"""Pointwise reconstruction losses for GB-LSR training.

Three loss families are supported via ``LossConfig.loss_name``:

  - ``"mse"``         : ``F.mse_loss(recon, target)``  (default).
  - ``"l1"``          : ``F.l1_loss(recon, target)``.
  - ``"charbonnier"`` : ``mean(sqrt((recon - target)^2 + eps^2))``,
                       a smooth L1 with ``eps`` controlling the
                       quadratic-to-linear transition near zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


LossName = Literal["mse", "l1", "charbonnier"]


@dataclass
class LossConfig:
    lambda_point: float = 1.0
    loss_name: LossName = "mse"
    charbonnier_eps: float = 1.0e-3


def pointwise_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstruction and target."""
    return F.mse_loss(recon, target)


def pointwise_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute (L1) error between reconstruction and target."""
    return F.l1_loss(recon, target)


def pointwise_charbonnier(
    recon: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Charbonnier loss: mean(sqrt((recon - target)^2 + eps^2)).

    At zero residual the per-element loss equals ``eps`` so the mean is also
    ``eps``. Smaller ``eps`` makes the loss more sensitive to small residuals
    (closer to L1); larger ``eps`` smooths the bottom of the well (closer to
    a quadratic near zero).
    """
    if eps <= 0:
        raise ValueError(f"charbonnier_eps must be > 0, got {eps!r}")
    diff = recon - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def pointwise_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_name: LossName,
    charbonnier_eps: float,
) -> torch.Tensor:
    """Dispatch to ``pointwise_mse`` / ``pointwise_l1`` / ``pointwise_charbonnier``."""
    if loss_name == "mse":
        return pointwise_mse(recon, target)
    if loss_name == "l1":
        return pointwise_l1(recon, target)
    if loss_name == "charbonnier":
        return pointwise_charbonnier(recon, target, charbonnier_eps)
    raise ValueError(
        f"unknown loss_name={loss_name!r}; expected one of {{'mse', 'l1', 'charbonnier'}}"
    )


def compute_losses(
    out: dict,
    target: torch.Tensor,
    cfg: LossConfig,
) -> dict[str, torch.Tensor]:
    """Compute the training-loss dict ``{L_point, L_total}`` from a model output.

    ``out["recon"]`` is compared against ``target`` using the loss surface
    in ``cfg.loss_name``; ``L_total = cfg.lambda_point * L_point``.
    """
    recon = out["recon"]
    L_point = pointwise_loss(recon, target, cfg.loss_name, cfg.charbonnier_eps)
    total = cfg.lambda_point * L_point
    return {"L_point": L_point, "L_total": total}
