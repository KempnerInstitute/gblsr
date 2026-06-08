"""Tests for ``gblsr.training.losses``: pointwise reconstruction losses.

Verifies that each of the three loss families (``mse``, ``l1``,
``charbonnier``) returns the expected scalar values on a closed-form
input (zero residual, constant residual), and that the dispatching
``compute_losses`` returns the correct ``L_point`` / ``L_total`` keys.
"""

from __future__ import annotations

import math

import pytest
import torch

from gblsr.training import (
    LossConfig,
    compute_losses,
    pointwise_charbonnier,
    pointwise_l1,
    pointwise_loss,
    pointwise_mse,
)


def test_pointwise_mse_zero_residual_is_zero() -> None:
    """``pointwise_mse(x, x)`` is exactly zero."""
    x = torch.randn(2, 3, 8, 8)
    assert pointwise_mse(x, x).item() == 0.0


def test_pointwise_mse_constant_residual_equals_squared() -> None:
    """``pointwise_mse(x + c, x)`` is exactly ``c ** 2``."""
    x = torch.randn(2, 3, 8, 8)
    c = 0.5
    loss = pointwise_mse(x + c, x).item()
    assert abs(loss - c * c) < 1e-6


def test_pointwise_l1_zero_residual_is_zero() -> None:
    """``pointwise_l1(x, x)`` is exactly zero."""
    x = torch.randn(2, 3, 8, 8)
    assert pointwise_l1(x, x).item() == 0.0


def test_pointwise_l1_constant_residual_equals_absolute() -> None:
    """``pointwise_l1(x + c, x)`` is exactly ``|c|``."""
    x = torch.randn(2, 3, 8, 8)
    c = -0.3
    loss = pointwise_l1(x + c, x).item()
    assert abs(loss - abs(c)) < 1e-6


def test_pointwise_charbonnier_zero_residual_equals_eps() -> None:
    """At zero residual, each per-element loss is ``sqrt(0 + eps^2) = eps``,
    so the mean is also ``eps``. This is the smoothed-L1 contract."""
    eps = 1e-3
    x = torch.randn(2, 3, 8, 8)
    loss = pointwise_charbonnier(x, x, eps=eps).item()
    assert abs(loss - eps) < 1e-7


def test_pointwise_charbonnier_constant_residual_smoothed_form() -> None:
    """For constant residual ``c``, per-element loss is
    ``sqrt(c^2 + eps^2)``; the mean is also that value."""
    eps = 1e-2
    c = 0.5
    x = torch.randn(2, 3, 8, 8)
    loss = pointwise_charbonnier(x + c, x, eps=eps).item()
    expected = math.sqrt(c * c + eps * eps)
    assert abs(loss - expected) < 1e-6


def test_pointwise_charbonnier_rejects_nonpositive_eps() -> None:
    """``charbonnier_eps <= 0`` raises ``ValueError`` (division-by-zero
    behavior in the gradient + degenerate at residual zero)."""
    x = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError, match="eps must be > 0"):
        pointwise_charbonnier(x, x, eps=0.0)


@pytest.mark.parametrize(
    "loss_name,expected_at_zero",
    [
        ("mse", 0.0),
        ("l1", 0.0),
        ("charbonnier", 1e-3),  # eps default in the test below
    ],
)
def test_pointwise_loss_dispatches(loss_name: str, expected_at_zero: float) -> None:
    """``pointwise_loss`` dispatches to the right family by name."""
    x = torch.randn(2, 3, 8, 8)
    loss = pointwise_loss(x, x, loss_name=loss_name, charbonnier_eps=1e-3).item()
    assert abs(loss - expected_at_zero) < 1e-6


def test_pointwise_loss_rejects_unknown_name() -> None:
    """An unrecognised ``loss_name`` raises ``ValueError``."""
    x = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError, match="unknown loss_name"):
        pointwise_loss(x, x, loss_name="bogus_loss", charbonnier_eps=1e-3)


def test_compute_losses_returns_l_point_and_l_total() -> None:
    """``compute_losses`` returns the dict ``{L_point, L_total}`` where
    ``L_total == lambda_point * L_point``."""
    recon = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    cfg = LossConfig(lambda_point=2.5, loss_name="mse", charbonnier_eps=1e-3)
    out = {"recon": recon}
    losses = compute_losses(out, target, cfg)
    assert set(losses.keys()) == {"L_point", "L_total"}
    assert abs(losses["L_total"].item() - 2.5 * losses["L_point"].item()) < 1e-6
