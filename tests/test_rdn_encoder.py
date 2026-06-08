"""Tests for the RDN encoder.

Verifies:
  - The encoder builds at three representative ``num_features``
    values and produces the expected trainable-parameter counts
    (computed from the architectural arithmetic).
  - A forward pass on a small input produces a tensor of the right
    shape (spatial dims preserved, output channel dim = num_features).
"""

import pytest
import torch

from gblsr.encoders import RDNConfig, RDNEncoder, build_rdn_encoder


def _trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "num_features,expected_params",
    [
        # Trainable-parameter counts derived from the architectural
        # arithmetic. ``growth_rate`` stays at the default (64).
        (48, 20_572_880),
        (64, 21_973_952),
        (96, 24_852_896),
    ],
)
def test_rdn_param_count(num_features: int, expected_params: int) -> None:
    """Trainable parameter count at the given ``num_features``."""
    encoder = build_rdn_encoder(num_features=num_features)
    assert _trainable_params(encoder) == expected_params


def test_rdn_forward_shape() -> None:
    """Forward pass preserves spatial dims and outputs num_features channels."""
    # Tiny variant so the test runs fast on CPU.
    encoder = RDNEncoder(
        RDNConfig(
            num_features=16,
            growth_rate=8,
            num_rdbs=2,
            num_layers_per_rdb=2,
        )
    )
    x = torch.randn(1, 3, 32, 32)
    y = encoder(x)
    assert y.shape == (1, 16, 32, 32)
    assert y.dtype == x.dtype


def test_rdn_out_dim_attribute() -> None:
    """The ``out_dim`` attribute exposes num_features for downstream sizing."""
    enc = build_rdn_encoder(num_features=48)
    assert enc.out_dim == 48
