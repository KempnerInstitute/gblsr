"""Tests for ``gblsr.models.basis``: basis functions, soft order mask,
patch reconstruction, AdaptivityHead.

These tests use synthetic inputs whose expected outputs are analytic
(constant 1, ``cos(pi*x)``, ``sin(pi*x)``, geometric midpoint of an
interval, etc.). When a test's expected value is derived from the
basis-math contract, the contract is stated in the docstring so a
reader can verify by inspection.
"""

from __future__ import annotations

import math

import torch

from gblsr.models.basis import (
    AdaptivityHead,
    BasisConfig,
    build_1d_basis,
    image_to_patches,
    make_patch_coords,
    patches_to_image,
    reconstruct_patch,
    soft_order_mask,
)


# ---------- coordinates + reshape helpers ----------


def test_make_patch_coords_shape_and_endpoints() -> None:
    """``make_patch_coords(P)`` returns a ``(P, P, 2)`` grid spanning
    ``[-1, 1]^2``; corners are exactly ``(-1, -1)`` and ``(1, 1)``."""
    P = 4
    coords = make_patch_coords(P)
    assert coords.shape == (P, P, 2)
    assert torch.allclose(coords[0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(coords[-1, -1], torch.tensor([1.0, 1.0]))


def test_image_to_patches_roundtrip() -> None:
    """``image_to_patches`` then ``patches_to_image`` is the identity
    map on any tensor with spatial dims divisible by ``P``."""
    B, C, H, W, P = 2, 3, 16, 24, 8
    x = torch.randn(B, C, H, W)
    p = image_to_patches(x, P)
    assert p.shape == (B, H // P, W // P, C, P, P)
    x_back = patches_to_image(p)
    assert torch.allclose(x_back, x, atol=1e-7)


# ---------- 1D basis ----------


def test_build_1d_basis_fourier_p_max_1_is_constant() -> None:
    """With ``p_max == 1`` the truncated Fourier basis is just the
    constant function 1."""
    P = 8
    coord = torch.linspace(-1.0, 1.0, P)
    s_e = torch.ones(1, 1)
    basis = build_1d_basis(coord, p_max=1, s_e=s_e, family="fourier")
    assert basis.shape[-2:] == (P, 1)
    assert torch.allclose(basis, torch.ones_like(basis), atol=1e-6)


def test_build_1d_basis_fourier_p_max_3_matches_analytic() -> None:
    """At ``p_max == 3`` the basis is exactly ``[1, cos(pi x s_e), sin(pi x s_e)]``."""
    P = 5
    coord = torch.linspace(-1.0, 1.0, P)
    s_e = torch.ones(1, 1)
    basis = build_1d_basis(coord, p_max=3, s_e=s_e, family="fourier")
    # basis shape: (1, P, 3)
    assert basis.shape == (1, P, 3)
    expected_const = torch.ones(P)
    expected_cos = torch.cos(math.pi * coord)
    expected_sin = torch.sin(math.pi * coord)
    assert torch.allclose(basis[0, :, 0], expected_const, atol=1e-6)
    assert torch.allclose(basis[0, :, 1], expected_cos, atol=1e-6)
    assert torch.allclose(basis[0, :, 2], expected_sin, atol=1e-6)


def test_build_1d_basis_cosine_first_mode_is_constant() -> None:
    """In the cosine family, mode ``k=0`` is ``cos(0) = 1`` everywhere."""
    P = 5
    coord = torch.linspace(-1.0, 1.0, P)
    s_e = torch.ones(1, 1)
    basis = build_1d_basis(coord, p_max=3, s_e=s_e, family="cosine")
    # Pick out mode 0 across all positions.
    mode_0 = basis[..., :, 0].squeeze()
    assert torch.allclose(mode_0, torch.ones_like(mode_0), atol=1e-6)


# ---------- soft order mask ----------


def test_soft_order_mask_p_soft_high_passes_all_modes() -> None:
    """With ``p_soft`` well above ``p_max`` and high sharpness, every
    mode is fully on (mask close to 1)."""
    p_max = 4
    p_soft = torch.tensor([[10.0]])
    mask = soft_order_mask(p_soft, p_max=p_max, sharpness=10.0)
    assert mask.shape[-2:] == (p_max, p_max)
    assert torch.all(mask > 0.99)


def test_soft_order_mask_p_soft_zero_keeps_only_dc() -> None:
    """With ``p_soft == 0`` and high sharpness, the mask is sigmoid(0)
    = 0.5 at the DC mode (0, 0) and essentially zero everywhere else
    (the cutoff places the boundary exactly at ``k_max = 0``)."""
    p_max = 4
    p_soft = torch.zeros(1, 1)
    mask = soft_order_mask(p_soft, p_max=p_max, sharpness=10.0)
    assert abs(mask[..., 0, 0].item() - 0.5) < 1e-5
    # Off-DC modes are far on the wrong side of the cutoff -> ~0.
    assert mask[..., 1, 1].item() < 1e-3
    assert mask[..., p_max - 1, p_max - 1].item() < 1e-12


# ---------- patch reconstruction ----------


def test_reconstruct_patch_dc_coefficient_yields_constant_patch() -> None:
    """A coefficient set only at the constant mode ``(0, 0)`` should
    reconstruct a spatially-constant patch with that coefficient value.

    The DC basis function is identically 1 in both axes, so the
    reconstruction at any position is ``coeff[0, 0] * 1 * 1 = coeff[0, 0]``,
    modulated by the soft order mask at ``(0, 0)``. With high
    ``cutoff_sharpness`` and ``p_soft`` well above zero, the mask at
    DC is effectively 1, so the reconstruction is exactly the
    coefficient value.
    """
    P = 8
    B, nH, nW, C, p_max = 1, 1, 1, 3, 4
    coeffs = torch.zeros(B, nH, nW, C, p_max, p_max)
    coeffs[..., 0, 0] = 0.7  # per-channel coefficient at the DC mode
    s_e = torch.full((B, nH, nW, 1), 0.5)
    p_soft = torch.full((B, nH, nW, 1), float(p_max + 5))  # well above cutoff
    cfg = BasisConfig(
        patch_size=P,
        p_max=p_max,
        family="fourier",
        s_e_range=(0.1, 2.0),
        p_soft_range=(1.0, float(p_max + 5)),
        cutoff_sharpness=100.0,  # very sharp -> mask at (0,0) is exactly 1
    )
    coords_1d = torch.linspace(-1.0, 1.0, P)
    patches = reconstruct_patch(coeffs, s_e, p_soft, coords_1d, cfg)
    assert patches.shape == (B, nH, nW, C, P, P)
    assert torch.allclose(patches, 0.7 * torch.ones_like(patches), atol=1e-5)


# ---------- AdaptivityHead ----------


def test_adaptivity_head_zero_init_linear_returns_arithmetic_midpoint() -> None:
    """Zero-init linear-sigmoid head emits ``s_e`` at the arithmetic
    midpoint of ``s_e_range``, independent of input features."""
    s_lo, s_hi = 0.25, 2.0
    head = AdaptivityHead(
        d_feat=8,
        s_e_range=(s_lo, s_hi),
        p_soft_range=(1.0, 16.0),
        bandwidth_mode="local_linear",
    )
    feat = torch.randn(1, 4, 4, 8)
    s_e, p_soft = head(feat)
    expected_mid = s_lo + 0.5 * (s_hi - s_lo)
    assert torch.allclose(s_e, torch.full_like(s_e, expected_mid), atol=1e-6)


def test_adaptivity_head_zero_init_logspace_returns_geometric_midpoint() -> None:
    """Zero-init log-space head emits ``s_e`` at the *geometric*
    midpoint ``sqrt(s_lo * s_hi)`` of the range."""
    s_lo, s_hi = 0.25, 2.0
    head = AdaptivityHead(
        d_feat=8,
        s_e_range=(s_lo, s_hi),
        p_soft_range=(1.0, 16.0),
        bandwidth_mode="local_logspace",
    )
    feat = torch.randn(1, 4, 4, 8)
    s_e, _ = head(feat)
    geometric_mid = math.sqrt(s_lo * s_hi)
    assert torch.allclose(s_e, torch.full_like(s_e, geometric_mid), atol=1e-6)


def test_adaptivity_head_output_shape() -> None:
    """``s_e`` and ``p_soft`` outputs each carry a trailing ``1`` dim
    so they broadcast cleanly against per-patch coefficient tensors."""
    head = AdaptivityHead(
        d_feat=8,
        s_e_range=(0.25, 2.0),
        p_soft_range=(1.0, 16.0),
    )
    feat = torch.randn(2, 4, 4, 8)
    s_e, p_soft = head(feat)
    assert s_e.shape == (2, 4, 4, 1)
    assert p_soft.shape == (2, 4, 4, 1)
