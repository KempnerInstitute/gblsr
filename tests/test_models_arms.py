"""Forward-pass tests for ``gblsr.models.arms``.

Verifies that all four ``LocalSpectralArm`` ``bandwidth_mode`` values
plus the ``BaselineArm`` (``global_fourier_mlp``) build, run forward,
and produce reconstructions at the expected output shapes. These
tests complement ``test_models_basis.py`` (which tests the basis math
in isolation) by exercising the full arm-level forward path on
synthetic inputs.

Kept tiny (``image_size=16``, ``patch_size=8``, ``d_feat=8``) so the
suite runs in well under a second on CPU.
"""

from __future__ import annotations

import pytest
import torch

from gblsr.models import (
    BANDWIDTH_MODES,
    BaselineArm,
    BasisConfig,
    Encoder,
    EncoderConfig,
    LocalSpectralArm,
    ModelConfig,
    build_model,
    n_params,
)


def _tiny_model_config(arm: str) -> ModelConfig:
    """Smallest model config that still exercises encoder + decoder paths."""
    return ModelConfig(
        arm=arm,
        image_size=16,
        patch_size=8,
        basis=BasisConfig(patch_size=8, p_max=4),
        encoder=EncoderConfig(d_feat=8),
        n_global_freq=16,
        decoder_hidden=16,
        decoder_layers=2,
    )


@pytest.mark.parametrize("bandwidth_mode", list(BANDWIDTH_MODES))
def test_local_spectral_arm_forward_shape_each_mode(bandwidth_mode: str) -> None:
    """``LocalSpectralArm.forward`` returns ``recon`` at the input HxW
    for every supported ``bandwidth_mode``."""
    mc = _tiny_model_config("local_spectral")
    model = LocalSpectralArm(mc, bandwidth_mode=bandwidth_mode).eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out["recon"].shape == (1, 3, 16, 16)
    # s_e and p_soft are emitted at the patch-grid shape ``(B, nH, nW, 1)``.
    assert out["s_e"].shape == (1, 2, 2, 1)
    assert out["p_soft"].shape == (1, 2, 2, 1)


def test_baseline_arm_forward_shape() -> None:
    """``BaselineArm`` (Global Fourier-MLP) returns ``recon`` at
    ``cfg.image_size`` (shape-locked)."""
    mc = _tiny_model_config("global_fourier_mlp")
    model = BaselineArm(mc).eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
    assert out["recon"].shape == (1, 3, 16, 16)
    # BaselineArm has no per-patch coefficients / bandwidth.
    assert out["coeffs"] is None
    assert out["s_e"] is None
    assert out["p_soft"] is None


def test_baseline_arm_rejects_mismatched_input_size() -> None:
    """``BaselineArm`` is shape-locked: a 32x32 input on a 16x16-config
    model raises ``NotImplementedError`` rather than silently producing
    a wrong-resolution reconstruction."""
    mc = _tiny_model_config("global_fourier_mlp")
    model = BaselineArm(mc).eval()
    x_wrong = torch.randn(1, 3, 32, 32)
    with pytest.raises(NotImplementedError, match="shape-locked"):
        with torch.no_grad():
            model(x_wrong)


def test_build_model_dispatches_to_correct_arm() -> None:
    """``build_model`` returns ``BaselineArm`` for ``global_fourier_mlp``
    and ``LocalSpectralArm`` for ``local_spectral``."""
    mc_b = _tiny_model_config("global_fourier_mlp")
    m_b = build_model(mc_b)
    assert isinstance(m_b, BaselineArm)

    mc_l = _tiny_model_config("local_spectral")
    m_l = build_model(mc_l)
    assert isinstance(m_l, LocalSpectralArm)


def test_build_model_raises_on_unknown_arm() -> None:
    """``build_model`` rejects arm names outside the Literal set with
    a clear ``ValueError`` (defensive against typos in user configs)."""
    mc = _tiny_model_config("local_spectral")
    # Sneak past the ``Literal`` type via attribute assignment, since
    # the runtime check is the only thing that fires.
    mc.arm = "unknown_arm_name"
    with pytest.raises(ValueError, match="unknown arm"):
        build_model(mc)


def test_encoder_forward_preserves_patch_grid_resolution() -> None:
    """``Encoder`` downsamples by exactly ``patch_size`` (it has
    ``log2(patch_size)`` stride-2 blocks)."""
    cfg = EncoderConfig(d_feat=8)
    enc = Encoder(image_size=32, patch_size=8, cfg=cfg).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        feat = enc(x)
    # 32 / 8 = 4 -> 4x4 patch grid.
    assert feat.shape == (1, 8, 4, 4)


def test_n_params_counts_only_trainable() -> None:
    """``n_params`` sums only ``requires_grad=True`` parameters."""
    mc = _tiny_model_config("local_spectral")
    model = LocalSpectralArm(mc, bandwidth_mode="global_scalar")
    total_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params(model) == total_count
    assert n_params(model) > 0

    # Freeze everything: n_params should drop to zero.
    for p in model.parameters():
        p.requires_grad_(False)
    assert n_params(model) == 0


def test_bandwidth_modes_is_canonical_tuple() -> None:
    """``BANDWIDTH_MODES`` is the source of truth for valid
    ``bandwidth_mode`` strings; the parametrize above relies on this."""
    assert BANDWIDTH_MODES == (
        "fixed_midpoint",
        "global_scalar",
        "local_linear",
        "local_logspace",
    )
