"""Tests for the arbitrary-scale SR extension (``gblsr.asr``).

Verifies:
  - The base and nf96 models reproduce the paper's trainable-parameter
    counts (GB-LSR-Scalar-ASR 22.024M; nf96 24.927M), with the total
    decomposing exactly into encoder + decoder coefficient head.
  - Forward (continuous-query) and ``predict_full`` (full-image) shapes.
  - ``predict_full`` tiling produces the same result as the untiled path.
  - The reported family variants (noLE, nf96) build and run by config,
    and the local ensemble is parameter-free.
  - The scalar bandwidth initialises at exactly 1.0.
"""

import pytest
import torch

from gblsr import GBLSRScalarASR
from gblsr.asr import GBLSRScalarASRDecoder, make_coord


def _trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _tiny() -> GBLSRScalarASR:
    """A small ASR model so forward-pass tests run fast on CPU."""
    return GBLSRScalarASR(
        encoder_cfg={
            "num_features": 16,
            "growth_rate": 8,
            "num_rdbs": 2,
            "num_layers_per_rdb": 2,
        },
        decoder_cfg={"p_max": 6},
    )


@pytest.mark.parametrize(
    "num_features,paper_millions",
    [
        # Paper trainable-parameter counts (pareto table, 3-decimal M):
        # GB-LSR-Scalar-ASR (base) = 22.024M; nf96 variant = 24.927M.
        (64, 22.024),
        (96, 24.927),
    ],
)
def test_asr_param_count_matches_paper(num_features: int, paper_millions: float) -> None:
    """Full-model trainable params match the paper, and split into enc + head."""
    model = GBLSRScalarASR(encoder_cfg={"num_features": num_features})
    total = _trainable(model)
    assert round(total / 1e6, 3) == paper_millions

    # Structural decomposition (derived, not memorised): the only trainable
    # parts are the RDN encoder and the decoder's coefficient head, which is
    # a Conv2d(num_features -> 3*p_max^2, 1x1, bias) plus the scalar log_s_e.
    enc = _trainable(model.encoder)
    P = model.decoder.p_max
    head = (3 * P * P) * num_features + (3 * P * P) + 1
    assert total == enc + head


def test_asr_forward_shape() -> None:
    """Continuous-query forward: (B, Q, 2) coords -> (B, Q, 3) RGB."""
    model = _tiny()
    lr = torch.randn(2, 3, 8, 8)
    coord = make_coord((12, 12)).unsqueeze(0).expand(2, -1, -1).contiguous()
    cell = torch.empty_like(coord)
    cell[:, :, 0] = 2.0 / 12
    cell[:, :, 1] = 2.0 / 12
    out = model(lr, coord, cell)
    assert out.shape == (2, 12 * 12, 3)


def test_asr_predict_full_shape() -> None:
    """predict_full renders a (B, 3, H_q, W_q) image at an arbitrary scale."""
    model = _tiny()
    lr = torch.randn(1, 3, 8, 8)
    out = model.predict_full(lr, H_q=20, W_q=20)
    assert out.shape == (1, 3, 20, 20)


def test_asr_predict_full_tiling_parity() -> None:
    """Tiled query evaluation equals the untiled path."""
    model = _tiny().eval()
    lr = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        full = model.predict_full(lr, H_q=16, W_q=16, tile_q=None)
        tiled = model.predict_full(lr, H_q=16, W_q=16, tile_q=37)
    assert torch.allclose(full, tiled, atol=1e-6)


def test_asr_noLE_variant_builds_and_is_param_free() -> None:
    """The noLE variant builds, runs, and adds no parameters vs base."""
    base = _tiny()
    noLE = GBLSRScalarASR(
        encoder_cfg={
            "num_features": 16,
            "growth_rate": 8,
            "num_rdbs": 2,
            "num_layers_per_rdb": 2,
        },
        decoder_cfg={"p_max": 6, "local_ensemble": False},
    )
    assert noLE.decoder.local_ensemble is False
    assert _trainable(noLE) == _trainable(base)  # local ensemble is param-free
    lr = torch.randn(1, 3, 8, 8)
    out = noLE.predict_full(lr, H_q=10, W_q=10)
    assert out.shape == (1, 3, 10, 10)


def test_asr_bandwidth_init_is_one() -> None:
    """The scalar bandwidth initialises at exactly 1.0 (softplus_inv init)."""
    dec = GBLSRScalarASRDecoder(in_dim=16)
    assert float(dec.bandwidth()) == pytest.approx(1.0, abs=1e-5)


def test_asr_top_level_export() -> None:
    """GBLSRScalarASR is importable from the package root."""
    import gblsr

    assert gblsr.GBLSRScalarASR is GBLSRScalarASR
