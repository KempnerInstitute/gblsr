"""Tests for the curated public API surface.

Verifies:
  - Every name in each subpackage's ``__all__`` resolves to a real
    attribute on the module. Catches typos in the re-export lists
    and accidental rename drift.
  - The canonical top-level imports the README documents resolve.
  - The README's canonical subpackage imports resolve.
"""

from __future__ import annotations

import importlib

import pytest


SUBPACKAGES = [
    "gblsr",
    "gblsr.models",
    "gblsr.encoders",
    "gblsr.latency",
    "gblsr.training",
    "gblsr.data",
    "gblsr.metrics",
]


@pytest.mark.parametrize("modpath", SUBPACKAGES)
def test_public_api_all_resolves(modpath: str) -> None:
    """Every name in ``<module>.__all__`` must resolve to an attribute."""
    mod = importlib.import_module(modpath)
    assert hasattr(mod, "__all__"), f"{modpath} should declare __all__"
    for name in mod.__all__:
        assert hasattr(mod, name), (
            f"{modpath}.__all__ lists {name!r}, but no such attribute "
            f"exists on {modpath} (typo or rename drift)."
        )


def test_canonical_top_level_imports() -> None:
    """The canonical top-level imports the README documents must resolve."""
    from gblsr import (
        BaselineArm,
        BasisConfig,
        EncoderConfig,
        LatencyConfig,
        LocalSpectralArm,
        ModelConfig,
        build_model,
        measure_latency,
    )

    for sym in (
        BaselineArm,
        BasisConfig,
        EncoderConfig,
        LatencyConfig,
        LocalSpectralArm,
        ModelConfig,
        build_model,
        measure_latency,
    ):
        assert sym is not None


def test_canonical_subpackage_imports() -> None:
    """The README's canonical subpackage imports resolve."""
    from gblsr.data import DataConfig, build_datasets, label_patches
    from gblsr.encoders import RDNConfig, RDNEncoder, build_rdn_encoder
    from gblsr.latency import (
        LatencyConfig,
        LatencyResult,
        measure_latency,
        measure_latency_grid,
    )
    from gblsr.metrics import (
        edge_lpips,
        local_spectrum_error,
        lpips_metric,
        psnr,
        ssim,
    )
    from gblsr.models import (
        BaselineArm,
        BasisConfig,
        EncoderConfig,
        GlobalFourierMLPDecoder,
        LocalSpectralArm,
        LocalSpectralDecoder,
        ModelConfig,
        build_model,
    )
    from gblsr.training import RunConfig, build_model_from_run_config, train_one

    for sym in (
        DataConfig,
        build_datasets,
        label_patches,
        RDNConfig,
        RDNEncoder,
        build_rdn_encoder,
        LatencyConfig,
        LatencyResult,
        measure_latency,
        measure_latency_grid,
        edge_lpips,
        local_spectrum_error,
        lpips_metric,
        psnr,
        ssim,
        BaselineArm,
        BasisConfig,
        EncoderConfig,
        GlobalFourierMLPDecoder,
        LocalSpectralArm,
        LocalSpectralDecoder,
        ModelConfig,
        build_model,
        RunConfig,
        build_model_from_run_config,
        train_one,
    ):
        assert sym is not None


def test_no_private_symbols_in_all() -> None:
    """``__all__`` should not list any underscore-prefixed names.

    Underscore prefix is the package's convention for private helpers
    (``_unpack_batch``, ``_resolve_mode``, ``_DenseLayer``, etc.).
    """
    for modpath in SUBPACKAGES:
        mod = importlib.import_module(modpath)
        all_list = getattr(mod, "__all__", [])
        private = [n for n in all_list if n.startswith("_") and n != "__version__"]
        assert not private, f"{modpath}.__all__ exports private names: {private!r}"
