"""GB-LSR: Global-Bandwidth Local Spectral Representation.

A fixed-grid local spectral image representation for continuous image
reconstruction. The image domain is partitioned into a fixed grid of
non-overlapping square patches; each patch carries a small block of
coefficients for a truncated Fourier basis, predicted from shared
convolutional-encoder features by a single linear projection. A single
trainable scalar bandwidth is shared globally across all patches, and
reconstruction at any continuous coordinate is a fixed-size basis
contraction whose cost is independent of image size.

Quick start
-----------

The most-used entry points are re-exported at the package top level::

    from gblsr import LocalSpectralArm, build_model, ModelConfig
    from gblsr import BasisConfig, EncoderConfig
    from gblsr import measure_latency, LatencyConfig

For specialized entry points, use the subpackages:

  - ``gblsr.models``     model classes, configs, factory, basis primitives
  - ``gblsr.encoders``   heavier encoders (RDN)
  - ``gblsr.latency``    fixed GPU latency protocol
  - ``gblsr.training``   training driver, losses, RunConfig
  - ``gblsr.metrics``    PSNR / SSIM / LPIPS / edge-LPIPS / local-spectrum error
  - ``gblsr.data``       loaders + region slicing
"""

from .latency.protocol import LatencyConfig, measure_latency
from .models.arms import (
    BaselineArm,
    EncoderConfig,
    LocalSpectralArm,
    ModelConfig,
    build_model,
)
from .models.basis import BasisConfig

__version__ = "0.0.0"

__all__ = [
    "__version__",
    # Models
    "LocalSpectralArm",
    "BaselineArm",
    "build_model",
    "ModelConfig",
    "EncoderConfig",
    "BasisConfig",
    # Latency protocol
    "measure_latency",
    "LatencyConfig",
]
