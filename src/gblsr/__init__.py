"""GB-LSR: Global-Bandwidth Local Spectral Representation.

A fixed-grid local spectral image representation for continuous image
reconstruction. The image domain is partitioned into a fixed grid of
non-overlapping square patches; each patch carries a small block of
coefficients for a truncated Fourier basis, predicted from shared
convolutional-encoder features by a single linear projection. A single
trainable scalar bandwidth is shared globally across all patches, and
reconstruction at any continuous coordinate is a fixed-size basis
contraction whose cost is independent of image size. A separate
arbitrary-scale super-resolution extension (:mod:`gblsr.asr`) reuses the
same local-spectral decoder behind an RDN encoder and a continuous-query
interface.

Quick start
-----------

The most-used entry points are re-exported at the package top level::

    from gblsr import LocalSpectralArm, build_model, ModelConfig
    from gblsr import BasisConfig, EncoderConfig
    from gblsr import measure_latency, LatencyConfig
    from gblsr import GBLSRScalarASR

For specialized entry points, use the subpackages:

  - ``gblsr.models``     model classes, configs, factory, basis primitives
  - ``gblsr.encoders``   heavier encoders (RDN)
  - ``gblsr.asr``        arbitrary-scale SR extension (RDN encoder + ASR decoder)
  - ``gblsr.latency``    fixed GPU latency protocol
  - ``gblsr.training``   training driver, losses, RunConfig
  - ``gblsr.metrics``    PSNR / SSIM / LPIPS / edge-LPIPS / local-spectrum error
  - ``gblsr.data``       loaders + region slicing
"""

from .asr.model import GBLSRScalarASR
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
    # Arbitrary-scale SR extension
    "GBLSRScalarASR",
    # Latency protocol
    "measure_latency",
    "LatencyConfig",
]
