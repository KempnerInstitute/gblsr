"""Public API for GB-LSR's core architecture (encoder, decoder, basis).

Re-exports the canonical model classes, configs, basis primitives, and
shape utilities from ``.arms``, ``.basis``, and ``.shape_utils``.
"""

from .arms import (
    BANDWIDTH_MODES,
    BaselineArm,
    Encoder,
    EncoderConfig,
    GlobalFourierMLPDecoder,
    LocalSpectralArm,
    LocalSpectralDecoder,
    ModelConfig,
    build_model,
    n_params,
)
from .basis import (
    AdaptivityHead,
    BasisConfig,
    build_1d_basis,
    global_fourier_features,
    image_to_patches,
    make_patch_coords,
    patches_to_image,
    reconstruct_patch,
    soft_order_mask,
)
from .shape_utils import (
    crop_to_original,
    make_valid_patch_mask,
    pad_to_multiple,
)

__all__ = [
    # Arms + factory
    "LocalSpectralArm",
    "BaselineArm",
    "LocalSpectralDecoder",
    "GlobalFourierMLPDecoder",
    "Encoder",
    "build_model",
    "n_params",
    "BANDWIDTH_MODES",
    # Configs
    "ModelConfig",
    "EncoderConfig",
    "BasisConfig",
    # Basis primitives
    "AdaptivityHead",
    "build_1d_basis",
    "global_fourier_features",
    "image_to_patches",
    "make_patch_coords",
    "patches_to_image",
    "reconstruct_patch",
    "soft_order_mask",
    # Shape utilities
    "pad_to_multiple",
    "crop_to_original",
    "make_valid_patch_mask",
]
