"""Image-feature encoders used by GB-LSR.

The ``Encoder`` in ``gblsr.models.arms`` is a small convolutional
encoder used directly by the default GB-LSR variants. ``RDNEncoder``
here is a heavier Residual Dense Network encoder, suitable for use
as the image-encoding stage of an arbitrary-scale super-resolution
pipeline.
"""

from .rdn import RDNConfig, RDNEncoder, build_rdn_encoder

__all__ = ["RDNConfig", "RDNEncoder", "build_rdn_encoder"]
