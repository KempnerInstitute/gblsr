"""Arbitrary-scale super-resolution (ASR) extension of GB-LSR.

The :class:`GBLSRScalarASR` model pairs the RDN encoder
(:class:`gblsr.encoders.rdn.RDNEncoder`) with the
:class:`GBLSRScalarASRDecoder`, exposing the LIIF / LTE continuous-query
interface so a single encoder feature map can be decoded at any output
resolution. The reported family variants (base, noLE, nf48, nf96) are all
reachable by configuration; see :class:`GBLSRScalarASR` for the recipe.

The native-reconstruction model lives in :mod:`gblsr.models.arms`; this
subpackage is the separate arbitrary-scale SR extension.
"""

from .coord import make_coord
from .decoder import GBLSRScalarASRDecoder
from .model import GBLSRScalarASR

__all__ = ["GBLSRScalarASR", "GBLSRScalarASRDecoder", "make_coord"]
