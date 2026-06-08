"""Smoke tests for the gblsr package.

Verifies the package is importable and every submodule loads without
raising. CI runs these on every change; if any import path breaks
during refactors, this test catches it before downstream code does.
"""

import gblsr


def test_version():
    assert gblsr.__version__ == "0.0.0"


def test_subpackages_import():
    from gblsr.cli import eval, latency, reconstruct, train  # noqa: F401
    from gblsr.data import loaders  # noqa: F401
    from gblsr.encoders import rdn  # noqa: F401
    from gblsr.latency import protocol  # noqa: F401
    from gblsr.metrics import quality  # noqa: F401
    from gblsr.models import arms, basis, shape_utils  # noqa: F401
    from gblsr.training import losses, trainer  # noqa: F401
