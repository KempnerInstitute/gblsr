#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.reconstruct``.

Equivalent to::

    uv run gblsr-reconstruct ...
    # or:
    python -m gblsr.cli.reconstruct ...
"""

import sys

from gblsr.cli.reconstruct import main


if __name__ == "__main__":
    sys.exit(main())
