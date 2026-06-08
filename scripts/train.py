#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.train``.

Equivalent to::

    uv run gblsr-train ...
    # or:
    python -m gblsr.cli.train ...
"""

import sys

from gblsr.cli.train import main


if __name__ == "__main__":
    sys.exit(main())
