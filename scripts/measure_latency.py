#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.latency``.

Equivalent to::

    uv run gblsr-measure-latency ...
    # or:
    python -m gblsr.cli.latency ...
"""

import sys

from gblsr.cli.latency import main


if __name__ == "__main__":
    sys.exit(main())
