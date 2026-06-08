#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.encode``.

Equivalent to::

    uv run gblsr-encode ...
    # or:
    python -m gblsr.cli.encode ...
"""

import sys

from gblsr.cli.encode import main


if __name__ == "__main__":
    sys.exit(main())
