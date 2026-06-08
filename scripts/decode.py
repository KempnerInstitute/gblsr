#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.decode``.

Equivalent to::

    uv run gblsr-decode ...
    # or:
    python -m gblsr.cli.decode ...
"""

import sys

from gblsr.cli.decode import main


if __name__ == "__main__":
    sys.exit(main())
