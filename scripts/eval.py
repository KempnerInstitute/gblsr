#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.eval``.

Equivalent to::

    uv run gblsr-eval ...
    # or:
    python -m gblsr.cli.eval ...
"""

import sys

from gblsr.cli.eval import main


if __name__ == "__main__":
    sys.exit(main())
