#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.eval``.

Equivalent to running::

    uv run gblsr-eval ...
    # or:
    python -m gblsr.cli.eval ...

provided so that users browsing the repo at the root level see a
``scripts/`` directory advertising the available command-line tools.
"""

import sys

from gblsr.cli.eval import main


if __name__ == "__main__":
    sys.exit(main())
