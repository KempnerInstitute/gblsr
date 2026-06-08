#!/usr/bin/env python
"""Thin repo-root shim for ``gblsr.cli.latency``.

Equivalent to running::

    uv run gblsr-measure-latency ...
    # or:
    python -m gblsr.cli.latency ...

provided so that users browsing the repo at the root level see a
``scripts/`` directory advertising the available command-line tools.
"""

import sys

from gblsr.cli.latency import main


if __name__ == "__main__":
    sys.exit(main())
