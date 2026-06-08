"""Fixed GPU latency protocol for inference-cost characterization.

See :mod:`gblsr.latency.protocol` for the full protocol description.
"""

from .protocol import (
    TIMED_DEFAULT,
    WARMUP_DEFAULT,
    LatencyConfig,
    LatencyResult,
    measure_latency,
    measure_latency_grid,
)

__all__ = [
    "LatencyConfig",
    "LatencyResult",
    "measure_latency",
    "measure_latency_grid",
    "WARMUP_DEFAULT",
    "TIMED_DEFAULT",
]
