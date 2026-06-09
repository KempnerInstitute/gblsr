"""Fixed GPU latency measurement protocol.

Per-image inference-cost measurement protocol. The protocol is
deliberately deployment-conservative: it reports the kind of latency a
naive PyTorch eager-mode deployment would see at single-image batch
size, with no specialized speedups enabled.

Protocol parameters (defaults)
------------------------------

  - Batch size: ``1``  (caller's responsibility; the sample tensor is
    passed in pre-batched).
  - Warm-up forward passes: ``WARMUP_DEFAULT = 10``.
  - Timed forward passes: ``TIMED_DEFAULT = 50``.
  - Timing on CUDA: ``torch.cuda.Event(enable_timing=True)`` with an
    explicit ``torch.cuda.synchronize()`` after each timed call so the
    elapsed time reflects on-device completion (not the host-side
    launch queue).
  - Timing on CPU: ``time.perf_counter()``.
  - No AMP, no ``torch.compile``, no CUDA Graphs. AMP is checked at
    call time (a global autocast context raises); ``torch.compile`` and
    CUDA Graphs cannot be detected reliably post-hoc, so the docstring
    is the contract for callers.
  - The model must be in ``eval()`` mode; ``measure_latency`` raises
    otherwise (silent training-mode timing is a common pitfall).
  - No file I/O or dataloader work inside the timed region: the sample
    tensor is pre-allocated on the target device by the caller.

Reported statistics
-------------------

For each ``(model, sample)`` pair, :func:`measure_latency` returns a
:class:`LatencyResult` containing the raw per-rep timings plus
median, mean, std, p90, and p95 in milliseconds. When the sample is
on CUDA and ``track_peak_memory=True`` (default), it also returns
peak device memory in MiB measured via
``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch


WARMUP_DEFAULT: int = 10
TIMED_DEFAULT: int = 50


@dataclass
class LatencyConfig:
    """Configuration for the fixed GPU latency protocol.

    Parameters
    ----------
    n_warmup : int
        Number of un-timed forward passes before the timed phase.
        Defaults to 10.
    n_timed : int
        Number of timed forward passes whose results enter the
        summary statistics. Defaults to 50.
    sync_before_timing : bool
        If True (default), call ``torch.cuda.synchronize`` after the
        warm-up phase so the timed phase starts from a clean queue.
        Has no effect on CPU.
    """

    n_warmup: int = WARMUP_DEFAULT
    n_timed: int = TIMED_DEFAULT
    sync_before_timing: bool = True


@dataclass
class LatencyResult:
    """Per-image latency measurement summary.

    Attributes
    ----------
    per_image_times_ms : list[float]
        Raw per-rep timings (sorted ascending) in milliseconds.
    median_ms, mean_ms, std_ms : float
        Summary statistics across the timed reps.
    p90_ms, p95_ms : float
        90th and 95th percentile of the per-rep timings.
    peak_memory_mb : float or None
        Peak device memory in MiB during the measurement, or ``None``
        on CPU / when ``track_peak_memory=False``.
    cfg : LatencyConfig
        The configuration used for this measurement (recorded so a
        ``LatencyResult`` is self-describing).
    """

    per_image_times_ms: list[float]
    median_ms: float
    mean_ms: float
    std_ms: float
    p90_ms: float
    p95_ms: float
    peak_memory_mb: float | None = None
    cfg: LatencyConfig = field(default_factory=LatencyConfig)


def _autocast_active() -> bool:
    """True if any common-device autocast context is currently active.

    PyTorch's autocast state is device-scoped; the no-argument form
    of ``torch.is_autocast_enabled()`` only checks the CUDA device.
    We OR over both ``"cuda"`` and ``"cpu"`` so a CPU-only test (or
    a CPU-side autocast on a mixed setup) cannot slip through.
    """

    for device_type in ("cuda", "cpu"):
        try:
            if torch.is_autocast_enabled(device_type):
                return True
        except TypeError:
            # Older PyTorch without device_type kwarg: no-arg form
            # only covers CUDA. Fall back, then break.
            if torch.is_autocast_enabled():
                return True
            break
    return False


def _validate_inference_settings(model: torch.nn.Module) -> None:
    """Reject obviously-wrong measurement setups.

    The protocol requires the model in ``eval()`` and no AMP. Both
    are common pitfalls when timing a model casually; silent
    training-mode timing reports wildly different numbers because
    BatchNorm / dropout flip their behavior.
    """

    if model.training:
        raise RuntimeError(
            "fixed GPU latency protocol expects the model in eval() mode; "
            "got training=True. Call model.eval() before measure_latency()."
        )
    if _autocast_active():
        raise RuntimeError(
            "fixed GPU latency protocol requires no AMP; an autocast "
            "context is currently active. Call measure_latency() outside "
            "any torch.amp.autocast(...) block."
        )


def _summarize(times_ms: list[float]) -> tuple[float, float, float, float, float]:
    """Return ``(median, mean, std, p90, p95)`` from a non-empty list of timings."""

    if not times_ms:
        raise RuntimeError("no timing samples collected")
    s = sorted(times_ms)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    mean = sum(s) / n
    var = sum((t - mean) ** 2 for t in s) / n
    std = math.sqrt(var)
    p90 = s[max(0, int(0.90 * (n - 1)))]
    p95 = s[max(0, int(0.95 * (n - 1)))]
    return float(median), float(mean), float(std), float(p90), float(p95)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    sample: torch.Tensor,
    cfg: LatencyConfig | None = None,
    track_peak_memory: bool = True,
) -> LatencyResult:
    """Measure per-image latency for one ``(model, sample)`` pair.

    Parameters
    ----------
    model : torch.nn.Module
        Model to time. Must already be in ``eval()`` mode and on the
        same device as ``sample``. The forward pass must accept
        ``sample`` as its only positional argument.
    sample : torch.Tensor
        Pre-allocated input tensor (already on the target device).
        Typically shape ``(1, 3, H, W)`` for single-image timing.
    cfg : LatencyConfig, optional
        Protocol configuration. Defaults to ``WARMUP=10, TIMED=50``.
    track_peak_memory : bool
        If True and ``sample`` is on CUDA, the result includes
        ``peak_memory_mb`` measured via
        ``torch.cuda.max_memory_allocated``. Defaults to True.

    Returns
    -------
    LatencyResult
        Aggregated summary statistics + raw per-rep timings + the
        ``LatencyConfig`` used.
    """

    cfg = cfg if cfg is not None else LatencyConfig()
    _validate_inference_settings(model)

    is_cuda = sample.is_cuda
    if is_cuda and track_peak_memory:
        torch.cuda.reset_peak_memory_stats(sample.device)

    # Warm-up phase (un-timed).
    for _ in range(cfg.n_warmup):
        _ = model(sample)
    if cfg.sync_before_timing and is_cuda:
        torch.cuda.synchronize(sample.device)

    # Timed phase.
    times_ms: list[float] = []
    if is_cuda:
        for _ in range(cfg.n_timed):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(sample)
            end.record()
            torch.cuda.synchronize(sample.device)
            times_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(cfg.n_timed):
            t0 = time.perf_counter()
            _ = model(sample)
            times_ms.append(float((time.perf_counter() - t0) * 1000.0))

    peak_mb: float | None = None
    if is_cuda and track_peak_memory:
        peak_bytes = torch.cuda.max_memory_allocated(sample.device)
        peak_mb = float(peak_bytes) / (1024.0 * 1024.0)

    median, mean, std, p90, p95 = _summarize(times_ms)
    return LatencyResult(
        per_image_times_ms=sorted(times_ms),
        median_ms=median,
        mean_ms=mean,
        std_ms=std,
        p90_ms=p90,
        p95_ms=p95,
        peak_memory_mb=peak_mb,
        cfg=cfg,
    )


def measure_latency_grid(
    model: torch.nn.Module,
    samples: dict[str, torch.Tensor],
    cfg: LatencyConfig | None = None,
    track_peak_memory: bool = True,
) -> dict[str, LatencyResult]:
    """Measure latency across multiple named samples with the same protocol.

    Convenient for sweeping a model over a ``(dataset, scale)`` or
    ``(dataset, image)`` grid. Each cell is measured independently
    (so peak-memory stats are per-cell, not cumulative).

    Returns
    -------
    dict[str, LatencyResult]
        Result mapped by input cell name.
    """

    return {
        name: measure_latency(model, sample, cfg=cfg, track_peak_memory=track_peak_memory)
        for name, sample in samples.items()
    }
