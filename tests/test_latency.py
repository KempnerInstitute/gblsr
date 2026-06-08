"""Tests for the fixed GPU latency protocol."""

import pytest
import torch
import torch.nn as nn

from gblsr.latency import (
    LatencyConfig,
    LatencyResult,
    measure_latency,
    measure_latency_grid,
)


class _IdentityModel(nn.Module):
    """No-op model whose forward pass returns the input.

    Used as a sentinel model in latency tests: we care about the
    protocol's accounting, not the model's behavior.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_measure_latency_returns_summary() -> None:
    """``measure_latency`` returns the expected summary fields."""
    model = _IdentityModel().eval()
    x = torch.randn(1, 3, 32, 32)
    cfg = LatencyConfig(n_warmup=2, n_timed=5)
    result = measure_latency(model, x, cfg=cfg, track_peak_memory=False)

    assert isinstance(result, LatencyResult)
    assert len(result.per_image_times_ms) == 5
    assert result.median_ms > 0
    assert result.mean_ms > 0
    assert result.std_ms >= 0
    # Percentiles must respect the ordering median <= p90 <= p95.
    assert result.p90_ms >= result.median_ms
    assert result.p95_ms >= result.p90_ms
    # The protocol used is recorded with the result.
    assert result.cfg.n_warmup == 2
    assert result.cfg.n_timed == 5
    # CPU run has no peak-memory entry.
    assert result.peak_memory_mb is None


def test_measure_latency_rejects_training_mode() -> None:
    """The protocol bails when the model is not in eval() mode.

    Silent training-mode timing is a common pitfall: BatchNorm /
    dropout flip behavior, producing misleading numbers.
    """
    model = _IdentityModel()  # not eval()
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(RuntimeError, match="eval"):
        measure_latency(model, x, cfg=LatencyConfig(n_warmup=1, n_timed=1))


def test_measure_latency_rejects_autocast() -> None:
    """The protocol bails inside a torch.amp.autocast context.

    AMP changes the dtype and kernel selection; the fixed protocol
    requires no AMP.
    """
    model = _IdentityModel().eval()
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(RuntimeError, match="autocast"):
        with torch.amp.autocast(device_type="cpu", enabled=True):
            measure_latency(model, x, cfg=LatencyConfig(n_warmup=1, n_timed=1))


def test_measure_latency_grid_returns_one_result_per_sample() -> None:
    """``measure_latency_grid`` returns one ``LatencyResult`` per input."""
    model = _IdentityModel().eval()
    samples = {
        "small": torch.randn(1, 3, 16, 16),
        "medium": torch.randn(1, 3, 32, 32),
    }
    results = measure_latency_grid(
        model,
        samples,
        cfg=LatencyConfig(n_warmup=1, n_timed=3),
        track_peak_memory=False,
    )
    assert set(results.keys()) == {"small", "medium"}
    for r in results.values():
        assert len(r.per_image_times_ms) == 3
        assert r.median_ms > 0
