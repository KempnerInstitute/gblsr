"""CLI: measure inference latency under the fixed GPU latency protocol.

Times a forward pass of the model described by a YAML config using the
fixed protocol from :mod:`gblsr.latency.protocol`
(default 10 warm-up + 50 timed reps, ``torch.cuda.Event`` timing,
no AMP / no ``torch.compile`` / no CUDA Graphs).

Usage::

    uv run gblsr-measure-latency --config configs/example.yaml
    # or:
    python -m gblsr.cli.latency --config configs/example.yaml

Optional arguments override the config (or supply settings the config
does not carry):

    --height / --width          : input spatial dimensions (default: cfg.image_size)
    --device                    : "cuda" | "cpu" (default: cuda if available)
    --arm-index                 : if the config has multiple arms, pick one
    --warmup / --timed          : protocol knobs (default: 10 / 50)
    --no-peak-memory            : skip the peak-memory measurement (CUDA only)
"""

from __future__ import annotations

import argparse
import sys

import torch

from gblsr.latency import LatencyConfig, measure_latency
from gblsr.models.arms import n_params
from gblsr.training.trainer import (
    build_model_from_run_config,
    expand_run_configs,
    load_config,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure GB-LSR inference latency under the fixed GPU latency protocol.",
    )
    p.add_argument("--config", required=True, help="Path to a RunConfig YAML.")
    p.add_argument("--device", default=None, help='"cuda" | "cpu" (default: auto).')
    p.add_argument("--height", type=int, default=None, help="Override input height in pixels.")
    p.add_argument("--width", type=int, default=None, help="Override input width in pixels.")
    p.add_argument(
        "--arm-index",
        type=int,
        default=0,
        help="If the config has multiple arms, which to time (default: 0).",
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="Number of warm-up forward passes (default: 10)."
    )
    p.add_argument(
        "--timed", type=int, default=50, help="Number of timed forward passes (default: 50)."
    )
    p.add_argument("--no-peak-memory", action="store_true", help="Skip CUDA peak-memory tracking.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    cfg_dict = load_config(args.config)
    runs = expand_run_configs(cfg_dict)
    if not runs:
        print("ERROR: config has no arms", file=sys.stderr)
        return 1
    if args.arm_index < 0 or args.arm_index >= len(runs):
        print(
            f"ERROR: --arm-index {args.arm_index} out of range (0..{len(runs) - 1})",
            file=sys.stderr,
        )
        return 1
    rc = runs[args.arm_index]

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model_from_run_config(rc).to(device).eval()

    H = args.height if args.height is not None else rc.image_size
    W = args.width if args.width is not None else rc.image_size
    sample = torch.randn(1, 3, H, W, device=device)

    print(f"arm            = {rc.arm}")
    print(f"bandwidth_mode = {rc.bandwidth_mode}")
    print(f"device         = {device}")
    print(f"input shape    = (1, 3, {H}, {W})")
    print(f"params         = {n_params(model):,}")
    print()

    lat_cfg = LatencyConfig(n_warmup=args.warmup, n_timed=args.timed)
    result = measure_latency(
        model,
        sample,
        cfg=lat_cfg,
        track_peak_memory=not args.no_peak_memory,
    )

    print("latency (ms):")
    print(f"  median = {result.median_ms:.3f}")
    print(f"  mean   = {result.mean_ms:.3f}")
    print(f"  std    = {result.std_ms:.3f}")
    print(f"  p90    = {result.p90_ms:.3f}")
    print(f"  p95    = {result.p95_ms:.3f}")
    if result.peak_memory_mb is not None:
        print(f"peak memory     = {result.peak_memory_mb:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
