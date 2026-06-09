"""CLI: evaluate a trained GB-LSR checkpoint on a held-out split.

Builds the model from a YAML config, loads weights from a checkpoint,
runs evaluation against the val split that the YAML's data config
points at, and writes the aggregate metric panel
(PSNR / SSIM / LPIPS / edge-LPIPS / local-spectrum error) to stdout
as JSON.

Usage::

    uv run gblsr-eval --config configs/example.yaml \\
                      --checkpoint runs/example/gb_lsr_scalar__seed0/model.pt
    # or:
    python -m gblsr.cli.eval --config ... --checkpoint ...

The checkpoint may be either a trainer-saved dict (with a ``state_dict``
key) or a bare ``state_dict``; both are detected automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from gblsr.data.loaders import fit_region_thresholds
from gblsr.training.trainer import (
    aggregate,
    build_loaders,
    build_model_from_run_config,
    evaluate,
    expand_run_configs,
    load_config,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained GB-LSR checkpoint on the val split.",
    )
    p.add_argument("--config", required=True, help="Path to a RunConfig YAML.")
    p.add_argument(
        "--checkpoint", required=True, help="Path to a checkpoint file (.pt) saved by the trainer."
    )
    p.add_argument("--device", default=None, help='"cuda" | "cpu" (default: auto).')
    p.add_argument(
        "--arm-index",
        type=int,
        default=0,
        help="If the config has multiple arms, which to evaluate (default: 0).",
    )
    p.add_argument(
        "--num-val-images",
        type=int,
        default=None,
        help="Override RunConfig.num_val_images (cap the val split).",
    )
    p.add_argument(
        "--region-threshold-samples",
        type=int,
        default=1024,
        help="Number of patches sampled to fit region thresholds (default: 1024).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON aggregate (default: stdout).",
    )
    return p


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """Load a checkpoint file and return a state_dict.

    Accepts both the trainer's dict format (with ``state_dict`` key)
    and a bare state_dict.
    """
    # weights_only=False: the trainer saves a metadata dict alongside
    # the state_dict (arm, seed, experiment_id, git_commit, config),
    # which is not loadable under torch.load's safe-loading restriction.
    # Caller-supplied checkpoints are assumed to come from a trusted
    # trainer; do not pass arbitrary remote files to this loader.
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"checkpoint at {checkpoint_path} is not a dict; got {type(obj).__name__}")


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

    model = build_model_from_run_config(rc).to(device)
    state_dict = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()

    _, va_loader = build_loaders(rc.data, rc.batch_size, rc.data.num_workers)
    region_th = fit_region_thresholds(
        va_loader,
        rc.patch_size,
        n_samples=args.region_threshold_samples,
        device=str(device),
    )

    max_n = args.num_val_images if args.num_val_images is not None else rc.num_val_images
    rows = evaluate(model, va_loader, rc, region_th, device, max_n)
    agg = aggregate(rows)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "arm": rc.arm,
        "bandwidth_mode": rc.bandwidth_mode,
        "seed": rc.seed,
        "num_val_images": len(rows),
        "aggregate": agg,
    }
    pretty = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(pretty + "\n")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
