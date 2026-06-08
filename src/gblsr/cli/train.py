"""CLI: train one or more ``(arm, seed)`` runs from a RunConfig YAML.

Wraps :func:`gblsr.training.trainer.train_one` with argparse + a
YAML-loading frontend. Each cell in the YAML's
``arms x seeds`` grid produces one ``train_one()`` invocation; the
combined per-cell aggregates are summarized in
``<run_root>/summary.json`` when more than one cell is trained.

Usage::

    uv run gblsr-train --config configs/example.yaml
    # or:
    python -m gblsr.cli.train --config configs/example.yaml

    # subset the grid:
    uv run gblsr-train --config configs/example.yaml --only-arm local_spectral
    uv run gblsr-train --config configs/example.yaml --only-seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gblsr.training.trainer import expand_run_configs, load_config, train_one


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train one or more GB-LSR (arm, seed) cells from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to a RunConfig YAML.")
    p.add_argument(
        "--only-arm",
        default=None,
        help="Optional filter: train only cells whose arm matches this value.",
    )
    p.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="Optional filter: train only cells whose seed matches this value.",
    )
    args = p.parse_args(argv)

    cfg_dict = load_config(args.config)
    runs = expand_run_configs(cfg_dict)
    if args.only_arm:
        runs = [r for r in runs if r.arm == args.only_arm]
    if args.only_seed is not None:
        runs = [r for r in runs if r.seed == args.only_seed]

    summary = []
    for rc in runs:
        print(f"\n=== {rc.experiment_id} / {rc.arm} / seed={rc.seed} ===", flush=True)
        out = train_one(rc)
        summary.append(
            {
                "arm": rc.arm,
                "seed": rc.seed,
                "run_dir": rc.run_dir,
                "aggregate": out["aggregate"],
            }
        )
    if len(runs) > 1:
        group_dir = Path(cfg_dict["run_root"])
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
