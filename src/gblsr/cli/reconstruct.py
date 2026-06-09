"""CLI: reconstruct an image with a trained GB-LSR checkpoint.

Loads the model architecture from a YAML config, restores weights
from a checkpoint, runs a single forward pass on the input image,
and writes the reconstruction to the output path.

``LocalSpectralArm`` (the main GB-LSR variant) handles inputs of
arbitrary spatial size: ``LocalSpectralArm.forward`` reflect-pads to
the nearest multiple of ``patch_size`` and crops the reconstruction
back to the input's original size. ``BaselineArm`` (Global
Fourier-MLP) is shape-locked to the configured ``image_size`` and
will raise on non-matching inputs.

Usage::

    uv run gblsr-reconstruct \\
        --config configs/example.yaml \\
        --checkpoint runs/example/gb_lsr_scalar__seed0/model.pt \\
        --input  path/to/image.png \\
        --output path/to/recon.png

    # or:
    python -m gblsr.cli.reconstruct --config ... --checkpoint ... --input ... --output ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from gblsr.training.trainer import (
    build_model_from_run_config,
    expand_run_configs,
    load_config,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct an image with a trained GB-LSR checkpoint.",
    )
    p.add_argument("--config", required=True, help="Path to a RunConfig YAML.")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint file (.pt) saved by the trainer.",
    )
    p.add_argument("--input", required=True, help="Path to the input image (PIL-readable).")
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the reconstructed image (extension picks the encoder).",
    )
    p.add_argument("--device", default=None, help='"cuda" | "cpu" (default: auto).')
    p.add_argument(
        "--arm-index",
        type=int,
        default=0,
        help="If the config has multiple arms, which to use (default: 0).",
    )
    return p


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """Load a checkpoint and return a state_dict.

    Accepts the trainer's dict format (``{"state_dict": ..., "config": ...,
    ...}``) and a bare ``state_dict`` dict.
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


def _load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an image, return a ``(1, 3, H, W)`` tensor in ``[0, 1]``."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)
    return t.unsqueeze(0).to(device)


def _save_tensor_image(t: torch.Tensor, path: str) -> None:
    """Save a ``(1, 3, H, W)`` or ``(3, H, W)`` tensor in ``[0, 1]``.

    The output file format is selected by the path extension via PIL.
    """
    if t.dim() == 4:
        t = t[0]
    t = t.detach().clamp(0.0, 1.0).cpu()
    img = TF.to_pil_image(t)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


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

    x = _load_image_tensor(args.input, device)
    with torch.no_grad():
        out = model(x)
        recon = out["recon"]

    _save_tensor_image(recon, args.output)

    print(f"arm        = {rc.arm}")
    print(f"input      = {args.input} (shape {tuple(x.shape)})")
    print(f"output     = {args.output} (shape {tuple(recon.shape)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
