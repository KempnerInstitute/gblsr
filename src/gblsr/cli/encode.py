"""CLI: encode an image to a GB-LSR feature tensor.

Use case: split the GB-LSR forward pass across machines. Encode on
machine A, transfer the (much smaller) feature file, decode on machine
B via ``gblsr-decode``. With the default GB-LSR config
(``patch_size=32``, ``d_feat=128``), the feature tensor is roughly
24x smaller than the raw uint8 input image.

Both ends must use the same model architecture and the same trained
checkpoint; the feature space is model-specific. The decode side
performs sanity checks on the blob's ``arm`` / ``bandwidth_mode`` /
``patch_size`` to catch checkpoint mismatches early.

Only the local-spectral arm (``arm: local_spectral``) is supported.
The Global Fourier-MLP baseline arm uses a shape-locked global-pool
decoder and is not amenable to per-patch feature transfer.

Output file format (torch.save dict)::

    {
        "feat":           (B, D, nH, nW) tensor on CPU,
        "pad_info":       dict (orig_h/w, padded_h/w, pad_t/b/l/r, ...),
        "orig_shape":     tuple of the input shape (B, C, H, W),
        "arm":            arm name from the RunConfig,
        "bandwidth_mode": bandwidth mode (sanity check on decode),
        "patch_size":     patch_size (sanity check on decode),
    }

Usage::

    uv run gblsr-encode \\
        --config configs/example.yaml \\
        --checkpoint runs/example/gb_lsr_scalar__seed0/model.pt \\
        --input  path/to/image.png \\
        --output path/to/features.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from gblsr.models import pad_to_multiple
from gblsr.training.trainer import (
    build_model_from_run_config,
    expand_run_configs,
    load_config,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encode an image to a GB-LSR feature tensor.",
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
        help="Path to write the feature blob (.pt file via torch.save).",
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

    Accepts the trainer's dict format (``{"state_dict": ...,
    "config": ..., ...}``) and a bare ``state_dict`` dict.
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

    if rc.arm != "local_spectral":
        print(
            f"ERROR: gblsr-encode supports only arm='local_spectral'; got arm={rc.arm!r}. "
            "The Global Fourier-MLP baseline arm uses a shape-locked global-pool decoder "
            "and is not amenable to per-patch feature transfer.",
            file=sys.stderr,
        )
        return 1

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model_from_run_config(rc).to(device)
    state_dict = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open(args.input).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(device)

    # Same pad-then-encode steps that LocalSpectralArm.forward performs
    # internally, but we stop after the encoder and serialize.
    P = rc.patch_size
    x_pad, pad_info = pad_to_multiple(x, P, mode="reflect")
    with torch.no_grad():
        feat = model.encoder(x_pad)

    blob = {
        "feat": feat.cpu(),
        "pad_info": pad_info,
        "orig_shape": tuple(x.shape),
        "arm": rc.arm,
        "bandwidth_mode": rc.bandwidth_mode,
        "patch_size": rc.patch_size,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)

    in_bytes = x.nelement() * x.element_size()
    feat_bytes = feat.nelement() * feat.element_size()
    on_disk = out_path.stat().st_size
    ratio_in_vs_feat = in_bytes / feat_bytes
    print(f"arm           = {rc.arm}")
    print(f"input         = {args.input} (shape {tuple(x.shape)}, {in_bytes:,} bytes fp32)")
    print(
        f"feat          = {tuple(feat.shape)} ({feat_bytes:,} bytes fp32; "
        f"{ratio_in_vs_feat:.2f}x smaller than input)"
    )
    print(f"output        = {out_path} ({on_disk:,} bytes on disk, includes pad_info + metadata)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
