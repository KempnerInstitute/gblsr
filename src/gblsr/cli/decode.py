"""CLI: decode a GB-LSR feature tensor back to an image.

Paired with ``gblsr-encode``. Decode hard-fails on arm /
bandwidth_mode / patch_size mismatch between the blob and the loaded
checkpoint. Only ``arm="local_spectral"`` is supported.

Usage::

    uv run gblsr-decode \\
        --config configs/example.yaml \\
        --checkpoint runs/example/gb_lsr_scalar__seed0/model.pt \\
        --input  path/to/features.pt \\
        --output path/to/recon.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from gblsr.models import crop_to_original
from gblsr.training.trainer import (
    build_model_from_run_config,
    expand_run_configs,
    load_config,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode a GB-LSR feature tensor back to an image.",
    )
    p.add_argument("--config", required=True, help="Path to a RunConfig YAML.")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint file (.pt) saved by the trainer.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to the feature blob (.pt file produced by gblsr-encode).",
    )
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


def _load_feat_blob(blob_path: str, device: torch.device) -> dict:
    """Load and validate a feature blob written by gblsr-encode.

    Raises ``RuntimeError`` if required keys are missing.
    """
    # weights_only=False: pad_info is a Python dict (not a Tensor), and
    # the metadata fields (arm / bandwidth_mode / patch_size / orig_shape)
    # are plain Python objects. Same trust assumption as the checkpoint
    # loader: caller-supplied blobs are assumed to come from a trusted
    # gblsr-encode run.
    blob = torch.load(blob_path, map_location=device, weights_only=False)
    required = {"feat", "pad_info", "arm", "bandwidth_mode", "patch_size"}
    missing = required - set(blob.keys())
    if missing:
        raise RuntimeError(
            f"feature blob at {blob_path} is missing required keys: "
            f"{sorted(missing)!r}. Was it produced by gblsr-encode?"
        )
    return blob


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
            f"ERROR: gblsr-decode supports only arm='local_spectral'; got arm={rc.arm!r}.",
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

    blob = _load_feat_blob(args.input, device)

    # Sanity-check the blob against the loaded model. A mismatch on
    # arm / bandwidth_mode / patch_size means the checkpoint that
    # produced the blob is not the checkpoint we just loaded, and the
    # decode would silently produce garbage.
    if blob["arm"] != rc.arm:
        print(
            f"ERROR: blob's arm ({blob['arm']!r}) does not match loaded model's "
            f"arm ({rc.arm!r}); the checkpoint that encoded this blob is not the "
            "one passed via --checkpoint.",
            file=sys.stderr,
        )
        return 1
    if blob["bandwidth_mode"] != rc.bandwidth_mode:
        print(
            f"ERROR: blob's bandwidth_mode ({blob['bandwidth_mode']!r}) does not match "
            f"loaded model's bandwidth_mode ({rc.bandwidth_mode!r}).",
            file=sys.stderr,
        )
        return 1
    if blob["patch_size"] != rc.patch_size:
        print(
            f"ERROR: blob's patch_size ({blob['patch_size']}) does not match loaded "
            f"model's patch_size ({rc.patch_size}); decode would not align.",
            file=sys.stderr,
        )
        return 1

    feat = blob["feat"].to(device)
    pad_info = blob["pad_info"]
    with torch.no_grad():
        out = model.decoder(feat)
        recon = crop_to_original(out["recon"], pad_info).clamp(0.0, 1.0)

    # Save (PIL picks the encoder by file extension).
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(recon[0].detach().cpu()).save(out_path)

    print(f"arm           = {rc.arm}")
    print(
        f"input (feat)  = {args.input} (shape {tuple(feat.shape)}, "
        f"{feat.nelement() * feat.element_size():,} bytes fp32)"
    )
    print(f"output        = {out_path} (shape {tuple(recon.shape)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
