"""Pad / crop helpers for the variable-resolution path.

These helpers make it possible to feed images of arbitrary spatial size
into a model whose internals require the spatial dims to be divisible
by some ``multiple`` (typically ``patch_size``, the encoder downsample
factor). Used by ``LocalSpectralArm`` in ``gblsr.models.arms`` to
support inputs whose ``H`` / ``W`` are not multiples of ``patch_size``.

Padding mode
------------
Default ``mode="reflect"`` matches the boundary handling used elsewhere
in the codebase. Reflect padding requires the per-side pad to be
strictly less than the corresponding spatial extent; for tiny inputs
(``H`` or ``W <= multiple``) we fall back to ``mode="replicate"`` so the
call always succeeds. The choice is recorded in
``pad_info["mode_used"]``.

Symmetry convention
-------------------
We split the total pad evenly across the two sides (left / right or
top / bottom), with any odd remainder going on the right / bottom side.
This matches ``F.pad``'s natural left/right/top/bottom ordering.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, dict]:
    """Pad spatial dims (last two) so they are divisible by `multiple`.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape (..., H, W). Typically (B, C, H, W).
    multiple : int
        Spatial divisor. Output H and W will be the smallest multiples of
        this value that are >= H and >= W.
    mode : str
        Initial padding mode passed to ``F.pad`` (default ``"reflect"``).
        Reflect requires pad < min(H, W); for tiny inputs (H <= multiple
        or W <= multiple, or any side pad >= H/W) we automatically fall
        back to ``"replicate"`` so the call always succeeds. The mode
        actually used is recorded in ``pad_info["mode_used"]``.

    Returns
    -------
    x_pad : torch.Tensor
        Padded tensor of shape (..., padded_h, padded_w).
    pad_info : dict
        Keys: ``orig_h``, ``orig_w``, ``padded_h``, ``padded_w``,
        ``pad_l``, ``pad_r``, ``pad_t``, ``pad_b``, ``mode_used``.
    """
    if x.ndim < 2:
        raise ValueError(f"pad_to_multiple expects ndim >= 2, got shape {tuple(x.shape)}")
    if multiple <= 0:
        raise ValueError(f"`multiple` must be positive, got {multiple}")

    H, W = int(x.shape[-2]), int(x.shape[-1])
    padded_h = ((H + multiple - 1) // multiple) * multiple
    padded_w = ((W + multiple - 1) // multiple) * multiple
    pad_h_total = padded_h - H
    pad_w_total = padded_w - W
    pad_t = pad_h_total // 2
    pad_b = pad_h_total - pad_t
    pad_l = pad_w_total // 2
    pad_r = pad_w_total - pad_l

    # Reflect requires per-side pad < spatial extent. Replicate has no
    # such constraint, so it is always a safe fallback.
    mode_used = mode
    if mode == "reflect" and (pad_t >= H or pad_b >= H or pad_l >= W or pad_r >= W):
        mode_used = "replicate"

    # `F.pad` only supports 4D / 5D tensors for reflect / replicate. The
    # arms feed (B, C, H, W) which is fine. For other ranks we promote.
    if x.ndim == 4:
        x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode_used)
    elif x.ndim == 3:
        x_pad = F.pad(x.unsqueeze(0), (pad_l, pad_r, pad_t, pad_b), mode=mode_used).squeeze(0)
    else:
        # Generic case: flatten leading dims to a batch axis.
        lead = x.shape[:-2]
        x_flat = x.reshape(-1, 1, H, W)
        x_pad_flat = F.pad(x_flat, (pad_l, pad_r, pad_t, pad_b), mode=mode_used)
        x_pad = x_pad_flat.reshape(*lead, padded_h, padded_w)

    pad_info = {
        "orig_h": H,
        "orig_w": W,
        "padded_h": padded_h,
        "padded_w": padded_w,
        "pad_l": pad_l,
        "pad_r": pad_r,
        "pad_t": pad_t,
        "pad_b": pad_b,
        "mode_used": mode_used,
        # Patch-grid metadata so downstream consumers can build the
        # valid_patch_mask without recomputing P or the grid shape.
        "patch_size": int(multiple),
        "num_patches_h": padded_h // int(multiple),
        "num_patches_w": padded_w // int(multiple),
    }
    return x_pad, pad_info


def crop_to_original(x_pad: torch.Tensor, pad_info: dict) -> torch.Tensor:
    """Inverse of :func:`pad_to_multiple`.

    Crops the last two spatial dims back to ``(orig_h, orig_w)`` using the
    pad offsets recorded in ``pad_info``.
    """
    pad_t = int(pad_info["pad_t"])
    pad_l = int(pad_info["pad_l"])
    H = int(pad_info["orig_h"])
    W = int(pad_info["orig_w"])
    return x_pad[..., pad_t : pad_t + H, pad_l : pad_l + W]


def make_valid_patch_mask(
    pad_info: dict,
    patch_size: int,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a boolean mask over the padded patch grid.

    A patch is valid iff its (P x P) extent in padded coordinates intersects
    the original image rectangle. ``pad_to_multiple`` uses symmetric padding
    (``pad_t`` / ``pad_l`` on top/left, ``pad_b`` / ``pad_r`` on bottom/right),
    so the original image lives at
    ``[pad_t : pad_t + orig_h, pad_l : pad_l + orig_w]`` inside the padded canvas.

    Validity rule (rectangle intersection):
        orig_y0 = pad_t,           orig_y1 = pad_t + orig_h
        orig_x0 = pad_l,           orig_x1 = pad_l + orig_w
        patch_y0 = row * P,        patch_y1 = patch_y0 + P
        patch_x0 = col * P,        patch_x1 = patch_x0 + P
        valid = (patch_y1 > orig_y0) AND (patch_y0 < orig_y1)
                AND (patch_x1 > orig_x0) AND (patch_x0 < orig_x1)

    Boundary patches that contain any original-image pixels are valid.
    Patches whose full PxP extent lies entirely inside the reflect/replicate
    pad region are invalid.

    Parameters
    ----------
    pad_info : dict
        Output of :func:`pad_to_multiple` (must contain ``orig_h``,
        ``orig_w``, ``padded_h``, ``padded_w``, ``pad_t``, ``pad_l``).
    patch_size : int
        Patch extent P in pixel coordinates. Must equal ``pad_info["patch_size"]``
        when the latter is present (we re-derive the grid shape from
        ``padded_h`` / ``padded_w`` to remain robust).
    batch_size : int | None
        If None, returns a 2D mask ``(num_patches_h, num_patches_w)``.
        Otherwise returns a 3D mask ``(batch_size, num_patches_h, num_patches_w)``
        broadcast across the batch axis (every batch element gets the
        same mask because we apply a single shared pad to the whole batch).
    device : torch.device | None
        Device for the returned tensor. Defaults to CPU.

    Returns
    -------
    torch.Tensor
        Boolean mask. Shape ``(num_patches_h, num_patches_w)`` if
        ``batch_size is None``, else ``(batch_size, num_patches_h, num_patches_w)``.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    padded_h = int(pad_info["padded_h"])
    padded_w = int(pad_info["padded_w"])
    pad_t = int(pad_info["pad_t"])
    pad_l = int(pad_info["pad_l"])
    orig_h = int(pad_info["orig_h"])
    orig_w = int(pad_info["orig_w"])

    if padded_h % patch_size != 0 or padded_w % patch_size != 0:
        raise ValueError(
            f"padded shape ({padded_h}, {padded_w}) is not a multiple of patch_size={patch_size}"
        )

    nH = padded_h // patch_size
    nW = padded_w // patch_size

    orig_y0 = pad_t
    orig_y1 = pad_t + orig_h
    orig_x0 = pad_l
    orig_x1 = pad_l + orig_w

    rows = torch.arange(nH, device=device)
    cols = torch.arange(nW, device=device)
    patch_y0 = rows * patch_size
    patch_y1 = patch_y0 + patch_size
    patch_x0 = cols * patch_size
    patch_x1 = patch_x0 + patch_size

    row_valid = (patch_y1 > orig_y0) & (patch_y0 < orig_y1)  # (nH,)
    col_valid = (patch_x1 > orig_x0) & (patch_x0 < orig_x1)  # (nW,)
    mask_2d = row_valid.unsqueeze(1) & col_valid.unsqueeze(0)  # (nH, nW)

    if batch_size is None:
        return mask_2d
    return mask_2d.unsqueeze(0).expand(batch_size, nH, nW)
