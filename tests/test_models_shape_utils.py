"""Tests for ``gblsr.models.shape_utils``: pad / crop / valid-patch-mask
helpers that enable the variable-resolution forward path.

These helpers are exercised indirectly inside ``LocalSpectralArm.forward``
(reflect-pad input -> run model -> crop back to original); the tests
here cover them directly so a bug in the geometry would surface as a
unit failure rather than a downstream shape mismatch.
"""

from __future__ import annotations

import torch

from gblsr.models import (
    crop_to_original,
    make_valid_patch_mask,
    pad_to_multiple,
)


def test_pad_to_multiple_pads_to_nearest_multiple() -> None:
    """A ``73 x 97`` input padded to multiples of 8 becomes ``80 x 104``.

    The pad is split evenly: top/left get ``floor``, bottom/right get
    the remainder.
    """
    x = torch.randn(1, 3, 73, 97)
    x_pad, info = pad_to_multiple(x, multiple=8, mode="reflect")
    assert x_pad.shape == (1, 3, 80, 104)
    assert info["orig_h"] == 73
    assert info["orig_w"] == 97
    assert info["padded_h"] == 80
    assert info["padded_w"] == 104
    # 80 - 73 = 7 -> pad_t=3, pad_b=4 ; 104 - 97 = 7 -> pad_l=3, pad_r=4
    assert info["pad_t"] == 3
    assert info["pad_b"] == 4
    assert info["pad_l"] == 3
    assert info["pad_r"] == 4
    assert info["num_patches_h"] == 10
    assert info["num_patches_w"] == 13


def test_pad_to_multiple_noop_when_already_aligned() -> None:
    """If H and W are already multiples of ``multiple``, ``pad_to_multiple``
    returns the input unchanged (zero pad on all sides)."""
    x = torch.randn(2, 3, 32, 64)
    x_pad, info = pad_to_multiple(x, multiple=8)
    assert torch.equal(x_pad, x)
    assert (info["pad_t"], info["pad_b"], info["pad_l"], info["pad_r"]) == (0, 0, 0, 0)


def test_pad_to_multiple_falls_back_to_replicate_for_tiny_inputs() -> None:
    """``reflect`` pad requires per-side pad < spatial extent. For a
    tiny ``2 x 3`` input padded to a multiple of 8, the required pad
    exceeds the spatial extent, so the helper falls back to
    ``replicate`` and records the fallback in ``pad_info["mode_used"]``."""
    x = torch.randn(1, 3, 2, 3)
    x_pad, info = pad_to_multiple(x, multiple=8, mode="reflect")
    assert x_pad.shape == (1, 3, 8, 8)
    assert info["mode_used"] == "replicate"


def test_pad_then_crop_round_trips() -> None:
    """``crop_to_original(pad_to_multiple(x))`` returns the original
    tensor unchanged (in shape and content)."""
    x = torch.randn(1, 3, 51, 73)
    x_pad, info = pad_to_multiple(x, multiple=8)
    x_back = crop_to_original(x_pad, info)
    assert x_back.shape == x.shape
    assert torch.equal(x_back, x)


def test_make_valid_patch_mask_marks_interior_patches_valid() -> None:
    """For a ``73 x 97`` input padded to ``80 x 104`` with ``patch_size=8``,
    the patch grid is ``10 x 13 = 130`` patches. The original image
    covers patches ``[0..9] x [0..12]`` (since padding is < patch_size),
    so all 130 patches contain at least one original-image pixel and
    should be valid."""
    info = {
        "orig_h": 73,
        "orig_w": 97,
        "padded_h": 80,
        "padded_w": 104,
        "pad_t": 3,
        "pad_b": 4,
        "pad_l": 3,
        "pad_r": 4,
        "patch_size": 8,
        "num_patches_h": 10,
        "num_patches_w": 13,
    }
    mask = make_valid_patch_mask(info, patch_size=8)
    assert mask.shape == (10, 13)
    assert mask.dtype == torch.bool
    assert bool(mask.all())  # every patch touches the original image


def test_make_valid_patch_mask_excludes_fully_padded_patches() -> None:
    """For an input where the pad on one side exceeds ``patch_size``,
    the patches that lie entirely inside the pad region are invalid."""
    # Original 4x4, padded to 32x32. patch_size=8 -> 4x4 grid.
    # Symmetric pad: pad_t=pad_b=pad_l=pad_r=14. Original lives at
    # [14:18, 14:18] inside the padded canvas. Patches at row index 0
    # span [0:8] which lies entirely inside the top pad (since 8 < 14)
    # -> invalid. Patch at row index 1 spans [8:16] which intersects
    # the original image at row 14-15 -> valid. Similarly for cols.
    info = {
        "orig_h": 4,
        "orig_w": 4,
        "padded_h": 32,
        "padded_w": 32,
        "pad_t": 14,
        "pad_b": 14,
        "pad_l": 14,
        "pad_r": 14,
        "patch_size": 8,
        "num_patches_h": 4,
        "num_patches_w": 4,
    }
    mask = make_valid_patch_mask(info, patch_size=8)
    assert mask.shape == (4, 4)
    # Only patches at (row 1 or 2) AND (col 1 or 2) intersect [14:18, 14:18].
    expected = torch.tensor(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_make_valid_patch_mask_with_batch_dim() -> None:
    """When ``batch_size`` is provided, the mask gets a leading batch
    axis (broadcast — same mask per batch element since the pad is
    shared across the batch)."""
    info = {
        "orig_h": 32,
        "orig_w": 32,
        "padded_h": 32,
        "padded_w": 32,
        "pad_t": 0,
        "pad_b": 0,
        "pad_l": 0,
        "pad_r": 0,
        "patch_size": 8,
        "num_patches_h": 4,
        "num_patches_w": 4,
    }
    mask = make_valid_patch_mask(info, patch_size=8, batch_size=5)
    assert mask.shape == (5, 4, 4)
    assert bool(mask.all())  # no padding -> every patch is original
