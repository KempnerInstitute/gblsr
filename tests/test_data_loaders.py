"""Tests for ``gblsr.data.loaders``: region slicing, collate functions,
and threshold serialization.

Region-slicing tests use synthetic patches whose expected outputs are
known analytically (a constant patch has zero high-frequency energy;
a Nyquist checkerboard has nearly all its energy above any reasonable
HF cutoff; a hard step edge has high Sobel-magnitude response).

Collate tests cover both the fixed-size (``collate``) and
variable-resolution (``varres_collate``) paths plus the
``make_collate`` dispatcher.

Threshold serialization tests round-trip a ``RegionThresholds``
through ``save_thresholds`` and ``load_thresholds``.
"""

from __future__ import annotations

import pytest
import torch

from gblsr.data.loaders import (
    REGIONS,
    RegionThresholds,
    VARRES_BATCH_GT_1_MSG,
    collate,
    label_patches,
    load_thresholds,
    make_collate,
    per_patch_edge_density,
    per_patch_hf_fraction,
    save_thresholds,
    varres_collate,
)


def test_regions_constant_is_canonical() -> None:
    """``REGIONS`` is the canonical 4-name list. Order matters
    (it is used as the label-int -> name decoder downstream)."""
    assert REGIONS == ["smooth", "edge", "texture", "mixed"]


def test_hf_fraction_constant_image_is_zero() -> None:
    """A constant-color patch has all its energy in the DC component;
    the HF mask is zero at DC, so the HF fraction is ~0."""
    P = 8
    x = torch.full((1, 3, P, P), 0.5)
    frac = per_patch_hf_fraction(x, patch_size=P, cutoff=0.25)
    assert frac.shape == (1, 1, 1)
    assert frac.item() < 1e-3


def test_hf_fraction_checkerboard_substantial_and_exceeds_flat() -> None:
    """A pixel-wise {0, 1} checkerboard concentrates its non-DC energy
    at the Nyquist frequency. Because the pattern has mean 0.5, the
    DC and Nyquist contributions are each about half of the total
    energy (each ``(0.5 * P)^2`` in ortho-norm), so the HF fraction
    is roughly 0.5. The exact value matters less than the fact that
    it is far above the flat-image baseline (~0)."""
    P = 8
    flat = torch.full((1, 3, P, P), 0.5)
    pattern = torch.zeros(P, P)
    pattern[::2, ::2] = 1.0
    pattern[1::2, 1::2] = 1.0
    x_checker = pattern.unsqueeze(0).unsqueeze(0).expand(1, 3, P, P).contiguous()
    f_flat = per_patch_hf_fraction(flat, patch_size=P).item()
    f_checker = per_patch_hf_fraction(x_checker, patch_size=P).item()
    assert 0.3 < f_checker < 0.7
    assert f_checker > f_flat + 0.3


def test_edge_density_step_exceeds_flat() -> None:
    """A hard interior step has higher edge density than a flat patch.

    NOTE: a uniform-gray patch does NOT have zero edge density, because
    Sobel uses ``padding=1`` (zero-padding) so every boundary pixel
    sees a step from the zero pad into the gray interior and responds
    non-trivially. The closed-form expected value is
    ``ed_flat ≈ (8P - 8) / P^2`` for a uniform-0.5 patch
    (≈ 0.875 at P=8, ≈ 0.123 at P=64). A patch with an interior step
    has additional gradient signal on top of that, so its mean ED is
    strictly larger.
    """
    P = 16
    flat = torch.full((1, 3, P, P), 0.5)
    step = torch.zeros(1, 3, P, P)
    step[:, :, :, P // 2 :] = 1.0
    ed_flat = per_patch_edge_density(flat, patch_size=P).item()
    ed_step = per_patch_edge_density(step, patch_size=P).item()
    assert ed_flat >= 0.0
    assert ed_step > ed_flat


def test_edge_density_hard_step_is_nonzero() -> None:
    """A hard vertical step (half-black / half-white) produces a
    large Sobel-gradient response at the step location, so the
    patch-mean edge density is well above zero."""
    P = 16
    pattern = torch.zeros(P, P)
    pattern[:, P // 2 :] = 1.0
    x = pattern.unsqueeze(0).unsqueeze(0).expand(1, 3, P, P).contiguous()
    ed = per_patch_edge_density(x, patch_size=P)
    assert ed.item() > 0.1


def test_label_patches_flat_assigned_smooth() -> None:
    """A flat patch labels as 0 (smooth) when the edge-density
    threshold is set above the Sobel-with-zero-padding boundary
    artifact baseline.

    For a uniform-0.5 patch the analytic ED is roughly
    ``(8P - 8) / P^2``: ≈ 0.875 at P=8, ≈ 0.123 at P=64. We use P=64
    here so the boundary effect is small (~0.12) and a comfortable
    ``edge_median = 0.5`` places the flat-patch ED below threshold,
    triggering ``is_smooth``."""
    P = 64
    x = torch.full((1, 3, P, P), 0.5)
    th = RegionThresholds(hf_q1=0.05, hf_q3=0.50, edge_median=0.5)
    labels = label_patches(x, patch_size=P, thresholds=th)
    assert labels.shape == (1, 1, 1)
    assert labels.dtype == torch.long
    assert int(labels[0, 0, 0]) == 0


def test_label_patches_checkerboard_assigned_texture() -> None:
    """A Nyquist checkerboard (hf ~ 1) -> label 2 (texture) given
    thresholds with q3 < 1."""
    P = 8
    pattern = torch.zeros(P, P)
    pattern[::2, ::2] = 1.0
    pattern[1::2, 1::2] = 1.0
    x = pattern.unsqueeze(0).unsqueeze(0).expand(1, 3, P, P).contiguous()
    th = RegionThresholds(hf_q1=0.05, hf_q3=0.30, edge_median=0.1)
    labels = label_patches(x, patch_size=P, thresholds=th)
    assert int(labels[0, 0, 0]) == 2


def test_label_patches_output_shape() -> None:
    """``label_patches`` returns ``(B, nH, nW)`` with the right
    block-grid shape for the input."""
    P = 8
    # 1x3x32x32 -> 4x4 patch grid per image.
    x = torch.zeros(1, 3, 32, 32)
    th = RegionThresholds(hf_q1=0.05, hf_q3=0.30, edge_median=0.1)
    labels = label_patches(x, patch_size=P, thresholds=th)
    assert labels.shape == (1, 4, 4)
    assert labels.dtype == torch.long


# ---------- collate functions ----------


def test_collate_stacks_locked_size_samples() -> None:
    """Fixed-size ``collate`` stacks ``[3, S, S]`` tensors along a new
    batch axis and returns the 3-tuple ``(x, names, paths)``."""
    samples = [
        (torch.randn(3, 16, 16), "dtd", "p1"),
        (torch.randn(3, 16, 16), "div2k", "p2"),
        (torch.randn(3, 16, 16), "dtd", "p3"),
    ]
    x, names, paths = collate(samples)
    assert x.shape == (3, 3, 16, 16)
    assert names == ["dtd", "div2k", "dtd"]
    assert paths == ["p1", "p2", "p3"]


def test_varres_collate_returns_meta_4_tuple() -> None:
    """``varres_collate`` produces ``(x[1, 3, H, W], names, paths, metas)``
    from a single variable-size sample."""
    t = torch.randn(3, 73, 97)
    meta = {
        "filename": "x.png",
        "dataset": "synthetic",
        "original_h": 73,
        "original_w": 97,
        "path": "/tmp/x.png",
    }
    batch = [(t, "synthetic", "/tmp/x.png", meta)]
    x, names, paths, metas = varres_collate(batch)
    assert x.shape == (1, 3, 73, 97)
    assert names == ["synthetic"]
    assert paths == ["/tmp/x.png"]
    assert metas == [meta]


def test_varres_collate_rejects_batch_gt_1() -> None:
    """Stacking rectangular tensors of different sizes is impossible,
    so ``varres_collate`` raises ``NotImplementedError`` for batches > 1."""
    a = (torch.randn(3, 16, 16), "n", "p", {})
    b = (torch.randn(3, 32, 48), "n", "p", {})
    with pytest.raises(NotImplementedError, match=VARRES_BATCH_GT_1_MSG.split(";")[0]):
        varres_collate([a, b])


def test_make_collate_dispatches() -> None:
    """``make_collate(varres=True)`` returns ``varres_collate``;
    ``make_collate(varres=False)`` returns ``collate``."""
    assert make_collate(True) is varres_collate
    assert make_collate(False) is collate


# ---------- threshold serialization ----------


def test_save_and_load_thresholds_round_trips(tmp_path) -> None:
    """``load_thresholds(save_thresholds(th, path))`` returns a
    ``RegionThresholds`` equal to ``th``."""
    th = RegionThresholds(hf_q1=0.123, hf_q3=0.789, edge_median=0.456)
    path = tmp_path / "subdir" / "th.json"  # exercises mkdir parents=True
    save_thresholds(th, str(path))
    assert path.exists()
    th2 = load_thresholds(str(path))
    assert th2.hf_q1 == 0.123
    assert th2.hf_q3 == 0.789
    assert th2.edge_median == 0.456
