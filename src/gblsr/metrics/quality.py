"""Metrics: PSNR, SSIM, LPIPS, edge-LPIPS, local-spectrum error.

All metrics accept ``(B, 3, H, W)`` tensors in ``[0, 1]`` and return
per-image scalars (shape ``(B,)``) unless otherwise noted. Region-sliced
variants operate on per-patch labels from
``data.loaders.label_patches`` and return ``(B, n_regions)``.

PSNR / SSIM / LPIPS / edge-LPIPS are shape-agnostic. The local-spectrum
error and the per-region decomposition require ``H % patch_size == 0``
and ``W % patch_size == 0``; the variable-resolution panel
(``compute_metric_panel_varres``) returns only the shape-agnostic
metrics and omits the per-patch / per-region rows.

LPIPS backend is ``lpips.LPIPS(net="alex")``.
"""

from __future__ import annotations

import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.basis import image_to_patches
from ..data.loaders import REGIONS


def psnr(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """Per-image peak signal-to-noise ratio in dB.

    Accepts ``(B, 3, H, W)`` tensors in ``[0, data_range]`` and returns a
    ``(B,)`` tensor of PSNR values.
    """
    mse = ((pred - target) ** 2).mean(dim=(-3, -2, -1))
    return 10.0 * torch.log10((data_range**2) / (mse + eps))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return g.view(1, 1, -1)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Classical SSIM with Gaussian weighting; averaged across channels and pixels per image."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    B, C, H, W = pred.shape
    dtype = pred.dtype
    device = pred.device
    w1d = _gaussian_window(window_size, sigma, device, dtype)
    w2d = (w1d.transpose(-1, -2) @ w1d).view(1, 1, window_size, window_size)
    w = w2d.expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_p = F.conv2d(pred, w, padding=pad, groups=C)
    mu_t = F.conv2d(target, w, padding=pad, groups=C)
    mu_p2 = mu_p**2
    mu_t2 = mu_t**2
    mu_pt = mu_p * mu_t
    sigma_p2 = F.conv2d(pred * pred, w, padding=pad, groups=C) - mu_p2
    sigma_t2 = F.conv2d(target * target, w, padding=pad, groups=C) - mu_t2
    sigma_pt = F.conv2d(pred * target, w, padding=pad, groups=C) - mu_pt
    num = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)
    ssim_map = num / (den + eps)
    return ssim_map.mean(dim=(-3, -2, -1))


def has_lpips() -> bool:
    """Return True iff the optional ``lpips`` package is importable.

    Used by callers that want to gate LPIPS / edge-LPIPS gracefully
    rather than failing hard. The import is not performed eagerly here.
    """
    return importlib.util.find_spec("lpips") is not None


class LPIPSWrapper:
    def __init__(self, net: str = "alex"):
        self.net = net
        self._models: dict[str, nn.Module] = {}

    def get(self, device: torch.device) -> nn.Module:
        key = str(device)
        if key not in self._models:
            import lpips  # noqa

            m = lpips.LPIPS(net=self.net, verbose=False).to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self._models[key] = m
        return self._models[key]


_LPIPS: LPIPSWrapper | None = None


def get_lpips() -> LPIPSWrapper:
    """Return the package-singleton ``LPIPSWrapper`` (lazy-initialized, AlexNet backbone)."""
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = LPIPSWrapper(net="alex")
    return _LPIPS


def lpips_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expects [0, 1] images, scales to [-1, 1] for LPIPS.  Returns (B,)."""
    model = get_lpips().get(pred.device)
    p2 = pred * 2.0 - 1.0
    t2 = target * 2.0 - 1.0
    with torch.no_grad():
        d = model(p2, t2).view(-1)
    return d


def edge_mask(x: torch.Tensor, threshold_percentile: float = 0.8, dilate: int = 1) -> torch.Tensor:
    """Return (B, 1, H, W) binary mask of edge regions."""
    gray = x.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = (gx**2 + gy**2).sqrt()
    B = mag.shape[0]
    flat = mag.view(B, -1)
    q = torch.quantile(flat, threshold_percentile, dim=1, keepdim=True)
    mask = (flat >= q).view_as(mag).to(x.dtype)
    if dilate > 0:
        kernel = torch.ones((1, 1, 2 * dilate + 1, 2 * dilate + 1), device=x.device, dtype=x.dtype)
        mask = (F.conv2d(mask, kernel, padding=dilate) > 0).to(x.dtype)
    return mask


def edge_lpips(
    pred: torch.Tensor, target: torch.Tensor, threshold_percentile: float = 0.8
) -> torch.Tensor:
    """LPIPS restricted to edge regions via gray-masked substitution."""
    m = edge_mask(target, threshold_percentile=threshold_percentile)
    gray = target.new_full(target.shape, 0.5)
    pred_m = pred * m + gray * (1.0 - m)
    targ_m = target * m + gray * (1.0 - m)
    return lpips_metric(pred_m, targ_m)


def local_spectrum_error(pred: torch.Tensor, target: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Per-patch log-PSD L1 error, averaged per image."""
    gray_p = pred.mean(dim=1, keepdim=True)
    gray_t = target.mean(dim=1, keepdim=True)
    p = image_to_patches(gray_p, patch_size).squeeze(3)
    t = image_to_patches(gray_t, patch_size).squeeze(3)
    fp = torch.fft.fft2(p, norm="ortho")
    ft = torch.fft.fft2(t, norm="ortho")
    pp = (fp.real**2 + fp.imag**2).clamp_min(1e-8)
    pt = (ft.real**2 + ft.imag**2).clamp_min(1e-8)
    diff = (torch.log(pp) - torch.log(pt)).abs()
    return diff.mean(dim=(-4, -3, -2, -1))


def per_region_mean(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """values: (B, nH, nW); labels: same. Returns (B, n_regions), NaN where region absent."""
    assert values.ndim == labels.ndim, (values.shape, labels.shape)
    B = values.shape[0]
    out = values.new_full((B, len(REGIONS)), float("nan"))
    for r in range(len(REGIONS)):
        mask = (labels == r).to(values.dtype)
        n = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(1.0)
        s = (values * mask).sum(dim=tuple(range(1, values.ndim)))
        m = s / n
        present = mask.sum(dim=tuple(range(1, mask.ndim))) > 0
        out[:, r] = torch.where(present, m, values.new_full((B,), float("nan")))
    return out


def compute_metric_panel(
    pred: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    patch_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-image and per-region metrics in a single dict."""
    B = pred.shape[0]
    ps = psnr(pred, target)
    ss = ssim(pred, target)
    lp = lpips_metric(pred, target)
    elp = edge_lpips(pred, target)
    lse = local_spectrum_error(pred, target, patch_size)

    P = patch_size
    gp = image_to_patches(pred, P)
    gt = image_to_patches(target, P)
    mse_patch = ((gp - gt) ** 2).mean(dim=(-3, -2, -1))
    psnr_patch = 10.0 * torch.log10(1.0 / (mse_patch + 1e-8))
    psnr_reg = per_region_mean(psnr_patch, patch_labels)

    gp_flat = gp.reshape(B, *gp.shape[1:4], -1)
    gt_flat = gt.reshape(B, *gt.shape[1:4], -1)
    mu_p = gp_flat.mean(-1)
    mu_t = gt_flat.mean(-1)
    var_p = gp_flat.var(-1, unbiased=False)
    var_t = gt_flat.var(-1, unbiased=False)
    cov = ((gp_flat - mu_p.unsqueeze(-1)) * (gt_flat - mu_t.unsqueeze(-1))).mean(-1)
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_patch_ch = ((2 * mu_p * mu_t + C1) * (2 * cov + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
    )
    ssim_patch = ssim_patch_ch.mean(dim=-1)
    ssim_reg = per_region_mean(ssim_patch, patch_labels)

    gray_p = pred.mean(dim=1, keepdim=True)
    gray_t = target.mean(dim=1, keepdim=True)
    pp = image_to_patches(gray_p, patch_size).squeeze(3)
    tt = image_to_patches(gray_t, patch_size).squeeze(3)
    fp_ = torch.fft.fft2(pp, norm="ortho")
    ft_ = torch.fft.fft2(tt, norm="ortho")
    pp_power = (fp_.real**2 + fp_.imag**2).clamp_min(1e-8)
    tt_power = (ft_.real**2 + ft_.imag**2).clamp_min(1e-8)
    diff = (torch.log(pp_power) - torch.log(tt_power)).abs().mean(dim=(-2, -1))
    lse_reg = per_region_mean(diff, patch_labels)

    return {
        "psnr": ps,
        "ssim": ss,
        "lpips": lp,
        "edge_lpips": elp,
        "local_spectrum_error": lse,
        "psnr_region": psnr_reg,
        "ssim_region": ssim_reg,
        "local_spectrum_error_region": lse_reg,
    }


def compute_metric_panel_varres(
    pred: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
) -> dict[str, torch.Tensor]:
    """Variable-resolution metric panel.

    Returns the shape-agnostic metrics:
        ``psnr`` / ``ssim`` / ``lpips`` / ``edge_lpips`` /
        ``local_spectrum_error`` (the last computed only when both spatial
        dims are multiples of ``patch_size``; otherwise ``NaN`` is filled
        per-image so downstream aggregation can skip cleanly).

    Per-region rows are NOT included: the region-label pipeline is
    patch-grid bound and is not exposed in the variable-resolution
    panel. Callers that need per-region rows must run in fixed-size
    mode (``image_size`` set to a concrete int).

    Inputs ``pred`` and ``target`` must have the *same* shape. Padded
    pixels are not included in the metric: callers are expected to pass
    in tensors already cropped to the original ``(orig_h, orig_w)`` per
    ``LocalSpectralArm.forward``'s contract.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"compute_metric_panel_varres: pred {tuple(pred.shape)} and target "
            f"{tuple(target.shape)} must match (no padded-pixel inclusion)"
        )
    B, _, H, W = pred.shape
    ps = psnr(pred, target)
    ss = ssim(pred, target)
    if has_lpips():
        lp = lpips_metric(pred, target)
        elp = edge_lpips(pred, target)
    else:
        lp = pred.new_full((B,), float("nan"))
        elp = pred.new_full((B,), float("nan"))
    if H % patch_size == 0 and W % patch_size == 0:
        lse = local_spectrum_error(pred, target, patch_size)
    else:
        lse = pred.new_full((B,), float("nan"))
    return {
        "psnr": ps,
        "ssim": ss,
        "lpips": lp,
        "edge_lpips": elp,
        "local_spectrum_error": lse,
    }
