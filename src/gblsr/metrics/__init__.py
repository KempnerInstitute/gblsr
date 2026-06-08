"""Public API for GB-LSR's image-quality metric panel.

Re-exports per-image metrics (PSNR, SSIM, LPIPS, edge-LPIPS,
local-spectrum error) and the per-image / per-region panel functions.
LPIPS uses the ``lpips`` package with the AlexNet backbone.
"""

from .quality import (
    compute_metric_panel,
    compute_metric_panel_varres,
    edge_lpips,
    edge_mask,
    get_lpips,
    has_lpips,
    local_spectrum_error,
    lpips_metric,
    per_region_mean,
    psnr,
    ssim,
)

__all__ = [
    # Per-image scalars
    "psnr",
    "ssim",
    "lpips_metric",
    "edge_lpips",
    "local_spectrum_error",
    # Panels
    "compute_metric_panel",
    "compute_metric_panel_varres",
    "per_region_mean",
    # Helpers
    "edge_mask",
    "has_lpips",
    "get_lpips",
]
