"""GB-LSR architecture: encoder, local-spectral decoder, and arm builders.

This module defines the building blocks of the GB-LSR family:

  - ``Encoder``: a small convolutional encoder that downsamples an image
    to the patch-grid resolution.
  - ``LocalSpectralDecoder``: the per-patch truncated-Fourier decoder
    with four bandwidth modes (``fixed_midpoint``, ``global_scalar``
    (the main variant), ``local_linear``, ``local_logspace``).
  - ``GlobalFourierMLPDecoder`` / ``BaselineArm``: a Global Fourier-MLP
    that pools encoder features to a global code and renders each pixel
    via a coordinate-MLP. Provided as a no-local-basis control variant.
  - ``LocalSpectralArm``: the full GB-LSR model (encoder + decoder).
  - ``build_model``: factory dispatching on ``cfg.arm``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn

from .basis import (
    BasisConfig,
    AdaptivityHead,
    global_fourier_features,
    patches_to_image,
    reconstruct_patch,
)
from .shape_utils import pad_to_multiple, crop_to_original, make_valid_patch_mask


@dataclass
class EncoderConfig:
    d_feat: int = 128
    groupnorm: int = 8


@dataclass
class ModelConfig:
    arm: Literal[
        "global_fourier_mlp",
        "local_spectral",
    ]
    image_size: int = 256
    patch_size: int = 32
    basis: BasisConfig = None
    encoder: EncoderConfig = None
    n_global_freq: int = 128
    decoder_hidden: int = 256
    decoder_layers: int = 3
    max_freq_cycles_per_image: float = 16.0


class Encoder(nn.Module):
    """Small CNN that downsamples to patch-grid resolution."""

    def __init__(self, image_size: int, patch_size: int, cfg: EncoderConfig):
        super().__init__()
        # The encoder is fully convolutional; the only runtime
        # requirement is that the input spatial dims be a multiple of
        # 2 ** n_down (== ``patch_size``). The padding helpers in
        # ``LocalSpectralArm.forward`` enforce that.
        assert (patch_size & (patch_size - 1)) == 0, "patch_size should be power of two"
        n_down = int(math.log2(patch_size))
        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(3, cfg.d_feat, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(cfg.groupnorm, cfg.d_feat))
        layers.append(nn.SiLU())
        for _ in range(n_down):
            layers.append(nn.Conv2d(cfg.d_feat, cfg.d_feat, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(cfg.groupnorm, cfg.d_feat))
            layers.append(nn.SiLU())
        layers.append(nn.Conv2d(cfg.d_feat, cfg.d_feat, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, D, nH, nW)."""
        return self.net(x)


class GlobalFourierMLPDecoder(nn.Module):
    """Baseline decoder: pool to global code, evaluate a coordinate Fourier-MLP per pixel."""

    def __init__(self, image_size: int, cfg: ModelConfig):
        super().__init__()
        self.image_size = image_size
        torch.manual_seed(0)
        self.register_buffer(
            "B_freq",
            torch.randn(cfg.n_global_freq, 2) * cfg.max_freq_cycles_per_image * 0.5,
        )
        d_in = 2 * cfg.n_global_freq + cfg.encoder.d_feat
        layers: list[nn.Module] = []
        for i in range(cfg.decoder_layers):
            din = d_in if i == 0 else cfg.decoder_hidden
            layers.append(nn.Linear(din, cfg.decoder_hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(cfg.decoder_hidden, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """(B, D, nH, nW) -> (B, 3, H, W)."""
        B = feat_map.shape[0]
        H = W = self.image_size
        z = feat_map.mean(dim=(-2, -1))
        device = feat_map.device
        u = torch.linspace(-1.0, 1.0, H, device=device)
        v = torch.linspace(-1.0, 1.0, W, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        coords = torch.stack([uu, vv], dim=-1)
        ff = global_fourier_features(coords, self.B_freq)
        ff = ff.view(1, H * W, -1).expand(B, -1, -1)
        z = z.view(B, 1, -1).expand(-1, H * W, -1)
        inp = torch.cat([ff, z], dim=-1)
        out = self.mlp(inp)
        return out.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()


BANDWIDTH_MODES = ("fixed_midpoint", "global_scalar", "local_linear", "local_logspace")


class LocalSpectralDecoder(nn.Module):
    """Local-spectral decoder: per-patch basis coefficients + adaptivity head.

    ``bandwidth_mode`` controls how the per-patch bandwidth ``s_e`` is produced:
      - ``"fixed_midpoint"`` : ``s_e`` pinned to the linear midpoint of
                               ``s_e_range`` (no training of the bandwidth).
      - ``"global_scalar"``  : one learnable scalar ``s_raw`` shared
                               across all patches, mapped via the
                               geometric (log-space) sigmoid to ``s_e``.
                               Broadcast to ``(B, nH, nW, 1)``.
      - ``"local_linear"``   : per-patch ``s_e`` from ``AdaptivityHead``
                               with a linear-sigmoid bandwidth head.
      - ``"local_logspace"`` : per-patch ``s_e`` from ``AdaptivityHead``
                               with a log-space sigmoid bandwidth head
                               (geometric midpoint at zero-init).

    If ``bandwidth_mode is None`` it is derived from ``adapt_bandwidth``:
        ``adapt_bandwidth=False`` -> ``"fixed_midpoint"``
        ``adapt_bandwidth=True``  -> ``"local_linear"``
    """

    def __init__(
        self,
        basis_cfg: BasisConfig,
        encoder_cfg: EncoderConfig,
        adapt_bandwidth: bool = True,
        adapt_order: bool = True,
        bandwidth_mode: str | None = None,
    ):
        super().__init__()
        self.basis_cfg = basis_cfg
        self.encoder_cfg = encoder_cfg
        self.adapt_bandwidth = adapt_bandwidth
        self.adapt_order = adapt_order
        if bandwidth_mode is None:
            bandwidth_mode = "local_linear" if adapt_bandwidth else "fixed_midpoint"
        if bandwidth_mode not in BANDWIDTH_MODES:
            raise ValueError(
                f"LocalSpectralDecoder: unknown bandwidth_mode={bandwidth_mode!r}; "
                f"expected one of {BANDWIDTH_MODES}"
            )
        self.bandwidth_mode = bandwidth_mode

        P = basis_cfg.p_max
        n_coeff = 3 * P * P
        self.coeff_proj = nn.Linear(encoder_cfg.d_feat, n_coeff)
        head_bandwidth_mode = (
            "local_logspace" if bandwidth_mode == "local_logspace" else "local_linear"
        )
        self.adaptivity = AdaptivityHead(
            encoder_cfg.d_feat,
            s_e_range=basis_cfg.s_e_range,
            p_soft_range=basis_cfg.p_soft_range,
            bandwidth_mode=head_bandwidth_mode,
        )
        if bandwidth_mode == "global_scalar":
            # One learnable scalar, zero-init -> sigmoid(0) = 0.5 -> s_e = sqrt(s_lo*s_hi).
            self.global_s_raw = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("global_s_raw", None)

        nn.init.normal_(self.coeff_proj.weight, std=0.01)
        nn.init.zeros_(self.coeff_proj.bias)

    def _global_scalar_s_e(self, template: torch.Tensor) -> torch.Tensor:
        """Return an (..., 1) tensor of shape matching `template`'s leading dims,
        with every entry equal to the log-space-mapped global scalar s_e."""
        s_lo, s_hi = self.basis_cfg.s_e_range
        log_lo = math.log(s_lo)
        log_hi = math.log(s_hi)
        s_scalar = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(self.global_s_raw))
        return s_scalar.expand_as(template)

    def forward(self, feat_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, D, nH, nW) -> dict with ``coeffs``, ``s_e``, ``p_soft``, ``patches``, ``recon``."""
        B, D, nH, nW = feat_map.shape
        feat = feat_map.permute(0, 2, 3, 1).contiguous()
        pmax = self.basis_cfg.p_max
        c = self.coeff_proj(feat)
        coeffs = c.view(B, nH, nW, 3, pmax, pmax)
        s_e, p_soft = self.adaptivity(feat)  # shape (..., 1)
        if self.bandwidth_mode == "fixed_midpoint":
            s_lo, s_hi = self.basis_cfg.s_e_range
            s_e = torch.full_like(s_e, 0.5 * (s_lo + s_hi))
        elif self.bandwidth_mode == "global_scalar":
            s_e = self._global_scalar_s_e(s_e)
        # else: local_linear / local_logspace use s_e from the adaptivity head.
        if not self.adapt_order:
            p_lo, p_hi = self.basis_cfg.p_soft_range
            p_soft = torch.full_like(p_soft, p_hi)
        coords_1d = torch.linspace(-1.0, 1.0, self.basis_cfg.patch_size, device=feat_map.device)
        patches = reconstruct_patch(coeffs, s_e, p_soft, coords_1d, self.basis_cfg)
        recon = patches_to_image(patches)
        return {
            "coeffs": coeffs,
            "s_e": s_e,
            "p_soft": p_soft,
            "patches": patches,
            "recon": recon,
        }


class BaselineArm(nn.Module):
    """Global Fourier-MLP baseline (no-local-basis control variant).

    The encoder pools to a global feature vector; a coordinate
    Fourier-MLP then renders an ``image_size``-sized reconstruction
    pixel-by-pixel. Shape-locked to ``cfg.image_size`` (no internal
    pad / crop).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.image_size, cfg.patch_size, cfg.encoder)
        self.decoder = GlobalFourierMLPDecoder(cfg.image_size, cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # The Global Fourier-MLP baseline bakes the coordinate grid into
        # its decoder, so the forward pass is shape-locked to
        # ``cfg.image_size``. Fail loudly on mismatched input rather
        # than silently produce a wrong-resolution reconstruction.
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if H != self.cfg.image_size or W != self.cfg.image_size:
            raise NotImplementedError(
                "BaselineArm (Global Fourier-MLP) is shape-locked to "
                f"image_size={self.cfg.image_size}; got input {H}x{W}."
            )
        feat = self.encoder(x)
        recon = self.decoder(feat)
        return {"recon": recon, "coeffs": None, "s_e": None, "p_soft": None, "patches": None}


class LocalSpectralArm(nn.Module):
    """Main GB-LSR model: convolutional encoder + ``LocalSpectralDecoder``.

    Variable-resolution: the forward pass reflect-pads to a multiple of
    ``patch_size``, runs the encoder + per-patch local-spectral decoder,
    and crops the reconstruction back to the input's original ``H x W``.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        adapt_bandwidth: bool = True,
        adapt_order: bool = True,
        bandwidth_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.image_size, cfg.patch_size, cfg.encoder)
        self.decoder = LocalSpectralDecoder(
            cfg.basis,
            cfg.encoder,
            adapt_bandwidth=adapt_bandwidth,
            adapt_order=adapt_order,
            bandwidth_mode=bandwidth_mode,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Variable-resolution forward.

        Pads to a multiple of ``patch_size``, runs encoder + local-spectral
        decoder on the padded tensor, and crops ``recon`` back to the
        input's spatial size. The auxiliary outputs (``coeffs``, ``s_e``,
        ``p_soft``, ``patches``) stay at the padded patch-grid layout
        since they live in patch space. ``valid_patch_mask`` of shape
        ``(B, num_patches_h, num_patches_w)`` lets consumers exclude
        fully-padded patches.
        """
        P = self.cfg.patch_size
        x_pad, pad_info = pad_to_multiple(x, P, mode="reflect")
        B = int(x.shape[0])
        feat = self.encoder(x_pad)
        out = self.decoder(feat)
        out["recon"] = crop_to_original(out["recon"], pad_info)
        out["pad_info"] = pad_info
        out["valid_patch_mask"] = make_valid_patch_mask(
            pad_info=pad_info,
            patch_size=P,
            batch_size=B,
            device=x.device,
        )
        return out


def build_model(
    cfg: ModelConfig,
    adapt_bandwidth: bool = True,
    adapt_order: bool = True,
    bandwidth_mode: str | None = None,
) -> nn.Module:
    """Factory dispatching on ``cfg.arm`` to ``BaselineArm`` or ``LocalSpectralArm``.

    Raises ``ValueError`` for any other arm name.
    """
    if cfg.arm == "global_fourier_mlp":
        return BaselineArm(cfg)
    elif cfg.arm == "local_spectral":
        return LocalSpectralArm(
            cfg,
            adapt_bandwidth=adapt_bandwidth,
            adapt_order=adapt_order,
            bandwidth_mode=bandwidth_mode,
        )
    else:
        raise ValueError(f"unknown arm: {cfg.arm}")


def n_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
