"""Residual Dense Network (RDN) encoder.

Image-feature encoder that maps an input image of shape
``(B, 3, H, W)`` to a dense feature map of shape
``(B, num_features, H, W)`` at the input's spatial resolution.
Suitable as the encoder stage of an arbitrary-scale super-resolution
pipeline, where downstream code consumes the dense feature map at
the input resolution. The full RDN paper additionally includes an
upsampling network for end-to-end super-resolution; that part is
omitted here.

Reference
---------
Zhang, Tian, Kong, Zhong, Fu. "Residual Dense Network for Image
Super-Resolution." CVPR 2018. arXiv:1802.08797.

The implementation here is written from the architectural
specification in the RDN paper. The four modules of the network and the
``num_features (G_0)`` / ``growth_rate (G)`` / ``num_rdbs (D)`` /
``num_layers_per_rdb (C)`` defaults all match the RDN paper. With the
defaults (``G_0 = G = 64``, ``D = 16``, ``C = 8``) the encoder has
21,973,952 (~21.97 M) trainable parameters.

Changing ``num_features`` adjusts only ``G_0`` (the SFENet output
dim and the encoder output dim); the per-dense-layer growth rate
``G`` is independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RDNConfig:
    """RDN encoder hyperparameters.

    Defaults follow the RDN paper's higher-capacity configuration
    (``G_0 = G = 64, D = 16, C = 8``), yielding ~21.97 M trainable
    parameters.

    Parameters
    ----------
    num_features : int
        ``G_0``: SFENet output channels, RDB input/output channels,
        and the final encoder output dim. Defaults to 64.
    growth_rate : int
        ``G``: channels added by each dense layer inside an RDB.
        Defaults to 64.
    num_rdbs : int
        ``D``: number of Residual Dense Blocks. Defaults to 16.
    num_layers_per_rdb : int
        ``C``: number of dense conv+ReLU layers inside each RDB.
        Defaults to 8.
    kernel_size : int
        Kernel size for all 3x3 convolutions in the network.
        Defaults to 3 (the only value used in the RDN paper).
    in_channels : int
        Input image channel count. Defaults to 3 (RGB).
    """

    num_features: int = 64
    growth_rate: int = 64
    num_rdbs: int = 16
    num_layers_per_rdb: int = 8
    kernel_size: int = 3
    in_channels: int = 3


class _DenseLayer(nn.Module):
    """One conv -> ReLU step inside a Residual Dense Block.

    The output is concatenated with the input along the channel axis
    to realize the dense connection pattern of the RDB.
    """

    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.relu(self.conv(x))], dim=1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block: C dense conv+ReLU layers, LFF, local residual.

    Input is a ``(B, G_0, H, W)`` feature map. Each of the C dense
    layers grows the channel dim by ``growth_rate``. A 1x1 Local
    Feature Fusion convolution then collapses the
    ``G_0 + C * growth_rate``-channel concatenation back to ``G_0``.
    The block output is the LFF result plus the block input
    (local residual).
    """

    def __init__(
        self,
        num_features: int,
        growth_rate: int,
        num_layers: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.dense_layers = nn.Sequential(
            *[
                _DenseLayer(num_features + i * growth_rate, growth_rate, kernel_size)
                for i in range(num_layers)
            ]
        )
        # Local Feature Fusion: 1x1 conv that collapses the dense
        # concatenation back to num_features channels.
        self.local_feature_fusion = nn.Conv2d(
            num_features + num_layers * growth_rate,
            num_features,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_out = self.dense_layers(x)
        return self.local_feature_fusion(dense_out) + x


class RDNEncoder(nn.Module):
    """RDN encoder.

    Module layout::

        SFENet  : Conv2d(3, G_0, 3) -> Conv2d(G_0, G_0, 3)
                  (first output saved as the global residual source)
        RDBs    : D Residual Dense Blocks, each with C dense layers
                  and one 1x1 Local Feature Fusion.
        GFF     : concatenate outputs of all D RDBs
                  -> Conv2d(D * G_0, G_0, 1) -> Conv2d(G_0, G_0, 3)
        Output  : GFF result + SFENet first-conv output (global residual)

    Spatial dimensions are preserved end-to-end (padding=1 on all 3x3
    convs, no stride > 1, no downsampling). The output shape is
    ``(B, num_features, H, W)``.

    The trainable parameter count for the default
    (``num_features = growth_rate = 64``, ``num_rdbs = 16``,
    ``num_layers_per_rdb = 8``) configuration is **21,973,952**
    (~21.97 M).

    The ``out_dim`` attribute exposes the output channel count so
    downstream modules (e.g. the GB-LSR decoder) can size their
    coefficient-projection heads without inspecting ``cfg``.
    """

    out_dim: int

    def __init__(self, cfg: RDNConfig | None = None):
        super().__init__()
        cfg = cfg if cfg is not None else RDNConfig()
        self.cfg = cfg
        self.out_dim = cfg.num_features

        G0 = cfg.num_features
        G = cfg.growth_rate
        D = cfg.num_rdbs
        C = cfg.num_layers_per_rdb
        kSize = cfg.kernel_size
        pad = (kSize - 1) // 2

        # Shallow Feature Extraction net (SFENet)
        self.sfe_1 = nn.Conv2d(cfg.in_channels, G0, kSize, padding=pad)
        self.sfe_2 = nn.Conv2d(G0, G0, kSize, padding=pad)

        # Residual Dense Blocks
        self.rdbs = nn.ModuleList([ResidualDenseBlock(G0, G, C, kSize) for _ in range(D)])

        # Global Feature Fusion (GFF): 1x1 then 3x3
        self.gff_1x1 = nn.Conv2d(D * G0, G0, kernel_size=1, padding=0)
        self.gff_3x3 = nn.Conv2d(G0, G0, kSize, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_channels, H, W) -> (B, num_features, H, W)``.

        Spatial dims (``H``, ``W``) are unchanged.
        """
        residual_source = self.sfe_1(x)
        features = self.sfe_2(residual_source)

        rdb_outputs: list[torch.Tensor] = []
        for rdb in self.rdbs:
            features = rdb(features)
            rdb_outputs.append(features)

        concat = torch.cat(rdb_outputs, dim=1)
        global_features = self.gff_3x3(self.gff_1x1(concat))

        # Global residual: add the SFENet first-conv output.
        return global_features + residual_source


def build_rdn_encoder(
    *,
    num_features: int = 64,
    growth_rate: int = 64,
    num_rdbs: int = 16,
    num_layers_per_rdb: int = 8,
) -> RDNEncoder:
    """Convenience factory for the RDN encoder.

    Representative trainable-parameter counts at varying
    ``num_features`` (with ``growth_rate`` held at the default 64):

    +-----------------+------------------------+
    | num_features    | trainable params (~M)  |
    +=================+========================+
    | 48              | 20.57                  |
    +-----------------+------------------------+
    | 64 (default)    | 21.97                  |
    +-----------------+------------------------+
    | 96              | 24.85                  |
    +-----------------+------------------------+
    """
    cfg = RDNConfig(
        num_features=num_features,
        growth_rate=growth_rate,
        num_rdbs=num_rdbs,
        num_layers_per_rdb=num_layers_per_rdb,
    )
    return RDNEncoder(cfg)
