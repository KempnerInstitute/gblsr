"""GB-LSR-Scalar-ASR composed model (RDN encoder + ASR decoder).

Pairs the :class:`gblsr.encoders.rdn.RDNEncoder` with the
:class:`gblsr.asr.decoder.GBLSRScalarASRDecoder` to form the
arbitrary-scale super-resolution model.

The reported family variants are all reachable by configuration, with no
separate code path:

  * **base**: ``GBLSRScalarASR()``
  * **noLE** (no 4-corner local ensemble)::

        GBLSRScalarASR(decoder_cfg={"local_ensemble": False})

  * **nf48 / nf96** (narrower / wider RDN encoder)::

        GBLSRScalarASR(encoder_cfg={"num_features": 96})

``encoder_cfg`` keys are :class:`gblsr.encoders.rdn.RDNConfig` fields
(``num_features``, ``growth_rate``, ``num_rdbs``, ``num_layers_per_rdb``);
``decoder_cfg`` keys are :class:`GBLSRScalarASRDecoder` arguments
(``p_max``, ``bandwidth_init``, ``local_ensemble``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.rdn import RDNConfig, RDNEncoder
from .coord import make_coord
from .decoder import GBLSRScalarASRDecoder


class GBLSRScalarASR(nn.Module):
    """RDN encoder + GB-LSR-Scalar-ASR decoder."""

    def __init__(
        self,
        encoder_cfg: dict | None = None,
        decoder_cfg: dict | None = None,
    ):
        super().__init__()
        self.encoder = RDNEncoder(RDNConfig(**(encoder_cfg or {})))
        in_dim = self.encoder.out_dim
        self.decoder = GBLSRScalarASRDecoder(in_dim=in_dim, **(decoder_cfg or {}))

    def forward(
        self,
        lr: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Training interface. Returns (B, Q, 3) RGB at the given coords."""
        feat = self.encoder(lr)
        return self.decoder(feat, coord, cell)

    def predict_full(
        self,
        lr: torch.Tensor,
        H_q: int,
        W_q: int,
        tile_q: int | None = None,
    ) -> torch.Tensor:
        """Eval interface. Renders a full (B, 3, H_q, W_q) HR image.

        ``tile_q`` optionally chunks the query grid into batches of at most
        ``tile_q`` coordinates to bound peak memory; the result is identical
        to the untiled path.
        """
        feat = self.encoder(lr)
        B = feat.shape[0]
        device = feat.device
        total_q = H_q * W_q
        coord = make_coord((H_q, W_q)).to(device)
        cell = torch.empty_like(coord)
        cell[:, 0] = 2.0 / H_q
        cell[:, 1] = 2.0 / W_q

        if tile_q is None or total_q <= tile_q:
            c = coord.unsqueeze(0).expand(B, -1, -1).contiguous()
            ce = cell.unsqueeze(0).expand(B, -1, -1).contiguous()
            pred = self.decoder(feat, c, ce)
        else:
            parts: list[torch.Tensor] = []
            for s in range(0, total_q, tile_q):
                c = coord[s : s + tile_q].unsqueeze(0).expand(B, -1, -1).contiguous()
                ce = cell[s : s + tile_q].unsqueeze(0).expand(B, -1, -1).contiguous()
                parts.append(self.decoder(feat, c, ce))
            pred = torch.cat(parts, dim=1)

        return pred.view(B, H_q, W_q, 3).permute(0, 3, 1, 2).contiguous()
