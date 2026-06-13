"""Coordinate helper for the arbitrary-scale SR decoder.

The LIIF / LTE convention uses (y, x) ordering in coords and
``align_corners=False``-style grid centres: ``v_i = -1 + 1/n + (2/n) * i``
for ``i = 0..n-1``. ``torch.nn.functional.grid_sample`` expects (x, y) order,
so callers use ``coord.flip(-1)`` at the sample site.
"""

from __future__ import annotations

import torch


def make_coord(shape: tuple[int, int], flatten: bool = True) -> torch.Tensor:
    """Centred grid in ``[-1 + 1/n, 1 - 1/n]``.

    Args:
      shape: ``(H, W)`` spatial shape.
      flatten: if True, return ``(H*W, 2)``; else ``(H, W, 2)``.

    Returns:
      Tensor of coordinates in (y, x) order.
    """
    coord_seqs = []
    for n in shape:
        r = 1.0 / n
        seq = -1.0 + r + (2.0 * r) * torch.arange(n, dtype=torch.float32)
        coord_seqs.append(seq)
    grid = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        grid = grid.view(-1, grid.shape[-1])
    return grid
