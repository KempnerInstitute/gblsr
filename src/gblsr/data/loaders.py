"""Image dataset loaders and per-patch region slicing.

Dataset loaders are provided for DTD, DIV2K, Kodak, Set14, Urban100, and
generic flat image folders. Each loader yields ``(image_tensor in [0, 1],
source_name, path, meta)`` so collation can build either a fixed-size
batch (``collate``) or a variable-resolution single-image batch
(``varres_collate``).

Region labels (smooth / edge / texture / mixed)
-----------------------------------------------
For each crop we compute per-patch PSD high-band energy fraction
(fraction of total patch power in the upper half of radial frequencies).
A patch is labeled:

    smooth   : HF fraction in [0, q1) and edge-density low
    edge     : edge-density high (Sobel magnitude mean above median)
    texture  : HF fraction in [q3, 1]
    mixed    : otherwise

q1, q3 are the 25th / 75th percentile of HF fraction computed on a
warm-up sample of 1024 patches from the train pool. These thresholds
are cached on disk so every arm uses identical labels for the same
split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass
class DataConfig:
    """Loader configuration.

    The loader supports two modes:

    Mode A (fixed-size, default):
        ``image_size`` is an int (default 256) and ``resize_policy`` is
        ``"center_crop_or_upsample_to_size"``. ``__getitem__`` returns
        ``[3, image_size, image_size]`` tensors. Used for matched-size
        batching at a fixed resolution.

    Mode B (original-resolution / variable-resolution):
        ``image_size`` is ``None`` AND/OR ``resize_policy == "original"``.
        ``__getitem__`` returns ``[3, H, W]`` at the *original* image
        resolution (rectangular OK; tiny images are NOT silently
        upsampled). Variable-resolution batches do not stack;
        ``batch_size > 1`` raises ``NotImplementedError``.
    """

    dtd_root: str
    div2k_root: str
    image_size: int | None = 256
    patch_size: int = 32
    dtd_split: str = "train"
    dtd_partition: int = 1
    dtd_max_images: int | None = None
    div2k_max_images: int | None = None
    val_fraction: float = 0.1
    num_workers: int = 4
    seed: int = 0
    resize_policy: str = "center_crop_or_upsample_to_size"


def _center_or_random_crop(img: Image.Image, size: int, rng: torch.Generator | None) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        s = max(size / w, size / h)
        img = img.resize((max(int(w * s) + 1, size), max(int(h * s) + 1, size)), Image.BICUBIC)
        w, h = img.size
    if rng is None:
        x = (w - size) // 2
        y = (h - size) // 2
    else:
        x = int(torch.randint(0, w - size + 1, (1,), generator=rng).item())
        y = int(torch.randint(0, h - size + 1, (1,), generator=rng).item())
    return img.crop((x, y, x + size, y + size))


class DTDImages(Dataset):
    """Iterate over DTD image files (flat list)."""

    def __init__(
        self, root: str, split: str = "train", partition: int = 1, max_images: int | None = None
    ):
        inner = Path(root) / "dtd" / "dtd"
        if not inner.exists():
            alt = Path(root) / "dtd"
            inner = alt if alt.exists() else inner
        labels_dir = inner / "labels"
        split_file = labels_dir / f"{split}{partition}.txt"
        if split_file.exists():
            rel_paths = [
                line.strip() for line in split_file.read_text().splitlines() if line.strip()
            ]
            self.paths = [str(inner / "images" / r) for r in rel_paths]
        else:
            imgs = sorted((inner / "images").rglob("*.jpg"))
            self.paths = [str(p) for p in imgs]
        if max_images:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


class DIV2KImages(Dataset):
    """Iterate over DIV2K HR PNG image files.

    Looks for images under ``<root>/DIV2K_train_HR/*.png`` first, falling
    back to ``<root>/*.png`` if that subdirectory is absent.
    """

    def __init__(self, root: str, max_images: int | None = None):
        p = Path(root)
        inner = p / "DIV2K_train_HR"
        inner = inner if inner.exists() else p
        paths = sorted(inner.glob("*.png"))
        self.paths = [str(pp) for pp in paths]
        if max_images:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


class KodakImages(Dataset):
    """Kodak: 24-image test set, layout <root>/kodim01.png ... kodim24.png."""

    def __init__(self, root: str):
        paths = sorted(Path(root).glob("kodim*.png"))
        if not paths:
            paths = sorted(Path(root).glob("*.png"))
        self.paths = [str(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


class FlatImageFolder(Dataset):
    """Generic flat folder of images. Used for Set14 / Urban100."""

    def __init__(self, root: str, exts=(".png", ".jpg", ".jpeg", ".bmp")):
        p = Path(root)
        paths = sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
        self.paths = [str(x) for x in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


VARRES_BATCH_GT_1_MSG = (
    "variable-resolution mode requires batch_size=1; stacking "
    "rectangular tensors of different sizes is not supported"
)


def _resolve_mode(cfg: "DataConfig") -> str:
    """Return ``"locked"`` or ``"varres"`` based on a DataConfig.

    Selects ``"varres"`` if ``image_size is None`` or ``resize_policy ==
    "original"``; otherwise ``"locked"``.
    """
    if cfg.image_size is None:
        return "varres"
    if cfg.resize_policy == "original":
        return "varres"
    return "locked"


class PatchCropDataset(Dataset):
    """Wraps an image path dataset.

    Locked mode (Mode A):
        Yields ``(image_tensor_in_[0,1] of shape [3, S, S], source_label,
        path, meta)`` where ``meta`` is a dict with ``filename``,
        ``dataset``, ``original_h`` (== S), ``original_w`` (== S).

    Variable-res mode (Mode B):
        Yields ``(image_tensor_in_[0,1] of shape [3, H, W], source_label,
        path, meta)`` where ``meta`` carries the *true* original
        ``original_h`` / ``original_w``. Tiny images are NOT silently
        upsampled.
    """

    def __init__(
        self,
        sources: Sequence[tuple[str, Dataset]],
        image_size: int | None,
        train: bool,
        seed: int,
        resize_policy: str = "center_crop_or_upsample_to_size",
    ):
        self.sources = sources
        self.image_size = image_size
        self.train = train
        self.seed = seed
        self.resize_policy = resize_policy
        self.varres = (image_size is None) or (resize_policy == "original")
        self.index: list[tuple[int, int]] = []
        for si, (_, ds) in enumerate(sources):
            for i in range(len(ds)):
                self.index.append((si, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, i = self.index[idx]
        name, ds = self.sources[si]
        path = ds[i]
        img = Image.open(path).convert("RGB")
        if self.varres:
            t = TF.to_tensor(img)
            orig_h, orig_w = int(t.shape[-2]), int(t.shape[-1])
        else:
            rng = None
            if self.train:
                rng = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
            img = _center_or_random_crop(img, int(self.image_size), rng)
            t = TF.to_tensor(img)
            orig_h, orig_w = int(t.shape[-2]), int(t.shape[-1])
        meta = {
            "filename": Path(path).name,
            "dataset": name,
            "original_h": orig_h,
            "original_w": orig_w,
            "path": path,
        }
        return t, name, path, meta


def build_datasets(cfg: DataConfig):
    """Train/val split built from a DTD + DIV2K pool.

    Honors ``cfg.image_size`` and ``cfg.resize_policy`` so locked-size
    configs keep the fixed-resolution path and configs that set
    ``image_size=None`` or ``resize_policy="original"`` opt into the
    variable-resolution loader instead.
    """
    dtd_train = DTDImages(
        cfg.dtd_root,
        split=cfg.dtd_split,
        partition=cfg.dtd_partition,
        max_images=cfg.dtd_max_images,
    )
    div2k_train = DIV2KImages(cfg.div2k_root, max_images=cfg.div2k_max_images)

    def split(ds: Dataset, val_fraction: float, name: str):
        n = len(ds)
        g = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_val = max(1, int(val_fraction * n))
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]
        train_subset = torch.utils.data.Subset(ds, train_idx)
        val_subset = torch.utils.data.Subset(ds, val_idx)
        return train_subset, val_subset

    dtd_tr, dtd_va = split(dtd_train, cfg.val_fraction, "dtd")
    div2k_tr, div2k_va = split(div2k_train, cfg.val_fraction, "div2k")

    train_sources = [("dtd", dtd_tr), ("div2k", div2k_tr)]
    val_sources = [("dtd", dtd_va), ("div2k", div2k_va)]

    train_ds = PatchCropDataset(
        train_sources,
        cfg.image_size,
        train=True,
        seed=cfg.seed,
        resize_policy=cfg.resize_policy,
    )
    val_ds = PatchCropDataset(
        val_sources,
        cfg.image_size,
        train=False,
        seed=cfg.seed,
        resize_policy=cfg.resize_policy,
    )
    return train_ds, val_ds


def collate(batch):
    """Locked-mode collate. Stacks ``[3, S, S]`` tensors along a batch axis.

    Returns the legacy ``(x, names, paths)`` 3-tuple so callers using
    the unpacking ``x, names, paths = batch`` continue to work
    unchanged. The dataset's optional 4th metadata element is dropped
    here; callers that need per-image meta in locked mode should use
    ``collate_with_meta`` (or rebuild it from ``names`` / ``paths``).
    """
    sample = batch[0]
    if len(sample) == 3:
        xs, names, paths = zip(*batch)
    else:
        # 4-tuple from PatchCropDataset; drop meta for backwards compat.
        xs, names, paths, _metas = zip(*batch)
    return torch.stack(xs, dim=0), list(names), list(paths)


def collate_with_meta(batch):
    """Locked-mode collate that preserves per-image metadata.

    Returns ``(x, names, paths, metas)`` where ``x`` is stacked as in
    :func:`collate`. Use this collate when a downstream caller needs
    per-image metadata propagated alongside the batch.
    """
    sample = batch[0]
    if len(sample) == 3:
        xs, names, paths = zip(*batch)
        metas = [
            {
                "filename": Path(p).name if isinstance(p, str) else str(p),
                "dataset": n,
                "original_h": int(t.shape[-2]),
                "original_w": int(t.shape[-1]),
                "path": p,
            }
            for t, n, p in zip(xs, names, paths)
        ]
    else:
        xs, names, paths, metas = zip(*batch)
        metas = list(metas)
    return torch.stack(xs, dim=0), list(names), list(paths), metas


def varres_collate(batch):
    """Variable-resolution collate.

    Returns ``(x, names, paths, metas)`` with ``x`` of shape ``[1, 3, H, W]``.
    Raises ``NotImplementedError`` if more than one sample is supplied,
    because variable-size tensors cannot be stacked along a batch axis
    without padding.
    """
    if len(batch) > 1:
        raise NotImplementedError(VARRES_BATCH_GT_1_MSG)
    sample = batch[0]
    if len(sample) == 3:
        t, name, path = sample
        meta = {
            "filename": Path(path).name if isinstance(path, str) else str(path),
            "dataset": name,
            "original_h": int(t.shape[-2]),
            "original_w": int(t.shape[-1]),
            "path": path,
        }
    else:
        t, name, path, meta = sample
    return t.unsqueeze(0), [name], [path], [meta]


def make_collate(varres: bool):
    """Return the appropriate collate fn for the loader mode."""
    return varres_collate if varres else collate


# ---------- region slicing ----------


def _radial_freq_grid(P: int, device, dtype) -> torch.Tensor:
    kx = torch.fft.fftfreq(P, d=1.0, device=device).to(dtype)
    ky = torch.fft.fftfreq(P, d=1.0, device=device).to(dtype)
    kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
    return (kxx**2 + kyy**2).sqrt()


def per_patch_hf_fraction(x: torch.Tensor, patch_size: int, cutoff: float = 0.25) -> torch.Tensor:
    """Return (B, nH, nW) HF energy fraction per patch."""
    gray = x.mean(dim=1, keepdim=True)
    from ..models.basis import image_to_patches

    p = image_to_patches(gray, patch_size).squeeze(3)
    fft = torch.fft.fft2(p, norm="ortho")
    power = fft.real**2 + fft.imag**2
    radial = _radial_freq_grid(patch_size, x.device, x.dtype)
    hf_mask = (radial > cutoff).to(x.dtype)
    hf_e = (power * hf_mask).sum(dim=(-2, -1))
    total_e = power.sum(dim=(-2, -1)) + 1e-8
    return hf_e / total_e


def per_patch_edge_density(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Sobel-magnitude mean per patch."""
    gray = x.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = (gx**2 + gy**2).sqrt()
    from ..models.basis import image_to_patches

    p = image_to_patches(mag, patch_size).squeeze(3)
    return p.mean(dim=(-2, -1))


REGIONS = ["smooth", "edge", "texture", "mixed"]


@dataclass
class RegionThresholds:
    hf_q1: float
    hf_q3: float
    edge_median: float


def fit_region_thresholds(
    loader,
    patch_size: int,
    n_samples: int = 1024,
    device: str = "cuda",
) -> RegionThresholds:
    """Warm-up pass to compute thresholds from ~n_samples patches."""
    hf_all, ed_all = [], []
    seen = 0
    for batch in loader:
        x = batch[0].to(device)
        hf = per_patch_hf_fraction(x, patch_size).flatten().cpu()
        ed = per_patch_edge_density(x, patch_size).flatten().cpu()
        hf_all.append(hf)
        ed_all.append(ed)
        seen += hf.numel()
        if seen >= n_samples:
            break
    hf = torch.cat(hf_all)[:n_samples]
    ed = torch.cat(ed_all)[:n_samples]
    return RegionThresholds(
        hf_q1=torch.quantile(hf, 0.25).item(),
        hf_q3=torch.quantile(hf, 0.75).item(),
        edge_median=torch.quantile(ed, 0.5).item(),
    )


def label_patches(
    x: torch.Tensor,
    patch_size: int,
    thresholds: RegionThresholds,
) -> torch.Tensor:
    """Return (B, nH, nW) integer labels: 0=smooth, 1=edge, 2=texture, 3=mixed."""
    hf = per_patch_hf_fraction(x, patch_size)
    ed = per_patch_edge_density(x, patch_size)
    lbl = torch.full_like(hf, 3, dtype=torch.long)
    is_smooth = (hf < thresholds.hf_q1) & (ed < thresholds.edge_median)
    is_texture = hf >= thresholds.hf_q3
    is_edge = (ed >= thresholds.edge_median) & (hf < thresholds.hf_q3)
    lbl = torch.where(is_smooth, torch.zeros_like(lbl), lbl)
    lbl = torch.where(is_edge, torch.ones_like(lbl), lbl)
    lbl = torch.where(is_texture, torch.full_like(lbl, 2), lbl)
    return lbl


def save_thresholds(th: RegionThresholds, path: str):
    """Write ``th`` to ``path`` as JSON (creates parent dirs as needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"hf_q1": th.hf_q1, "hf_q3": th.hf_q3, "edge_median": th.edge_median}, indent=2)
    )


def load_thresholds(path: str) -> RegionThresholds:
    """Inverse of :func:`save_thresholds`: read JSON, return ``RegionThresholds``."""
    d = json.loads(Path(path).read_text())
    return RegionThresholds(**d)
