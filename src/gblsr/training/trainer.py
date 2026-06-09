"""GB-LSR training driver.

Launch example:
    python -m gblsr.training.trainer --config configs/example.yaml

Produces under ``<run_dir>``:
    experiment.json       aggregated per-run metadata + metrics
    rows.json             per-image val rows
    aggregate.json        aggregate val metrics
    report.md             human-readable run summary
    region_thresholds.json
    figures/recon_error.png
    figures/bandwidth_order.png   (only when the arm has s_e / p_soft)
    figures/region_mask.png
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.loaders import (
    DataConfig,
    REGIONS,
    RegionThresholds,
    VARRES_BATCH_GT_1_MSG,
    _resolve_mode,
    build_datasets,
    collate,
    fit_region_thresholds,
    label_patches,
    save_thresholds,
    varres_collate,
)
from ..metrics.quality import compute_metric_panel, compute_metric_panel_varres
from ..models.arms import EncoderConfig, ModelConfig, build_model, n_params
from ..models.basis import BasisConfig
from .losses import LossConfig, compute_losses


def set_seed(seed: int):
    """Seed Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@dataclass
class RunConfig:
    experiment_id: str
    run_dir: str
    arm: str
    seed: int
    total_steps: int = 200
    batch_size: int = 8
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 20
    val_every: int = 0
    num_val_images: int = 64
    num_val_curve_images: int = 64
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            dtd_root="", div2k_root="", image_size=256, patch_size=32
        )
    )
    image_size: int = 256
    patch_size: int = 32
    d_feat: int = 128
    n_encoder_layers: int = 3
    p_max: int = 16
    family: Literal["fourier", "cosine"] = "fourier"
    s_e_lo: float = 0.25
    s_e_hi: float = 2.0
    p_soft_lo: float = 1.0
    p_soft_hi: float = 16.0
    adapt_bandwidth: bool = True
    adapt_order: bool = True
    # bandwidth_mode overrides adapt_bandwidth when set. Values:
    #   None (default) : map adapt_bandwidth -> {"fixed_midpoint", "local_linear"}
    #   "fixed_midpoint" | "global_scalar" | "local_linear" | "local_logspace"
    bandwidth_mode: str | None = None
    lambda_point: float = 1.0
    # Pointwise loss surface; default ``"mse"``. ``charbonnier_eps``
    # is consumed only when ``loss_name == "charbonnier"``.
    loss_name: Literal["mse", "l1", "charbonnier"] = "mse"
    charbonnier_eps: float = 1.0e-3
    n_global_freq: int = 128
    decoder_hidden: int = 256
    decoder_layers: int = 3
    dump_locality: bool = False
    save_checkpoint: bool = False


def load_config(path: str) -> dict:
    """Read a YAML config file and return the parsed dict.

    Wraps ``yaml.safe_load``; raises whatever PyYAML raises on malformed
    YAML or filesystem errors.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def expand_run_configs(cfg_dict: dict) -> list[RunConfig]:
    """YAML with base + per-arm overrides + seeds -> one RunConfig per (arm, seed)."""
    base = dict(cfg_dict.get("base", {}))
    arms = cfg_dict.get("arms", [])
    seeds = cfg_dict.get("seeds", [0])
    experiment_id = cfg_dict["experiment_id"]
    run_root = cfg_dict["run_root"]

    runs = []
    for arm_entry in arms:
        for s in seeds:
            merged = dict(base)
            merged.update(arm_entry)
            merged["seed"] = s
            merged["experiment_id"] = experiment_id
            arm_name = arm_entry["arm"]
            # Allow arm entries to carry a friendly slug for the run dir
            slug = arm_entry.get("slug", arm_name)
            merged["run_dir"] = str(Path(run_root) / f"{slug}__seed{s}")
            merged.pop("slug", None)
            data_kv = merged.pop("data", {})
            data_cfg = DataConfig(**data_kv)
            rc = RunConfig(
                **{k: v for k, v in merged.items() if k != "data"},
                data=data_cfg,
            )
            runs.append(rc)
    return runs


def build_loaders(data_cfg: DataConfig, batch_size: int, num_workers: int):
    """Build train / val DataLoaders with the right collate for the mode.

    - Locked (Mode A): ``collate`` packs ``[B, 3, S, S]`` and returns the
      legacy 3-tuple ``(x, names, paths)`` for back-compat with existing
      callers.
    - Variable-res (Mode B): ``varres_collate`` returns
      ``(x[1, 3, H, W], names, paths, metas)`` and raises
      ``NotImplementedError`` when ``batch_size > 1``.
    """
    mode = _resolve_mode(data_cfg)
    train_ds, val_ds = build_datasets(data_cfg)
    if mode == "varres":
        if batch_size != 1:
            raise NotImplementedError(VARRES_BATCH_GT_1_MSG)
        coll = varres_collate
    else:
        coll = collate
    tr = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=coll,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    va = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=coll,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    return tr, va


def _unpack_batch(batch):
    """Unpack either a 3-tuple (legacy) or 4-tuple (with metas) batch.

    Returns (x, names, paths, metas) where metas is None for legacy 3-tuples.
    """
    if len(batch) == 3:
        x, names, paths = batch
        metas = None
    else:
        x, names, paths, metas = batch
    return x, names, paths, metas


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def measure_inference_ms(
    model, sample: torch.Tensor, device, n_warmup: int = 5, n_timed: int = 20
) -> float:
    """Median wall-clock (ms) for a single-image forward pass.

    sample: (1, 3, H, W) tensor already on device. Uses cuda.synchronize around
    each timed call when device is cuda. Returns the median over n_timed runs.
    """
    was_training = model.training
    model.eval()
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    for _ in range(n_warmup):
        _ = model(sample)
    if is_cuda:
        torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(n_timed):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(sample)
        if is_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    if was_training:
        model.train()
    return float(np.median(np.asarray(times_ms, dtype=np.float64)))


@torch.no_grad()
def evaluate(
    model, val_loader, rc: RunConfig, region_th: RegionThresholds, device, max_images: int
):
    """Run the model on at most ``max_images`` validation images.

    Returns a list of per-image metric dicts. In locked (fixed-size) mode
    each dict also carries per-region metrics keyed by region name; in
    variable-resolution mode the per-region rows are omitted.
    """
    model.eval()
    rows = []
    n_seen = 0
    for batch in val_loader:
        x, names, paths, _metas = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        out = model(x)
        recon = out["recon"].clamp(0.0, 1.0)
        # Eval-mode dispatch is config-driven, NOT inferred from runtime
        # tensor shape. Mode A (fixed-size) computes per-patch region
        # labels and the full panel; Mode B (varres) computes the
        # whole-image panel only, with no per-region rows.
        mode = _resolve_mode(rc.data)
        if mode == "locked":
            labels = label_patches(x, rc.patch_size, region_th)
            panel = compute_metric_panel(recon, x, rc.patch_size, labels)
        elif mode == "varres":
            panel = compute_metric_panel_varres(recon, x, rc.patch_size)
        else:
            raise ValueError(f"unknown data mode: {mode!r}")
        for b in range(x.shape[0]):
            row = {
                "name": names[b],
                "path": paths[b],
                "psnr": float(panel["psnr"][b]),
                "ssim": float(panel["ssim"][b]),
                "lpips": float(panel["lpips"][b]),
                "edge_lpips": float(panel["edge_lpips"][b]),
                "local_spectrum_error": float(panel["local_spectrum_error"][b]),
            }
            if "psnr_region" in panel:
                for r_i, r_name in enumerate(REGIONS):
                    row[f"psnr_{r_name}"] = float(panel["psnr_region"][b, r_i])
                    row[f"ssim_{r_name}"] = float(panel["ssim_region"][b, r_i])
                    row[f"local_spectrum_error_{r_name}"] = float(
                        panel["local_spectrum_error_region"][b, r_i]
                    )
            rows.append(row)
            n_seen += 1
            if n_seen >= max_images:
                break
        if n_seen >= max_images:
            break
    model.train()
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-image metric rows to ``{metric: {mean, std, n}}``.

    ``name`` and ``path`` columns are skipped. NaN values are skipped per
    metric (so a metric that is NaN for some images still gets the mean
    over the non-NaN entries, with ``n`` reflecting the valid count).
    """
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k not in ("name", "path")]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], float) and not math.isnan(r[k])]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            agg[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
    return agg


@torch.no_grad()
def dump_locality_artifacts(
    model,
    val_loader,
    rc: RunConfig,
    region_th: RegionThresholds,
    device,
    run_dir: Path,
    max_images: int,
) -> dict:
    """Write per-patch s_e / p_soft / region labels for up to max_images val images.

    Output: {run_dir}/locality.npz with keys
        s_e:           float32 [N, nH, nW]
        p_soft:        float32 [N, nH, nW]
        region_labels: int8    [N, nH, nW]  (0=smooth, 1=edge, 2=texture, 3=mixed)
        image_names:   <U128   [N]
        image_paths:   <U512   [N]

    ``BaselineArm`` (no s_e / p_soft) writes an empty-arrays npz with the same keys so
    downstream code can check `s_e.size > 0`.
    """
    model.eval()
    s_e_chunks: list[np.ndarray] = []
    p_soft_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    name_chunks: list[str] = []
    path_chunks: list[str] = []
    n_seen = 0
    has_adaptivity = True
    for batch in val_loader:
        x, names, paths, _metas = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        out = model(x)
        if out.get("s_e") is None:
            has_adaptivity = False
            break
        labels = label_patches(x, rc.patch_size, region_th)
        s_e = out["s_e"].squeeze(-1).detach().cpu().numpy().astype(np.float32)
        p_soft = out["p_soft"].squeeze(-1).detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy().astype(np.int8)
        B = x.shape[0]
        remaining = max_images - n_seen
        take = min(B, remaining)
        s_e_chunks.append(s_e[:take])
        p_soft_chunks.append(p_soft[:take])
        label_chunks.append(labels_np[:take])
        name_chunks.extend(names[:take])
        path_chunks.extend(paths[:take])
        n_seen += take
        if n_seen >= max_images:
            break
    model.train()

    out_path = run_dir / "locality.npz"
    if has_adaptivity and s_e_chunks:
        s_e_arr = np.concatenate(s_e_chunks, axis=0)
        p_soft_arr = np.concatenate(p_soft_chunks, axis=0)
        labels_arr = np.concatenate(label_chunks, axis=0)
        np.savez_compressed(
            out_path,
            s_e=s_e_arr,
            p_soft=p_soft_arr,
            region_labels=labels_arr,
            image_names=np.array(name_chunks, dtype=np.str_),
            image_paths=np.array(path_chunks, dtype=np.str_),
        )
        summary = {
            "n_images": int(s_e_arr.shape[0]),
            "n_patches_h": int(s_e_arr.shape[1]),
            "n_patches_w": int(s_e_arr.shape[2]),
            "s_e_global_min": float(s_e_arr.min()),
            "s_e_global_max": float(s_e_arr.max()),
            "s_e_global_mean": float(s_e_arr.mean()),
            "p_soft_global_min": float(p_soft_arr.min()),
            "p_soft_global_max": float(p_soft_arr.max()),
            "p_soft_global_mean": float(p_soft_arr.mean()),
        }
        print(
            f"[{rc.arm} seed={rc.seed}] wrote locality.npz "
            f"({summary['n_images']} images, {summary['n_patches_h']}x{summary['n_patches_w']} patches); "
            f"s_e [{summary['s_e_global_min']:.4f}, {summary['s_e_global_max']:.4f}] "
            f"mean={summary['s_e_global_mean']:.4f}; "
            f"p_soft [{summary['p_soft_global_min']:.4f}, {summary['p_soft_global_max']:.4f}] "
            f"mean={summary['p_soft_global_mean']:.4f}",
            flush=True,
        )
        return summary
    # BaselineArm: empty locality (caller can check has_adaptivity=False).
    np.savez_compressed(
        out_path,
        s_e=np.zeros((0, 0, 0), dtype=np.float32),
        p_soft=np.zeros((0, 0, 0), dtype=np.float32),
        region_labels=np.zeros((0, 0, 0), dtype=np.int8),
        image_names=np.array([], dtype=np.str_),
        image_paths=np.array([], dtype=np.str_),
    )
    print(f"[{rc.arm} seed={rc.seed}] arm has no s_e/p_soft; wrote empty locality.npz", flush=True)
    return {"n_images": 0, "has_adaptivity": False}


def save_figures(
    model, val_loader, rc: RunConfig, region_th, device, run_dir: Path, n_images: int = 6
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        x, names, paths, _metas = _unpack_batch(batch)
        x = x.to(device)
        out = model(x)
        recon = out["recon"].clamp(0.0, 1.0)
        n_images = min(n_images, x.shape[0])

        fig, axes = plt.subplots(n_images, 3, figsize=(9, 3 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, 3)
        for i in range(n_images):
            gt = x[i].detach().cpu().permute(1, 2, 0).numpy()
            rec = recon[i].detach().cpu().permute(1, 2, 0).numpy()
            err = np.abs(gt - rec).mean(axis=-1)
            axes[i, 0].imshow(gt)
            axes[i, 0].set_title(f"GT [{names[i]}]")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.clip(rec, 0, 1))
            axes[i, 1].set_title(f"{rc.arm} recon")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(err, cmap="inferno")
            axes[i, 2].set_title("|GT - recon|")
            axes[i, 2].axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "recon_error.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        if out.get("s_e") is not None:
            fig, axes = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))
            if n_images == 1:
                axes = axes.reshape(1, 2)
            s_e = out["s_e"].squeeze(-1).detach().cpu().numpy()
            p_soft = out["p_soft"].squeeze(-1).detach().cpu().numpy()
            for i in range(n_images):
                im0 = axes[i, 0].imshow(s_e[i], cmap="viridis")
                axes[i, 0].set_title(f"s_e  [{names[i]}]")
                axes[i, 0].axis("off")
                plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
                im1 = axes[i, 1].imshow(p_soft[i], cmap="magma")
                axes[i, 1].set_title("p_soft")
                axes[i, 1].axis("off")
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(fig_dir / "bandwidth_order.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        labels = label_patches(x, rc.patch_size, region_th).cpu().numpy()
        fig, axes = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, 2)
        cmap_regions = matplotlib.colors.ListedColormap(
            ["#b0c4de", "#ff7f0e", "#2ca02c", "#9467bd"]
        )
        for i in range(n_images):
            gt = x[i].detach().cpu().permute(1, 2, 0).numpy()
            axes[i, 0].imshow(gt)
            axes[i, 0].set_title("GT")
            axes[i, 0].axis("off")
            im = axes[i, 1].imshow(
                labels[i], cmap=cmap_regions, vmin=-0.5, vmax=3.5, interpolation="nearest"
            )
            axes[i, 1].set_title("region (smooth/edge/texture/mixed)")
            axes[i, 1].axis("off")
            plt.colorbar(
                im,
                ax=axes[i, 1],
                fraction=0.046,
                pad=0.04,
                ticks=[0, 1, 2, 3],
                format=lambda x, _: REGIONS[int(x)],
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "region_mask.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def build_model_from_run_config(rc: RunConfig) -> torch.nn.Module:
    """Build a model from a :class:`RunConfig`.

    Does not move the model to a device or apply seeding; that is the
    caller's responsibility.
    """
    enc_cfg = EncoderConfig(d_feat=rc.d_feat, n_layers=rc.n_encoder_layers)
    basis_cfg = BasisConfig(
        patch_size=rc.patch_size,
        p_max=rc.p_max,
        family=rc.family,
        s_e_range=(rc.s_e_lo, rc.s_e_hi),
        p_soft_range=(rc.p_soft_lo, rc.p_soft_hi),
    )
    mc = ModelConfig(
        arm=rc.arm,
        image_size=rc.image_size,
        patch_size=rc.patch_size,
        basis=basis_cfg,
        encoder=enc_cfg,
        n_global_freq=rc.n_global_freq,
        decoder_hidden=rc.decoder_hidden,
        decoder_layers=rc.decoder_layers,
    )
    return build_model(
        mc,
        adapt_bandwidth=rc.adapt_bandwidth,
        adapt_order=rc.adapt_order,
        bandwidth_mode=rc.bandwidth_mode,
    )


def train_one(rc: RunConfig) -> dict:
    """Run one ``(arm, seed)`` training cell end-to-end.

    Creates ``rc.run_dir``, trains for ``rc.total_steps`` AdamW steps,
    evaluates on the val split, dumps ``experiment.json`` / ``rows.json``
    / ``aggregate.json`` / ``report.md`` / ``region_thresholds.json``
    (and figures + optional ``model.pt`` when ``rc.save_checkpoint``),
    and returns the experiment-json dict.
    """
    run_dir = Path(rc.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(rc.seed)

    model = build_model_from_run_config(rc).to(device)

    tr_loader, va_loader = build_loaders(rc.data, rc.batch_size, rc.data.num_workers)

    th_path = run_dir / "region_thresholds.json"
    print(f"[{rc.arm} seed={rc.seed}] fitting region thresholds ...", flush=True)
    region_th = fit_region_thresholds(va_loader, rc.patch_size, n_samples=1024, device=str(device))
    save_thresholds(region_th, str(th_path))

    opt = torch.optim.AdamW(
        model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay, betas=(0.9, 0.95)
    )
    loss_cfg = LossConfig(
        lambda_point=rc.lambda_point,
        loss_name=rc.loss_name,
        charbonnier_eps=rc.charbonnier_eps,
    )

    def _val_curve_point(step_val: int, n_images: int) -> dict:
        mid_rows = evaluate(model, va_loader, rc, region_th, device, n_images)
        mid_agg = aggregate(mid_rows)
        point = {"step": step_val, "n_val_images": len(mid_rows)}
        for m in ("psnr", "ssim", "lpips", "edge_lpips", "local_spectrum_error"):
            if m in mid_agg:
                point[m] = mid_agg[m]["mean"]
        print(
            f"[{rc.arm} seed={rc.seed}] val-curve step {step_val:6d}  "
            + " ".join(
                f"{m}={point.get(m, float('nan')):.4f}"
                for m in ("psnr", "lpips", "local_spectrum_error")
            ),
            flush=True,
        )
        return point

    model.train()
    step = 0
    t0 = time.time()
    train_iter = _cycle(tr_loader)
    log_rows = []
    train_curve: list[dict] = []
    while step < rc.total_steps:
        batch = next(train_iter)
        x, _, _, _ = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        out = model(x)
        losses = compute_losses(out, x, loss_cfg)
        L = losses["L_total"]
        if not torch.isfinite(L):
            breakdown = {k: float(v) for k, v in losses.items()}
            raise RuntimeError(f"non-finite loss at step {step}: losses={breakdown}")
        opt.zero_grad(set_to_none=True)
        L.backward()
        if rc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip)
        opt.step()

        if step % rc.log_every == 0 or step == rc.total_steps - 1:
            msg = {k: float(v.detach()) for k, v in losses.items()}
            msg["step"] = step
            msg["elapsed_s"] = round(time.time() - t0, 2)
            print(
                f"[{rc.arm} seed={rc.seed}] step {step:6d}  "
                + " ".join(f"{k}={v:.4f}" for k, v in msg.items() if k not in ("step", "elapsed_s"))
                + f"  elapsed={msg['elapsed_s']}s",
                flush=True,
            )
            log_rows.append(msg)
        step += 1

        if rc.val_every > 0 and step % rc.val_every == 0 and step < rc.total_steps:
            train_curve.append(_val_curve_point(step, rc.num_val_curve_images))

    print(f"[{rc.arm} seed={rc.seed}] final eval on {rc.num_val_images} images ...", flush=True)
    rows = evaluate(model, va_loader, rc, region_th, device, rc.num_val_images)
    agg = aggregate(rows)
    # Append final full-sized eval as the last train_curve datapoint so curves
    # are self-contained without having to merge aggregate + curve.
    final_curve_point = {"step": rc.total_steps, "n_val_images": len(rows)}
    for m in ("psnr", "ssim", "lpips", "edge_lpips", "local_spectrum_error"):
        if m in agg:
            final_curve_point[m] = agg[m]["mean"]
    train_curve.append(final_curve_point)

    # Time on the first val sample so timing reflects the resolution
    # eval ran on; fall back to ``image_size`` when the loader is empty.
    timing_h = rc.image_size if rc.image_size else rc.patch_size
    timing_w = timing_h
    sample_for_timing = torch.zeros(1, 3, timing_h, timing_w, device=device)
    try:
        probe_batch = next(iter(va_loader))
        x_probe, _, _, _ = _unpack_batch(probe_batch)
        if x_probe.shape[0] >= 1:
            sample_for_timing = x_probe[:1].to(device, non_blocking=True)
    except StopIteration:
        pass
    inference_ms_per_image = measure_inference_ms(model, sample_for_timing, device)
    print(
        f"[{rc.arm} seed={rc.seed}] inference_ms_per_image (median) = {inference_ms_per_image:.3f} ms",
        flush=True,
    )

    adaptivity = None
    model.eval()
    with torch.no_grad():
        try:
            ad_batch = next(iter(va_loader))
            x0, _, _, _ = _unpack_batch(ad_batch)
            out0 = model(x0.to(device))
            if out0.get("s_e") is not None:
                se = out0["s_e"].flatten().detach().cpu()
                ps = out0["p_soft"].flatten().detach().cpu()
                adaptivity = {
                    "s_e": {
                        "min": float(se.min()),
                        "max": float(se.max()),
                        "mean": float(se.mean()),
                        "std": float(se.std(unbiased=False)),
                    },
                    "p_soft": {
                        "min": float(ps.min()),
                        "max": float(ps.max()),
                        "mean": float(ps.mean()),
                        "std": float(ps.std(unbiased=False)),
                    },
                }
        except StopIteration:
            pass
    model.train()

    save_figures(model, va_loader, rc, region_th, device, run_dir)

    locality_summary = None
    if rc.dump_locality:
        locality_summary = dump_locality_artifacts(
            model,
            va_loader,
            rc,
            region_th,
            device,
            run_dir,
            rc.num_val_images,
        )

    experiment_json = {
        "experiment_id": rc.experiment_id,
        "arm": rc.arm,
        "seed": rc.seed,
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_params_trainable": n_params(model),
        "inference_ms_per_image": inference_ms_per_image,
        "config": {
            k: v if not hasattr(v, "__dict__") else asdict(v) for k, v in asdict(rc).items()
        },
        "region_thresholds": {
            "hf_q1": region_th.hf_q1,
            "hf_q3": region_th.hf_q3,
            "edge_median": region_th.edge_median,
        },
        "train_log": log_rows,
        "train_curve": train_curve,
        "aggregate": agg,
        "adaptivity_final": adaptivity,
        "num_val_images": len(rows),
        "locality_dump": locality_summary,
    }
    (run_dir / "experiment.json").write_text(json.dumps(experiment_json, indent=2))
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    (run_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    _write_report_md(run_dir, experiment_json, agg)

    if rc.save_checkpoint:
        ckpt_path = run_dir / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "arm": rc.arm,
                "seed": rc.seed,
                "experiment_id": rc.experiment_id,
                "git_commit": experiment_json["git_commit"],
                "config": experiment_json["config"],
            },
            ckpt_path,
        )
        print(
            f"[{rc.arm} seed={rc.seed}] wrote checkpoint {ckpt_path} "
            f"({ckpt_path.stat().st_size / 1024 / 1024:.2f} MB)",
            flush=True,
        )
    return experiment_json


def _write_report_md(run_dir: Path, experiment_json: dict, agg: dict):
    arm = experiment_json["arm"]
    seed = experiment_json["seed"]
    lines = [
        f"# {arm}  seed={seed}",
        "",
        f"**Experiment:** `{experiment_json['experiment_id']}`",
        f"**Commit:** `{experiment_json['git_commit']}`  ",
        f"**Timestamp:** {experiment_json['timestamp_utc']}",
        f"**Trainable params:** {experiment_json['n_params_trainable']:,}",
        f"**Inference ms/image (median):** {experiment_json['inference_ms_per_image']:.3f}",
        f"**Val images:** {experiment_json['num_val_images']}",
        "",
        "## Aggregate val metrics",
        "",
        "| metric | mean | std | n |",
        "| --- | --- | --- | --- |",
    ]
    for k, v in agg.items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {v['n']} |")
    lines.append("")
    lines.append("## Figures")
    for name in ["recon_error.png", "bandwidth_order.png", "region_mask.png"]:
        p = run_dir / "figures" / name
        if p.exists():
            lines.append(f"- `figures/{name}`")
    (run_dir / "report.md").write_text("\n".join(lines))
