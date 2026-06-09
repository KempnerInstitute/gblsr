"""End-to-end smoke test for ``gblsr.training.trainer.train_one``
plus the full chained train -> eval CLI -> reconstruct CLI workflow.

Creates a tiny synthetic DTD + DIV2K directory layout in ``tmp_path``,
sets up a deliberately small ``RunConfig`` (image_size=64, 2 training
steps, 2 val images) with ``save_checkpoint=True``, runs
``train_one`` to completion, then exercises the eval and reconstruct
CLIs against the resulting ``model.pt``.

This is the only test in the suite that goes through the full
trainer pipeline. It is slower than the smoke / import tests
(~30 s on CPU, dominated by LPIPS' first-call AlexNet weight load),
so any further trainer-level tests should probably live alongside
this one rather than across multiple files.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gblsr.cli import eval as cli_eval
from gblsr.cli import reconstruct as cli_reconstruct
from gblsr.data.loaders import DataConfig
from gblsr.training.trainer import RunConfig, train_one


def _populate_tiny_datasets(tmp_path: Path) -> tuple[str, str]:
    """Create minimal DTD- and DIV2K-shaped directories with a few images.

    DTD loader looks for ``<root>/dtd/dtd/images/<class>/*.jpg``;
    DIV2K loader looks for ``<root>/DIV2K_train_HR/*.png`` (or PNGs
    directly under root). We populate both with synthetic 96x96 images
    so the training crop (image_size=64) works without upsampling.
    """
    dtd_root = tmp_path / "dtd_root"
    div2k_root = tmp_path / "div2k_root"

    dtd_images_dir = dtd_root / "dtd" / "dtd" / "images" / "synth"
    dtd_images_dir.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (96, 96), color=(i * 30, 100, 200)).save(
            dtd_images_dir / f"img_{i:02d}.jpg"
        )

    div2k_hr_dir = div2k_root / "DIV2K_train_HR"
    div2k_hr_dir.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (96, 96), color=(50, i * 30, 150)).save(div2k_hr_dir / f"{i:04d}.png")

    return str(dtd_root), str(div2k_root)


def _write_chain_yaml(tmp_path: Path, dtd_root: str, div2k_root: str) -> Path:
    """Write a tiny YAML config matching the chain-test's ``RunConfig``.

    Saved separately because the eval / reconstruct CLIs take a YAML
    path on the command line (not a ``RunConfig`` object), so the chain
    test needs the same hyperparameters expressed in both forms.
    """
    cfg_path = tmp_path / "chain.yaml"
    cfg_path.write_text(
        f"""\
experiment_id: e2e_chain
run_root: {tmp_path}/runs_chain_eval
base:
  total_steps: 2
  batch_size: 1
  num_val_images: 2
  num_val_curve_images: 0
  image_size: 64
  patch_size: 8
  d_feat: 16
  p_max: 4
  s_e_lo: 0.25
  s_e_hi: 2.0
  p_soft_lo: 1.0
  p_soft_hi: 4.0
  decoder_hidden: 16
  decoder_layers: 1
  n_global_freq: 16
  data:
    dtd_root: {dtd_root}
    div2k_root: {div2k_root}
    image_size: 64
    patch_size: 8
    num_workers: 0
    val_fraction: 0.34
arms:
  - arm: local_spectral
    bandwidth_mode: global_scalar
seeds: [0]
"""
    )
    return cfg_path


def test_train_eval_reconstruct_chain_e2e_cpu(tmp_path: Path) -> None:
    """End-user workflow: train -> save model.pt -> eval CLI -> reconstruct CLI.

    Verifies the trainer's saved checkpoint is loadable by both the
    eval CLI and the reconstruct CLI, and that the eval CLI's printed
    aggregate panel contains the same metric keys the trainer wrote
    into ``aggregate.json``.
    """
    dtd_root, div2k_root = _populate_tiny_datasets(tmp_path)
    run_dir = tmp_path / "run"

    rc = RunConfig(
        experiment_id="e2e_chain",
        run_dir=str(run_dir),
        arm="local_spectral",
        seed=0,
        total_steps=2,
        batch_size=1,
        log_every=1,
        val_every=0,
        num_val_images=2,
        num_val_curve_images=0,
        image_size=64,
        patch_size=8,
        d_feat=16,
        p_max=4,
        s_e_lo=0.25,
        s_e_hi=2.0,
        p_soft_lo=1.0,
        p_soft_hi=4.0,
        decoder_hidden=16,
        decoder_layers=1,
        n_global_freq=16,
        bandwidth_mode="global_scalar",
        save_checkpoint=True,
        data=DataConfig(
            dtd_root=dtd_root,
            div2k_root=div2k_root,
            image_size=64,
            patch_size=8,
            num_workers=0,
            val_fraction=0.34,
        ),
    )

    # Step 1: train, save checkpoint.
    out = train_one(rc)
    ckpt = run_dir / "model.pt"
    assert ckpt.exists(), f"trainer did not save checkpoint at {ckpt}"
    assert ckpt.stat().st_size > 0

    # Standard artifacts written by train_one.
    for fname in (
        "experiment.json",
        "rows.json",
        "aggregate.json",
        "report.md",
        "region_thresholds.json",
    ):
        assert (run_dir / fname).exists(), f"missing {fname}"
    exp = json.loads((run_dir / "experiment.json").read_text())
    assert exp["arm"] == "local_spectral"
    assert exp["seed"] == 0
    assert exp["n_params_trainable"] > 0
    agg = out["aggregate"]
    for metric in ("psnr", "ssim", "lpips", "edge_lpips"):
        assert metric in agg, f"aggregate missing {metric}"
        assert "mean" in agg[metric]

    # Step 2: gblsr-eval CLI loads the same checkpoint.
    cfg_path = _write_chain_yaml(tmp_path, dtd_root, div2k_root)
    eval_out = tmp_path / "eval.json"
    rc_code = cli_eval.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--device",
            "cpu",
            "--num-val-images",
            "2",
            "--region-threshold-samples",
            "32",
            "--output",
            str(eval_out),
        ]
    )
    assert rc_code == 0
    eval_payload = json.loads(eval_out.read_text())
    assert eval_payload["arm"] == "local_spectral"
    assert eval_payload["bandwidth_mode"] == "global_scalar"
    for metric in ("psnr", "ssim", "lpips", "edge_lpips"):
        assert metric in eval_payload["aggregate"], f"eval CLI aggregate missing {metric}"

    # Step 3: gblsr-reconstruct CLI loads the same checkpoint.
    img_in = tmp_path / "input.png"
    img_out = tmp_path / "recon.png"
    Image.new("RGB", (64, 64), color=(80, 160, 240)).save(img_in)
    rc_code = cli_reconstruct.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--input",
            str(img_in),
            "--output",
            str(img_out),
            "--device",
            "cpu",
        ]
    )
    assert rc_code == 0
    assert img_out.exists()
    assert Image.open(img_out).size == (64, 64)
