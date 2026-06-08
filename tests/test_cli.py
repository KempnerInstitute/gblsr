"""Tests for the CLI entry points: gblsr-train, gblsr-eval,
gblsr-measure-latency, gblsr-reconstruct.

Verifies:
  - Every CLI module imports and exposes a ``main`` callable.
  - ``--help`` runs without raising and exits cleanly for each CLI.
  - ``gblsr-measure-latency`` runs end-to-end on CPU against a tiny
    inline config (no actual checkpoint or dataset needed).
  - ``gblsr-reconstruct`` runs end-to-end on CPU against a tiny
    random-init checkpoint and a synthetic input PNG.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from gblsr.cli import eval as cli_eval
from gblsr.cli import latency as cli_latency
from gblsr.cli import reconstruct as cli_reconstruct
from gblsr.cli import train as cli_train
from gblsr.training.trainer import build_model_from_run_config, expand_run_configs, load_config


_TINY_CONFIG = """\
experiment_id: tiny
run_root: /tmp/gblsr_cli_test_unused
base:
  total_steps: 1
  batch_size: 1
  image_size: 16
  patch_size: 8
  d_feat: 16
  n_encoder_layers: 1
  p_max: 4
  s_e_lo: 0.25
  s_e_hi: 2.0
  p_soft_lo: 1.0
  p_soft_hi: 4.0
  decoder_hidden: 16
  decoder_layers: 2
  n_global_freq: 16
  data:
    dtd_root: /unused/by/latency
    div2k_root: /unused/by/latency
    image_size: 16
    patch_size: 8
arms:
  - arm: local_spectral
    bandwidth_mode: global_scalar
seeds: [0]
"""


def test_latency_main_callable() -> None:
    assert callable(cli_latency.main)


def test_eval_main_callable() -> None:
    assert callable(cli_eval.main)


def test_latency_help_exits_zero() -> None:
    """``--help`` prints + exits 0 (argparse default)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_latency.main(["--help"])
    assert exc_info.value.code == 0


def test_eval_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_eval.main(["--help"])
    assert exc_info.value.code == 0


def test_reconstruct_main_callable() -> None:
    assert callable(cli_reconstruct.main)


def test_reconstruct_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_reconstruct.main(["--help"])
    assert exc_info.value.code == 0


def test_train_main_callable() -> None:
    assert callable(cli_train.main)


def test_train_help_exits_zero() -> None:
    """``gblsr-train --help`` prints + exits 0 (argparse default)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_train.main(["--help"])
    assert exc_info.value.code == 0


def test_reconstruct_end_to_end_cpu(capsys, tmp_path) -> None:
    """End-to-end CPU run of ``gblsr-reconstruct``.

    Builds a tiny model from an inline config, saves a random-init
    checkpoint to disk, writes a synthetic PNG input, then runs the
    reconstruct CLI and asserts the output file exists with the
    expected dimensions.
    """
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(_TINY_CONFIG)
    cfg_dict = load_config(str(cfg_path))
    rc = expand_run_configs(cfg_dict)[0]

    # Build the matching model and dump its random-init state_dict
    # as a checkpoint.
    model = build_model_from_run_config(rc).eval()
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    # Synthetic input image (PIL-saved, gets reloaded by the CLI).
    input_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(80, 120, 200)).save(input_path)

    output_path = tmp_path / "recon.png"

    rc_code = cli_reconstruct.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ]
    )
    assert rc_code == 0
    assert output_path.exists()

    # The output should be the same spatial size as the input.
    out_img = Image.open(output_path)
    assert out_img.size == (16, 16)

    captured = capsys.readouterr().out
    assert "arm" in captured
    assert "input" in captured
    assert "output" in captured


def test_latency_end_to_end_cpu(capsys, tmp_path) -> None:
    """End-to-end CPU run of ``gblsr-measure-latency``.

    Builds a deliberately tiny model (image_size=16, p_max=4, etc.)
    so the test finishes in well under a second. Confirms that the
    full pipeline (load_config -> expand_run_configs ->
    build_model_from_run_config -> measure_latency -> print) works
    end-to-end.
    """
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(_TINY_CONFIG)

    rc = cli_latency.main(
        [
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--timed",
            "2",
            "--no-peak-memory",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr().out
    # Sanity checks on the printed report.
    assert "arm" in captured
    assert "local_spectral" in captured
    assert "bandwidth_mode" in captured
    assert "global_scalar" in captured
    assert "latency (ms)" in captured
    assert "median" in captured
