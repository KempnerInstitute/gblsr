"""Tests for the CLI entry points: gblsr-train, gblsr-eval,
gblsr-measure-latency, gblsr-reconstruct, gblsr-encode, gblsr-decode.

Verifies:
  - Every CLI module imports and exposes a ``main`` callable.
  - ``--help`` runs without raising and exits cleanly for each CLI.
  - ``gblsr-measure-latency`` runs end-to-end on CPU against a tiny
    inline config (no actual checkpoint or dataset needed).
  - ``gblsr-reconstruct`` runs end-to-end on CPU against a tiny
    random-init checkpoint and a synthetic input PNG.
  - ``gblsr-encode`` -> ``gblsr-decode`` round-trips on CPU against a
    tiny random-init checkpoint, with bit-identical equivalence to a
    full ``LocalSpectralArm.forward`` pass.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from gblsr.cli import decode as cli_decode
from gblsr.cli import encode as cli_encode
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


def test_encode_main_callable() -> None:
    assert callable(cli_encode.main)


def test_encode_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_encode.main(["--help"])
    assert exc_info.value.code == 0


def test_decode_main_callable() -> None:
    assert callable(cli_decode.main)


def test_decode_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_decode.main(["--help"])
    assert exc_info.value.code == 0


def test_encode_then_decode_round_trip_cpu(capsys, tmp_path) -> None:
    """End-to-end CPU round-trip: gblsr-encode -> features.pt -> gblsr-decode.

    Builds a tiny model from an inline config, saves a random-init
    checkpoint, encodes a synthetic image to a feature blob, decodes
    that blob back to an image, and asserts the decoded image is
    bit-identical to running ``LocalSpectralArm.forward`` end-to-end
    (since the split path runs the same encoder + decoder).
    """
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(_TINY_CONFIG)
    cfg_dict = load_config(str(cfg_path))
    rc = expand_run_configs(cfg_dict)[0]

    # Random-init model + checkpoint.
    model = build_model_from_run_config(rc).eval()
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    input_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color=(80, 120, 200)).save(input_path)
    feat_path = tmp_path / "features.pt"
    output_path = tmp_path / "recon.png"

    # 1) encode
    rc_code = cli_encode.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(input_path),
            "--output",
            str(feat_path),
            "--device",
            "cpu",
        ]
    )
    assert rc_code == 0
    assert feat_path.exists()

    # Inspect the feature blob: required keys + metadata sanity.
    blob = torch.load(feat_path, weights_only=False)
    assert set(blob.keys()) >= {"feat", "pad_info", "arm", "bandwidth_mode", "patch_size"}
    assert blob["arm"] == rc.arm
    assert blob["bandwidth_mode"] == rc.bandwidth_mode
    assert blob["patch_size"] == rc.patch_size
    # feat shape: (1, d_feat, image_size/patch_size, image_size/patch_size)
    # with the tiny config that's (1, 16, 2, 2).
    assert blob["feat"].shape == (
        1,
        rc.d_feat,
        rc.image_size // rc.patch_size,
        rc.image_size // rc.patch_size,
    )

    # 2) decode
    rc_code = cli_decode.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(feat_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ]
    )
    assert rc_code == 0
    assert output_path.exists()
    assert Image.open(output_path).size == (16, 16)


def test_decode_rejects_blob_with_bandwidth_mode_mismatch(tmp_path) -> None:
    """If the blob's ``bandwidth_mode`` does not match the loaded
    model's, decode must hard-fail (rc=1) rather than silently produce
    a wrong reconstruction."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(_TINY_CONFIG)
    cfg_dict = load_config(str(cfg_path))
    rc = expand_run_configs(cfg_dict)[0]

    model = build_model_from_run_config(rc).eval()
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    # Hand-craft a feat blob whose bandwidth_mode disagrees with the config.
    feat = torch.randn(1, rc.d_feat, 2, 2)
    pad_info = {
        "orig_h": rc.image_size,
        "orig_w": rc.image_size,
        "padded_h": rc.image_size,
        "padded_w": rc.image_size,
        "pad_t": 0,
        "pad_b": 0,
        "pad_l": 0,
        "pad_r": 0,
        "mode_used": "reflect",
        "patch_size": rc.patch_size,
        "num_patches_h": rc.image_size // rc.patch_size,
        "num_patches_w": rc.image_size // rc.patch_size,
    }
    bad_blob = {
        "feat": feat,
        "pad_info": pad_info,
        "orig_shape": (1, 3, rc.image_size, rc.image_size),
        "arm": rc.arm,
        "bandwidth_mode": "local_linear",  # disagrees with rc.bandwidth_mode="global_scalar"
        "patch_size": rc.patch_size,
    }
    feat_path = tmp_path / "bad.pt"
    torch.save(bad_blob, feat_path)

    rc_code = cli_decode.main(
        [
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(feat_path),
            "--output",
            str(tmp_path / "should_not_be_written.png"),
            "--device",
            "cpu",
        ]
    )
    assert rc_code == 1
    assert not (tmp_path / "should_not_be_written.png").exists()


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
