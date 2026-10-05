# GB-LSR

[![CI](https://github.com/KempnerInstitute/gblsr/actions/workflows/ci.yml/badge.svg)](https://github.com/KempnerInstitute/gblsr/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2606.19617-b31b1b.svg)](https://arxiv.org/abs/2606.19617)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-ffcc4d.svg)](https://huggingface.co/KempnerInstituteAI/gblsr)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/KempnerInstitute/gblsr/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Global-Bandwidth Local Spectral Representation for continuous image
reconstruction.

Paper: [arXiv:2606.19617](https://arxiv.org/abs/2606.19617)

A fixed-grid local spectral image representation: the image domain is
partitioned into a fixed grid of non-overlapping square patches, each
patch carries a small block of coefficients for a truncated Fourier
basis predicted from shared convolutional-encoder features by a single
linear projection, and a single trainable scalar bandwidth is shared
across every patch and every image. As in earlier local spectral
decoders, decoding at a continuous coordinate is a fixed-size basis
contraction whose cost is set by the spectral cutoff; GB-LSR learns the
bandwidth of that basis instead of fixing it.

## Repository layout

```
gblsr/                           repo root
├── pyproject.toml               project metadata + dependencies + CLI entries
├── uv.lock                      reproducible dependency pins (uv)
├── README.md                    (this file)
├── LICENSE                      BSD 3-Clause
├── CITATION.cff                 GitHub-readable citation metadata
├── src/gblsr/                   Python package
│   ├── models/                  Basis, encoder, local-spectral decoder
│   ├── encoders/                Heavyweight image encoders (e.g. RDN)
│   ├── asr/                     Arbitrary-scale SR extension (RDN + ASR decoder)
│   ├── latency/                 Fixed GPU latency protocol
│   ├── training/                Training driver + losses
│   ├── data/                    Image loaders + region slicing
│   ├── metrics/                 PSNR / SSIM / LPIPS / edge-LPIPS / LSE
│   └── cli/                     Console-script entry points
├── scripts/                     Repo-root shims for the CLI commands
├── configs/                     Example YAML configs
└── tests/                       Unit + smoke + end-to-end tests
```

## Installation

Python 3.10+.

### Using uv (recommended for development)

[uv](https://docs.astral.sh/uv/) gives reproducible installs via the
checked-in `uv.lock` and is the canonical dev workflow for this repo:

```bash
uv sync                   # resolve + install from uv.lock (reproducible)
uv sync --extra dev       # also installs pytest + ruff
uv run pytest             # run any command in the env without activating it
uv run gblsr-train --config configs/example.yaml
```

To bump dependencies, edit `pyproject.toml` and run `uv lock` to
refresh the lock file.

### Using pip

If you do not have uv installed, plain `pip` works too (you do not get
the pinned-versions reproducibility of `uv.lock`, but the package will
install fine):

```bash
pip install -e .          # editable install for development
pip install -e ".[dev]"   # also installs pytest + ruff
```

`uv sync` installs torch from the PyTorch CUDA 12.1 index; with pip, install a
torch build that matches your CUDA driver.

## Command-line tools

Six console commands are installed by `pip install -e .` / `uv sync`:

| Command                    | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `gblsr-train`              | Train one or more `(arm, seed)` combinations from a RunConfig YAML.    |
| `gblsr-eval`               | Evaluate a saved checkpoint on the val split; prints aggregate as JSON.|
| `gblsr-measure-latency`    | Measure per-image inference latency under the fixed GPU latency protocol. |
| `gblsr-reconstruct`        | Run a trained checkpoint on one image and write the reconstruction.    |
| `gblsr-encode`             | Encode an image to a compact feature tensor (encoder-only forward).    |
| `gblsr-decode`             | Decode a feature tensor back to an image (decoder-only forward).       |

Usage:

```bash
uv run gblsr-train --config configs/example.yaml
uv run gblsr-eval --config configs/example.yaml --checkpoint <path/to/model.pt>
uv run gblsr-measure-latency --config configs/example.yaml --device cuda
uv run gblsr-reconstruct \
    --config configs/example.yaml \
    --checkpoint <path/to/model.pt> \
    --input image.png --output recon.png
uv run gblsr-encode \
    --config configs/example.yaml \
    --checkpoint <path/to/model.pt> \
    --input image.png --output features.pt
uv run gblsr-decode \
    --config configs/example.yaml \
    --checkpoint <path/to/model.pt> \
    --input features.pt --output recon.png
```

Also reachable as `python -m gblsr.cli.<name>` or `scripts/<name>.py`; for
`gblsr-measure-latency` these are `gblsr.cli.latency` and
`scripts/measure_latency.py`.

## Using gblsr from Python

The most-used entry points are re-exported at the package top level:

```python
from gblsr import LocalSpectralArm, build_model, ModelConfig
from gblsr import BasisConfig, EncoderConfig
from gblsr import measure_latency, LatencyConfig
```

For specialized entry points, use the subpackages:

```python
from gblsr.models   import (
    LocalSpectralArm, BaselineArm,
    LocalSpectralDecoder, GlobalFourierMLPDecoder,
    build_model, ModelConfig, EncoderConfig, BasisConfig,
)
from gblsr.encoders import RDNEncoder, RDNConfig, build_rdn_encoder
from gblsr.latency  import measure_latency, LatencyConfig, LatencyResult
from gblsr.training import RunConfig, train_one, build_model_from_run_config
from gblsr.metrics  import psnr, ssim, lpips_metric, edge_lpips, local_spectrum_error
from gblsr.data     import DataConfig, build_datasets, label_patches
```

The package and each subpackage except ``gblsr.cli`` declare an
``__all__`` listing the public symbols; deep imports (e.g.
``gblsr.models.arms.LocalSpectralArm``) also work.

### Quick API check

```python
import torch
from gblsr import LocalSpectralArm, ModelConfig, EncoderConfig, BasisConfig
from gblsr.latency import measure_latency

# GB-LSR-Scalar at patch size 32, as in the design study
mc = ModelConfig(
    arm="local_spectral",
    image_size=256,
    patch_size=32,
    basis=BasisConfig(patch_size=32, p_max=16),
    encoder=EncoderConfig(d_feat=128),
)
model = LocalSpectralArm(mc, bandwidth_mode="global_scalar", adapt_order=False).eval()

# Forward pass
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
recon = out["recon"]
print(recon.shape)              # (1, 3, 256, 256)

# Inference latency (10 warmup + 50 timed reps under the fixed protocol)
result = measure_latency(model, x, track_peak_memory=False)
print(f"median latency: {result.median_ms:.2f} ms")
```

## Variants

The GB-LSR family is parameterized by ``bandwidth_mode`` and
``adapt_order`` in ``LocalSpectralDecoder``:

| ``bandwidth_mode``  | Description                                              |
|---------------------|----------------------------------------------------------|
| ``fixed_midpoint``  | Bandwidth pinned to a single fixed value (no training). |
| ``global_scalar``   | Main variant: one global trainable scalar shared across all patches. |
| ``local_linear``    | Per-patch bandwidth from a linear-sigmoid head.         |
| ``local_logspace``  | Per-patch bandwidth from a log-space sigmoid head.      |

With ``adapt_order=True`` (the default), the decoder also predicts a
per-patch cutoff order. The paper's GB-LSR-Scalar, GB-LSR-Fixed, and
GB-LSR-Bandwidth are ``global_scalar``, ``fixed_midpoint``, and
``local_logspace`` with ``adapt_order=False``; GB-LSR-Full is
``local_logspace`` with ``adapt_order=True``.

## Arbitrary-scale super-resolution (ASR) extension

`gblsr.asr` decodes one RDN-encoder feature map at any output resolution via
a LIIF/LTE-style continuous-query interface. All reported variants are
config-only (no separate code path):

```python
import torch
from gblsr import GBLSRScalarASR

model = GBLSRScalarASR()                                       # base, 22.02M params
noLE  = GBLSRScalarASR(decoder_cfg={"local_ensemble": False})  # drop 4-corner ensemble
nf96  = GBLSRScalarASR(encoder_cfg={"num_features": 96})       # wider encoder, 24.93M

hr = model.predict_full(torch.rand(1, 3, 64, 64), H_q=256, W_q=256)  # (1,3,256,256)
```

`encoder_cfg` takes `RDNConfig` fields; `decoder_cfg` takes
`GBLSRScalarASRDecoder` arguments (`p_max`, `bandwidth_init`,
`local_ensemble`); variants compose (e.g. `nf96+noLE`).

Scope: ships the method (encoder + decoder + `predict_full`). The training
recipe (1M steps on DIV2K) is in the paper; the harness that timed the
super-resolution comparison and the methods the paper compares with are
not bundled.

## Pretrained models

Trained weights for four super-resolution models and two design-study
models from the paper (training seed 0) are on the Hugging Face Hub at
[KempnerInstituteAI/gblsr](https://huggingface.co/KempnerInstituteAI/gblsr),
tag `arxiv-v2`, under CC BY-NC 4.0. The model card gives their scores
and loading code.

## Datasets

The data loaders read external datasets from disk (paths configured per
run). Canonical sources:

- DTD: <https://www.robots.ox.ac.uk/~vgg/data/dtd/>
- DIV2K: <https://data.vision.ee.ethz.ch/cvl/DIV2K/>
- Kodak: <https://r0k.us/graphics/kodak/>
- Set14 / Set5 / Urban100: <https://github.com/jbhuang0604/SelfExSR>
- BSDS300 / B100: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/>

Kodak / Set14 / Urban100 are the test sets of the native-reconstruction
design study; Set5 / Set14 / B100 / Urban100 and the DIV2K validation
split are the arbitrary-scale SR evaluation datasets.

## Production speedup

Defaults are deployment-conservative (batch=1, no AMP, no
`torch.compile`, no CUDA Graphs); `gblsr-measure-latency` uses the
same defaults. For production, layer these on top of
`LocalSpectralArm`:

- `torch.compile` (`model = torch.compile(model)`): the first call
  compiles, and a new input shape can trigger a recompile.
- Batching: pass `(B, 3, H, W)`; per-image cost amortizes.
- CUDA Graphs: capture + replay at a fixed input shape.
- AMP (bf16/fp16): use only inside a larger AMP pipeline.

## Citation

If you use GB-LSR in your work, please cite:

```bibtex
@article{shad2026gblsr,
  title   = {{GB-LSR}: Local Spectral Decoding with a Learned Global
             Bandwidth for Arbitrary-Scale Super-Resolution},
  author  = {Shad, Max and Khoshnevis, Naeem},
  journal = {arXiv preprint arXiv:2606.19617},
  year    = {2026},
}
```

## License

BSD 3-Clause. Copyright (c) 2026, President and Fellows of Harvard
College. See [LICENSE](LICENSE) for the full text.
