# GB-LSR

**Global-Bandwidth Local Spectral Representation** for continuous image
reconstruction.

**Paper**: [arXiv:XXX](https://arxiv.org/abs/XXX) *(placeholder — update once the arXiv ID is assigned)*

A fixed-grid local spectral image representation: the image domain is
partitioned into a fixed grid of non-overlapping square patches, each
patch carries a small block of coefficients for a truncated Fourier
basis predicted from shared convolutional-encoder features by a single
linear projection, and a single trainable scalar bandwidth is shared
globally across all patches. Reconstruction at any continuous
coordinate is a fixed-size basis contraction whose cost is independent
of image size.

## Repository layout

```
gblsr/                           repo root
├── pyproject.toml               project metadata + dependencies + CLI entries
├── uv.lock                      reproducible dependency pins (uv)
├── README.md                    (this file)
├── LICENSE                      BSD 3-Clause
├── CITATION.cff                 GitHub-readable citation metadata
├── python-pkg-plan.md           package design / layout notes
├── src/gblsr/                   Python package
│   ├── models/                  Basis, encoder, local-spectral decoder
│   ├── encoders/                Heavyweight image encoders (e.g. RDN)
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

CUDA toolkit and a matching `torch` build must be installed separately.

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

Each command is also reachable as a module (``python -m gblsr.cli.train
| eval | latency | reconstruct | encode | decode``) and via the repo-root
shim scripts (`scripts/train.py`, `scripts/eval.py`, `scripts/measure_latency.py`,
`scripts/reconstruct.py`, `scripts/encode.py`, `scripts/decode.py`).

### Splitting the forward pass across machines

`gblsr-encode` + `gblsr-decode` together let you split a GB-LSR
forward pass across machines: encode on machine A, transfer the
(much smaller) feature blob over the network, decode on machine B.
The encoder output for the default config is ~24x smaller than the
raw input image, and the decode side reconstructs bit-identically
to a single-machine ``LocalSpectralArm.forward`` pass on the same
checkpoint. Both ends must use the same trained checkpoint; the
feature blob carries metadata (arm, bandwidth_mode, patch_size) and
the decode side hard-fails on any mismatch. Only the local-spectral
arm is supported (the Global Fourier-MLP baseline arm uses a
shape-locked global-pool decoder that is not amenable to per-patch
feature transfer).

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

Each ``__init__.py`` declares an ``__all__`` listing the canonical
public symbols; the deep-import paths (e.g.
``gblsr.models.arms.LocalSpectralArm``) continue to work for power
users.

### Minimal end-to-end example

```python
import torch
from gblsr import LocalSpectralArm, ModelConfig, EncoderConfig, BasisConfig
from gblsr.latency import measure_latency

# Build a tiny model
mc = ModelConfig(
    arm="local_spectral",
    image_size=256,
    patch_size=32,
    basis=BasisConfig(patch_size=32, p_max=16),
    encoder=EncoderConfig(d_feat=128, n_layers=3),
)
model = LocalSpectralArm(mc, bandwidth_mode="global_scalar").eval()

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

## Production speedup

The shipped configuration matches a deployment-conservative inference
setup: batch size 1, no AMP, no `torch.compile`, no CUDA Graphs.
This makes `gblsr-measure-latency` an honest reflection of
"ordinary PyTorch eager-mode inference at single-image batch" — the
worst-case latency a naive deployment would see.

For production use, in order of empirically observed impact on
``LocalSpectralArm`` (single H200, batch=1, ``global_scalar``
bandwidth mode, 256x256 input):

- **`torch.compile`**: ``model = torch.compile(model)``. The
  largest single-knob speedup — measured **2.44x** at 256x256 on
  H200 (1.43 ms -> 0.58 ms). One-time compile cost is ~60 s and
  is per-shape; recompilation is triggered by input-shape changes.
- **Batching**: replace ``(1, 3, H, W)`` input with ``(B, 3, H, W)``
  for larger ``B``. Decoder cost shifts from per-image overhead to
  per-pixel multiply-adds, which amortize well.
- **CUDA Graphs**: capture the forward pass once at a fixed input
  shape and replay it on subsequent inputs. Useful for repeated
  identical-shape inference.
- **AMP** (``torch.amp.autocast`` bf16 or fp16): **not recommended
  for this model.** The model is small enough (~1 M parameters,
  per-patch projections of size 32x16x16) that autocast overhead
  (dtype casts in/out of the autocast region, weight casts) exceeds
  the half-precision kernel savings. Empirically AMP bf16 measures
  0.79-0.95x of fp32 (i.e. slower) at every input size tested
  (64, 256, 512, 1024). Stacking ``torch.compile`` + bf16 also does
  not beat ``torch.compile`` alone. Recommended only if you are
  stacking the model into a larger pipeline that already runs under
  AMP.

These knobs are independent and can be combined; for this model,
``torch.compile`` alone delivers essentially all of the available
speedup.

## Variants

The GB-LSR family is parameterized by ``bandwidth_mode`` in
``LocalSpectralDecoder``:

| ``bandwidth_mode``  | Description                                              |
|---------------------|----------------------------------------------------------|
| ``fixed_midpoint``  | Bandwidth pinned to a single fixed value (no training). |
| ``global_scalar``   | **Main variant**: one global trainable scalar shared across all patches. |
| ``local_linear``    | Per-patch bandwidth from a linear-sigmoid head.         |
| ``local_logspace``  | Per-patch bandwidth from a log-space sigmoid head.      |

## Datasets

The data loaders read external datasets from disk (paths configured per
run). Canonical sources:

- DTD: <https://www.robots.ox.ac.uk/~vgg/data/dtd/>
- DIV2K: <https://data.vision.ee.ethz.ch/cvl/DIV2K/>
- Kodak: <https://r0k.us/graphics/kodak/>
- Set14 / Set5 / Urban100: <https://github.com/jbhuang0604/SelfExSR>
- BSDS500 / B100: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/>

## Citation

If you use GB-LSR in your work, please cite:

```bibtex
@article{shad2026gblsr,
  title   = {GB-LSR: A Fast Local-Spectral Image Representation with a
             Single Global Bandwidth for Continuous Reconstruction and
             Super-Resolution},
  author  = {Shad, Max and Khoshnevis, Naeem},
  journal = {arXiv preprint arXiv:XXX},
  year    = {2026},
}
```

*(Replace ``XXX`` with the actual arXiv identifier once assigned.)*

The same citation metadata is also provided in machine-readable
form in [CITATION.cff](CITATION.cff) (Citation File Format 1.2.0),
which GitHub renders as a "Cite this repository" widget in the
sidebar and which tools like Zotero / arXiv2BibTeX consume directly.

## License

BSD 3-Clause. Copyright (c) 2026, President and Fellows of Harvard
College. See [LICENSE](LICENSE) for the full text.

## Patent notice

The methods implemented in this software are the subject of one or
more pending patent applications owned and controlled by the
President and Fellows of Harvard College (the "Patent Rights"). The
BSD 3-Clause license above grants rights to the copyrighted source
code only; it does not grant, by implication, estoppel, or
otherwise, any license under the Patent Rights.

For patent licensing inquiries, including any commercial use that
requires rights under the Patent Rights, contact the Harvard Office
of Technology Development (OTD) at `otd@harvard.edu`.
