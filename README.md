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

Also reachable as `python -m gblsr.cli.<name>` or `scripts/<name>.py`.

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

### Quick API check

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

Defaults are deployment-conservative (batch=1, no AMP, no
`torch.compile`, no CUDA Graphs); `gblsr-measure-latency` reproduces
the paper protocol. For production, layer these on top of
`LocalSpectralArm`, in measured impact order on H200:

- **`torch.compile`** (`model = torch.compile(model)`): ~2.4x at
  256x256 (1.43 ms -> 0.58 ms). One-time ~60 s compile per input
  shape.
- **Batching**: pass `(B, 3, H, W)`; per-image cost amortizes.
- **CUDA Graphs**: capture + replay at a fixed input shape.
- **AMP** (bf16/fp16): *not recommended for this model* — measured
  0.79–0.95x of fp32 (autocast overhead exceeds kernel savings for a
  ~1 M-param model). Use only inside a larger AMP pipeline.

`torch.compile` alone delivers essentially all the available speedup;
stacking with bf16 does not beat it.

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

## License

BSD 3-Clause. Copyright (c) 2026, President and Fellows of Harvard
College. See [LICENSE](LICENSE) for the full text.
