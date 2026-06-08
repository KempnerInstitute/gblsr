"""Public API for GB-LSR training: driver, configs, losses.

Re-exports the canonical training entry points so users can write
``from gblsr.training import RunConfig, train_one``.
"""

from .losses import (
    LossConfig,
    LossName,
    compute_losses,
    pointwise_charbonnier,
    pointwise_l1,
    pointwise_loss,
    pointwise_mse,
)
from .trainer import (
    RunConfig,
    aggregate,
    build_loaders,
    build_model_from_run_config,
    evaluate,
    expand_run_configs,
    load_config,
    set_seed,
    train_one,
)

__all__ = [
    # Trainer / driver
    "RunConfig",
    "train_one",
    "build_model_from_run_config",
    "build_loaders",
    "load_config",
    "expand_run_configs",
    "evaluate",
    "aggregate",
    "set_seed",
    # Losses
    "LossConfig",
    "LossName",
    "compute_losses",
    "pointwise_loss",
    "pointwise_mse",
    "pointwise_l1",
    "pointwise_charbonnier",
]
