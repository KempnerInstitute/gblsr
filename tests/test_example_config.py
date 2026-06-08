"""Schema test for ``configs/example.yaml``.

Loads the example config and runs it through ``expand_run_configs``
to confirm the YAML matches the current ``RunConfig`` / ``DataConfig``
fields. Catches drift between the example config and the actual
trainer schema during refactors (the smoke test cost is millisecond
since no training, no data, no model is involved).
"""

from pathlib import Path

from gblsr.training.trainer import expand_run_configs, load_config


# Resolve relative to the repo root (tests/ is a sibling of configs/).
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"


def test_example_config_exists() -> None:
    assert EXAMPLE_CONFIG_PATH.exists(), f"example config missing at {EXAMPLE_CONFIG_PATH}"


def test_example_config_parses_and_expands() -> None:
    cfg_dict = load_config(str(EXAMPLE_CONFIG_PATH))
    runs = expand_run_configs(cfg_dict)

    # The YAML lists one arm and one seed -> one RunConfig.
    assert len(runs) == 1

    rc = runs[0]
    assert rc.experiment_id == "gblsr_example"
    assert rc.arm == "local_spectral"
    assert rc.bandwidth_mode == "global_scalar"
    assert rc.seed == 0
    # The run dir should incorporate the slug + seed.
    assert "gb_lsr_scalar__seed0" in rc.run_dir


def test_example_config_has_valid_loss_name() -> None:
    """Loss must be one of the LossConfig literals."""
    cfg_dict = load_config(str(EXAMPLE_CONFIG_PATH))
    runs = expand_run_configs(cfg_dict)
    assert runs[0].loss_name in {"mse", "l1", "charbonnier"}


def test_example_config_data_block() -> None:
    """The ``data`` block populates ``DataConfig``'s required fields."""
    cfg_dict = load_config(str(EXAMPLE_CONFIG_PATH))
    runs = expand_run_configs(cfg_dict)
    data = runs[0].data
    # Required fields must be present (even if as placeholder paths).
    assert isinstance(data.dtd_root, str) and data.dtd_root
    assert isinstance(data.div2k_root, str) and data.div2k_root
    assert data.image_size == 256
    assert data.patch_size == 32
