"""Public API for GB-LSR data loading + region slicing.

Re-exports the canonical dataset classes, the ``DataConfig`` schema,
the collate functions for fixed-size and variable-resolution modes,
and the per-patch region-labeling utilities.
"""

from .loaders import (
    DataConfig,
    DIV2KImages,
    DTDImages,
    FlatImageFolder,
    KodakImages,
    PatchCropDataset,
    REGIONS,
    RegionThresholds,
    build_datasets,
    collate,
    collate_with_meta,
    fit_region_thresholds,
    label_patches,
    load_thresholds,
    make_collate,
    per_patch_edge_density,
    per_patch_hf_fraction,
    save_thresholds,
    varres_collate,
)

__all__ = [
    # Config + dataset classes
    "DataConfig",
    "DTDImages",
    "DIV2KImages",
    "KodakImages",
    "FlatImageFolder",
    "PatchCropDataset",
    "build_datasets",
    # Collation
    "collate",
    "collate_with_meta",
    "varres_collate",
    "make_collate",
    # Region slicing
    "REGIONS",
    "RegionThresholds",
    "label_patches",
    "fit_region_thresholds",
    "save_thresholds",
    "load_thresholds",
    "per_patch_hf_fraction",
    "per_patch_edge_density",
]
