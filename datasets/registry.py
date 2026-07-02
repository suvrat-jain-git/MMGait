"""
registry.py — Dataset Registry

A single lookup table mapping a --dataset CLI flag value to the function
that builds its dataloaders, and the config files it needs merged.

This is the seam that lets train.py (and any other entry point) stay
completely dataset-agnostic: it never imports datasets.fvg_b or
datasets.oulp_mvlp directly, never has an if/elif chain on dataset name.
It looks up DATASET_REGISTRY[args.dataset], calls the returned builder
function, and gets back the same {'train', 'val', 'protocols', ...,
'meta'} dict shape regardless of which dataset was actually loaded
(see datasets/base.py DatasetMeta for the contract every builder must
satisfy).

Adding a new dataset means writing its loader module and registering it
here -- nothing in train.py, the trainer, the model factory, or the
evaluators needs to change, since all of them already consume datasets
generically via DatasetMeta and the dict-batch format.

NOTE: the second dataset registered here was originally named
'oulp_bag' and built against an incorrect assumption about which
OU-ISIR dataset was actually being used (a bag/no-bag covariate
dataset). This was corrected to 'oulp_mvlp' -- the OU-MVLP Multi-View
Large Population dataset, whose covariate is VIEW ANGLE, confirmed
directly from the official download page. See datasets/oulp_mvlp.py's
module docstring for the full correction.
"""

from typing import Callable, Dict, NamedTuple


class DatasetEntry(NamedTuple):
    """
    builder:      callable(cfg) -> dict with 'train'/'val'/'protocols'/
                  'meta'/etc. keys (see any build_*_dataloaders function)
    config_files: list of yaml config file paths (relative to repo root)
                  that should be merged into cfg['dataset'] for this
                  dataset specifically. Kept separate from model.yaml/
                  train.yaml, which are dataset-independent.
    """
    builder: Callable
    config_files: list


def _build_fvgb(cfg):
    from datasets.fvg_b import build_fvgb_dataloaders
    return build_fvgb_dataloaders(cfg)


def _build_oulp_mvlp(cfg):
    from datasets.oulp_mvlp import build_oulp_mvlp_dataloaders
    return build_oulp_mvlp_dataloaders(cfg)


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    'fvgb': DatasetEntry(
        builder=_build_fvgb,
        config_files=['configs/datasets/fvgb.yaml'],
    ),
    'oulp_mvlp': DatasetEntry(
        builder=_build_oulp_mvlp,
        config_files=['configs/datasets/oulp_mvlp.yaml'],
    ),
}


def get_dataset_entry(name: str) -> DatasetEntry:
    """
    Look up a dataset by its --dataset flag value.

    Raises a clear, actionable error (listing valid options) rather than
    a bare KeyError, since this is a user-facing CLI-argument lookup.
    """
    if name not in DATASET_REGISTRY:
        valid = ', '.join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. Valid options: {valid}"
        )
    return DATASET_REGISTRY[name]
