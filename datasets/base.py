"""
base.py — Dataset-Agnostic Interface Contract

Defines the common interface every dataset loader (FVG-B, OU-LP-Bag, future
datasets) must satisfy, so the rest of the pipeline (model, trainer, losses,
evaluators) never needs to know which dataset is active.

Design principle:
    The MODEL decides which heads exist based on DATASET METADATA, not flags.
    A dataset reports what labels it can provide via `DatasetMeta`. The
    training script reads this metadata once at startup and instantiates
    only the heads/losses that have corresponding data. This avoids:
        - dead parameters (heads that exist but never receive gradients)
        - silent zero-loss terms (a loss computed against missing labels)
        - crashes from None labels reaching a loss function unexpectedly

Every sample loader (train/val/gallery/probe) must return a `Sample`
namedtuple. Fields that the active dataset does not support MUST be `None`
— this is the explicit, type-checkable way of signalling "this dataset has
no age labels" rather than e.g. returning -1 or 0 and hoping it's not
silently treated as a valid label downstream.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import namedtuple
import torch


# ── Sample contract ──────────────────────────────────────────────────────────

# A single sample returned by any Dataset.__getitem__ in this codebase.
#
#   frames:       [T, 1, H, W] float32 tensor in [0, 1]
#   id_label:     int — 0-indexed identity label within the current split
#                 (or original subject_id for gallery/probe datasets — see
#                 note in each dataset's docstring, mirrors existing FVG-B
#                 convention of remapped ids for train/val, raw ids for
#                 gallery/probe)
#   gender_label: int (0=Male, 1=Female) or None if dataset has no gender
#                 labels for this dataset/subject
#   age_label:    float (regression target, raw age in years) or None if
#                 dataset/subject has no age annotation
#   age_bin:      int (classification target, index into AGE_BINS) or None
#                 under the same condition as age_label
Sample = namedtuple(
    'Sample', ['frames', 'id_label', 'gender_label', 'age_label', 'age_bin']
)


# ── Standard age bins (literature convention) ────────────────────────────────
# 7-bin convention used widely in face/gait age-estimation literature:
#   child, teen, young adult, adult, middle-aged, mature, senior
AGE_BINS = [
    (0, 12,  'child'),
    (13, 18, 'teen'),
    (19, 25, 'young_adult'),
    (26, 35, 'adult'),
    (36, 45, 'middle_aged'),
    (46, 60, 'mature'),
    (61, 200, 'senior'),
]
NUM_AGE_BINS = len(AGE_BINS)


def age_to_bin(age: float) -> int:
    """Map a raw age value to its bin index per AGE_BINS."""
    for i, (lo, hi, _name) in enumerate(AGE_BINS):
        if lo <= age <= hi:
            return i
    raise ValueError(f"Age {age} does not fall into any defined AGE_BINS range")


# ── Dataset metadata contract ────────────────────────────────────────────────

@dataclass
class DatasetMeta:
    """
    Reported by every dataset's build_dataloaders() function.

    The training script reads this ONCE at startup to decide which heads
    and loss terms to instantiate. This is the single source of truth for
    "does this dataset support gender/age" — no other part of the codebase
    should infer this by inspecting label values at runtime.

    Fields:
        name:              dataset identifier string, e.g. 'fvgb', 'oulp_mvlp'
        has_gender:        True if every training/val/eval sample has a
                           non-None gender_label
        has_age:           True if AT LEAST the age-labeled subset has
                           non-None age_label for those samples. Datasets
                           may have a partial age-labeled subset (e.g.
                           OU-LP-Bag's intersection-with-OULP-Age list) —
                           in that case has_age=True but individual
                           samples may still carry age_label=None, and the
                           age loss must mask those out per-batch (see
                           losses/combined_loss.py age masking).
        num_identities:    number of identity classes in the training split
        image_size:        (H, W) — both datasets normalised to common size
        sequence_length:   T — both datasets normalised to common length
        protocols:         list of named evaluation protocol keys this
                           dataset defines (e.g. FVG-B's WS/BGHT/CL/MP/ALL).
                           OU-LP-Bag will define its own (e.g. bag/no-bag).
    """
    name: str
    has_gender: bool
    has_age: bool
    num_identities: int
    image_size: tuple
    sequence_length: int
    protocols: List[str]


# ── Per-batch label availability mask helper ─────────────────────────────────

def build_label_masks(batch_age_labels: List[Optional[float]]) -> torch.Tensor:
    """
    Build a boolean mask for which samples in a batch have a valid age
    label, for datasets where age annotation is a partial subset (e.g.
    OU-LP-Bag's intersection list) rather than dataset-wide.

    Args:
        batch_age_labels: list of length B, each entry float age or None

    Returns:
        mask: [B] bool tensor — True where age_label is not None

    Usage in combined_loss.py:
        Only the masked subset of a batch contributes to the age loss.
        If mask.sum() == 0 for a batch, the age loss term is skipped
        for that batch (returns 0, no gradient) rather than crashing.
    """
    return torch.tensor([lbl is not None for lbl in batch_age_labels])


# ── Collate function ──────────────────────────────────────────────────────

def gait_collate_fn(samples: List[Sample]) -> Dict[str, Any]:
    """
    Custom collate function for batches of `Sample` namedtuples.

    PyTorch's default_collate raises TypeError on None values — every
    dataset in this codebase may have age_label=None either for ALL
    samples (FVG-B, no age annotation at all) or for SOME samples
    (OU-LP-Bag, only the OULP-Age intersection subset is labeled).
    default_collate cannot handle either case, so we collate manually.

    Returns a dict (not a Sample) because PyTorch's `pin_memory` and
    multiprocessing workers handle dicts/tensors more predictably than
    nested namedtuples containing None across worker process boundaries.

    Returns:
        {
            'frames':        [B, T, 1, H, W] float32 tensor
            'id_label':      [B] long tensor
            'gender_label':  [B] long tensor, OR None if every sample's
                             gender_label is None (gender-free dataset)
            'age_label':     [B] float tensor with NaN where original was
                             None, OR None if every sample's age_label
                             is None. NaN (not zero) avoids silently
                             treating "no label" as "label=0".
            'age_bin':       [B] long tensor with -1 where original was
                             None, OR None if every sample's age_bin is
                             None. -1 is an invalid class index so any
                             unmasked use immediately errors instead of
                             silently training against bin 0.
            'age_mask':      [B] bool tensor — True where age_label was
                             not None. Always present (all-False if no
                             sample in the batch has an age label), so
                             downstream code never needs to check for
                             a missing key.
        }
    """
    frames       = torch.stack([s.frames for s in samples], dim=0)
    id_label     = torch.tensor([s.id_label for s in samples], dtype=torch.long)

    gender_vals  = [s.gender_label for s in samples]
    if all(g is None for g in gender_vals):
        gender_label = None
    else:
        # Mixed None/non-None within one dataset would indicate a bug in
        # the dataset loader (gender should be dataset-wide, not partial)
        assert all(g is not None for g in gender_vals), (
            "gender_label is None for some but not all samples in a batch. "
            "Gender must be dataset-wide (all-or-nothing) — if this fires, "
            "check the dataset loader's gender_map construction."
        )
        gender_label = torch.tensor(gender_vals, dtype=torch.long)

    age_vals = [s.age_label for s in samples]
    age_mask = build_label_masks(age_vals)
    if not age_mask.any():
        age_label = None
        age_bin   = None
    else:
        age_label = torch.tensor(
            [a if a is not None else float('nan') for a in age_vals],
            dtype=torch.float32,
        )
        age_bin = torch.tensor(
            [b if b is not None else -1 for b in [s.age_bin for s in samples]],
            dtype=torch.long,
        )

    return {
        'frames':       frames,
        'id_label':     id_label,
        'gender_label': gender_label,
        'age_label':    age_label,
        'age_bin':      age_bin,
        'age_mask':     age_mask,
    }
