"""
test_dataset.py — Dataset Abstraction Layer Tests (V2)

Tests the contract that every dataset loader in this codebase must
satisfy (datasets/base.py), and both concrete dataset loaders
(datasets/fvg_b.py, datasets/oulp_mvlp.py) against synthetic on-disk
directory trees matching their real structure.

Coverage:
    - Sample namedtuple / DatasetMeta contract
    - Age binning (datasets/base.py AGE_BINS, age_to_bin)
    - gait_collate_fn across all three label-availability cases:
      always-None (FVG-B-style), partially-None (OU-LP-Bag-style),
      always-present
    - The dataset registry (datasets/registry.py) dispatch mechanism
    - Full FVG-B build_fvgb_dataloaders() against a synthetic directory
    - Full OU-MVLP build_oulp_mvlp_dataloaders() against a synthetic
      directory, including the defensive label-file parsing and the
      standalone diagnostic mode's underlying functions
"""

import sys
import os
import csv
import shutil
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.base import (
    Sample, DatasetMeta, AGE_BINS, NUM_AGE_BINS, age_to_bin,
    gait_collate_fn, build_label_masks,
)


# -- Sample / DatasetMeta contract tests -----------------------------------------

class TestSampleContract:

    def test_sample_fields(self):
        s = Sample(
            frames=torch.zeros(10, 1, 64, 64),
            id_label=5, gender_label=1, age_label=None, age_bin=None,
        )
        assert s.frames.shape == (10, 1, 64, 64)
        assert s.id_label == 5
        assert s.gender_label == 1
        assert s.age_label is None
        assert s.age_bin is None


class TestDatasetMeta:

    def test_required_fields(self):
        meta = DatasetMeta(
            name='test', has_gender=True, has_age=False,
            num_identities=10, image_size=(64, 64),
            sequence_length=30, protocols=['A', 'B'],
        )
        assert meta.name == 'test'
        assert meta.has_gender is True
        assert meta.has_age is False
        assert meta.protocols == ['A', 'B']


class TestAgeBinning:

    def test_all_bins_covered(self):
        """Every integer age 0-100 must map to exactly one bin."""
        for age in range(0, 101):
            bin_idx = age_to_bin(age)
            assert 0 <= bin_idx < NUM_AGE_BINS

    def test_bin_boundaries(self):
        """Boundary values between adjacent bins must not raise and
        must be assigned to the bin whose range explicitly contains them."""
        for lo, hi, name in AGE_BINS:
            assert age_to_bin(lo) is not None
            assert age_to_bin(hi) is not None

    def test_negative_age_rejected(self):
        with pytest.raises(ValueError):
            age_to_bin(-1)

    def test_num_age_bins_matches_list_length(self):
        assert NUM_AGE_BINS == len(AGE_BINS)
        assert NUM_AGE_BINS == 7   # the literature convention chosen


# -- gait_collate_fn tests (the bug this codebase actually hit) -----------------

class TestGaitCollateFn:
    """
    These specifically target the real bug found during stage 1
    integration testing: PyTorch's default_collate cannot handle None
    fields at all, which would have crashed on the very first FVG-B
    batch had this not been caught before wiring it into the
    DataLoaders. These tests exist so that bug can never silently
    return if gait_collate_fn is ever modified.
    """

    def test_all_none_age_fvgb_style(self):
        samples = [
            Sample(torch.rand(5,1,8,8), 1, 0, None, None),
            Sample(torch.rand(5,1,8,8), 2, 1, None, None),
        ]
        batch = gait_collate_fn(samples)
        assert batch['age_label'] is None
        assert batch['age_bin'] is None
        assert not batch['age_mask'].any()
        assert batch['gender_label'] is not None
        assert batch['frames'].shape == (2, 5, 1, 8, 8)

    def test_partial_age_oulp_mvlp_style(self):
        samples = [
            Sample(torch.rand(5,1,8,8), 1, 1, 25.0, age_to_bin(25.0)),
            Sample(torch.rand(5,1,8,8), 2, 0, None, None),
            Sample(torch.rand(5,1,8,8), 3, 1, 60.0, age_to_bin(60.0)),
        ]
        batch = gait_collate_fn(samples)
        assert batch['age_mask'].tolist() == [True, False, True]
        assert torch.isnan(batch['age_label'][1])
        assert batch['age_bin'][1].item() == -1

    def test_fully_labeled_age(self):
        samples = [
            Sample(torch.rand(5,1,8,8), 1, 0, 30.0, age_to_bin(30.0)),
            Sample(torch.rand(5,1,8,8), 2, 1, 45.0, age_to_bin(45.0)),
        ]
        batch = gait_collate_fn(samples)
        assert batch['age_mask'].all()

    def test_mixed_none_gender_rejected(self):
        """
        Gender must be dataset-wide (all-or-nothing), unlike age which
        may be partial. Mixed None/non-None gender indicates a bug in
        the calling dataset loader's gender_map construction.
        """
        samples = [
            Sample(torch.rand(5,1,8,8), 1, 0, None, None),
            Sample(torch.rand(5,1,8,8), 2, None, None, None),
        ]
        with pytest.raises(AssertionError):
            gait_collate_fn(samples)

    def test_no_gender_no_age_dataset(self):
        """A hypothetical identity-only dataset -- both fields all-None."""
        samples = [
            Sample(torch.rand(5,1,8,8), 1, None, None, None),
            Sample(torch.rand(5,1,8,8), 2, None, None, None),
        ]
        batch = gait_collate_fn(samples)
        assert batch['gender_label'] is None
        assert batch['age_label'] is None


class TestBuildLabelMasks:

    def test_mask_correctness(self):
        labels = [1.0, None, 2.0, None, 3.0]
        mask = build_label_masks(labels)
        assert mask.tolist() == [True, False, True, False, True]


# -- Dataset registry tests --------------------------------------------------------

class TestDatasetRegistry:

    def test_known_datasets_registered(self):
        from datasets.registry import DATASET_REGISTRY
        assert 'fvgb' in DATASET_REGISTRY
        assert 'oulp_mvlp' in DATASET_REGISTRY

    def test_unknown_dataset_raises_clear_error(self):
        from datasets.registry import get_dataset_entry
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_entry('not_a_real_dataset')

    def test_fvgb_entry_resolves(self):
        from datasets.registry import get_dataset_entry
        entry = get_dataset_entry('fvgb')
        assert callable(entry.builder)
        assert len(entry.config_files) > 0


# -- Synthetic FVG-B fixture + end-to-end test ------------------------------------

@pytest.fixture(scope='module')
def synthetic_fvgb_root():
    """
    Build a minimal but structurally faithful synthetic FVG-B directory
    tree, matching the real FVG-B convention exactly:
    crop_sil/session{N}/{subject_id:03d}/{seq_id:02d}/{frame:05d}.png
    """
    root = Path(tempfile.mkdtemp()) / 'fvgb_synthetic'
    sil_root = root / 'crop_sil'

    subjects = list(range(1, 13))   # 12 subjects, all <=147 -> session1
    genders  = {sid: ('M' if sid % 2 == 1 else 'F') for sid in subjects}
    N_FRAMES = 15

    for sid in subjects:
        for seq_id in [f'{i:02d}' for i in range(1, 13)]:
            seq_dir = sil_root / 'session1' / f'{sid:03d}' / seq_id
            seq_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(1, N_FRAMES + 1):
                img = Image.fromarray(
                    (np.random.rand(32, 32) * 255).astype(np.uint8), mode='L'
                )
                img.save(seq_dir / f'{frame_idx:05d}.png')

    train_ids = subjects[:9]
    test_ids  = subjects[9:]

    with open(root / 'train_id_list.txt', 'w') as f:
        f.write('\n'.join(str(s) for s in train_ids))
    with open(root / 'test_id_list.txt', 'w') as f:
        f.write('\n'.join(str(s) for s in test_ids))
    with open(root / 'annotated_gender_information.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sid in subjects:
            writer.writerow([sid, genders[sid]])

    yield str(root)
    shutil.rmtree(root.parent, ignore_errors=True)


class TestFVGBDataset:

    def test_build_dataloaders_end_to_end(self, synthetic_fvgb_root):
        from datasets.fvg_b import build_fvgb_dataloaders

        cfg = {
            'dataset': {
                'root': synthetic_fvgb_root,
                'sequence_length': 8,
                'image_size': [32, 32],
                'val_fraction': 0.2,
            },
            'training': {
                'batch_size': 4, 'num_workers': 0, 'P': 4, 'K': 2,
            },
        }
        result = build_fvgb_dataloaders(cfg)

        assert result['meta'].name == 'fvgb'
        assert result['meta'].has_gender is True
        assert result['meta'].has_age is False
        assert result['meta'].image_size == (32, 32)
        assert result['meta'].sequence_length == 8

        batch = next(iter(result['train']))
        assert batch['frames'].dim() == 5   # [B, T, 1, H, W]
        assert batch['age_label'] is None
        assert batch['gender_label'] is not None

        val_batch = next(iter(result['val']))
        assert val_batch['age_label'] is None

    def test_ws_protocol_present(self, synthetic_fvgb_root):
        from datasets.fvg_b import build_fvgb_dataloaders
        cfg = {
            'dataset': {
                'root': synthetic_fvgb_root, 'sequence_length': 8,
                'image_size': [32, 32], 'val_fraction': 0.2,
            },
            'training': {'batch_size': 4, 'num_workers': 0, 'P': 4, 'K': 2},
        }
        result = build_fvgb_dataloaders(cfg)
        assert result['protocols']['WS'] is not None
        gal_batch = next(iter(result['protocols']['WS']['gallery']))
        # Gallery/probe use RAW subject ids (not remapped), per the
        # documented convention in datasets/fvg_b.py
        assert gal_batch['id_label'].max().item() <= max(range(1, 13))


# -- Synthetic OU-LP-Bag fixture + end-to-end test --------------------------------

@pytest.fixture(scope='module')
def synthetic_oulp_mvlp_root():
    """
    Build a minimal synthetic OU-MVLP directory tree matching the REAL
    confirmed structure (see datasets/oulp_mvlp.py module docstring):
    Silhouette_{view}-{seq}/{subject}/{frame}.png, with subjects placed
    deliberately around the real TRAIN_ID_MAX=5153 boundary so the
    standard subject-ID-range train/test split is exercised correctly
    without needing all 10,307 real subjects on disk.

    Uses 2 of the 14 views for speed -- VIEWS is monkeypatched on the
    oulp_mvlp module within each test that needs it, not globally,
    to avoid leaking state across tests in the same pytest session.
    """
    root = Path(tempfile.mkdtemp()) / 'oulp_mvlp_synthetic'

    test_views = ['000', '090']
    train_subjects = list(range(5142, 5154))   # 12 subjects, all <=5153
    test_subjects  = list(range(5154, 5162))   # 8 subjects, all >5153
    subjects = train_subjects + test_subjects
    genders  = {sid: ('M' if sid % 2 == 1 else 'F') for sid in subjects}
    N_FRAMES = 10

    for view in test_views:
        for seq in ['00', '01']:
            folder = root / f'Silhouette_{view}-{seq}'
            for sid in subjects:
                sid_str = str(sid).zfill(5)
                subj_dir = folder / sid_str
                subj_dir.mkdir(parents=True, exist_ok=True)
                for frame_idx in range(1, N_FRAMES + 1):
                    img = Image.fromarray(
                        (np.random.rand(32, 32) * 255).astype(np.uint8), mode='L'
                    )
                    img.save(subj_dir / f'{frame_idx:04d}.png')

    with open(root / 'gender_labels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sid in subjects:
            writer.writerow([str(sid).zfill(5), genders[sid]])

    age_labeled_subjects = subjects[::2]   # half the subjects
    with open(root / 'age_gender_intersection.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sid in age_labeled_subjects:
            age = 20 + (sid * 3) % 60
            writer.writerow([str(sid).zfill(5), age, genders[sid]])

    yield {
        'root': str(root), 'views': test_views,
        'train_subjects': train_subjects, 'test_subjects': test_subjects,
        'age_labeled_subjects': age_labeled_subjects,
    }
    shutil.rmtree(root.parent, ignore_errors=True)


class TestOULPMVLPDataset:
    """
    Tests for datasets/oulp_mvlp.py -- REPLACES the earlier
    TestOULPBagDataset, which tested against an incorrect understanding
    of this dataset (a bag/no-bag covariate that doesn't exist; the
    real covariate is view angle). See datasets/oulp_mvlp.py's module
    docstring for the full correction.
    """

    @pytest.fixture(autouse=True)
    def _patch_views(self, synthetic_oulp_mvlp_root):
        """
        Every test in this class needs datasets.oulp_mvlp.VIEWS reduced
        to the 2 views actually present in the synthetic fixture (the
        real default is all 14, which the synthetic data doesn't have).
        autouse=True applies this to every test method automatically,
        and restores the original VIEWS list afterward so this doesn't
        leak into other test files/sessions.
        """
        import datasets.oulp_mvlp as oulp_mvlp_module
        original_views = oulp_mvlp_module.VIEWS
        oulp_mvlp_module.VIEWS = synthetic_oulp_mvlp_root['views']
        yield
        oulp_mvlp_module.VIEWS = original_views

    def test_label_file_parsing_functions(self, synthetic_oulp_mvlp_root):
        from datasets.oulp_mvlp import (
            _load_gender_map, _load_age_gender_intersection,
            _discover_subjects_on_disk,
        )
        root = synthetic_oulp_mvlp_root['root']

        gender_map = _load_gender_map(os.path.join(root, 'gender_labels.csv'))
        assert len(gender_map) == 20   # 12 train + 8 test subjects

        age_map = _load_age_gender_intersection(
            os.path.join(root, 'age_gender_intersection.csv')
        )
        assert len(age_map) == 10   # half the subjects

        found = _discover_subjects_on_disk(root)
        assert found == set(synthetic_oulp_mvlp_root['train_subjects']
                            + synthetic_oulp_mvlp_root['test_subjects'])

    def test_malformed_gender_file_rejected(self):
        from datasets.oulp_mvlp import _load_gender_map
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("garbage,not,a,real,file\nmore,nonsense\n")
            path = f.name
        try:
            with pytest.raises(RuntimeError, match="Could not parse"):
                _load_gender_map(path)
        finally:
            os.unlink(path)

    def test_missing_view_folder_rejected(self, synthetic_oulp_mvlp_root):
        """Regression test: requesting a view with no on-disk folder
        must fail with a clear error, not silently return empty/wrong
        data."""
        from datasets.oulp_mvlp import _collect_sequences_for_view_seq
        with pytest.raises(RuntimeError, match="not found"):
            _collect_sequences_for_view_seq(
                synthetic_oulp_mvlp_root['root'], '999', '00'
            )

    def test_build_dataloaders_end_to_end(self, synthetic_oulp_mvlp_root):
        from datasets.oulp_mvlp import build_oulp_mvlp_dataloaders

        cfg = {
            'dataset': {
                'root': synthetic_oulp_mvlp_root['root'],
                'sequence_length': 8, 'image_size': [32, 32],
                'val_fraction': 0.2,
                'gender_label_file': 'gender_labels.csv',
                'age_gender_intersection_file': 'age_gender_intersection.csv',
                'cross_view': False,
            },
            'training': {'batch_size': 4, 'num_workers': 0, 'P': 4, 'K': 2},
        }
        result = build_oulp_mvlp_dataloaders(cfg)

        assert result['meta'].name == 'oulp_mvlp'
        assert result['meta'].has_gender is True
        assert result['meta'].has_age is True
        # protocols are now per-view ('view_000', 'view_090'), NOT
        # ['bag', 'no_bag'] -- the core correction this rewrite makes
        assert set(result['meta'].protocols) == {'view_000', 'view_090'}

        # CRITICAL regression test for the real bug found during this
        # stage's own integration testing: test_ids must be the
        # subjects ACTUALLY ON DISK in the test ID range, not the
        # theoretical 5154-10307 range -- using the unfiltered
        # theoretical range here silently produced misleading
        # "10295 test subjects" log output when only 8 actually existed.
        assert result['test_ids'] == synthetic_oulp_mvlp_root['test_subjects']

        batch = next(iter(result['train']))
        assert batch['gender_label'] is not None
        assert batch['age_mask'] is not None
        assert batch['age_mask'].dtype == torch.bool

    def test_same_view_protocols_present(self, synthetic_oulp_mvlp_root):
        from datasets.oulp_mvlp import build_oulp_mvlp_dataloaders
        cfg = {
            'dataset': {
                'root': synthetic_oulp_mvlp_root['root'],
                'sequence_length': 8, 'image_size': [32, 32],
                'val_fraction': 0.2,
                'gender_label_file': 'gender_labels.csv',
                'age_gender_intersection_file': 'age_gender_intersection.csv',
                'cross_view': False,
            },
            'training': {'batch_size': 4, 'num_workers': 0, 'P': 4, 'K': 2},
        }
        result = build_oulp_mvlp_dataloaders(cfg)
        for view in synthetic_oulp_mvlp_root['views']:
            assert result['protocols'][f'view_{view}'] is not None

    def test_view_loaders_always_present(self, synthetic_oulp_mvlp_root):
        """
        view_loaders must be present regardless of the cross_view flag
        -- evaluators/gait_eval.py's cross-view aggregation needs
        access to all views directly, and train.py's mid-training
        same_view check needs meta.protocols populated regardless.
        """
        from datasets.oulp_mvlp import build_oulp_mvlp_dataloaders
        cfg = {
            'dataset': {
                'root': synthetic_oulp_mvlp_root['root'],
                'sequence_length': 8, 'image_size': [32, 32],
                'val_fraction': 0.2,
                'gender_label_file': 'gender_labels.csv',
                'age_gender_intersection_file': 'age_gender_intersection.csv',
                'cross_view': False,   # even with cross_view OFF...
            },
            'training': {'batch_size': 4, 'num_workers': 0, 'P': 4, 'K': 2},
        }
        result = build_oulp_mvlp_dataloaders(cfg)
        # ...view_loaders must still be populated
        assert 'view_loaders' in result
        for view in synthetic_oulp_mvlp_root['views']:
            assert result['view_loaders'][view] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
