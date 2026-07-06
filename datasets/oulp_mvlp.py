import os
import csv
import random
import re
import sys
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.sampler import PKSampler
from datasets.base import Sample, DatasetMeta, gait_collate_fn, age_to_bin
import torchvision.transforms.functional as TF


# -- Constants ------------------------------------------------------------------

VIEWS = ['000', '015', '030', '045', '060', '075', '090',
         '180', '195', '210', '225', '240', '255', '270']

GALLERY_SEQ = '01'
PROBE_SEQ   = '00'

# Standard OU-MVLP subject-ID-range train/test split (literature
# convention, confirmed across multiple published papers -- see module
# docstring). NOT a random/stratified split like FVG-B/the corrected
# oulp_bag.py used -- deviating from this would make results
# non-comparable to published baselines.
TRAIN_ID_MAX = 5153   # subjects 1..5153 inclusive -> train
                       # subjects 5154..10307      -> test


# -- Label file parsing (defensive, fail-loud -- same approach as the
#    original corrected oulp_bag.py, since the exact file format still
#    could not be directly verified) -----------------------------------------------

def _format_subject_id(sid, width=5):
    """Zero-pad a subject ID to match the on-disk folder naming."""
    return str(sid).zfill(width)


def _load_subject_info(path):
    """
    Read subject_info_OUMVLP.csv which contains ID, gender, age for
    ALL 10,307 subjects. Format confirmed from real downloaded file:
        ID,gender,age       <- header line
        1,M,-               <- age '-' means no label for this subject
        2,F,22
        3,F,26

    Returns:
        gender_map: {subject_id (int) -> gender_label (int)}  M=0, F=1
        age_map:    {subject_id (int) -> age (float)}
                    only subjects with a real age label (not '-')
    """
    gender_map = {}
    age_map    = {}

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                sid = int(row[0].strip())
            except ValueError:
                continue   # skip header / comment lines

            g = row[1].strip().upper()
            if g == 'M':
                gender_map[sid] = 0
            elif g == 'F':
                gender_map[sid] = 1
            else:
                continue

            if len(row) >= 3:
                age_token = row[2].strip()
                if age_token != '-':
                    try:
                        age_map[sid] = float(age_token)
                    except ValueError:
                        pass

    if not gender_map:
        raise RuntimeError(
            f"Could not parse any (ID, gender) pairs from {path}. "
            f"Expected format: ID,gender,age with header row. "
            f"Check the file exists and has the correct format."
        )

    n_male   = sum(1 for g in gender_map.values() if g == 0)
    n_female = sum(1 for g in gender_map.values() if g == 1)
    print(f"Subject info loaded: {len(gender_map)} subjects "
          f"({n_male} M, {n_female} F), "
          f"{len(age_map)} with age labels")
    return gender_map, age_map


def _load_train_test_split(path):
    """
    Read ID_list.csv which contains paired train/test subject IDs.
    Format confirmed from real downloaded file:
        Training & testing subject ID are shown...   <- comment lines
        Note: Sequence 00 is used as Probe...
                  while both sequences...
                                                     <- blank line
        Training subject ID, Testing subject ID      <- header
        1,2                                          <- train_id, test_id
        3,4
        ...
        10305,10306

    Returns:
        train_ids: sorted list of training subject IDs
        test_ids:  sorted list of testing subject IDs
    """
    train_ids = []
    test_ids  = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            # Handle the last row which has empty train ID (,10307)
            # -- subject 10307 is test-only with no train pair
            try:
                tsid = int(row[1].strip())
            except ValueError:
                continue
            test_ids.append(tsid)
            try:
                tid = int(row[0].strip())
                train_ids.append(tid)
            except ValueError:
                pass   # empty train column -- test-only subject, skip for train

    if not train_ids:
        raise RuntimeError(
            f"Could not parse any train/test ID pairs from {path}. "
            f"Expected format: two numeric columns (train_id, test_id) "
            f"after comment lines. Check the file exists and has the "
            f"correct format."
        )

    print(f"Train/test split loaded: "
          f"{len(train_ids)} train, {len(test_ids)} test subjects")
    return sorted(train_ids), sorted(test_ids)


# Backward-compatible aliases
def _load_gender_map(path):
    gender_map, _ = _load_subject_info(path)
    return gender_map


def _load_age_gender_intersection(path):
    _, age_map = _load_subject_info(path)
    return age_map




# -- Sequence discovery on disk -------------------------------------------------

def _collect_sequences_for_view_seq(root, view, seq, subject_ids=None):
    """
    Collect sequences for ONE (view, sequence) combination, e.g.
    Silhouette_000-01/. This is the unit the rest of this module builds
    on -- unlike FVG-B/the original oulp_bag.py, there is no single
    "walk everything and filter" helper here, because view+sequence are
    baked into the TOP-LEVEL folder name (Silhouette_{view}-{seq}/),
    not a filterable subdirectory level.

    Args:
        root:        dataset root Path
        view:        one of VIEWS (e.g. '000')
        seq:         '00' or '01'
        subject_ids: list of subject IDs to include (None = all found)

    Returns: list of dicts {'subject_id', 'view', 'seq', 'frame_dir'}
    """
    folder = Path(root) / f'Silhouette_{view}-{seq}'
    if not folder.exists():
        raise RuntimeError(
            f"Expected silhouette folder not found: {folder}. Confirm "
            f"the root path and folder-naming convention against your "
            f"actual downloaded data (see this module's docstring "
            f"'FORMAT ASSUMPTIONS' section)."
        )

    subject_set = set(subject_ids) if subject_ids is not None else None
    sequences = []

    for subj_dir in sorted(folder.iterdir()):
        if not subj_dir.is_dir():
            continue
        try:
            sid = int(subj_dir.name)
        except ValueError:
            continue
        if subject_set is not None and sid not in subject_set:
            continue

        frames = sorted(subj_dir.glob('*.png'))
        if len(frames) == 0:
            continue
        sequences.append({
            'subject_id': sid, 'view': view, 'seq': seq,
            'frame_dir': subj_dir,
        })

    return sequences


def _discover_subjects_on_disk(root, reference_view=None, reference_seq='01'):
    """
    Scan ONE reference Silhouette_{view}-{seq} folder to determine which
    subject IDs actually exist on disk. Used purely for accurate
    reporting (see build_oulp_mvlp_dataloaders' train/test split log
    message) -- NOT used to filter train_ids_all/test_ids themselves,
    since _collect_sequences_for_view_seq already correctly filters
    against real data independently for every (view, subject_ids) call
    site; this helper exists only so log messages don't misleadingly
    report theoretical range sizes as if they were real counts.

    Args:
        root:           dataset root
        reference_view: which view to scan (default: VIEWS[0])
        reference_seq:  which sequence to scan (default: gallery seq)

    Returns:
        set of subject IDs (ints) found, or empty set if the reference
        folder doesn't exist (reported, not raised -- this is a
        best-effort reporting helper, not a hard requirement)
    """
    view = reference_view or VIEWS[0]
    folder = Path(root) / f'Silhouette_{view}-{reference_seq}'
    if not folder.exists():
        return set()

    found = set()
    for subj_dir in folder.iterdir():
        if not subj_dir.is_dir():
            continue
        try:
            found.add(int(subj_dir.name))
        except ValueError:
            continue
    return found


def _sample_frames(frame_dir, T, training=True):
    """Identical contiguous-window sampling logic to every other
    dataset loader in this codebase -- see datasets/fvg_b.py for the
    full rationale."""
    all_frames = sorted(frame_dir.glob('*.png'))
    N = len(all_frames)

    if N == 0:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    if N >= T:
        start = random.randint(0, N - T) if training else (N - T) // 2
        selected = all_frames[start: start + T]
    else:
        selected = [all_frames[i % N] for i in range(T)]

    return selected


# -- Dataset classes -------------------------------------------------------------

class OULPMVLPDataset(Dataset):
    """
    Training / validation dataset. Pools sequences across ALL 14 views
    and BOTH sequence indices for the given subject_ids -- training
    sees every view, not a single one, so the model learns
    view-invariant (or at least view-robust) gait representations.

    Returns a Sample (see datasets/base.py), same contract as every
    other dataset loader in this codebase.
    """

    def __init__(self, root, subject_ids, id_remap, gender_map, age_map,
                 T=30, image_size=64, augment=False):
        super().__init__()
        self.gender_map = gender_map
        self.age_map    = age_map
        self.id_remap   = id_remap
        self.T          = T
        self.image_size = image_size
        self.augment    = augment

        self.sequences = []
        for view in VIEWS:
            for seq in ['00', '01']:
                try:
                    self.sequences.extend(
                        _collect_sequences_for_view_seq(
                            root, view, seq, subject_ids
                        )
                    )
                except RuntimeError:
                    # A specific view-seq folder may legitimately not
                    # exist for partial/subset downloads -- skip rather
                    # than fail the whole dataset construction, but
                    # this is worth a visible warning.
                    print(f"[WARNING] Skipping Silhouette_{view}-{seq} "
                          f"(folder not found)")

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No sequences found for given subject_ids under {root}."
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        sid = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=self.augment)

        do_flip = self.augment and (random.random() < 0.5)

        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            if do_flip:
                img = TF.hflip(img)
            tensors.append(TF.to_tensor(img))

        sequence_tensor = torch.stack(tensors, dim=0)

        age = self.age_map.get(sid)
        age_bin = age_to_bin(age) if age is not None else None

        return Sample(
            frames=sequence_tensor,
            id_label=self.id_remap[sid],
            gender_label=self.gender_map[sid],
            age_label=age,
            age_bin=age_bin,
        )


class OULPMVLPViewDataset(Dataset):
    """
    Gallery or probe dataset for ONE specific view angle. Used for both
    the same_view and cross_view evaluation modes -- in same_view mode,
    one of these is built per view for gallery and one for probe
    (matched view); in cross_view mode, ALL 14 are built for gallery
    and ALL 14 for probe, and the evaluator computes the full pairwise
    matrix across them.

    Returns a Sample with RAW subject_id as id_label (not remapped),
    matching the gallery/probe convention used everywhere else in this
    codebase.
    """

    def __init__(self, root, subject_ids, gender_map, age_map,
                 view, seq, T=30, image_size=64):
        super().__init__()
        self.gender_map = gender_map
        self.age_map    = age_map
        self.T          = T
        self.image_size = image_size
        self.view       = view
        self.seq        = seq

        self.sequences = _collect_sequences_for_view_seq(
            root, view, seq, subject_ids
        )

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No sequences found for view={view} seq={seq}."
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        sid = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=False)
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensors.append(TF.to_tensor(img))

        age = self.age_map.get(sid)
        age_bin = age_to_bin(age) if age is not None else None

        return Sample(
            frames=torch.stack(tensors, dim=0),
            id_label=sid,
            gender_label=self.gender_map[sid],
            age_label=age,
            age_bin=age_bin,
        )


# -- Factory functions -----------------------------------------------------------

def _id_list_from_range(lo, hi):
    return list(range(lo, hi + 1))


def build_view_loaders(root, test_ids, gender_map, age_map,
                       T=30, image_size=64, batch_size=16, num_workers=4):
    """
    Build one gallery DataLoader and one probe DataLoader PER VIEW
    (14 each, 28 total), keyed by view string. This is the structure
    the cross_view evaluation mode needs (all pairwise combinations);
    same_view mode is a simple subset of this (only the matching-view
    pairs are actually used).

    Returns:
        dict: {view: {'gallery': DataLoader, 'probe': DataLoader}}
              (or {view: None} if that view's folder was unavailable)
    """
    view_loaders = {}
    for view in VIEWS:
        try:
            gallery_ds = OULPMVLPViewDataset(
                root, test_ids, gender_map, age_map,
                view=view, seq=GALLERY_SEQ, T=T, image_size=image_size,
            )
            probe_ds = OULPMVLPViewDataset(
                root, test_ids, gender_map, age_map,
                view=view, seq=PROBE_SEQ, T=T, image_size=image_size,
            )
        except RuntimeError as e:
            print(f"  [WARNING] View {view} skipped: {e}")
            view_loaders[view] = None
            continue

        gallery_loader = DataLoader(
            gallery_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=gait_collate_fn,
        )
        probe_loader = DataLoader(
            probe_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=gait_collate_fn,
        )
        view_loaders[view] = {
            'gallery': gallery_loader, 'probe': probe_loader,
            'gallery_ids': test_ids, 'probe_ids': test_ids,
        }

    return view_loaders


def build_same_view_protocols(view_loaders):
    """
    Build the 'protocols' dict shape every OTHER dataset loader in this
    codebase already uses (one gallery + one probe per protocol name),
    for the same_view evaluation mode -- one protocol entry per view,
    named 'view_{view}', where gallery and probe are both that view.

    This is what gets returned in build_oulp_mvlp_dataloaders()'s
    'protocols' key when --cross_view is NOT passed, so every existing
    evaluator (gait_eval.py's evaluate_protocol, the analysis scripts'
    primary-protocol lookup via meta.protocols[0], etc.) works
    completely unmodified -- same_view mode requires ZERO changes
    anywhere outside this dataset loader.
    """
    protocols = {}
    for view, loaders in view_loaders.items():
        if loaders is None:
            protocols[f'view_{view}'] = None
            continue
        protocols[f'view_{view}'] = loaders   # gallery=probe=same view
    return protocols


def build_oulp_mvlp_dataloaders(cfg):
    """
    Build all dataloaders from config. Mirrors every other dataset
    loader's build_*_dataloaders() signature and return-dict shape
    (see datasets/fvg_b.py, the corrected datasets/oulp_bag.py).

    Required cfg keys (see configs/datasets/oulp_mvlp.yaml):
        cfg['dataset']['root']
        cfg['dataset']['sequence_length']
        cfg['dataset']['image_size']
        cfg['dataset']['id_list_file']          -- ID_list.csv
        cfg['dataset']['subject_info_file']      -- subject_info_OUMVLP.csv
        cfg['dataset'].get('cross_view', False)  -- the protocol flag
        cfg['training']['batch_size'], ['num_workers'], ['P'], ['K']
        cfg['dataset'].get('val_fraction', 0.1)

    Returns: same dict shape as every other loader, PLUS:
        'view_loaders': the full {view: {gallery, probe}} dict, always
                        present regardless of cross_view flag, since
                        evaluators/gait_eval.py's cross-view aggregation
                        function needs access to all 14 views directly
                        rather than only the matched-view 'protocols'
                        entries.
    """
    root        = cfg['dataset']['root']
    T           = cfg['dataset']['sequence_length']
    image_size  = cfg['dataset']['image_size'][0]
    batch_size  = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']
    val_frac    = cfg['dataset'].get('val_fraction', 0.1)
    cross_view  = cfg['dataset'].get('cross_view', False)

    # Load train/test split from ID_list.csv (paired columns format,
    # confirmed from real downloaded file -- see _load_train_test_split)
    train_ids_all, test_ids = _load_train_test_split(
        os.path.join(root, cfg['dataset']['id_list_file'])
    )

    # Load gender + age from subject_info_OUMVLP.csv (single combined
    # file, confirmed from real downloaded data -- age '-' means no
    # label for that subject, handled in _load_subject_info)
    gender_map, age_map = _load_subject_info(
        os.path.join(root, cfg['dataset']['subject_info_file'])
    )

    # Val split: stratified by gender, carved from the TRAINING subject
    # pool (test subjects are reserved entirely for gallery/probe,
    # consistent with every other dataset loader's convention).
    male_ids   = sorted([sid for sid in train_ids_all if gender_map.get(sid) == 0])
    female_ids = sorted([sid for sid in train_ids_all if gender_map.get(sid) == 1])

    val_ids, tr_ids = [], []
    for group in [male_ids, female_ids]:
        n_val = max(1, int(len(group) * val_frac)) if len(group) > 1 else 0
        val_ids.extend(group[-n_val:] if n_val > 0 else [])
        tr_ids.extend(group[:-n_val] if n_val > 0 else group)

    val_ids = sorted(val_ids)
    tr_ids  = sorted(tr_ids)

    all_train_ids = sorted(tr_ids + val_ids)
    id_remap      = {sid: i for i, sid in enumerate(all_train_ids)}
    num_classes   = len(all_train_ids)

    train_ds = OULPMVLPDataset(
        root, tr_ids, id_remap, gender_map, age_map,
        T=T, image_size=image_size, augment=True,
    )
    val_ds = OULPMVLPDataset(
        root, val_ids, id_remap, gender_map, age_map,
        T=T, image_size=image_size, augment=False,
    )

    P = cfg['training'].get('P', 8)
    K = cfg['training'].get('K', 4)
    try:
        pk_sampler = PKSampler(train_ds, P=P, K=K, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_sampler=pk_sampler,
            num_workers=num_workers, pin_memory=True,
            collate_fn=gait_collate_fn,
        )
        print(f"PKSampler: P={P} identities x K={K} samples = batch_size={P*K}")
    except RuntimeError as e:
        print(f"[WARNING] PKSampler failed ({e}), falling back to random sampler")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=gait_collate_fn,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=gait_collate_fn,
    )

    print("Building per-view gallery/probe loaders "
          "(14 views x 2 = 28 DataLoaders)...")
    view_loaders = build_view_loaders(
        root, test_ids, gender_map, age_map,
        T=T, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )

    # protocols dict shape: same_view mode populates it directly so
    # every existing evaluator works unmodified; cross_view mode ALSO
    # populates it (so meta.protocols[0] / the "primary protocol"
    # pattern used by train.py's mid-training Rank-1 check still has
    # something cheap to evaluate every 10 epochs without running the
    # full 182-pair cross-view aggregation mid-training) but the
    # cross-view-specific aggregation itself is computed on demand by
    # evaluators/gait_eval.py's evaluate_cross_view_protocol(), not
    # pre-computed here.
    protocols = build_same_view_protocols(view_loaders)
    protocol_names = list(protocols.keys())

    n_age_labeled = len(age_map)
    print(f"Age-labeled subjects (intersection with age/gender file): "
          f"{n_age_labeled} / {len(gender_map)} total subjects")
    print(f"Train/test split (standard OU-MVLP subject-ID-range split, "
          f"(from ID_list.csv):")
    print(f"  Train: {len(all_train_ids)} subjects")
    print(f"  Test:  {len(test_ids)} subjects")
    print(f"Evaluation mode: {'cross_view (literature-standard, 182 pairs)' if cross_view else 'same_view (default, 14 independent protocols)'}")

    meta = DatasetMeta(
        name='oulp_mvlp',
        has_gender=True,
        has_age=True,
        num_identities=num_classes,
        image_size=(image_size, image_size),
        sequence_length=T,
        protocols=protocol_names,
    )

    return {
        'train':        train_loader,
        'val':          val_loader,
        'protocols':    protocols,
        'view_loaders': view_loaders,
        'cross_view':   cross_view,
        'num_classes':  num_classes,
        'id_remap':     id_remap,
        'test_ids':     test_ids,
        'gender_map':   gender_map,
        'age_map':      age_map,
        'meta':         meta,
    }


# -- Standalone diagnostic --------------------------------------------------------

if __name__ == '__main__':
    """
    Run this directly against your real downloaded OU-MVLP root to
    verify every format assumption flagged in this module's docstring
    BEFORE the first real training run:

        python datasets/oulp_mvlp.py /path/to/oulp_mvlp_root \
            gender_labels.csv age_gender_intersection.csv
    """
    if len(sys.argv) != 4:
        print(__doc__)
        print(
            "\nUsage: python datasets/oulp_mvlp.py <root> "
            "<id_list_file> <subject_info_file>"
        )
        sys.exit(1)

    root, id_list_file, subject_info_file = sys.argv[1:4]

    print("=== OU-MVLP format verification ===\n")

    print(f"[1/4] Checking Silhouette_{{view}}-{{seq}} folders exist...")
    found_views = []
    for view in VIEWS:
        for seq in ['00', '01']:
            folder = Path(root) / f'Silhouette_{view}-{seq}'
            if folder.exists():
                found_views.append(f'{view}-{seq}')
    print(f"      Found {len(found_views)}/28 expected Silhouette_* folders")
    if len(found_views) < 28:
        missing = set(f'{v}-{s}' for v in VIEWS for s in ['00','01']) - set(found_views)
        print(f"      MISSING: {sorted(missing)[:10]}"
              f"{'...' if len(missing) > 10 else ''}")

    print(f"\n[2/4] Loading train/test split from {id_list_file}...")
    train_ids, test_ids = _load_train_test_split(os.path.join(root, id_list_file))
    print(f"      Train: {len(train_ids)} subjects, Test: {len(test_ids)} subjects")
    print(f"      (expect 5153 train, 5154 test)")

    print(f"\n[3/4] Loading gender + age from {subject_info_file}...")
    gender_map, age_map = _load_subject_info(os.path.join(root, subject_info_file))
    print(f"      Age range: {min(age_map.values()):.0f}-{max(age_map.values()):.0f} "
          f"(expect roughly 2-87 per official page)")

    print(f"\n[4/4] Scanning one view-seq folder for subject/frame structure...")
    if found_views:
        view, seq = found_views[0].split('-')
        sample_sequences = _collect_sequences_for_view_seq(root, view, seq)
        print(f"      Silhouette_{view}-{seq}: {len(sample_sequences)} subjects found")
        if sample_sequences:
            s = sample_sequences[0]
            n_frames = len(list(s['frame_dir'].glob('*.png')))
            print(f"      Example subject {s['subject_id']}: {n_frames} frames")

    print("\n=== Diagnostic complete ===")
    print("If folder count is not 28/28, or gender/age counts look wrong,")
    print("update the corresponding function in this file before running")
    print("real training.")
