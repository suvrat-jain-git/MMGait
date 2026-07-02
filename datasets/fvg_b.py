"""
fvg_b.py — FVG-B Dataset Loader (Official Protocol-Aware Version)

Directory structure on disk:
    <root>/
        crop_sil/
            session{1,2,3}/
                {subject_id:03d}/       e.g. 001, 002, ...
                    {sequence_id:02d}/  e.g. 01, 02, ..., 12
                        {frame:05d}.png e.g. 00001.png
        annotated_gender_information.csv   (subject_id,M/F — no header)
        train_id_list.txt                  (one subject_id per line)
        test_id_list.txt                   (one subject_id per line)

Session / Subject mapping (from FVG-B report Table 2):
    Subjects   1–147  → Session 1
    Subjects 148–226  → Session 2
    Session 3 subjects (bold in Table 2) → always in test set

Sequence ID meaning (from FVG-B report Table 1):
    Session 1:
        01,02,03 → Normal walk       (3 viewing angles: -45°, 0°, 45°)
        04,05,06 → Fast walk
        07,08,09 → Slow walk
        10,11,12 → Bag / Hat
    Session 2:
        01,02,03 → Normal walk
        04,05,06 → Fast walk
        07,08,09 → Change clothes
        10,11,12 → Multiple person

Official Evaluation Protocols (FVG-B report Table 3):
    Gallery is ALWAYS sequence '02' of Session 1 or 2.
    Probe sequences depend on the protocol:

    Protocol  Gallery       Probe
    --------  -------       -----
    WS        Sess1 seq02   Sess1 seq04-09
    BGHT      Sess1 seq02   Sess1 seq10-12
    CL        Sess2 seq02   Sess2 seq07-09
    MP        Sess2 seq02   Sess2 seq10-12
    ALL       Sess1 seq02   Sess1 seq01,03-12
              Sess2 seq02   Sess2 seq01,03-12
              Session3      Session3 seq01-12

Database Split:
    1% and 5% of test subjects are randomly selected as gallery subjects.
    Remaining test subjects are probe subjects.
    Random seed = 0 (original seed not publicly available; documented here).

Identity labels:
    Remapped to 0-indexed integers within the training split.
    Val split: last 10% of sorted train IDs (deterministic).
"""

import os
import csv
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from datasets.sampler import PKSampler
from datasets.base import Sample, DatasetMeta, gait_collate_fn
import torchvision.transforms.functional as TF


# ── Constants ──────────────────────────────────────────────────────────────

GALLERY_SEED = 0   # Seed for 1%/5% gallery subject selection
                   # Original paper seed not publicly available.
                   # Documented here for full reproducibility.

# Sequence IDs used as gallery anchor (always seq '02')
GALLERY_SEQ = '02'

# Protocol definitions: {name: {session: [probe_seq_ids]}}
# Session is an int key. ALL protocol has 3 sessions.
# Protocol definitions derived directly from FVG-B report Table 3.
#
# Table 3 (reconstructed):
#   Protocol | Sess1 Gal | Sess1 Prb   | Sess2 Gal | Sess2 Prb   | Sess3 Prb
#   ---------|-----------|-------------|-----------|-------------|----------
#   WS       |     2     | 4,5,6,7,8,9 |     2     | 4,5,6       |    -
#   BGHT     |     2     | 10,11,12    |     -     |    -        |    -
#   CL       |     -     |    -        |     2     | 7,8,9       |    -
#   MP       |     -     |    -        |     2     | 10,11,12    |    -
#   ALL      |     2     | 1,3-12      |     2     | 1,3-12      | 1-12
#
# gallery_sessions: sessions from which seq '02' is taken as gallery.
# probe: {session_int: [probe_seq_id_strings]}
PROTOCOLS = {
    'WS': {
        # Walk Speed: normal vs fast/slow walk, both sessions
        'gallery_sessions': [1, 2],
        'probe': {
            1: ['04','05','06','07','08','09'],
            2: ['04','05','06'],
        },
    },
    'BGHT': {
        # Bag/Hat: accessories, session 1 only
        'gallery_sessions': [1],
        'probe': {
            1: ['10','11','12'],
        },
    },
    'CL': {
        # Clothes change: session 2 only
        'gallery_sessions': [2],
        'probe': {
            2: ['07','08','09'],
        },
    },
    'MP': {
        # Multiple Person: session 2 only
        'gallery_sessions': [2],
        'probe': {
            2: ['10','11','12'],
        },
    },
    'ALL': {
        # All variations: both sessions + session 3
        # Gallery = Session1 seq02 AND Session2 seq02 per subject
        'gallery_sessions': [1, 2],
        'probe': {
            1: ['01','03','04','05','06','07','08','09','10','11','12'],
            2: ['01','03','04','05','06','07','08','09','10','11','12'],
            3: ['01','02','03','04','05','06','07','08','09','10','11','12'],
        },
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_id_list(path):
    """Read a text file of subject IDs (one per line). Returns sorted list of ints."""
    with open(path, 'r') as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    return sorted(ids)


def _load_gender_map(path):
    """
    Read annotated_gender_information.csv.
    Format: subject_id,M/F  (no header row)
    Returns dict: {subject_id (int) -> gender_label (int)}
        M = 0, F = 1
    """
    gender_map = {}
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            sid    = int(row[0].strip())
            gender = 0 if row[1].strip().upper() == 'M' else 1
            gender_map[sid] = gender
    return gender_map


def _infer_session(subject_id):
    """
    Infer which session a subject belongs to from their ID.
    Per FVG-B report Table 2:
        Subjects   1–147 → Session 1
        Subjects 148–226 → Session 2
    Session 3 subjects are a small subset (bold in Table 2) — they appear
    in both session directories; we handle them by checking disk.
    """
    if subject_id <= 147:
        return 1
    return 2


def _collect_sequences(root, subject_ids, sessions=None, seq_ids=None):
    """
    Collect sequences matching the given filters.

    Args:
        root:        dataset root Path
        subject_ids: list of subject IDs to include (None = all)
        sessions:    list of session ints to include (None = all)
        seq_ids:     list of sequence ID strings to include (None = all)

    Returns:
        list of dicts:
        {
            'subject_id': int,
            'session':    int,
            'seq_id':     str,
            'frame_dir':  Path,
        }
    """
    sil_root    = Path(root) / 'crop_sil'
    subject_set = set(subject_ids) if subject_ids is not None else None
    session_set = set(sessions)    if sessions    is not None else None
    seq_set     = set(seq_ids)     if seq_ids     is not None else None

    sequences = []

    for session_dir in sorted(sil_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('session'):
            continue
        session_num = int(session_dir.name.replace('session', ''))
        if session_set is not None and session_num not in session_set:
            continue

        for subj_dir in sorted(session_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            try:
                sid = int(subj_dir.name)
            except ValueError:
                continue
            if subject_set is not None and sid not in subject_set:
                continue

            for seq_dir in sorted(subj_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                if seq_set is not None and seq_dir.name not in seq_set:
                    continue
                frames = sorted(seq_dir.glob('*.png'))
                if len(frames) == 0:
                    continue
                sequences.append({
                    'subject_id': sid,
                    'session':    session_num,
                    'seq_id':     seq_dir.name,
                    'frame_dir':  seq_dir,
                })

    return sequences


def _sample_frames(frame_dir, T, training=True):
    """
    Load exactly T CONTIGUOUS frames from frame_dir.

    >= T frames:
        training=True  — pick a random contiguous window of T frames.
                         Different window each call; gives temporal augmentation.
        training=False — pick the centre-aligned window of T frames.
                         Deterministic; used for gallery and probe at eval time.
    < T frames:
        Tile from the start until T is reached (same for train and eval).

    Why contiguous frames matter:
        The motion branch computes abs(X[t] - X[t-1]).
        Consecutive frames produce smooth limb-movement differences.
        Uniformly spread frames produce large discontinuous jumps that do
        not represent real gait dynamics.

    Returns list of T Path objects in TEMPORAL ORDER.
    """
    all_frames = sorted(frame_dir.glob('*.png'))
    N = len(all_frames)

    if N == 0:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    if N >= T:
        if training:
            # Random start — any window that fits within [0, N-T]
            start = random.randint(0, N - T)
        else:
            # Centre-aligned — deterministic for evaluation
            start = (N - T) // 2
        selected = all_frames[start: start + T]
    else:
        # Tile: repeat sequence from start until we have T frames
        selected = [all_frames[i % N] for i in range(T)]

    return selected


def _gallery_probe_split(test_ids, split_pct, seed=GALLERY_SEED,
                          eligible_gallery_ids=None):
    """
    Randomly split test subjects into gallery and probe sets.

    Args:
        test_ids:             sorted list of ALL test subject IDs
        split_pct:            fraction to use as gallery (0.01=1%, 0.05=5%)
        seed:                 random seed (default GALLERY_SEED = 0)
        eligible_gallery_ids: if provided, gallery is sampled ONLY from
                              this subset of test_ids. Used for CL/MP to
                              ensure gallery subjects have session 2 data.
                              All remaining test_ids become probe.

    Returns:
        gallery_ids: list of subject IDs selected as gallery subjects
        probe_ids:   remaining subject IDs (probe)

    Note:
        The original paper seed is not publicly available.
        seed=0 is used and documented in GALLERY_SEED above.
    """
    rng = random.Random(seed)

    if eligible_gallery_ids is not None:
        # Sample gallery only from eligible subjects (e.g. session 2 subjects)
        pool = sorted(eligible_gallery_ids)
    else:
        pool = sorted(test_ids)

    n_gallery   = max(1, int(len(sorted(test_ids)) * split_pct))
    n_gallery   = min(n_gallery, len(pool))   # can't sample more than pool size
    gallery_ids = rng.sample(pool, n_gallery)
    gallery_ids = sorted(gallery_ids)
    # Probe = ALL test subjects.
    # Gallery subjects appear in both gallery AND probe —
    # their gallery sequences are seq '02', their probe sequences
    # are all other sequences. This is the correct FVG-B protocol:
    # every test subject must be identifiable from the gallery.
    probe_ids = sorted(test_ids)
    return gallery_ids, probe_ids


# ── Dataset classes ─────────────────────────────────────────────────────────

class FVGBDataset(Dataset):
    """
    Training / validation dataset.

    Each item is one sequence. Returns:
        frames:       [T, 1, H, W]  float32 in [0, 1]
        id_label:     int — 0-indexed within training split
        gender_label: int — 0=Male, 1=Female
    """

    def __init__(self, root, subject_ids, id_remap, gender_map,
                 T=30, image_size=224, augment=False):
        super().__init__()
        self.gender_map = gender_map
        self.id_remap   = id_remap
        self.T          = T
        self.image_size = image_size
        self.augment    = augment

        # Training sequences: all sessions, all sequences for these subjects
        self.sequences = _collect_sequences(root, subject_ids)

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No sequences found for given subject_ids under {root}."
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        # Contiguous window — random start during training,
        # centre-aligned during validation (augment flag doubles as the signal)
        frames = _sample_frames(seq['frame_dir'], self.T, training=self.augment)

        # One flip decision per sequence — same flip for all T frames
        do_flip = self.augment and (random.random() < 0.5)

        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            if do_flip:
                img = TF.hflip(img)
            tensors.append(TF.to_tensor(img))   # [1, H, W] float32

        # [T, 1, H, W]
        sequence_tensor = torch.stack(tensors, dim=0)

        return Sample(
            frames=sequence_tensor,
            id_label=self.id_remap[sid],
            gender_label=self.gender_map[sid],
            age_label=None,   # FVG-B has no age annotations
            age_bin=None,
        )


class FVGBGalleryDataset(Dataset):
    """
    Gallery dataset for one protocol at one database split.

    Gallery = sequence '02' of the protocol's gallery session,
              for the randomly selected gallery subjects.

    Returns:
        frames:       [T, 1, H, W]
        subject_id:   int — original subject ID (not remapped)
        gender_label: int
    """

    def __init__(self, root, gallery_ids, gender_map,
                 gallery_sessions, T=30, image_size=224):
        """
        Args:
            gallery_ids:      list of subject IDs selected as gallery
            gallery_sessions: list of session ints to pull gallery seq from.
                              WS/BGHT/CL/MP use [1] or [2].
                              ALL uses [1, 2] — two gallery seqs per subject.
        """
        super().__init__()
        self.gender_map = gender_map
        self.T          = T
        self.image_size = image_size

        # Gallery is always seq '02', collected across all gallery_sessions
        self.sequences = _collect_sequences(
            root, gallery_ids,
            sessions=gallery_sessions,
            seq_ids=[GALLERY_SEQ],
        )

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No gallery sequences found. sessions={gallery_sessions} seq={GALLERY_SEQ}"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=False)
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensors.append(TF.to_tensor(img))
        return Sample(
            frames=torch.stack(tensors, dim=0),
            id_label=sid,   # raw subject id, NOT remapped (gallery/probe convention)
            gender_label=self.gender_map[sid],
            age_label=None,   # FVG-B has no age annotations
            age_bin=None,
        )


class FVGBProbeDataset(Dataset):
    """
    Probe dataset for one protocol at one database split.

    Probe = protocol-specific sessions and sequence IDs,
            for the probe subjects (test - gallery).

    Returns:
        frames:       [T, 1, H, W]
        subject_id:   int — original subject ID
        gender_label: int
    """

    def __init__(self, root, probe_ids, gender_map,
                 probe_sessions_seqs, T=30, image_size=224):
        """
        Args:
            probe_ids:          list of subject IDs for probe
            probe_sessions_seqs: dict {session_int: [seq_id_strings]}
                                 from PROTOCOLS[name]['probe']
        """
        super().__init__()
        self.gender_map = gender_map
        self.T          = T
        self.image_size = image_size

        # Collect across all (session, seq_ids) pairs for this protocol
        self.sequences = []
        for session, seq_ids in probe_sessions_seqs.items():
            seqs = _collect_sequences(
                root, probe_ids,
                sessions=[session],
                seq_ids=seq_ids,
            )
            self.sequences.extend(seqs)

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No probe sequences found for probe_sessions_seqs={probe_sessions_seqs}"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        sid    = seq['subject_id']
        frames = _sample_frames(seq['frame_dir'], self.T, training=False)
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            tensors.append(TF.to_tensor(img))
        return Sample(
            frames=torch.stack(tensors, dim=0),
            id_label=sid,   # raw subject id, NOT remapped (gallery/probe convention)
            gender_label=self.gender_map[sid],
            age_label=None,   # FVG-B has no age annotations
            age_bin=None,
        )


# ── Factory functions ───────────────────────────────────────────────────────

def build_protocol_loaders(root, test_ids, gender_map, split_pct,
                           T=30, image_size=224, batch_size=16,
                           num_workers=4):
    """
    Build gallery and probe DataLoaders for all 5 official protocols
    at a given database split percentage.

    Args:
        root:        dataset root path
        test_ids:    sorted list of test subject IDs
        gender_map:  dict {subject_id -> 0/1}
        split_pct:   0.01 for 1% split, 0.05 for 5% split
        T:           sequence length
        image_size:  spatial size (H = W)
        batch_size:  DataLoader batch size
        num_workers: DataLoader workers

    Returns:
        dict keyed by protocol name, each containing:
            {
                'gallery':     DataLoader,
                'probe':       DataLoader,
                'gallery_ids': list of subject IDs,
                'probe_ids':   list of subject IDs,
            }
    """
    # Per FVG-B protocol (Table 3):
    # Gallery = seq '02' for ALL test subjects (not a random subset).
    # Probe   = protocol-specific sequences for ALL test subjects.
    # The 1%/5% split in the paper refers to the train/test subject
    # split ratio, not a gallery subject selection.
    #
    # Every test subject has:
    #   - One gallery entry (seq 02 from the gallery session)
    #   - Multiple probe entries (protocol-specific sequences)

    # Determine which sessions exist for each test subject on disk
    sil_root = Path(root) / 'crop_sil'
    sess1_test = sorted([
        sid for sid in test_ids
        if (sil_root / 'session1' / f'{sid:03d}').exists()
    ])
    sess2_test = sorted([
        sid for sid in test_ids
        if (sil_root / 'session2' / f'{sid:03d}').exists()
    ])

    loaders = {}
    for protocol_name, protocol_cfg in PROTOCOLS.items():
        gallery_sessions    = protocol_cfg['gallery_sessions']
        probe_sessions_seqs = protocol_cfg['probe']

        # Gallery subjects = test subjects who have the required session data
        if gallery_sessions == [1]:
            gallery_ids = sess1_test
        elif gallery_sessions == [2]:
            gallery_ids = sess2_test
        else:
            # Both sessions — include subjects from either session
            gallery_ids = sorted(set(sess1_test) | set(sess2_test))

        # Probe subjects = ALL test subjects
        probe_ids = sorted(test_ids)

        try:
            gallery_ds = FVGBGalleryDataset(
                root, gallery_ids, gender_map,
                gallery_sessions=gallery_sessions,
                T=T, image_size=image_size,
            )
            probe_ds = FVGBProbeDataset(
                root, probe_ids, gender_map,
                probe_sessions_seqs=probe_sessions_seqs,
                T=T, image_size=image_size,
            )
        except RuntimeError as e:
            print(f"  [WARNING] Protocol {protocol_name} skipped: {e}")
            loaders[protocol_name] = None
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

        loaders[protocol_name] = {
            'gallery':     gallery_loader,
            'probe':       probe_loader,
            'gallery_ids': gallery_ids,
            'probe_ids':   probe_ids,
        }

    return loaders


def build_fvgb_dataloaders(cfg):
    """
    Build all dataloaders from config.

    Args:
        cfg: merged config dict. Required keys:
            cfg['dataset']['root']
            cfg['dataset']['sequence_length']
            cfg['dataset']['image_size']       [H, W]
            cfg['training']['batch_size']
            cfg['training']['num_workers']
            cfg['dataset'].get('val_fraction', 0.1)

    Returns:
        dict with keys:
            'train':         DataLoader
            'val':           DataLoader
            'protocols_1pct': dict — all 5 protocols at 1% split
            'protocols_5pct': dict — all 5 protocols at 5% split
            'num_classes':   int — number of training identities
            'id_remap':      dict — {subject_id -> 0-indexed label}
            'test_ids':      list — test subject IDs
            'gender_map':    dict
    """
    root        = cfg['dataset']['root']
    T           = cfg['dataset']['sequence_length']
    image_size  = cfg['dataset']['image_size'][0]   # assume square
    batch_size  = cfg['training']['batch_size']
    num_workers = cfg['training']['num_workers']
    val_frac    = cfg['dataset'].get('val_fraction', 0.1)

    # ── Load splits and labels ─────────────────────────────────────────────
    train_ids  = _load_id_list(os.path.join(root, 'train_id_list.txt'))
    test_ids   = _load_id_list(os.path.join(root, 'test_id_list.txt'))
    gender_map = _load_gender_map(
        os.path.join(root, 'annotated_gender_information.csv')
    )

    # ── Val split: stratified by gender AND session ───────────────────────
    # Split train IDs into 4 groups: session1_male, session1_female,
    # session2_male, session2_female. Take val_frac from each group.
    # This ensures:
    #   1. Val gender ratio matches training set
    #   2. Val contains both Session 1 and Session 2 subjects
    #      (avoids val being all-Session2 which causes domain mismatch)
    sil_root = Path(root) / 'crop_sil'

    def get_session(sid):
        for sess in [1, 2, 3]:
            if (sil_root / f'session{sess}' / f'{sid:03d}').exists():
                return sess
        return 1  # fallback

    # Group by (session, gender)
    groups = {(s, g): [] for s in [1, 2] for g in [0, 1]}
    for sid in train_ids:
        sess   = get_session(sid)
        gender = gender_map[sid]
        key    = (min(sess, 2), gender)  # treat session3 as session2
        groups[key].append(sid)

    val_ids = []
    tr_ids  = []
    for key, sids in groups.items():
        sids_sorted = sorted(sids)
        n_val = max(1, int(len(sids_sorted) * val_frac)) if len(sids_sorted) > 1 else 0
        val_ids.extend(sids_sorted[-n_val:] if n_val > 0 else [])
        tr_ids.extend(sids_sorted[:-n_val]  if n_val > 0 else sids_sorted)

    val_ids = sorted(val_ids)
    tr_ids  = sorted(tr_ids)

    # Report split composition
    male_ids   = sorted([sid for sid in train_ids if gender_map[sid] == 0])
    female_ids = sorted([sid for sid in train_ids if gender_map[sid] == 1])

    # ── Identity remapping ─────────────────────────────────────────────────
    # 0-indexed labels from training subjects only
    # id_remap covers ALL train_ids (tr_ids + val_ids).
    # Both are subsets of the original sorted train list so labels
    # are contiguous 0..N-1 and always < num_classes.
    all_train_ids = sorted(tr_ids + val_ids)
    id_remap      = {sid: i for i, sid in enumerate(all_train_ids)}
    num_classes   = len(all_train_ids)

    # ── Training datasets ──────────────────────────────────────────────────
    train_ds = FVGBDataset(
        root, tr_ids, id_remap, gender_map,
        T=T, image_size=image_size, augment=True,
    )
    val_ds = FVGBDataset(
        root, val_ids, id_remap, gender_map,
        T=T, image_size=image_size, augment=False,
    )

    # Identity-balanced sampler: P identities × K samples per batch.
    # Guarantees valid positive pairs for triplet loss every batch.
    # P=8, K=4 → effective batch_size=32. Overrides cfg batch_size for train.
    P = cfg['training'].get('P', 8)
    K = cfg['training'].get('K', 4)
    try:
        pk_sampler = PKSampler(train_ds, P=P, K=K, drop_last=True)
        # Use batch_sampler — PKSampler already yields complete batches of P*K.
        # batch_sampler is incompatible with batch_size/shuffle/drop_last args.
        train_loader = DataLoader(
            train_ds,
            batch_sampler=pk_sampler,
            num_workers=num_workers, pin_memory=True,
            collate_fn=gait_collate_fn,
        )
        print(f"PKSampler: P={P} identities × K={K} samples = batch_size={P*K}")
    except RuntimeError as e:
        # Fallback to standard random sampling if not enough identities
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

    # ── Protocol loaders (all test subjects as gallery) ──────────────────
    proto_kwargs = dict(
        T=T, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    # split_pct parameter is now unused — kept for API compatibility
    protocols = build_protocol_loaders(
        root, test_ids, gender_map, split_pct=1.0, **proto_kwargs
    )

    meta = DatasetMeta(
        name='fvgb',
        has_gender=True,
        has_age=False,
        num_identities=num_classes,
        image_size=(image_size, image_size),
        sequence_length=T,
        protocols=list(PROTOCOLS.keys()),
    )

    return {
        'train':          train_loader,
        'val':            val_loader,
        'protocols':      protocols,
        # Keep 1pct/5pct keys pointing to same protocols for train.py compat
        'protocols_1pct': protocols,
        'protocols_5pct': protocols,
        'num_classes':    num_classes,
        'id_remap':       id_remap,
        'test_ids':       test_ids,
        'gender_map':     gender_map,
        'meta':           meta,
    }
