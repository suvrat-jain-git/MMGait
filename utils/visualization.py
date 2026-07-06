import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for server
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib not installed. pip install matplotlib")


def plot_training_curves(
    csv_path: str,
    out_dir:  str = 'experiments/plots',
) -> None:
    """
    Plot training and validation loss curves from the CSV log.

    Produces one figure with subplots for:
        - Total loss (train + val)
        - Identity loss
        - Triplet loss
        - Gender loss
        - Orthogonality loss
        - Val gender accuracy

    Args:
        csv_path: path to training_log.csv
        out_dir:  directory to save the PNG
    """
    _check_mpl()
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas not installed. pip install pandas")

    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ('train_total',    'val_total',       'Total Loss'),
        ('train_identity', None,              'Identity Loss (train)'),
        ('train_triplet',  None,              'Triplet Loss (train)'),
        ('train_gender',   None,              'Gender Loss (train)'),
        ('train_adversarial', None,           'Orthogonality Loss (train)'),
        (None,             'val_gender_acc',  'Val Gender Accuracy'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (train_col, val_col, title) in zip(axes, plots):
        if train_col and train_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label='Train', color='steelblue')
        if val_col and val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], label='Val',
                    color='tomato', alpha=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('BioKinematicNet Training Curves', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = out_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {out_path}")


def plot_cmc_curves(
    cmc_dict: dict,
    out_dir:  str = 'experiments/plots',
    max_rank: int = 20,
) -> None:
    """
    Plot CMC curves for multiple protocols on one figure.

    Args:
        cmc_dict: {protocol_name: np.ndarray of shape [max_rank]}
        out_dir:  directory to save PNG
        max_rank: x-axis limit
    """
    _check_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colours = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    fig, ax  = plt.subplots(figsize=(8, 6))

    for (name, cmc), colour in zip(cmc_dict.items(), colours):
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc * 100, label=name, color=colour, linewidth=2)
        ax.scatter(1, cmc[0] * 100, color=colour, zorder=5, s=40)

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Recognition Rate (%)', fontsize=12)
    ax.set_title('CMC Curves — BioKinematicNet', fontsize=13)
    ax.set_xlim(1, max_rank)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = out_dir / 'cmc_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CMC curves saved to {out_path}")


def plot_gender_confusion(
    preds:   np.ndarray,
    labels:  np.ndarray,
    out_dir: str = 'experiments/plots',
) -> None:
    """
    Plot gender classification confusion matrix.

    Args:
        preds:   [N] predicted gender labels
        labels:  [N] true gender labels
        out_dir: output directory
    """
    _check_mpl()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = ['Male', 'Female']
    cm = np.zeros((2, 2), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Gender Classification Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)

    plt.tight_layout()
    out_path = out_dir / 'gender_confusion.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gender confusion matrix saved to {out_path}")


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    label_type: str = 'identity',
    out_dir:    str = 'experiments/plots',
    n_subjects: int = 20,
) -> None:
    """
    Plot t-SNE of embeddings coloured by subject, gender, or age bin.

    Args:
        embeddings: [N, D] feature matrix
        labels:     [N]    subject IDs, gender labels (0/1), or age bin
                    indices (0..NUM_AGE_BINS-1)
        label_type: 'identity', 'gender', or 'age'. Determines both the
                    legend label naming (see _label_name below) and
                    whether identities are subsampled to n_subjects.
        out_dir:    output directory
        n_subjects: max subjects to plot (identity mode only)
    """
    _check_mpl()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn not installed. pip install scikit-learn")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subset for identity mode
    if label_type == 'identity':
        unique = np.unique(labels)[:n_subjects]
        mask   = np.isin(labels, unique)
        embeddings = embeddings[mask]
        labels     = labels[mask]

    print("Running t-SNE...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(unique_labels))

    def _label_name(lbl):
        """
        Human-readable legend name for a given label value, depending
        on label_type. Generalised beyond the original hardcoded
        ['Male','Female'][lbl] lookup (which only worked for exactly 2
        classes) to also support 'age' (NUM_AGE_BINS classes) and any
        future label_type without needing another hardcoded list here.
        """
        if label_type == 'identity':
            return f'Subject {lbl}'
        if label_type == 'gender':
            return ['Male', 'Female'][int(lbl)]
        if label_type == 'age':
            try:
                from datasets.base import AGE_BINS
                return AGE_BINS[int(lbl)][2]   # e.g. 'young_adult'
            except (ImportError, IndexError):
                return f'Age bin {lbl}'
        return str(lbl)   # fallback for any other label_type

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = _label_name(lbl)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=name, alpha=0.7, s=20)

    ax.set_title(f't-SNE -- coloured by {label_type}', fontsize=13)
    ax.axis('off')
    if label_type in ('gender', 'age'):
        ax.legend(fontsize=11)

    out_path = out_dir / f'tsne_{label_type}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {out_path}")


# -- Pipeline illustration: silhouette -> GEI -> motion-map panel -------------

def plot_pipeline_panel(
    sequence,
    gei,
    motion_pre,
    motion_post,
    out_dir: str = 'experiments/plots',
    frame_indices=None,
    title: str = 'BioKinematicNet Input Pipeline',
):
    """
    Side-by-side panel showing the data transformation pipeline:
        raw silhouette frames -> GEI -> motion map (pre-suppression)
                                      -> motion map (post-suppression)

    This is the architecture-illustration figure for the paper -- it
    visually demonstrates what each branch of the model actually sees,
    and specifically shows the effect of the static-suppression step
    (motion_suppressed = motion - beta*GEI) by displaying both the
    pre- and post-suppression motion maps side by side, so a reader can
    see directly that static body-shape signal is being removed from
    the motion branch's input.

    Args:
        sequence:     [T, 1, H, W] numpy array or tensor, the raw
                      silhouette sequence for ONE sample (already
                      detached/cpu/numpy by the caller -- this function
                      does no tensor-library-specific work itself, so it
                      stays usable regardless of how the caller obtained
                      the data)
        gei:          [1, H, W] numpy array, the GEI for this sample
        motion_pre:   [1, T-1, H, W] numpy array, motion map BEFORE
                      static suppression
        motion_post:  [1, T-1, H, W] numpy array, motion map AFTER
                      static suppression (motion_pre - beta*gei)
        out_dir:      output directory
        frame_indices: which raw silhouette frame indices to display in
                      the top row (default: 6 evenly spaced frames
                      across the sequence). Showing every frame would
                      make the panel too wide to be useful in a paper.
        title:        figure suptitle

    Layout:
        Row 1: N raw silhouette frames (evenly sampled across T)
        Row 2: GEI | motion map (pre-suppression, mean over T-1) |
               motion map (post-suppression, mean over T-1)

        The motion maps are shown as their TEMPORAL MEAN (averaged over
        the T-1 frame-difference maps) rather than a single time-slice,
        since a single motion frame from a near-static portion of the
        gait cycle would be visually uninformative -- the temporal mean
        shows where motion concentrates over the whole sequence, which
        is what's actually relevant to interpreting the suppression
        effect.
    """
    _check_mpl()

    sequence    = np.asarray(sequence)
    gei         = np.asarray(gei)
    motion_pre  = np.asarray(motion_pre)
    motion_post = np.asarray(motion_post)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = sequence.shape[0]
    if frame_indices is None:
        n_show = min(6, T)
        frame_indices = np.linspace(0, T - 1, n_show, dtype=int)

    n_frames_shown = len(frame_indices)

    # Temporal mean of the motion maps, for a single representative image
    # per motion-map type (see docstring rationale above).
    motion_pre_mean  = motion_pre[0].mean(axis=0)              # [H, W]
    # Clip post-suppression to [0, inf) -- negative values result from
    # subtracting the GEI (background regions where motion - GEI < 0).
    # Without clipping, inferno colormap maps negatives to orange/yellow
    # making the background look like high-activation, which is
    # visually misleading. After clipping, background is dark (zero)
    # and only genuine motion regions show as bright.
    motion_post_mean = np.clip(motion_post[0].mean(axis=0), 0, None)  # [H, W]
    gei_img          = gei[0]                                 # [H, W]

    n_cols = max(n_frames_shown, 3)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))

    # Row 1: raw silhouette frames
    for col in range(n_cols):
        ax = axes[0, col]
        if col < n_frames_shown:
            t = frame_indices[col]
            ax.imshow(sequence[t, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f't={t}', fontsize=9)
        ax.axis('off')

    # Row 2: GEI, motion-pre, motion-post (left-aligned, rest blank)
    row2_imgs  = [gei_img, motion_pre_mean, motion_post_mean]
    row2_names = ['GEI\n(morphology branch input)',
                  'Motion energy\n(pre-suppression)',
                  'Motion energy\n(post-suppression)\nstatic shape removed']
    for col in range(n_cols):
        ax = axes[1, col]
        if col < len(row2_imgs):
            cmap = 'gray' if col == 0 else 'inferno'
            im = ax.imshow(row2_imgs[col], cmap=cmap)
            ax.set_title(row2_names[col], fontsize=9)
            if col > 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout(h_pad=2.5)

    out_path = out_dir / 'pipeline_panel.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pipeline panel saved to {out_path}")
    return out_path
