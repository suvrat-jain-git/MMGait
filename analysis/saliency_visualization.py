"""
saliency_visualization.py — Grad-CAM-style Saliency Overlays for Fm/Fk

Produces gradient-based attention maps showing WHICH spatial regions of
the silhouette each branch's feature extractor responds most strongly
to, overlaid on the original GEI (for Fm) and on the mean motion map
(for Fk). This is direct visual evidence for the disentanglement claim:
if Fm attends to torso/shoulder regions (body shape) while Fk attends
to limb/joint regions (articulation), that supports the morphology-vs-
motion functional split independently of the numeric disentanglement
metrics (linear probe gap, R^2, etc.) computed elsewhere in this
codebase.

METHOD (Grad-CAM, adapted for a non-classification feature extractor):
    Standard Grad-CAM (Selvaraju et al., 2017) computes:
        L^c = ReLU( sum_k [ alpha_k^c * A^k ] )
    where A^k is the k-th channel of the last spatial feature map,
    alpha_k^c = global-average-pooled gradient of a target class score
    w.r.t. A^k.

    Fm and Fk are not classification logits -- they are feature vectors
    consumed by multiple downstream heads (gender, identity, age). There
    is no single "class score" to differentiate. Following the standard
    approach for visualising what a feature-extraction layer attends to
    (used e.g. for embedding/metric-learning networks), the target
    scalar used here is ||F||^2 (squared L2 norm) of the feature vector
    itself -- this asks "which spatial regions, if perturbed, would
    most change the MAGNITUDE of this feature", which is a reasonable
    proxy for "where is this branch looking" in the absence of a
    classification target.

KNOWN LIMITATION -- custom backbone only:
    This module hooks model.morph_encoder.encoder and
    model.motion_encoder.encoder directly by attribute name, which
    matches the custom MorphologyEncoder/MotionEncoder implementations
    (both expose a `.encoder` Sequential before the final pooling layer
    -- see models/morphology/morphology_encoder.py and
    models/motion/motion_encoder.py). The GaitBase backbone
    (models/backbones/gaitbase_backbone.py) has a DIFFERENT internal
    module structure (stem/layer1-4, no single `.encoder` attribute) and
    is NOT supported by this hook -- running this script against a
    --morph_backbone gaitbase checkpoint will raise a clear
    AttributeError rather than silently producing an empty or
    meaningless saliency map for the morphology branch. The motion
    branch (always the custom MotionEncoder regardless of
    --morph_backbone, per the GaitBase ablation design -- see
    models/biokinematic_net.py) is unaffected by this limitation.

Usage:
    python analysis/saliency_visualization.py --dataset fvgb \
        --checkpoint experiments/best.pth --n_samples 3
"""

import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class GradCAMHook:
    """
    Registers a forward hook (to capture the activation) and a full
    backward hook (to capture the gradient) on a given submodule, for
    one Grad-CAM computation. Designed to be used as a context manager
    so hooks are always removed afterward, even if an exception occurs
    mid-computation -- leaving stale hooks registered on a model that's
    reused elsewhere (e.g. for further training) would silently corrupt
    later forward/backward passes, which is exactly the kind of bug
    that's painful to track down after the fact.
    """

    def __init__(self, module):
        self.module = module
        self.activation = None
        self.gradient   = None
        self._fwd_handle = None
        self._bwd_handle = None

    def __enter__(self):
        def forward_hook(module, input, output):
            self.activation = output

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        self._fwd_handle = self.module.register_forward_hook(forward_hook)
        self._bwd_handle = self.module.register_full_backward_hook(backward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fwd_handle.remove()
        self._bwd_handle.remove()
        # Do not suppress exceptions -- re-raise whatever happened inside
        return False


def compute_gradcam_2d(activation, gradient):
    """
    Standard Grad-CAM combination for a 2D (spatial) activation/gradient
    pair, as produced by the morphology branch (2D conv backbone).

    Args:
        activation: [1, C, H, W] -- captured forward activation
        gradient:   [1, C, H, W] -- captured backward gradient

    Returns:
        cam: [H, W] numpy array, normalised to [0, 1]
    """
    weights = gradient.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
    cam = (weights * activation).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    cam = cam[0, 0].detach().cpu().numpy()

    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max
    return cam


def compute_gradcam_3d(activation, gradient):
    """
    Grad-CAM combination for a 3D (spatiotemporal) activation/gradient
    pair, as produced by the motion branch (3D conv backbone). Averaged
    over the temporal dimension after weighting, producing a single 2D
    spatial saliency map -- consistent with how the motion map itself is
    visualised as a temporal mean elsewhere in this codebase (see
    utils/visualization.py plot_pipeline_panel).

    Args:
        activation: [1, C, D, H, W]
        gradient:   [1, C, D, H, W]

    Returns:
        cam: [H, W] numpy array, normalised to [0, 1]
    """
    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)   # [1, C, 1, 1, 1]
    cam = (weights * activation).sum(dim=1, keepdim=True)   # [1, 1, D, H, W]
    cam = F.relu(cam)
    cam = cam.mean(dim=2)   # average over temporal dim -> [1, 1, H, W]
    cam = cam[0, 0].detach().cpu().numpy()

    cam_max = cam.max()
    if cam_max > 1e-8:
        cam = cam / cam_max
    return cam


def compute_branch_saliency(model, x):
    """
    Compute Grad-CAM saliency maps for both Fm (morphology) and Fk
    (motion) branches for a single input sequence.

    Args:
        model: BioKinematicNet instance (must use morph_backbone='custom'
              -- see module docstring's KNOWN LIMITATION)
        x:     [1, T, 1, H, W] -- single sequence

    Returns:
        cam_fm: [H_fm, W_fm] numpy array (spatial resolution of the last
                conv block of the morphology encoder, typically smaller
                than the input H,W due to stride-2 downsampling)
        cam_fk: [H_fk, W_fk] numpy array, same idea for motion
    """
    if not hasattr(model.morph_encoder, 'encoder'):
        raise AttributeError(
            "model.morph_encoder has no '.encoder' attribute -- this "
            "model was likely built with --morph_backbone gaitbase, "
            "which has a different internal structure than the custom "
            "MorphologyEncoder this Grad-CAM hook expects. See this "
            "module's docstring 'KNOWN LIMITATION' section. Re-run "
            "with a --morph_backbone custom checkpoint."
        )

    model.zero_grad()

    # requires_grad_(True) on the input silences a PyTorch UserWarning
    # ("Full backward hook is firing when gradients are computed with
    # respect to module outputs since no inputs require gradients") that
    # is otherwise harmless here -- this hook only reads grad_output
    # (the gradient w.r.t. the hooked module's OUTPUT), never grad_input,
    # and grad_output is numerically identical regardless of whether the
    # original input tensor requires grad. Still set explicitly to avoid
    # an alarming-looking warning in the script's console output for
    # anyone running this for the first time.
    x = x.clone().requires_grad_(True)

    with GradCAMHook(model.morph_encoder.encoder) as fm_hook, \
         GradCAMHook(model.motion_encoder.encoder) as fk_hook:

        out = model(x, mode='train')
        Fm = out['Fm']
        Fk = out['Fk']

        # Target scalar: squared L2 norm of each feature vector (see
        # module docstring's METHOD section for why this is the target
        # in the absence of a classification logit to differentiate).
        target = (Fm ** 2).sum() + (Fk ** 2).sum()
        target.backward()

        cam_fm = compute_gradcam_2d(fm_hook.activation, fm_hook.gradient)
        cam_fk = compute_gradcam_3d(fk_hook.activation, fk_hook.gradient)

    model.zero_grad()
    return cam_fm, cam_fk


def plot_saliency_overlay(gei, cam_fm, motion_mean, cam_fk, out_path, title):
    """
    Side-by-side panel: GEI with Fm's saliency overlaid, motion-map mean
    with Fk's saliency overlaid.

    Args:
        gei:         [H, W] numpy array
        cam_fm:      [H_fm, W_fm] numpy array (resized to match gei's
                     resolution before overlay, since the saliency map's
                     spatial size is the LAST CONV BLOCK's resolution,
                     not the original input resolution, due to the
                     encoder's internal stride-2 downsampling)
        motion_mean: [H, W] numpy array
        cam_fk:      [H_fk, W_fk] numpy array, same resizing note
        out_path:    where to save the PNG
        title:       figure title
    """
    if not HAS_MPL:
        raise ImportError("matplotlib not installed. pip install matplotlib")

    from scipy.ndimage import zoom

    def _resize_cam(cam, target_shape):
        if cam.shape == target_shape:
            return cam
        zoom_factors = (target_shape[0] / cam.shape[0],
                        target_shape[1] / cam.shape[1])
        return zoom(cam, zoom_factors, order=1)

    cam_fm_resized = _resize_cam(cam_fm, gei.shape)
    cam_fk_resized = _resize_cam(cam_fk, motion_mean.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    axes[0].imshow(gei, cmap='gray')
    axes[0].imshow(cam_fm_resized, cmap='jet', alpha=0.45)
    axes[0].set_title('Fm saliency (morphology)\noverlaid on GEI', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(motion_mean, cmap='gray')
    axes[1].imshow(cam_fk_resized, cmap='jet', alpha=0.45)
    axes[1].set_title('Fk saliency (motion)\noverlaid on mean motion map', fontsize=10)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saliency overlay saved to {out_path}")


def run_saliency_visualization(checkpoint_path, cfg, device, dataset_entry,
                               out_dir, n_samples=3, use_graph=True,
                               morph_backbone='custom'):
    if morph_backbone != 'custom':
        raise ValueError(
            f"saliency_visualization.py only supports morph_backbone="
            f"'custom' (got '{morph_backbone}') -- see this module's "
            f"docstring KNOWN LIMITATION section for why GaitBase isn't "
            f"hooked."
        )

    loaders = dataset_entry.builder(cfg)
    meta    = loaders['meta']

    from models.factory import build_model_config
    from models.biokinematic_net import BioKinematicNet
    from models.morphology.gei import generate_gei
    from models.motion.motion_generator import generate_motion

    model_cfg = build_model_config(
        cfg['model'], cfg['heads'], meta,
        use_graph=use_graph, morph_backbone=morph_backbone,
    )
    model = BioKinematicNet(model_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()   # eval mode for BatchNorm/Dropout determinism, but
                    # gradients still flow normally for Grad-CAM since
                    # we never call torch.no_grad()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    val_iter  = iter(loaders['val'])
    collected = 0
    out_paths = []

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    while collected < n_samples:
        try:
            batch = next(val_iter)
        except StopIteration:
            print(f"Val set exhausted after {collected} samples "
                  f"(requested {n_samples})")
            break

        frames = batch['frames']
        B = frames.shape[0]

        for i in range(B):
            if collected >= n_samples:
                break

            x = frames[i:i+1].to(device)

            cam_fm, cam_fk = compute_branch_saliency(model, x)

            with torch.no_grad():
                gei         = generate_gei(x)[0, 0].cpu().numpy()
                motion      = generate_motion(x)[0, 0].mean(dim=0).cpu().numpy()

            sid = batch['id_label'][i].item()
            out_path = Path(out_dir) / f'saliency_overlay_{collected+1}.png'
            plot_saliency_overlay(
                gei, cam_fm, motion, cam_fk, out_path,
                title=f'Branch Saliency (sample {collected+1}, subject id {sid})',
            )
            out_paths.append(str(out_path))
            collected += 1

    return out_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--no_graph', action='store_true',
                        help='Must match the flag used when this checkpoint was trained.')
    parser.add_argument('--morph_backbone', default='custom', choices=['custom'],
                        help="Only 'custom' is supported -- see module docstring.")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from datasets.registry import get_dataset_entry
    dataset_entry = get_dataset_entry(args.dataset)

    cfg = {}
    for path in ['configs/model.yaml', 'configs/heads.yaml', 'configs/train.yaml']:
        with open(path) as f:
            loaded = yaml.safe_load(f)
            if path == 'configs/heads.yaml':
                cfg['heads'] = loaded
            else:
                cfg.update(loaded)
    for path in dataset_entry.config_files:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    run_saliency_visualization(
        args.checkpoint, cfg, device, dataset_entry, args.plot_dir,
        n_samples=args.n_samples, use_graph=not args.no_graph,
        morph_backbone=args.morph_backbone,
    )


if __name__ == '__main__':
    main()
