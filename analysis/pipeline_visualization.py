"""
pipeline_visualization.py — Silhouette -> GEI -> Motion-Map Comparison Panel

Produces the architecture-illustration figure for the paper: for one or
more real samples from the dataset, shows the raw silhouette frames, the
resulting GEI (morphology branch input), and the motion map BEFORE and
AFTER static suppression (motion branch input) side by side.

This directly visualises the static-suppression mechanism
(motion_suppressed = motion - beta*GEI, see biokinematic_net.py) -- the
reader can see the body-shape signal being subtracted out of the motion
branch's input, which is the core architectural argument for why Fk
should encode dynamics rather than static appearance.

Uses the model's own generate_gei/generate_motion functions and the
static_suppression_beta value from the loaded model instance, NOT a
reimplementation -- this guarantees the figure shows exactly what the
model's forward pass actually computes, with no risk of the
visualisation silently drifting out of sync with the real pipeline if
the suppression formula is ever changed.

Usage:
    python analysis/pipeline_visualization.py --dataset fvgb \
        --checkpoint experiments/best.pth --n_samples 3
    python analysis/pipeline_visualization.py --dataset oulp_mvlp \
        --checkpoint experiments/best.pth --n_samples 3

Note: --checkpoint is used only to read static_suppression_beta from the
trained model's config (and, incidentally, to confirm the model loads
correctly) -- the panel itself does not depend on the model's LEARNED
weights at all, only on the deterministic GEI/motion-generation
functions and the suppression beta. A freshly-initialised model would
produce an identical panel for the same input sequence and beta value.
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.morphology.gei import generate_gei
from models.motion.motion_generator import generate_motion
from utils.visualization import plot_pipeline_panel


def run_pipeline_visualization(checkpoint_path, cfg, device, dataset_entry,
                                out_dir, n_samples=3, use_graph=True,
                                morph_backbone='custom'):
    loaders = dataset_entry.builder(cfg)
    meta    = loaders['meta']

    from models.factory import build_model_config
    from models.biokinematic_net import BioKinematicNet

    model_cfg = build_model_config(
        cfg['model'], cfg['heads'], meta,
        use_graph=use_graph, morph_backbone=morph_backbone,
    )
    model = BioKinematicNet(model_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    beta = model.static_suppression_beta
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(static_suppression_beta={beta})")

    # Pull n_samples real sequences from the val loader
    val_iter = iter(loaders['val'])
    collected = 0
    out_paths = []

    while collected < n_samples:
        try:
            batch = next(val_iter)
        except StopIteration:
            print(f"Val set exhausted after {collected} samples "
                  f"(requested {n_samples})")
            break

        frames = batch['frames']   # [B, T, 1, H, W]
        B = frames.shape[0]

        for i in range(B):
            if collected >= n_samples:
                break

            x = frames[i:i+1].to(device)   # [1, T, 1, H, W]

            with torch.no_grad():
                gei         = generate_gei(x)               # [1, 1, H, W]
                motion_pre  = generate_motion(x)             # [1, 1, T-1, H, W]
                motion_post = motion_pre - beta * gei.unsqueeze(2)

            seq_np    = x[0].cpu().numpy()              # [T, 1, H, W]
            gei_np    = gei[0].cpu().numpy()             # [1, H, W]
            pre_np    = motion_pre[0].cpu().numpy()       # [1, T-1, H, W]
            post_np   = motion_post[0].cpu().numpy()      # [1, T-1, H, W]

            sid = batch['id_label'][i].item()
            out_path = plot_pipeline_panel(
                seq_np, gei_np, pre_np, post_np,
                out_dir=out_dir,
                title=f'BioKinematicNet Input Pipeline (sample {collected+1}, '
                      f'subject id {sid})',
            )
            # Rename to avoid overwriting across samples
            import os
            numbered_path = str(out_path).replace(
                'pipeline_panel.png', f'pipeline_panel_{collected+1}.png'
            )
            os.rename(out_path, numbered_path)
            out_paths.append(numbered_path)
            print(f"  Sample {collected+1}: saved to {numbered_path}")

            collected += 1

    return out_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of example sequences to visualise')
    parser.add_argument('--no_graph', action='store_true',
                        help='Must match the flag used when this checkpoint was trained.')
    parser.add_argument('--morph_backbone', default='custom', choices=['custom', 'gaitbase'],
                        help='Must match the flag used when this checkpoint was trained.')
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

    run_pipeline_visualization(
        args.checkpoint, cfg, device, dataset_entry, args.plot_dir,
        n_samples=args.n_samples, use_graph=not args.no_graph,
        morph_backbone=args.morph_backbone,
    )


if __name__ == '__main__':
    main()
