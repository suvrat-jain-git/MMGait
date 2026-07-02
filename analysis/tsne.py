"""
tsne.py — t-SNE Visualisation of Feature Spaces (V2: dataset-agnostic)

Produces t-SNE plots for:
    1. Fm coloured by identity (do morphology features cluster by person?)
    2. Fm coloured by gender  (do morphology features cluster by gender?)
    3. Fk coloured by identity (do motion features cluster by person?)
    4. Fk coloured by gender  (do motion features cluster by gender?)
    5. Final embedding coloured by identity
    6. CONDITIONALLY (only if the loaded model has an age_head and the
       val set has age-labeled samples): Fm and Fk coloured by age bin,
       same rationale as the gender plots above but for age.

Expected results:
    Fm coloured by gender:    clear separation (morphology encodes gender)
    Fk coloured by gender:    mixed (motion should NOT encode gender)
    Fk coloured by identity:  clustering (motion discriminates identity)

Usage:
    python analysis/tsne.py --dataset fvgb --checkpoint experiments/best.pth
    python analysis/tsne.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) version:
    - extract_features() reads dict batches, conditionally collects
      gender/age labels depending on what the model+batch provide.
    - Model construction via models/factory.py + dataset registry.
    - Age t-SNE plots only attempted if there are age-labeled samples
      AND the val set has more than a handful of them (t-SNE on <10
      points is not meaningful) -- same sample-size guard philosophy as
      evaluators/gender_eval.py's age block.
"""

import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import plot_embedding_tsne


def extract_features(model, loader, device):
    all_Fm = []; all_Fk = []; all_emb = []
    all_gender = []; all_ids = []
    all_age_bin = []; all_age_mask = []

    has_gender = False
    has_age    = False

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            out    = model(frames, mode='train')
            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_emb.append(out['embedding'].cpu())
            id_label = batch['id_label']
            all_ids.extend(
                id_label.tolist() if hasattr(id_label, 'tolist')
                else list(id_label)
            )

            if 'gender_logits' in out and batch.get('gender_label') is not None:
                has_gender = True
                all_gender.extend(batch['gender_label'].tolist())

            if 'age_bin_logits' in out:
                has_age = True
                B = out['age_bin_logits'].shape[0]
                if batch.get('age_bin') is not None:
                    all_age_bin.extend(batch['age_bin'].tolist())
                    all_age_mask.extend(batch['age_mask'].tolist())
                else:
                    all_age_bin.extend([-1] * B)
                    all_age_mask.extend([False] * B)

    result = {
        'Fm':         torch.cat(all_Fm,  dim=0).numpy(),
        'Fk':         torch.cat(all_Fk,  dim=0).numpy(),
        'embedding':  torch.cat(all_emb, dim=0).numpy(),
        'id_labels':  np.array(all_ids),
    }
    if has_gender:
        result['gender_labels'] = np.array(all_gender)
    if has_age:
        result['age_bin']  = np.array(all_age_bin)
        result['age_mask'] = np.array(all_age_mask)
    return result


def run_tsne(checkpoint_path, cfg, device, plot_dir, dataset_entry,
            n_subjects=20, use_graph=True, morph_backbone='custom'):
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    print("\nExtracting val features for t-SNE...")
    feats = extract_features(model, loaders['val'], device)

    print(f"Val samples: {len(feats['id_labels'])}")
    print(f"Unique identities: {len(np.unique(feats['id_labels']))}")

    plots = [
        ('embedding', feats['id_labels'], 'identity', 'Embedding -- by Identity'),
        ('Fm',        feats['id_labels'], 'identity', 'Fm -- by Identity'),
        ('Fk',        feats['id_labels'], 'identity', 'Fk -- by Identity'),
    ]
    if 'gender_labels' in feats:
        plots.append(('Fm', feats['gender_labels'], 'gender', 'Fm -- by Gender'))
        plots.append(('Fk', feats['gender_labels'], 'gender', 'Fk -- by Gender'))
    else:
        print("\n(No gender labels for this dataset/checkpoint -- "
              "skipping gender t-SNE plots.)")

    if 'age_bin' in feats:
        n_age_labeled = feats['age_mask'].sum()
        if n_age_labeled >= 10:
            mask = feats['age_mask']
            plots.append((
                'Fm', feats['age_bin'][mask], 'age',
                'Fm -- by Age Bin', mask,
            ))
            plots.append((
                'Fk', feats['age_bin'][mask], 'age',
                'Fk -- by Age Bin', mask,
            ))
        else:
            print(f"\n(Only {n_age_labeled} age-labeled val samples -- "
                  f"too few for a meaningful t-SNE plot, skipping age "
                  f"visualisations.)")

    for plot_spec in plots:
        if len(plot_spec) == 4:
            feat_key, labels, label_type, title = plot_spec
            features = feats[feat_key]
        else:
            feat_key, labels, label_type, title, mask = plot_spec
            features = feats[feat_key][mask]

        print(f"\nRunning t-SNE: {title}...")
        fname = f"tsne_{feat_key}_{label_type}"
        try:
            plot_embedding_tsne(
                features, labels,
                label_type=label_type,
                out_dir=plot_dir,
                n_subjects=n_subjects,
            )
            import os
            src = f"{plot_dir}/tsne_{label_type}.png"
            dst = f"{plot_dir}/{fname}.png"
            if os.path.exists(src) and src != dst:
                os.rename(src, dst)
            print(f"  Saved: {dst}")
        except Exception as e:
            print(f"  Skipped: {e}")

    print("\nt-SNE complete.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    parser.add_argument('--n_subjects', type=int, default=20)
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

    run_tsne(
        args.checkpoint, cfg, device, args.plot_dir, dataset_entry,
        args.n_subjects, use_graph=not args.no_graph,
        morph_backbone=args.morph_backbone,
    )


if __name__ == '__main__':
    main()
