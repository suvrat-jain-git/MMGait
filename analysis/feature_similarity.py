"""
feature_similarity.py — Cross-Branch Feature Similarity Analysis
(V2: dataset-agnostic)

Measures how similar Fm and Fk are to each other, and how this
similarity relates to the orthogonality loss we applied.

Analyses:
    1. Cosine similarity between Fm and Fk per sample
       (distribution -- are they orthogonal on average?)

    2. Similarity breakdown by gender (only if model has a gender head)
       Are male/female samples more or less orthogonal?

    3. Cross-branch linear predictability
       Can we predict Fm from Fk (or vice versa)?
       If yes -> branches are not truly independent.
       Metric: R^2 of linear regression Fk -> Fm

    4. Correlation between orthogonality and retrieval performance
       Do samples with more orthogonal Fm/Fk have better/worse Rank-1?
       (uses the dataset's PRIMARY protocol, not a hardcoded 'WS')

Usage:
    python analysis/feature_similarity.py --dataset fvgb --checkpoint experiments/best.pth
    python analysis/feature_similarity.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) version:
    - extract_features() reads dict batches, conditionally collects
      gender labels.
    - Model construction via models/factory.py + dataset registry.
    - Section 4 (orthogonality vs retrieval) uses meta.protocols[0]
      instead of a hardcoded 'WS' string.
    - Section 2 (similarity by gender) is skipped cleanly if the
      checkpoint's model has no gender_head.
"""

import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import cosine_distance_matrix, compute_rank_k


def extract_features(model, loader, device):
    all_Fm = []; all_Fk = []; all_emb = []
    all_Fm_prime = []; all_Fk_prime = []
    all_gender = []; all_ids = []

    has_gender = False

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            out    = model(frames, mode='train')
            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_Fm_prime.append(out['Fm_prime'].cpu())
            all_Fk_prime.append(out['Fk_prime'].cpu())
            all_emb.append(out['embedding'].cpu())
            if 'gender_logits' in out and batch.get('gender_label') is not None:
                has_gender = True
                all_gender.extend(batch['gender_label'].tolist())
            id_label = batch['id_label']
            all_ids.extend(
                id_label.tolist() if hasattr(id_label, 'tolist')
                else list(id_label)
            )

    result = {
        'Fm':         torch.cat(all_Fm,       dim=0),
        'Fk':         torch.cat(all_Fk,       dim=0),
        'Fm_prime':   torch.cat(all_Fm_prime, dim=0),
        'Fk_prime':   torch.cat(all_Fk_prime, dim=0),
        'embedding':  torch.cat(all_emb,      dim=0),
        'id_labels':  torch.tensor(all_ids),
    }
    if has_gender:
        result['gender_labels'] = torch.tensor(all_gender)
    return result


def linear_r_squared(X, Y):
    """
    R^2 of linear regression X -> Y (closed-form solution).
    R^2 close to 0 -> X cannot predict Y (branches are independent).
    R^2 close to 1 -> X can predict Y (branches share information).
    """
    ones = torch.ones(X.shape[0], 1)
    X_b  = torch.cat([X, ones], dim=1)

    try:
        W     = torch.linalg.lstsq(X_b, Y).solution
        Y_hat = X_b @ W
    except Exception:
        return float('nan')

    ss_res = ((Y - Y_hat) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2     = (1 - ss_res / ss_tot).mean().item()
    return r2


def run_feature_similarity_analysis(checkpoint_path, cfg, device, dataset_entry,
                                    use_graph=True, morph_backbone='custom'):
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

    print("\nExtracting val features...")
    feats = extract_features(model, loaders['val'], device)

    Fm_norm = F.normalize(feats['Fm'], dim=1)
    Fk_norm = F.normalize(feats['Fk'], dim=1)

    print(f"\n{'='*60}")
    print("CROSS-BRANCH FEATURE SIMILARITY")
    print(f"{'='*60}")

    print("\n1. Fm vs Fk Cosine Similarity Distribution")
    sim_per_sample = (Fm_norm * Fk_norm).sum(dim=1)
    print(f"   Mean:   {sim_per_sample.mean():.4f}  "
          f"(0=orthogonal, 1=identical, -1=opposite)")
    print(f"   Std:    {sim_per_sample.std():.4f}")
    print(f"   Min:    {sim_per_sample.min():.4f}")
    print(f"   Max:    {sim_per_sample.max():.4f}")
    print(f"   % near-orthogonal (|sim|<0.1): "
          f"{(sim_per_sample.abs() < 0.1).float().mean()*100:.1f}%")

    Fm_prime_norm = F.normalize(feats['Fm_prime'], dim=1)
    Fk_prime_norm = F.normalize(feats['Fk_prime'], dim=1)
    sim_post = (Fm_prime_norm * Fk_prime_norm).sum(dim=1)
    print(f"   Pre-graph  Fm  vs Fk:  mean={sim_per_sample.mean():.4f}  "
          f"(BT loss applied here)")
    print(f"   Post-graph Fm' vs Fk': mean={sim_post.mean():.4f}  "
          f"(after graph mixing)")
    print(f"   Graph mixing increases similarity by: "
          f"{(sim_post.mean()-sim_per_sample.mean()):.4f}")

    if 'gender_labels' in feats:
        print("\n2. Fm vs Fk Similarity by Gender")
        gender_labels = feats['gender_labels']
        for cls, name in [(0, 'Male'), (1, 'Female')]:
            mask = gender_labels == cls
            sim  = sim_per_sample[mask].mean().item()
            print(f"   {name}: mean similarity = {sim:.4f}")
    else:
        print("\n2. (No gender labels for this dataset/checkpoint -- skipped)")

    print("\n3. Cross-Branch Linear Predictability (R^2)")
    print("   Testing: can Fk linearly predict Fm (and vice versa)?")
    print("   (R^2 near 0 = branches are independent)")

    r2_fk_to_fm = linear_r_squared(feats['Fk'], feats['Fm'])
    r2_fm_to_fk = linear_r_squared(feats['Fm'], feats['Fk'])
    print(f"   R^2 (Fk -> Fm): {r2_fk_to_fm:.4f}")
    print(f"   R^2 (Fm -> Fk): {r2_fm_to_fk:.4f}")

    if max(r2_fk_to_fm, r2_fm_to_fk) < 0.3:
        print("   PASSED -- low R^2, branches are largely independent")
    elif max(r2_fk_to_fm, r2_fm_to_fk) < 0.6:
        print("   ~ Moderate R^2 -- partial information sharing between branches")
    else:
        print("   FAILED -- high R^2, branches share significant information")

    print("\n4. Per-Sequence Orthogonality vs Retrieval Performance")
    primary_protocol = meta.protocols[0]
    protocol_data = loaders['protocols'].get(primary_protocol)
    if protocol_data is not None:
        print(f"   Computing per-sequence Rank-1 on {primary_protocol} probe...")
        probe_feats = extract_features(model, protocol_data['probe'], device)
        gal_feats   = extract_features(model, protocol_data['gallery'], device)

        subj_emb = defaultdict(list)
        for emb, sid in zip(gal_feats['embedding'], gal_feats['id_labels'].tolist()):
            subj_emb[sid].append(emb)
        gal_ids = sorted(subj_emb.keys())
        gal_emb = torch.stack([
            torch.stack(subj_emb[s]).mean(0) for s in gal_ids
        ])

        dist = cosine_distance_matrix(probe_feats['embedding'], gal_emb)
        gal_t = torch.tensor(gal_ids)
        probe_t = probe_feats['id_labels']

        correct = []
        for i in range(len(probe_t)):
            top1 = gal_t[dist[i].argmin()]
            correct.append((top1 == probe_t[i]).item())

        Fm_p = F.normalize(probe_feats['Fm'], dim=1)
        Fk_p = F.normalize(probe_feats['Fk'], dim=1)
        sim_p = (Fm_p * Fk_p).sum(dim=1)

        correct_t = torch.tensor(correct, dtype=torch.float)
        if correct_t.sum() > 0 and correct_t.sum() < len(correct_t):
            sim_corr = np.corrcoef(sim_p.numpy(), correct_t.numpy())[0, 1]
            print(f"   Correlation (Fm.Fk similarity vs Rank-1 correct): "
                  f"{sim_corr:.4f}")
        else:
            print("   (All-correct or all-incorrect retrievals -- "
                  "correlation undefined, skipped)")
        if (correct_t==1).any():
            print(f"   Mean sim (correct retrievals):   "
                  f"{sim_p[correct_t==1].mean():.4f}")
        if (correct_t==0).any():
            print(f"   Mean sim (incorrect retrievals): "
                  f"{sim_p[correct_t==0].mean():.4f}")
    else:
        print(f"   ({primary_protocol} protocol not available -- skipped)")

    return {
        'protocol':              meta.protocols[0],
        'fm_fk_cosine_sim_mean': sim_per_sample.mean().item(),
        'fm_fk_cosine_sim_std':  sim_per_sample.std().item(),
        'r2_fk_to_fm':           r2_fk_to_fm,
        'r2_fm_to_fk':           r2_fm_to_fk,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
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

    results = run_feature_similarity_analysis(
        args.checkpoint, cfg, device, dataset_entry,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )

    import json
    out = args.checkpoint.replace('.pth', '_feature_similarity.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
