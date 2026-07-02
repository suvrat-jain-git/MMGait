"""
gender_feature_analysis.py — Gender Encoding in Feature Space
(V2: dataset-agnostic)

Analyses how much gender information is encoded in Fm vs Fk by:

    1. Per-subject gender probe accuracy
       Which subjects are correctly gendered? Are errors consistent?

    2. Inter-class distance analysis
       Are male and female Fm features more separated than Fk features?
       Measures: mean intra-class distance, mean inter-class distance,
                 Fisher discriminant ratio

    3. Feature activation analysis
       Which dimensions of Fm are most gender-discriminative?
       (highest absolute difference between male and female mean features)

    4. Correlation between gender accuracy and retrieval accuracy
       Are subjects with poor gender prediction also harder to retrieve?

Usage:
    python analysis/gender_feature_analysis.py --dataset fvgb --checkpoint experiments/best.pth
    python analysis/gender_feature_analysis.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) version:
    - extract_all_features() reads dict batches.
    - Model construction via models/factory.py + dataset registry.
    - If the loaded checkpoint's model has no gender_head (e.g. a
      hypothetical future identity-only dataset), this script exits
      cleanly with a clear message rather than crashing on a missing
      'gender_logits' key -- this analysis is fundamentally about
      gender, so there's nothing useful to report without it, but the
      failure mode should be informative, not a stack trace.
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


def extract_all_features(model, loader, device):
    all_Fm = []; all_Fk = []
    all_gender_logits = []
    all_gender = []; all_ids = []

    has_gender = False

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            out    = model(frames, mode='train')
            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            if 'gender_logits' in out and batch.get('gender_label') is not None:
                has_gender = True
                all_gender_logits.append(out['gender_logits'].cpu())
                all_gender.extend(batch['gender_label'].tolist())
            id_label = batch['id_label']
            all_ids.extend(
                id_label.tolist() if hasattr(id_label, 'tolist')
                else list(id_label)
            )

    result = {
        'Fm':         torch.cat(all_Fm, dim=0),
        'Fk':         torch.cat(all_Fk, dim=0),
        'id_labels':  torch.tensor(all_ids),
    }
    if has_gender:
        result['gender_logits'] = torch.cat(all_gender_logits, dim=0)
        result['gender_labels'] = torch.tensor(all_gender)
    return result


def fisher_discriminant_ratio(features, labels):
    """
    FDR = (mu_1 - mu_0)^2 / (sigma_1^2 + sigma_0^2), averaged over dims.
    Higher FDR = classes are more separable.
    """
    mask0 = labels == 0
    mask1 = labels == 1
    mu0   = features[mask0].mean(dim=0)
    mu1   = features[mask1].mean(dim=0)
    var0  = features[mask0].var(dim=0).clamp(min=1e-8)
    var1  = features[mask1].var(dim=0).clamp(min=1e-8)
    fdr   = ((mu1 - mu0) ** 2) / (var0 + var1)
    return fdr.mean().item()


def run_gender_feature_analysis(checkpoint_path, cfg, device, dataset_entry,
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

    print("\nExtracting features from val set...")
    feats = extract_all_features(model, loaders['val'], device)

    if 'gender_logits' not in feats:
        print(
            "\nThis checkpoint's model has no gender_head -- nothing to "
            "analyse for gender feature encoding. Exiting cleanly."
        )
        return {}

    gender_labels = feats['gender_labels']
    id_labels     = feats['id_labels']
    Fm = F.normalize(feats['Fm'], dim=1)
    Fk = F.normalize(feats['Fk'], dim=1)

    preds = feats['gender_logits'].argmax(dim=1)

    print(f"\n{'='*60}")
    print("GENDER FEATURE ANALYSIS")
    print(f"{'='*60}")

    print("\n1. Fisher Discriminant Ratio (gender separability)")
    fdr_fm = fisher_discriminant_ratio(Fm, gender_labels)
    fdr_fk = fisher_discriminant_ratio(Fk, gender_labels)
    print(f"   Fm FDR: {fdr_fm:.4f}  (higher = more gender-discriminative)")
    print(f"   Fk FDR: {fdr_fk:.4f}")
    print(f"   Ratio Fm/Fk: {fdr_fm/fdr_fk:.2f}x")
    if fdr_fm > fdr_fk:
        print(f"   PASSED -- Fm is more gender-discriminative than Fk")
    else:
        print(f"   FAILED -- Fk is more gender-discriminative than Fm")

    print("\n2. Intra/Inter-class Gender Distances")
    for name, feat in [('Fm', Fm), ('Fk', Fk)]:
        dist = 1.0 - torch.mm(feat, feat.t())
        male_mask   = gender_labels == 0
        female_mask = gender_labels == 1

        male_intra   = dist[male_mask][:, male_mask].mean().item()
        female_intra = dist[female_mask][:, female_mask].mean().item()
        intra        = (male_intra + female_intra) / 2
        inter        = dist[male_mask][:, female_mask].mean().item()

        print(f"   {name}: intra={intra:.4f}  inter={inter:.4f}  "
              f"ratio={inter/intra:.2f}x")

    print("\n3. Top 10 Gender-Discriminative Dimensions of Fm")
    fm_unnorm = feats['Fm']
    male_mean   = fm_unnorm[gender_labels==0].mean(dim=0)
    female_mean = fm_unnorm[gender_labels==1].mean(dim=0)
    diff        = (female_mean - male_mean).abs()
    top10_dims  = diff.argsort(descending=True)[:10]
    print(f"   Dimension indices: {top10_dims.tolist()}")
    print(f"   Max diff: {diff[top10_dims[0]]:.4f}  "
          f"Min diff (top 10): {diff[top10_dims[9]]:.4f}")

    print("\n4. Per-Subject Gender Accuracy (val set)")
    subj_correct = defaultdict(list)
    gender_map_local = {}
    for idx in range(len(id_labels)):
        sid  = id_labels[idx].item()
        pred = preds[idx].item()
        true = gender_labels[idx].item()
        subj_correct[sid].append(pred == true)
        gender_map_local[sid] = true

    n_perfect = sum(1 for v in subj_correct.values() if all(v))
    n_zero    = sum(1 for v in subj_correct.values() if not any(v))
    print(f"   Subjects with 100% correct: {n_perfect}/{len(subj_correct)}")
    print(f"   Subjects with 0% correct:   {n_zero}/{len(subj_correct)}")

    print("\n   Per-subject breakdown:")
    for sid in sorted(subj_correct.keys()):
        results = subj_correct[sid]
        acc     = sum(results) / len(results)
        g       = 'M' if gender_map_local[sid] == 0 else 'F'
        bar     = '#' * int(acc * 10) + '.' * (10 - int(acc * 10))
        print(f"   Subject {sid:>5} ({g}): [{bar}] {acc*100:.0f}%")

    return {
        'fm_fdr':    fdr_fm,
        'fk_fdr':    fdr_fk,
        'fdr_ratio': fdr_fm / fdr_fk if fdr_fk > 0 else float('inf'),
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

    results = run_gender_feature_analysis(
        args.checkpoint, cfg, device, dataset_entry,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )

    if results:
        import json
        out = args.checkpoint.replace('.pth', '_gender_analysis.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
