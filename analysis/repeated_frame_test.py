"""
repeated_frame_test.py — What happens with static (repeated) frames?
(V2: dataset-agnostic)

Hypothesis:
    If Fk genuinely encodes motion, feeding a static sequence (same frame
    repeated T times) should produce near-zero motion features because
    abs(X[t] - X[t-1]) = 0 for all t.

    Fm (morphology) should be unaffected since GEI = mean(X) and a
    repeated frame has the same mean as a single frame.

Method:
    1. Take a real sequence [T, 1, H, W]
    2. Replace it with [frame_0 repeated T times]
    3. Extract features and compare:
        - Fk magnitude: should collapse (motion = 0)
        - Fm magnitude: should stay similar
        - Retrieval performance: should drop (no motion info)

This is a strong sanity check. If motion features don't collapse with
static input, the motion encoder is not genuinely motion-sensitive.

Usage:
    python analysis/repeated_frame_test.py --dataset fvgb --checkpoint experiments/best.pth
    python analysis/repeated_frame_test.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) version: same as
analysis/frame_shuffle_test.py -- dict batches, factory/registry model
construction, primary-protocol lookup instead of hardcoded 'WS'.
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

from utils.metrics import cosine_distance_matrix, compute_rank_k, compute_map


def make_static_sequence(frames: torch.Tensor) -> torch.Tensor:
    """Replace each sequence with its first frame repeated T times."""
    first_frame = frames[:, 0:1, :, :, :]
    static      = first_frame.expand_as(frames)
    return static.clone()


def extract_features(model, loader, device, static=False):
    all_Fm  = []; all_Fk  = []; all_emb = []; all_ids = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            if static:
                frames = make_static_sequence(frames)

            out = model(frames, mode='train')
            all_Fm.extend(out['Fm'].cpu().unbind(0))
            all_Fk.extend(out['Fk'].cpu().unbind(0))
            all_emb.extend(out['embedding'].cpu().unbind(0))
            id_label = batch['id_label']
            all_ids.extend(
                id_label.tolist() if hasattr(id_label, 'tolist')
                else list(id_label)
            )

    return {
        'Fm':        torch.stack(all_Fm),
        'Fk':        torch.stack(all_Fk),
        'embedding': torch.stack(all_emb),
        'ids':       all_ids,
    }


def run_repeated_frame_test(checkpoint_path, cfg, device, dataset_entry,
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

    primary_protocol = meta.protocols[0]
    protocol_data = loaders['protocols'].get(primary_protocol)
    if protocol_data is None:
        print(f"{primary_protocol} protocol not available")
        return None

    print(f"\n=== Repeated Frame Test (Static Sequence, protocol: {primary_protocol}) ===\n")

    print("Extracting original features...")
    orig = extract_features(model, protocol_data['probe'], device, static=False)
    gal  = extract_features(model, protocol_data['gallery'], device, static=False)

    subj_emb = defaultdict(list)
    for emb, sid in zip(gal['embedding'], gal['ids']):
        subj_emb[sid].append(emb)
    gal_ids = sorted(subj_emb.keys())
    gal_emb = torch.stack([torch.stack(subj_emb[s]).mean(0) for s in gal_ids])

    dist_orig = cosine_distance_matrix(orig['embedding'], gal_emb)
    r1_orig   = compute_rank_k(dist_orig, orig['ids'], gal_ids, k=1)
    map_orig  = compute_map(dist_orig, orig['ids'], gal_ids)

    print("Extracting static (repeated frame) features...")
    stat = extract_features(model, protocol_data['probe'], device, static=True)

    dist_stat = cosine_distance_matrix(stat['embedding'], gal_emb)
    r1_stat   = compute_rank_k(dist_stat, stat['ids'], gal_ids, k=1)
    map_stat  = compute_map(dist_stat, stat['ids'], gal_ids)

    fm_norm_orig = orig['Fm'].norm(dim=1).mean().item()
    fk_norm_orig = orig['Fk'].norm(dim=1).mean().item()
    fm_norm_stat = stat['Fm'].norm(dim=1).mean().item()
    fk_norm_stat = stat['Fk'].norm(dim=1).mean().item()

    fm_sim = F.cosine_similarity(orig['Fm'], stat['Fm']).mean().item()
    fk_sim = F.cosine_similarity(orig['Fk'], stat['Fk']).mean().item()

    print(f"\n{'='*60}")
    print("REPEATED FRAME TEST RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<40} {'Original':>10} {'Static':>10}")
    print(f"{'-'*40} {'-'*10} {'-'*10}")
    print(f"{primary_protocol + ' Rank-1':<40} {r1_orig*100:>9.2f}% {r1_stat*100:>9.2f}%")
    print(f"{primary_protocol + ' mAP':<40} {map_orig*100:>9.2f}% {map_stat*100:>9.2f}%")
    print(f"{'Fm feature norm':<40} {fm_norm_orig:>10.4f} {fm_norm_stat:>10.4f}")
    print(f"{'Fk feature norm':<40} {fk_norm_orig:>10.4f} {fk_norm_stat:>10.4f}")
    print(f"{'Fm cosine sim (orig vs static)':<40} {'--':>10} {fm_sim:>10.4f}")
    print(f"{'Fk cosine sim (orig vs static)':<40} {'--':>10} {fk_sim:>10.4f}")

    print(f"\nInterpretation:")
    fk_drop = (fk_norm_orig - fk_norm_stat) / fk_norm_orig
    fm_drop = (fm_norm_orig - fm_norm_stat) / fm_norm_orig
    print(f"  Fk norm drop: {fk_drop*100:.1f}%  Fm norm drop: {fm_drop*100:.1f}%")

    if fk_sim < fm_sim - 0.05:
        print(f"  PASSED -- Fk changes more with static input ({fk_sim:.3f}) "
              f"than Fm ({fm_sim:.3f})")
        print(f"    Motion branch is sensitive to absence of motion.")
    else:
        print(f"  ~ Fm and Fk show similar response to static input.")

    r1_drop = (r1_orig - r1_stat) * 100
    print(f"  Rank-1 drop: {r1_drop:.2f}%")

    return {
        'protocol':         primary_protocol,
        'r1_original':      r1_orig,
        'r1_static':        r1_stat,
        'map_original':     map_orig,
        'map_static':       map_stat,
        'fm_norm_original': fm_norm_orig,
        'fm_norm_static':   fm_norm_stat,
        'fk_norm_original': fk_norm_orig,
        'fk_norm_static':   fk_norm_stat,
        'fm_cosine_sim':    fm_sim,
        'fk_cosine_sim':    fk_sim,
        'rank1_drop':       r1_orig - r1_stat,
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

    results = run_repeated_frame_test(
        args.checkpoint, cfg, device, dataset_entry,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )

    if results is not None:
        import json
        out = args.checkpoint.replace('.pth', '_repeated_frame_test.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
