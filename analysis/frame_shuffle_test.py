"""
frame_shuffle_test.py — Does temporal order matter? (V2: dataset-agnostic)

Hypothesis:
    If Fk (motion branch) encodes genuine gait dynamics, shuffling the
    frame order should degrade motion features significantly.
    If Fm (morphology branch) encodes body shape, shuffling should
    have minimal effect since GEI is computed as a mean.

Method:
    1. Extract embeddings with original frame order -> Fk_orig, Fm_orig
    2. Extract embeddings with randomly shuffled frames -> Fk_shuf, Fm_shuf
    3. Compare:
        - Cosine similarity between Fk_orig and Fk_shuf (should DROP)
        - Cosine similarity between Fm_orig and Fm_shuf (should stay HIGH)
        - Gait retrieval Rank-1 with shuffled frames (should DROP)

Expected result:
    Fm is shuffle-invariant (body shape doesn't depend on frame order).
    Fk is shuffle-sensitive (motion patterns depend on temporal order).
    This validates that the two branches capture different information.

Usage:
    python analysis/frame_shuffle_test.py --dataset fvgb --checkpoint experiments/best.pth
    python analysis/frame_shuffle_test.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) version:
    - extract_features_with_shuffle() reads dict batches instead of
      unpacking a (frames, subject_ids, _) tuple.
    - Model construction via models/factory.py + the dataset registry.
    - Uses the dataset's PRIMARY protocol (meta.protocols[0]) instead of
      a hardcoded 'WS' string -- FVG-B's primary is 'WS', OU-LP-Bag's is
      'bag' (see datasets/oulp_mvlp.py PROTOCOLS list).
"""

import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import cosine_distance_matrix, compute_rank_k, compute_map


def shuffle_frames(frames: torch.Tensor) -> torch.Tensor:
    """Randomly shuffle frames along the temporal dimension, per-sample."""
    B, T = frames.shape[:2]
    shuffled = frames.clone()
    for b in range(B):
        perm = torch.randperm(T)
        shuffled[b] = frames[b, perm]
    return shuffled


def extract_features_with_shuffle(model, loader, device, shuffle=False):
    """Extract Fm, Fk, embedding and subject IDs, optionally shuffling frames."""
    all_Fm  = []; all_Fk  = []; all_emb = []; all_ids = []

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            if shuffle:
                frames = shuffle_frames(frames)

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


def run_shuffle_test(checkpoint_path, cfg, device, dataset_entry,
                     n_shuffle_runs=5, use_graph=True, morph_backbone='custom'):
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

    print(f"\n=== Frame Shuffle Test (protocol: {primary_protocol}) ===")
    print(f"Averaging over {n_shuffle_runs} shuffle permutations\n")

    print("Extracting original features...")
    orig = extract_features_with_shuffle(
        model, protocol_data['probe'], device, shuffle=False
    )
    gal  = extract_features_with_shuffle(
        model, protocol_data['gallery'], device, shuffle=False
    )

    subj_emb = defaultdict(list)
    for emb, sid in zip(gal['embedding'], gal['ids']):
        subj_emb[sid].append(emb)
    gal_ids = sorted(subj_emb.keys())
    gal_emb = torch.stack([torch.stack(subj_emb[s]).mean(0) for s in gal_ids])

    dist_orig = cosine_distance_matrix(orig['embedding'], gal_emb)
    r1_orig   = compute_rank_k(dist_orig, orig['ids'], gal_ids, k=1)
    map_orig  = compute_map(dist_orig, orig['ids'], gal_ids)
    print(f"Original  -- Rank-1: {r1_orig*100:.2f}%  mAP: {map_orig*100:.2f}%")

    fm_sims_list  = []
    fk_sims_list  = []
    r1_shuf_list  = []
    map_shuf_list = []

    for run in range(n_shuffle_runs):
        torch.manual_seed(run)
        shuf = extract_features_with_shuffle(
            model, protocol_data['probe'], device, shuffle=True
        )

        fm_sim = F.cosine_similarity(orig['Fm'], shuf['Fm']).mean().item()
        fk_sim = F.cosine_similarity(orig['Fk'], shuf['Fk']).mean().item()
        fm_sims_list.append(fm_sim)
        fk_sims_list.append(fk_sim)

        dist_shuf = cosine_distance_matrix(shuf['embedding'], gal_emb)
        r1_shuf   = compute_rank_k(dist_shuf, shuf['ids'], gal_ids, k=1)
        map_shuf  = compute_map(dist_shuf, shuf['ids'], gal_ids)
        r1_shuf_list.append(r1_shuf)
        map_shuf_list.append(map_shuf)

    fm_sim_avg  = np.mean(fm_sims_list)
    fk_sim_avg  = np.mean(fk_sims_list)
    r1_shuf_avg = np.mean(r1_shuf_list)
    map_shuf_avg= np.mean(map_shuf_list)

    print(f"Shuffled  -- Rank-1: {r1_shuf_avg*100:.2f}%  "
          f"mAP: {map_shuf_avg*100:.2f}%")

    print(f"\n{'='*55}")
    print("SHUFFLE SENSITIVITY")
    print(f"{'='*55}")
    print(f"{'Metric':<35} {'Original':>10} {'Shuffled':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10}")
    print(f"{primary_protocol + ' Rank-1':<35} {r1_orig*100:>9.2f}% {r1_shuf_avg*100:>9.2f}%")
    print(f"{primary_protocol + ' mAP':<35} {map_orig*100:>9.2f}% {map_shuf_avg*100:>9.2f}%")
    print(f"{'Fm cosine sim (orig vs shuf)':<35} {'--':>10} {fm_sim_avg:>10.4f}")
    print(f"{'Fk cosine sim (orig vs shuf)':<35} {'--':>10} {fk_sim_avg:>10.4f}")

    print(f"\nInterpretation:")
    if fk_sim_avg < fm_sim_avg - 0.05:
        print(f"  PASSED -- Fk ({fk_sim_avg:.3f}) < Fm ({fm_sim_avg:.3f})")
        print(f"    Motion branch is more sensitive to frame order than morphology.")
        print(f"    Confirms Fk captures temporal gait dynamics.")
    else:
        print(f"  ~ Fk ({fk_sim_avg:.3f}) approx Fm ({fm_sim_avg:.3f})")
        print(f"    Both branches show similar shuffle sensitivity.")

    r1_drop = (r1_orig - r1_shuf_avg) * 100
    print(f"\n  Rank-1 drop from shuffling: {r1_drop:.2f}%")
    if r1_drop > 5:
        print(f"  PASSED -- Temporal order matters for identity retrieval.")
    else:
        print(f"  ~ Retrieval is relatively robust to frame shuffling.")

    return {
        'protocol':          primary_protocol,
        'r1_original':       r1_orig,
        'r1_shuffled':       r1_shuf_avg,
        'map_original':      map_orig,
        'map_shuffled':      map_shuf_avg,
        'fm_cosine_sim':     fm_sim_avg,
        'fk_cosine_sim':     fk_sim_avg,
        'rank1_drop':        r1_orig - r1_shuf_avg,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_runs', type=int, default=5)
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

    results = run_shuffle_test(
        args.checkpoint, cfg, device, dataset_entry, args.n_runs,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )

    if results is not None:
        import json
        out = args.checkpoint.replace('.pth', '_shuffle_test.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
