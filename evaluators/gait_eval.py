"""
gait_eval.py — Gait Recognition Evaluator (V2: dataset-agnostic)

Computes for all applicable protocols of the active --dataset:
    Rank-1, Rank-5, mAP    -- retrieval metrics
    EER                     -- verification metric
    CMC curve               -- full rank curve saved as PNG

Usage:
    python evaluators/gait_eval.py --dataset fvgb --checkpoint experiments/best.pth
    python evaluators/gait_eval.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) evaluator:
    - extract_embeddings() now reads batch['frames']/batch['id_label']
      from the dict-batch format (datasets/base.py gait_collate_fn)
      instead of unpacking a (frames, subject_ids, _) tuple.
    - run_evaluation() builds the model via models/factory.py from the
      active dataset's DatasetMeta, instead of hardcoding
      build_fvgb_dataloaders + an unconditional gender-head injection.
      A model checkpoint trained without gender (or with --no_graph,
      or a different morph_backbone) loads correctly because the model
      is constructed identically to how it was during training -- via
      the same factory function, given the same flags.
    - Protocol iteration now comes from loaders['protocols'], whatever
      keys that dict has for the active dataset (FVG-B: WS/BGHT/CL/MP/
      ALL; OU-LP-Bag: its own protocol names) -- no hardcoded protocol
      list anywhere in this file.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    cosine_distance_matrix,
    compute_rank_k,
    compute_map,
    compute_cmc_curve,
    compute_eer,
)
from utils.visualization import plot_cmc_curves


# -- Embedding extraction -----------------------------------------------------

def extract_embeddings(model, loader, device):
    """
    Args:
        model:  BioKinematicNet, any head configuration
        loader: DataLoader yielding dict batches (gait_collate_fn format)
        device: torch.device

    Returns:
        embeddings: [N, 512] tensor
        ids:        list of N subject IDs (raw, not remapped --
                    gallery/probe datasets use raw subject_id as
                    id_label, per the Sample contract documented in
                    datasets/base.py and datasets/fvg_b.py)
    """
    all_emb = []
    all_ids = []
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            emb    = model(frames, mode='inference')
            all_emb.append(emb.cpu())
            id_label = batch['id_label']
            all_ids.extend(
                id_label.tolist() if hasattr(id_label, 'tolist')
                else list(id_label)
            )
    return torch.cat(all_emb, dim=0), all_ids


def aggregate_gallery_by_subject(gallery_emb, gallery_ids):
    subj_to_embs = defaultdict(list)
    for emb, sid in zip(gallery_emb, gallery_ids):
        subj_to_embs[sid].append(emb)
    agg_ids = sorted(subj_to_embs.keys())
    agg_emb = torch.stack([
        torch.stack(subj_to_embs[sid]).mean(dim=0)
        for sid in agg_ids
    ])
    return agg_emb, agg_ids


# -- Protocol evaluation --------------------------------------------------------

def evaluate_protocol(model, protocol_data, device, protocol_name,
                      cmc_max_rank=20):
    print(f"  Extracting gallery embeddings...", flush=True)
    gallery_emb, gallery_ids = extract_embeddings(
        model, protocol_data['gallery'], device
    )

    print(f"  Extracting probe embeddings...", flush=True)
    probe_emb, probe_ids = extract_embeddings(
        model, protocol_data['probe'], device
    )

    print(f"  Gallery sequences: {len(gallery_ids)}  "
          f"Probe sequences: {len(probe_ids)}", flush=True)

    gallery_emb_agg, gallery_ids_agg = aggregate_gallery_by_subject(
        gallery_emb, gallery_ids
    )
    print(f"  Gallery subjects: {len(gallery_ids_agg)}", flush=True)

    dist = cosine_distance_matrix(probe_emb, gallery_emb_agg)

    rank1 = compute_rank_k(dist, probe_ids, gallery_ids_agg, k=1)
    rank5 = compute_rank_k(dist, probe_ids, gallery_ids_agg, k=5)
    mAP   = compute_map(dist, probe_ids, gallery_ids_agg)

    cmc = compute_cmc_curve(dist, probe_ids, gallery_ids_agg,
                            max_rank=cmc_max_rank)

    eer, eer_threshold = compute_eer(dist, probe_ids, gallery_ids_agg)

    return {
        'rank1':         rank1,
        'rank5':         rank5,
        'mAP':           mAP,
        'eer':           eer,
        'eer_threshold': eer_threshold,
        'cmc':           cmc,
        'n_probe':       len(probe_ids),
        'n_gallery':     len(gallery_ids_agg),
    }


# -- Cross-view protocol evaluation (OU-MVLP --cross_view mode) -----------------

def evaluate_cross_view_protocol(model, view_loaders, device,
                                 views=None, cmc_max_rank=20):
    """
    Literature-standard OU-MVLP cross-view evaluation: every probe view
    is evaluated against every OTHER gallery view (excluding the
    identical view), and Rank-1/EER are reported both per (probe_view,
    gallery_view) pair AND averaged per probe view across all 13
    non-identical gallery views -- this per-probe-view average is the
    number that's directly comparable to published OU-MVLP baselines.

    Efficiency note: each view's gallery and probe embeddings are
    extracted EXACTLY ONCE (14 gallery extractions + 14 probe
    extractions = 28 total), then reused across all 14*13=182 pairwise
    comparisons via simple distance-matrix computation on the cached
    embeddings. Re-extracting embeddings per pair would mean running
    the model 182 times instead of 28 -- a ~6.5x wasted-compute factor
    avoided by caching.

    Args:
        model:        BioKinematicNet
        view_loaders: dict {view: {'gallery': DataLoader, 'probe':
                      DataLoader, ...}} as returned by
                      datasets/oulp_mvlp.py's build_view_loaders()
        device:       torch.device
        views:        list of view strings to include (default: all
                      keys in view_loaders with non-None loaders)
        cmc_max_rank: max rank for CMC curves (per-pair CMC is computed
                      but not averaged -- only Rank-1/EER are averaged
                      across pairs, matching the literature convention;
                      averaging full CMC curves across views is not a
                      standard reported metric)

    Returns:
        dict with:
            'per_pair':        {(probe_view, gallery_view): metrics dict}
                               for all 182 non-identical pairs
            'per_probe_view':  {probe_view: {'rank1_mean', 'rank1_std',
                               'eer_mean', 'eer_std'}} -- averaged over
                               the 13 gallery views for that probe view
            'overall':         {'rank1_mean', 'rank1_std', 'eer_mean',
                               'eer_std'} -- averaged over ALL 182 pairs,
                               the single headline number for the paper
    """
    available_views = [v for v in (views or view_loaders.keys())
                       if view_loaders.get(v) is not None]
    if len(available_views) < 2:
        print(
            f"[WARNING] Cross-view evaluation needs at least 2 available "
            f"views, found {len(available_views)}. Skipping."
        )
        return None

    print(f"Cross-view evaluation across {len(available_views)} views "
          f"({len(available_views)}x{len(available_views)-1} = "
          f"{len(available_views)*(len(available_views)-1)} pairs)...")

    # -- Step 1: extract every view's gallery/probe embeddings ONCE ----------
    gallery_cache = {}   # view -> (agg_emb, agg_ids)
    probe_cache   = {}   # view -> (emb, ids)

    for view in available_views:
        print(f"  Extracting embeddings for view {view}...", flush=True)
        gal_emb, gal_ids = extract_embeddings(
            model, view_loaders[view]['gallery'], device
        )
        gal_emb_agg, gal_ids_agg = aggregate_gallery_by_subject(gal_emb, gal_ids)
        gallery_cache[view] = (gal_emb_agg, gal_ids_agg)

        prb_emb, prb_ids = extract_embeddings(
            model, view_loaders[view]['probe'], device
        )
        probe_cache[view] = (prb_emb, prb_ids)

    # -- Step 2: compute all non-identical (probe_view, gallery_view) pairs --
    per_pair = {}
    for probe_view in available_views:
        prb_emb, prb_ids = probe_cache[probe_view]
        for gallery_view in available_views:
            if gallery_view == probe_view:
                continue   # excluded per the standard literature convention

            gal_emb_agg, gal_ids_agg = gallery_cache[gallery_view]
            dist = cosine_distance_matrix(prb_emb, gal_emb_agg)

            rank1 = compute_rank_k(dist, prb_ids, gal_ids_agg, k=1)
            rank5 = compute_rank_k(dist, prb_ids, gal_ids_agg, k=5)
            mAP   = compute_map(dist, prb_ids, gal_ids_agg)
            eer, eer_thresh = compute_eer(dist, prb_ids, gal_ids_agg)

            per_pair[(probe_view, gallery_view)] = {
                'rank1': rank1, 'rank5': rank5, 'mAP': mAP,
                'eer': eer, 'eer_threshold': eer_thresh,
            }

    # -- Step 3: average per probe view, and overall ---------------------------
    import numpy as np

    per_probe_view = {}
    for probe_view in available_views:
        pair_results = [
            v for (pv, gv), v in per_pair.items() if pv == probe_view
        ]
        rank1_vals = [r['rank1'] for r in pair_results]
        eer_vals   = [r['eer']   for r in pair_results]
        per_probe_view[probe_view] = {
            'rank1_mean': float(np.mean(rank1_vals)),
            'rank1_std':  float(np.std(rank1_vals)),
            'eer_mean':   float(np.mean(eer_vals)),
            'eer_std':    float(np.std(eer_vals)),
            'n_gallery_views': len(pair_results),
        }

    all_rank1 = [r['rank1'] for r in per_pair.values()]
    all_eer   = [r['eer']   for r in per_pair.values()]
    overall = {
        'rank1_mean': float(np.mean(all_rank1)),
        'rank1_std':  float(np.std(all_rank1)),
        'eer_mean':   float(np.mean(all_eer)),
        'eer_std':    float(np.std(all_eer)),
        'n_pairs':    len(per_pair),
    }

    print(f"\nCross-view OVERALL: Rank-1 = {overall['rank1_mean']*100:.2f}% "
          f"+/- {overall['rank1_std']*100:.2f}%  "
          f"(averaged over {overall['n_pairs']} pairs)")

    return {
        'per_pair':       per_pair,
        'per_probe_view': per_probe_view,
        'overall':        overall,
    }


# -- Main -----------------------------------------------------------------------

def run_evaluation(checkpoint_path, cfg, device, dataset_entry,
                   use_graph=True, morph_backbone='custom'):
    """
    Args:
        checkpoint_path: path to a .pth checkpoint
        cfg:             merged config dict (model/heads/train/dataset yaml)
        device:          torch.device
        dataset_entry:   a datasets.registry.DatasetEntry for the active
                         --dataset value
        use_graph:       must match the flag used when this checkpoint
                         was TRAINED -- the model architecture (whether
                         the graph module exists at all) depends on it.
                         Mismatching this against training will either
                         crash on state_dict load (graph params present
                         in checkpoint but no graph module to receive
                         them) or silently use a differently-shaped model.
        morph_backbone:  same caveat as use_graph -- must match training.
    """
    print("Building dataloaders...", flush=True)
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    results  = {}
    cmc_dict = {}

    print(f"\n{'='*60}")
    print(f"Protocol Evaluation ({meta.name}, all test subjects as gallery)")
    print(f"{'='*60}")

    for protocol_name, protocol_data in loaders.get('protocols', {}).items():
        if protocol_data is None:
            print(f"\n[{protocol_name}] SKIPPED")
            results[protocol_name] = None
            continue

        print(f"\n[{protocol_name}]")
        metrics = evaluate_protocol(model, protocol_data, device,
                                    protocol_name)
        results[protocol_name]  = metrics
        cmc_dict[protocol_name] = metrics['cmc']

        print(f"  Rank-1: {metrics['rank1']*100:.2f}%")
        print(f"  Rank-5: {metrics['rank5']*100:.2f}%")
        print(f"  mAP:    {metrics['mAP']*100:.2f}%")
        print(f"  EER:    {metrics['eer']*100:.2f}%  "
              f"(threshold={metrics['eer_threshold']:.4f})")

    # -- Cross-view evaluation (OU-MVLP --cross_view mode only) -----------------
    # Only active when the dataset config explicitly requests it (see
    # datasets/oulp_mvlp.py's cfg['dataset']['cross_view'] flag) AND the
    # dataset actually exposes per-view loaders -- FVG-B and the
    # same_view OU-MVLP mode have no 'view_loaders' key at all, so this
    # block is a complete no-op for every other dataset/mode.
    cross_view_results = None
    if loaders.get('cross_view') and 'view_loaders' in loaders:
        print(f"\n{'='*60}")
        print("CROSS-VIEW EVALUATION (literature-standard OU-MVLP protocol)")
        print(f"{'='*60}")
        cross_view_results = evaluate_cross_view_protocol(
            model, loaders['view_loaders'], device,
        )

    return results, cmc_dict, cross_view_results


def print_results_table(results):
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Protocol':<10} {'Rank-1':>8} {'Rank-5':>8} "
          f"{'mAP':>8} {'EER':>8} {'N_probe':>8} {'N_gallery':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for pname, metrics in results.items():
        if metrics is None:
            print(f"{pname:<10} {'N/A':>8} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8}")
        else:
            print(
                f"{pname:<10} "
                f"{metrics['rank1']*100:>7.2f}% "
                f"{metrics['rank5']*100:>7.2f}% "
                f"{metrics['mAP']*100:>7.2f}% "
                f"{metrics['eer']*100:>7.2f}% "
                f"{metrics['n_probe']:>8} "
                f"{metrics['n_gallery']:>10}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plot_dir', default='experiments/plots')
    parser.add_argument('--no_graph', action='store_true',
                        help='Must match the flag used when this '
                             'checkpoint was trained.')
    parser.add_argument('--morph_backbone', default='custom',
                        choices=['custom', 'gaitbase'],
                        help='Must match the flag used when this '
                             'checkpoint was trained.')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from datasets.registry import get_dataset_entry
    dataset_entry = get_dataset_entry(args.dataset)

    cfg = {}
    for path in ['configs/model.yaml', 'configs/heads.yaml',
                 'configs/train.yaml']:
        with open(path) as f:
            loaded = yaml.safe_load(f)
            if path == 'configs/heads.yaml':
                cfg['heads'] = loaded
            else:
                cfg.update(loaded)
    for path in dataset_entry.config_files:
        with open(path) as f:
            cfg.update(yaml.safe_load(f))

    results, cmc_dict, cross_view_results = run_evaluation(
        args.checkpoint, cfg, device, dataset_entry,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )
    print_results_table(results)

    if cmc_dict:
        try:
            plot_cmc_curves(cmc_dict, out_dir=args.plot_dir)
        except Exception as e:
            print(f"CMC plot skipped: {e}")

    out_path = args.checkpoint.replace('.pth', '_gait_results.json')
    serialisable = {}
    for pname, metrics in results.items():
        if metrics is None:
            serialisable[pname] = None
        else:
            serialisable[pname] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }

    if cross_view_results is not None:
        # JSON object keys must be strings -- per_pair's keys are
        # (probe_view, gallery_view) tuples, which json.dump cannot
        # serialise directly. Flatten to "probe_view->gallery_view"
        # string keys instead of silently dropping this data.
        serialisable['cross_view'] = {
            'per_pair': {
                f"{pv}->{gv}": v
                for (pv, gv), v in cross_view_results['per_pair'].items()
            },
            'per_probe_view': cross_view_results['per_probe_view'],
            'overall': cross_view_results['overall'],
        }

    with open(out_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
