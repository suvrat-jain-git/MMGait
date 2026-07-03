import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluators.gait_eval import evaluate_protocol
from evaluators.gender_eval import extract_features, train_linear_probe_avg
from utils.metrics import compute_gender_metrics
from datasets.base import NUM_AGE_BINS

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


GAIT_METRIC_KEYS = ['rank1', 'rank5', 'mAP', 'eer']


# -- Model loading --------------------------------------------------------------

def load_model(checkpoint_path, cfg, meta, device, use_graph=True,
               morph_backbone='custom'):
    from models.factory import build_model_config
    from models.biokinematic_net import BioKinematicNet

    model_cfg = build_model_config(
        cfg['model'], cfg['heads'], meta,
        use_graph=use_graph, morph_backbone=morph_backbone,
    )
    model = BioKinematicNet(model_cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']} from {checkpoint_path}")
    return model


# -- Per-seed evaluation ---------------------------------------------------------

def evaluate_one_seed(checkpoint_path, cfg, meta, loaders, device,
                      use_graph=True, morph_backbone='custom'):
    """
    Run gait + gender (+ age, if applicable) evaluation for one checkpoint.

    Returns:
        gait_results:   {protocol_name: {rank1, rank5, mAP, eer, ...} or None}
        gender_results: {balanced_accuracy, F1_Male, F1_Female,
                         linear_probe_Fm, linear_probe_Fk, gap}
                         or {} if this checkpoint's model has no gender head
        age_results:    same shape as gender_results but for age, or {}
                        if no age head / insufficient age-labeled val samples
    """
    model = load_model(checkpoint_path, cfg, meta, device,
                       use_graph=use_graph, morph_backbone=morph_backbone)

    gait_results = {}
    for protocol_name, protocol_data in loaders.get('protocols', {}).items():
        if protocol_data is None:
            gait_results[protocol_name] = None
            continue
        metrics = evaluate_protocol(model, protocol_data, device,
                                    protocol_name, cmc_max_rank=20)
        gait_results[protocol_name] = metrics

    feats = extract_features(model, loaders['val'], device)

    gender_results = {}
    if 'gender_logits' in feats:
        preds = feats['gender_logits'].argmax(dim=1)
        gm    = compute_gender_metrics(preds, feats['gender_labels'])
        acc_fm = train_linear_probe_avg(feats['Fm'], feats['gender_labels'], n_classes=2)
        acc_fk = train_linear_probe_avg(feats['Fk'], feats['gender_labels'], n_classes=2)
        gender_results = {
            'balanced_accuracy': gm['balanced_accuracy'],
            'F1_Male':           gm['F1_Male'],
            'F1_Female':         gm['F1_Female'],
            'linear_probe_Fm':   acc_fm,
            'linear_probe_Fk':   acc_fk,
            'gap':               acc_fm - acc_fk,
        }

    age_results = {}
    if 'age_bin_logits' in feats:
        mask = feats['age_mask']
        if mask.sum().item() >= 10:
            age_preds = feats['age_bin_logits'][mask].argmax(dim=1)
            age_bin   = feats['age_bin'][mask]
            age_acc   = (age_preds == age_bin).float().mean().item()
            acc_fm_age = train_linear_probe_avg(
                feats['Fm'][mask], age_bin, n_classes=NUM_AGE_BINS
            )
            acc_fk_age = train_linear_probe_avg(
                feats['Fk'][mask], age_bin, n_classes=NUM_AGE_BINS
            )
            age_results = {
                'bin_accuracy':    age_acc,
                'linear_probe_Fm': acc_fm_age,
                'linear_probe_Fk': acc_fk_age,
                'gap':             acc_fm_age - acc_fk_age,
            }

    return gait_results, gender_results, age_results


def evaluate_all_seeds(checkpoint_paths, cfg, meta, loaders, device,
                       use_graph=True, morph_backbone='custom', label='config'):
    """
    Run evaluate_one_seed() across a list of checkpoint paths, collecting
    per-metric lists for later aggregation/comparison.

    Returns:
        gait_lists:   {protocol: {metric_key: [v_seed1, v_seed2, ...]}}
        gender_lists: {metric_key: [v_seed1, ...]}
        age_lists:    {metric_key: [v_seed1, ...]}
    """
    gait_lists   = defaultdict(lambda: defaultdict(list))
    gender_lists = defaultdict(list)
    age_lists    = defaultdict(list)

    for i, ckpt_path in enumerate(checkpoint_paths):
        if not Path(ckpt_path).exists():
            print(f"  [WARNING] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"\n--- {label} seed {i+1}/{len(checkpoint_paths)}: {ckpt_path} ---")
        gait_res, gender_res, age_res = evaluate_one_seed(
            ckpt_path, cfg, meta, loaders, device,
            use_graph=use_graph, morph_backbone=morph_backbone,
        )

        for protocol, metrics in gait_res.items():
            if metrics is None:
                continue
            for key in GAIT_METRIC_KEYS:
                gait_lists[protocol][key].append(metrics[key])

        for key, val in gender_res.items():
            gender_lists[key].append(val)
        for key, val in age_res.items():
            age_lists[key].append(val)

    return gait_lists, gender_lists, age_lists


# -- Aggregation -----------------------------------------------------------------

def aggregate(values):
    """Mean +/- std from a list of floats."""
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def print_aggregate_report(gait_lists, gender_lists, age_lists, meta, label='Results'):
    print(f"\n{'='*70}")
    print(f"{label} (mean +/- std across {_n_seeds(gait_lists, gender_lists)} seeds)")
    print(f"{'='*70}")

    print(f"\nGait Recognition:")
    print(f"{'Protocol':<10} {'Rank-1':>16} {'Rank-5':>16} "
          f"{'mAP':>16} {'EER':>16}")
    print(f"{'-'*10} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
    for protocol in meta.protocols:
        if protocol not in gait_lists or not gait_lists[protocol]['rank1']:
            print(f"{protocol:<10} {'N/A':>16}")
            continue
        r1_m,  r1_s  = aggregate(gait_lists[protocol]['rank1'])
        r5_m,  r5_s  = aggregate(gait_lists[protocol]['rank5'])
        map_m, map_s = aggregate(gait_lists[protocol]['mAP'])
        eer_m, eer_s = aggregate(gait_lists[protocol]['eer'])
        print(f"{protocol:<10} "
              f"{r1_m*100:>6.2f}+/-{r1_s*100:.2f}% "
              f"{r5_m*100:>6.2f}+/-{r5_s*100:.2f}% "
              f"{map_m*100:>6.2f}+/-{map_s*100:.2f}% "
              f"{eer_m*100:>6.2f}+/-{eer_s*100:.2f}%")

    if gender_lists:
        print(f"\nGender & Disentanglement:")
        for key in ['balanced_accuracy', 'F1_Male', 'F1_Female',
                    'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            if not gender_lists[key]:
                continue
            m, s = aggregate(gender_lists[key])
            print(f"  {key:<25} {m*100:>6.2f} +/- {s*100:.2f}%")

    if age_lists:
        print(f"\nAge & Disentanglement:")
        for key in ['bin_accuracy', 'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            if not age_lists[key]:
                continue
            m, s = aggregate(age_lists[key])
            print(f"  {key:<25} {m*100:>6.2f} +/- {s*100:.2f}%")


def _n_seeds(gait_lists, gender_lists):
    for protocol_dict in gait_lists.values():
        for vals in protocol_dict.values():
            if vals:
                return len(vals)
    for vals in gender_lists.values():
        if vals:
            return len(vals)
    return 0


def results_to_dict(gait_lists, gender_lists, age_lists, meta):
    """Serialise aggregate results to a plain dict for JSON output."""
    out = {}
    for protocol in meta.protocols:
        if protocol not in gait_lists or not gait_lists[protocol]['rank1']:
            continue
        out[protocol] = {}
        for key in GAIT_METRIC_KEYS:
            m, s = aggregate(gait_lists[protocol][key])
            out[protocol][f'{key}_mean'] = m
            out[protocol][f'{key}_std']  = s
    if gender_lists:
        out['gender'] = {}
        for key in ['balanced_accuracy', 'F1_Male', 'F1_Female',
                    'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            if not gender_lists[key]:
                continue
            m, s = aggregate(gender_lists[key])
            out['gender'][f'{key}_mean'] = m
            out['gender'][f'{key}_std']  = s
    if age_lists:
        out['age'] = {}
        for key in ['bin_accuracy', 'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            if not age_lists[key]:
                continue
            m, s = aggregate(age_lists[key])
            out['age'][f'{key}_mean'] = m
            out['age'][f'{key}_std']  = s
    return out


# -- Significance testing ---------------------------------------------------------

def paired_wilcoxon(values_a, values_b):
    """
    Paired Wilcoxon signed-rank test between two equal-length lists of
    per-seed metric values (same seeds, two model configurations).

    Returns:
        dict with statistic, p_value, n, and a plain-English significance
        flag (p < 0.05), or None if scipy is unavailable or the inputs
        are unusable (different lengths, fewer than 1 paired observation).
    """
    if not HAS_SCIPY:
        return None
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return None

    x = np.array(values_a)
    y = np.array(values_b)

    try:
        stat, p = wilcoxon(x, y, method='exact')
    except Exception as e:
        # scipy can still raise for genuinely degenerate inputs (e.g. all
        # differences zero AND n too small for the exact null distribution
        # to be computed in some scipy versions) -- report this rather
        # than crash the whole comparison run over one metric.
        return {'error': str(e), 'n': len(x)}

    return {
        'statistic':   float(stat),
        'p_value':      float(p),
        'n':            len(x),
        'mean_a':       float(x.mean()),
        'mean_b':       float(y.mean()),
        'mean_diff':    float(y.mean() - x.mean()),
        'significant':  bool(p < 0.05),
    }


def print_comparison_report(gait_lists_a, gender_lists_a, age_lists_a,
                            gait_lists_b, gender_lists_b, age_lists_b,
                            meta, label_a='Config A', label_b='Config B'):
    print(f"\n{'='*80}")
    print(f"PAIRED COMPARISON: {label_a} vs {label_b} "
          f"(Wilcoxon signed-rank, n={_n_seeds(gait_lists_a, gender_lists_a)})")
    print(f"{'='*80}")

    if not HAS_SCIPY:
        print(
            "\n[WARNING] scipy not installed -- cannot run Wilcoxon test. "
            "Install with: pip install scipy"
        )
        return {}

    comparison = {}

    print(f"\nGait Recognition:")
    print(f"{'Protocol':<8} {'Metric':<8} {label_a:>12} {label_b:>12} "
          f"{'p-value':>10} {'Sig?':>6}")
    print(f"{'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")
    for protocol in meta.protocols:
        if protocol not in gait_lists_a or protocol not in gait_lists_b:
            continue
        comparison[protocol] = {}
        for key in GAIT_METRIC_KEYS:
            vals_a = gait_lists_a[protocol][key]
            vals_b = gait_lists_b[protocol][key]
            result = paired_wilcoxon(vals_a, vals_b)
            comparison[protocol][key] = result
            if result is None:
                continue
            if 'error' in result:
                print(f"{protocol:<8} {key:<8} "
                      f"{'(test failed: ' + result['error'][:30] + ')':>40}")
                continue
            sig = 'YES' if result['significant'] else 'no'
            print(f"{protocol:<8} {key:<8} "
                  f"{result['mean_a']*100:>11.2f}% {result['mean_b']*100:>11.2f}% "
                  f"{result['p_value']:>10.4f} {sig:>6}")

    if gender_lists_a and gender_lists_b:
        print(f"\nGender & Disentanglement:")
        comparison['gender'] = {}
        for key in ['balanced_accuracy', 'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            vals_a = gender_lists_a.get(key, [])
            vals_b = gender_lists_b.get(key, [])
            result = paired_wilcoxon(vals_a, vals_b)
            comparison['gender'][key] = result
            if result is None or 'error' in result:
                continue
            sig = 'YES' if result['significant'] else 'no'
            print(f"  {key:<25} {result['mean_a']*100:>7.2f}% -> "
                  f"{result['mean_b']*100:>7.2f}%  p={result['p_value']:.4f}  "
                  f"sig={sig}")

    if age_lists_a and age_lists_b:
        print(f"\nAge & Disentanglement:")
        comparison['age'] = {}
        for key in ['bin_accuracy', 'linear_probe_Fm', 'linear_probe_Fk', 'gap']:
            vals_a = age_lists_a.get(key, [])
            vals_b = age_lists_b.get(key, [])
            result = paired_wilcoxon(vals_a, vals_b)
            comparison['age'][key] = result
            if result is None or 'error' in result:
                continue
            sig = 'YES' if result['significant'] else 'no'
            print(f"  {key:<25} {result['mean_a']*100:>7.2f}% -> "
                  f"{result['mean_b']*100:>7.2f}%  p={result['p_value']:.4f}  "
                  f"sig={sig}")

    print(
        f"\nNote: 'Sig?'=YES means p < 0.05 (two-sided Wilcoxon signed-rank, "
        f"exact method). With only {_n_seeds(gait_lists_a, gender_lists_a)} "
        f"paired seeds, the minimum achievable p-value is 1/(2^n) = "
        f"{1/(2**_n_seeds(gait_lists_a, gender_lists_a)):.4f} -- a "
        f"non-significant result with this few seeds should NOT be read "
        f"as 'no difference exists', only as 'not enough evidence at "
        f"this sample size'."
    )

    return comparison


# -- Main -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'])
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to config A checkpoints (one per seed, '
                             'recommend 5: e.g. seeds 42,123,456,789,2024)')
    parser.add_argument('--no_graph', action='store_true',
                        help='Flag config A checkpoints were trained with.')
    parser.add_argument('--no_gender', action='store_true',
                        help='Flag config A checkpoints were trained without '
                             'gender supervision (--no_gender in train.py).')
    parser.add_argument('--morph_backbone', default='custom',
                        choices=['custom', 'gaitbase'],
                        help='Flag config A checkpoints were trained with.')

    parser.add_argument('--compare_checkpoints_b', nargs='+', default=None,
                        help='If provided, enables COMPARE mode: paths to '
                             'config B checkpoints (same number of seeds '
                             'as --checkpoints), paired Wilcoxon test run '
                             'against config A.')
    parser.add_argument('--no_graph_b', action='store_true',
                        help='Flag config B checkpoints were trained with.')
    parser.add_argument('--no_gender_b', action='store_true',
                        help='Flag config B checkpoints were trained without '
                             'gender supervision.')
    parser.add_argument('--morph_backbone_b', default='custom',
                        choices=['custom', 'gaitbase'],
                        help='Flag config B checkpoints were trained with.')
    parser.add_argument('--label_a', default='Config A')
    parser.add_argument('--label_b', default='Config B')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default='experiments/multi_seed_results.json')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    if len(args.checkpoints) < 5:
        print(
            f"[WARNING] Only {len(args.checkpoints)} checkpoints provided. "
            f"The intended protocol for this codebase is 5 seeds -- fewer "
            f"seeds weakens both the mean+/-std estimate and (in compare "
            f"mode) the Wilcoxon test's statistical power. Proceeding "
            f"anyway, but treat results accordingly."
        )

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

    print("Building dataloaders (shared across all checkpoints/seeds)...")
    if args.no_gender:
        cfg['dataset']['force_no_gender'] = True
        print("Gender supervision: DISABLED (no gender head in checkpoints)")
    loaders = dataset_entry.builder(cfg)
    meta    = loaders['meta']

    print(f"\n{'='*50}")
    print(f"Evaluating {args.label_a} ({len(args.checkpoints)} checkpoints)")
    print(f"{'='*50}")
    gait_a, gender_a, age_a = evaluate_all_seeds(
        args.checkpoints, cfg, meta, loaders, device,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
        label=args.label_a,
    )
    print_aggregate_report(gait_a, gender_a, age_a, meta, label=args.label_a)

    out_data = {args.label_a: results_to_dict(gait_a, gender_a, age_a, meta)}

    if args.compare_checkpoints_b is not None:
        if len(args.compare_checkpoints_b) != len(args.checkpoints):
            print(
                f"\n[WARNING] --checkpoints has {len(args.checkpoints)} "
                f"entries but --compare_checkpoints_b has "
                f"{len(args.compare_checkpoints_b)}. Wilcoxon requires "
                f"PAIRED observations (same number of seeds on each "
                f"side) -- truncating to the shorter list."
            )
            n = min(len(args.checkpoints), len(args.compare_checkpoints_b))
            args.checkpoints = args.checkpoints[:n]
            args.compare_checkpoints_b = args.compare_checkpoints_b[:n]

        print(f"\n{'='*50}")
        print(f"Evaluating {args.label_b} ({len(args.compare_checkpoints_b)} checkpoints)")
        print(f"{'='*50}")

        # If config B has different gender setting, rebuild dataloaders
        if args.no_gender_b != args.no_gender:
            cfg_b = dict(cfg)
            cfg_b['dataset'] = dict(cfg.get('dataset', {}))
            cfg_b['dataset']['force_no_gender'] = args.no_gender_b
            loaders_b = dataset_entry.builder(cfg_b)
            meta_b = loaders_b['meta']
            print(f"Gender supervision for B: {'DISABLED' if args.no_gender_b else 'ENABLED'}")
        else:
            loaders_b = loaders
            meta_b = meta

        gait_b, gender_b, age_b = evaluate_all_seeds(
            args.compare_checkpoints_b, cfg, meta_b, loaders_b, device,
            use_graph=not args.no_graph_b, morph_backbone=args.morph_backbone_b,
            label=args.label_b,
        )
        print_aggregate_report(gait_b, gender_b, age_b, meta_b, label=args.label_b)

        comparison = print_comparison_report(
            gait_a, gender_a, age_a, gait_b, gender_b, age_b, meta,
            label_a=args.label_a, label_b=args.label_b,
        )

        out_data[args.label_b] = results_to_dict(gait_b, gender_b, age_b, meta)
        out_data['wilcoxon_comparison'] = comparison

    with open(args.out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
