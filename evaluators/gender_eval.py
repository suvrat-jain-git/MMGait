"""
gender_eval.py — Gender + Age Disentanglement Evaluator (V2: dataset-agnostic)

Evaluates, for the active --dataset:
    1. Gender head accuracy from Fm' (gender_logits)
       Metrics: accuracy, balanced accuracy, F1 Male, F1 Female
    2. Linear probe on Fm -- should be HIGH (morphology encodes gender)
    3. Linear probe on Fk -- should be ~50% (motion should NOT encode gender)
    4. EER for gender verification from Fm embeddings
    5. CONDITIONALLY (only if the loaded model has an age_head, i.e. was
       trained on a has_age dataset): the same four-part analysis for
       age -- age head classification accuracy/MAE, linear probe on Fm
       vs Fk for age-bin prediction, disentanglement gap. Age linear
       probes are computed ONLY over samples with a valid age label
       (the partial-label subset), matching the masking convention used
       throughout the rest of this codebase.

Usage:
    python evaluators/gender_eval.py --dataset fvgb --checkpoint experiments/best.pth
    python evaluators/gender_eval.py --dataset oulp_mvlp --checkpoint experiments/best.pth

Changes from the original (FVG-B-only) evaluator:
    - extract_features() reads dict batches (datasets/base.py
      gait_collate_fn format) instead of unpacking a (frames, id_labels,
      gender_labels) tuple, and ALSO collects age_bin_logits/age_value/
      age_label/age_mask when the model produces them.
    - Model construction goes through models/factory.py + the dataset
      registry, exactly like the rewritten gait_eval.py, instead of
      hardcoding build_fvgb_dataloaders + unconditional gender injection.
    - The age disentanglement block (probe gap, FDR-style "Fm > Fk"
      check) only runs when the checkpoint's model actually has an
      age_head -- running this evaluator against an FVG-B checkpoint
      (no age) produces exactly the original gender-only report, with
      no age section printed or saved.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_gender_metrics, compute_eer
from utils.visualization import plot_gender_confusion
from datasets.base import NUM_AGE_BINS


# -- Feature extraction --------------------------------------------------------

def extract_features(model, loader, device):
    """
    Args:
        model:  BioKinematicNet, any head configuration
        loader: DataLoader yielding dict batches

    Returns:
        dict always containing: Fm, Fk, Fm_prime, id_labels
        CONDITIONALLY containing (only if the model produced them):
            gender_logits, gender_labels
            age_bin_logits, age_value, age_label, age_bin, age_mask
    """
    all_Fm = []; all_Fk = []; all_Fm_prime = []
    all_gender_logits = []; all_gender_labels = []
    all_age_bin_logits = []; all_age_values = []
    all_age_labels = []; all_age_bins = []; all_age_masks = []
    all_id_labels = []

    has_gender = False
    has_age    = False

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            out    = model(frames, mode='train')

            all_Fm.append(out['Fm'].cpu())
            all_Fk.append(out['Fk'].cpu())
            all_Fm_prime.append(out['Fm_prime'].cpu())
            all_id_labels.extend(
                batch['id_label'].tolist() if hasattr(batch['id_label'], 'tolist')
                else list(batch['id_label'])
            )

            if 'gender_logits' in out and batch.get('gender_label') is not None:
                has_gender = True
                all_gender_logits.append(out['gender_logits'].cpu())
                all_gender_labels.extend(batch['gender_label'].tolist())

            if 'age_bin_logits' in out:
                has_age = True
                B = out['age_bin_logits'].shape[0]
                all_age_bin_logits.append(out['age_bin_logits'].cpu())
                all_age_values.append(out['age_value'].cpu())
                if batch.get('age_mask') is not None:
                    all_age_masks.append(batch['age_mask'])
                else:
                    all_age_masks.append(torch.zeros(B, dtype=torch.bool))
                if batch.get('age_bin') is not None:
                    all_age_bins.extend(batch['age_bin'].tolist())
                    all_age_labels.extend(batch['age_label'].tolist())
                else:
                    all_age_bins.extend([-1] * B)
                    all_age_labels.extend([float('nan')] * B)

    result = {
        'Fm':         torch.cat(all_Fm,       dim=0),
        'Fk':         torch.cat(all_Fk,       dim=0),
        'Fm_prime':   torch.cat(all_Fm_prime, dim=0),
        'id_labels':  torch.tensor(all_id_labels),
    }
    if has_gender:
        result['gender_logits'] = torch.cat(all_gender_logits, dim=0)
        result['gender_labels'] = torch.tensor(all_gender_labels)
    if has_age:
        result['age_bin_logits'] = torch.cat(all_age_bin_logits, dim=0)
        result['age_value']      = torch.cat(all_age_values,      dim=0)
        result['age_bin']        = torch.tensor(all_age_bins)
        result['age_label']      = torch.tensor(all_age_labels)
        result['age_mask']       = torch.cat(all_age_masks, dim=0)
    return result


# -- Linear probe ----------------------------------------------------------------

def train_linear_probe(features, labels, n_classes, n_epochs=100, lr=0.01, seed=42):
    """
    Args:
        features:  [N, D]
        labels:    [N] long tensor of class indices
        n_classes: number of classes (2 for gender, NUM_AGE_BINS for age)
    """
    N = len(labels)
    gen  = torch.Generator().manual_seed(seed)
    idx  = torch.randperm(N, generator=gen)
    n_tr = int(N * 0.8)
    X_tr, y_tr = features[idx[:n_tr]], labels[idx[:n_tr]]
    X_te, y_te = features[idx[n_tr:]], labels[idx[n_tr:]]

    torch.manual_seed(seed)
    clf = nn.Linear(features.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    clf.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        F.cross_entropy(clf(X_tr), y_tr).backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        preds = clf(X_te).argmax(dim=1)
        acc   = (preds == y_te).float().mean().item()
    return acc


def train_linear_probe_avg(features, labels, n_classes, n_runs=5):
    accs = [train_linear_probe(features, labels, n_classes, seed=s)
            for s in range(n_runs)]
    return float(np.mean(accs))


# -- Main evaluation -----------------------------------------------------------

def run_gender_evaluation(checkpoint_path, cfg, device, plot_dir,
                          dataset_entry, use_graph=True, morph_backbone='custom'):
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

    print("Extracting features from val set...", flush=True)
    feats = extract_features(model, loaders['val'], device)

    results = {}

    # ---- GENDER ANALYSIS (only if model has a gender head) -----------------
    if 'gender_logits' in feats:
        print(f"\nVal samples: {len(feats['gender_labels'])}")
        print(f"Gender distribution: "
              f"Male={(feats['gender_labels']==0).sum().item()}  "
              f"Female={(feats['gender_labels']==1).sum().item()}")

        print("\n=== Gender Head (Fm') ===")
        preds = feats['gender_logits'].argmax(dim=1)
        gm    = compute_gender_metrics(preds, feats['gender_labels'])
        print(f"  Accuracy:          {gm['accuracy']*100:.2f}%")
        print(f"  Balanced Accuracy: {gm['balanced_accuracy']*100:.2f}%")
        print(f"  F1 Male:           {gm['F1_Male']*100:.2f}%"
              f"  (P={gm['precision_Male']*100:.1f}%"
              f"  R={gm['recall_Male']*100:.1f}%)")
        print(f"  F1 Female:         {gm['F1_Female']*100:.2f}%"
              f"  (P={gm['precision_Female']*100:.1f}%"
              f"  R={gm['recall_Female']*100:.1f}%)")

        print("\n=== Gender Verification EER (Fm') ===")
        Fm_norm = F.normalize(feats['Fm_prime'], dim=1)
        dist_matrix = 1.0 - torch.mm(Fm_norm, Fm_norm.t())
        gender_labels_list = feats['gender_labels'].tolist()
        eer_val, eer_thresh = compute_eer(
            dist_matrix, gender_labels_list, gender_labels_list
        )
        print(f"  EER: {eer_val*100:.2f}%  (threshold={eer_thresh:.4f})")

        print("\n=== Linear Probe on Fm (morphology) ===")
        print("Training linear probe (avg 5 seeds)... (expected HIGH accuracy)")
        acc_fm = train_linear_probe_avg(feats['Fm'], feats['gender_labels'], n_classes=2)
        print(f"Linear probe accuracy on Fm: {acc_fm*100:.2f}%")

        print("\n=== Linear Probe on Fk (motion) ===")
        print("Training linear probe (avg 5 seeds)... (expected ~50% if disentangled)")
        acc_fk = train_linear_probe_avg(feats['Fk'], feats['gender_labels'], n_classes=2)
        print(f"Linear probe accuracy on Fk: {acc_fk*100:.2f}%")

        gap = acc_fm - acc_fk
        print(f"\n{'='*60}")
        print("GENDER EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<30} {'Fm (head)':>12} {'Fm (probe)':>12} "
              f"{'Fk (probe)':>12}")
        print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        print(f"{'Accuracy':<30} {gm['accuracy']*100:>11.2f}%"
              f" {acc_fm*100:>11.2f}% {acc_fk*100:>11.2f}%")
        print(f"{'Balanced Accuracy':<30} "
              f"{gm['balanced_accuracy']*100:>11.2f}%"
              f" {'N/A':>12} {'N/A':>12}")
        print(f"{'EER (Fm verification)':<30} {eer_val*100:>11.2f}%"
              f" {'N/A':>12} {'N/A':>12}")
        print(f"\nDisentanglement check (linear probe gap):")
        print(f"  Linear probe Fm - Fk gap: {gap*100:.2f}%")
        if gap > 0.15:
            print("  PASSED -- morphology encodes gender more than motion does")
        else:
            print("  FAILED -- motion branch may be encoding gender too")

        try:
            plot_gender_confusion(
                preds.numpy(), feats['gender_labels'].numpy(), out_dir=plot_dir
            )
        except Exception as e:
            print(f"Confusion matrix plot skipped: {e}")

        results.update({
            'gender_head_accuracy':          gm['accuracy'],
            'gender_head_balanced_accuracy': gm['balanced_accuracy'],
            'gender_head_F1_Male':           gm['F1_Male'],
            'gender_head_F1_Female':         gm['F1_Female'],
            'gender_eer':                    eer_val,
            'linear_probe_Fm_gender':        acc_fm,
            'linear_probe_Fk_gender':        acc_fk,
            'disentanglement_gap_gender':    gap,
        })
    else:
        print("\nModel has no gender_head -- skipping gender analysis "
              "(this is expected for a no-gender dataset).")

    # ---- AGE ANALYSIS (only if model has an age head AND val set has
    #      at least some age-labeled samples) ---------------------------------
    if 'age_bin_logits' in feats:
        mask = feats['age_mask']
        n_age_labeled = mask.sum().item()
        print(f"\n{'='*60}")
        print(f"AGE DISENTANGLEMENT ANALYSIS "
              f"({n_age_labeled}/{len(mask)} val samples have age labels)")
        print(f"{'='*60}")

        if n_age_labeled < 10:
            print(
                f"  SKIPPED -- only {n_age_labeled} age-labeled samples in "
                f"val set, too few for a meaningful linear probe (need >=10 "
                f"for an 80/20 train/test split to be sensible). This is a "
                f"sample-size issue with the partial age-label coverage, "
                f"not a bug -- consider evaluating age on a larger val "
                f"split or the full age-labeled subset directly if this "
                f"keeps happening."
            )
        else:
            Fm_age   = feats['Fm'][mask]
            Fk_age   = feats['Fk'][mask]
            age_bin  = feats['age_bin'][mask]

            print("\n=== Age Head (Fm') ===")
            age_preds = feats['age_bin_logits'][mask].argmax(dim=1)
            age_acc   = (age_preds == age_bin).float().mean().item()
            age_mae   = F.l1_loss(feats['age_value'][mask], feats['age_label'][mask]).item()
            print(f"  Bin classification accuracy: {age_acc*100:.2f}%")
            print(f"  Regression MAE: {age_mae:.2f} years")

            print("\n=== Linear Probe on Fm (morphology) for age bin ===")
            acc_fm_age = train_linear_probe_avg(Fm_age, age_bin, n_classes=NUM_AGE_BINS)
            print(f"Linear probe accuracy on Fm: {acc_fm_age*100:.2f}%")

            print("\n=== Linear Probe on Fk (motion) for age bin ===")
            acc_fk_age = train_linear_probe_avg(Fk_age, age_bin, n_classes=NUM_AGE_BINS)
            print(f"Linear probe accuracy on Fk: {acc_fk_age*100:.2f}%")

            gap_age = acc_fm_age - acc_fk_age
            print(f"\nAge disentanglement gap (Fm - Fk): {gap_age*100:.2f}%")
            print(
                "  Note: gait dynamics genuinely correlate with age "
                "(cadence/stride length change with age), so a smaller "
                "gap here than for gender is expected and should NOT be "
                "over-interpreted as a disentanglement failure -- see "
                "models/heads/age_head.py docstring for the documented "
                "limitation."
            )

            results.update({
                'age_head_bin_accuracy':   age_acc,
                'age_head_mae_years':      age_mae,
                'linear_probe_Fm_age':     acc_fm_age,
                'linear_probe_Fk_age':     acc_fk_age,
                'disentanglement_gap_age': gap_age,
            })
    elif meta.has_age:
        print(
            "\nDataset reports has_age=True but the loaded checkpoint's "
            "model has no age_head -- this means the model was trained "
            "with a config that didn't build one. No age analysis run."
        )

    return results


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

    results = run_gender_evaluation(
        args.checkpoint, cfg, device, args.plot_dir, dataset_entry,
        use_graph=not args.no_graph, morph_backbone=args.morph_backbone,
    )

    out_path = args.checkpoint.replace('.pth', '_gender_age_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
