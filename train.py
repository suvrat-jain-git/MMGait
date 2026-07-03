import argparse
import random
import numpy as np
import torch
import yaml
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_configs(dataset_config_files):
    """
    Merge model.yaml + heads.yaml + train.yaml + the active dataset's own
    yaml file(s) into a single dict.

    Args:
        dataset_config_files: list of yaml paths from the dataset
                              registry entry for the active --dataset
    """
    cfg = {}
    for path in ['configs/model.yaml', 'configs/heads.yaml', 'configs/train.yaml']:
        with open(path, 'r') as f:
            loaded = yaml.safe_load(f)
            # heads.yaml has no top-level 'heads' key wrapper -- it's
            # consumed directly by models/factory.py, not merged into cfg
            if path == 'configs/heads.yaml':
                cfg['heads'] = loaded
            else:
                cfg.update(loaded)

    for path in dataset_config_files:
        with open(path, 'r') as f:
            cfg.update(yaml.safe_load(f))

    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fvgb', 'oulp_mvlp'],
                        help='Which dataset to train on')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed -- use 42, 123, 456, 789, 2024 '
                             'for the 5-seed multi-seed evaluation protocol')
    parser.add_argument('--no_graph', action='store_true',
                        help='Disable the Bio-Kinematic Graph (ablation): '
                             'Fm and Fk pass straight through with no '
                             'cross-branch interaction at all')
    parser.add_argument('--no_barlow', action='store_true',
                        help='Disable Barlow Twins disentanglement loss '
                             '(sets w_orthogonality=0.0). Used for the '
                             'No-Barlow-Twins ablation.')
    parser.add_argument('--save_dir', default=None,
                        help='Override checkpoint save directory from config. '
                             'Use this when running multiple seeds in parallel '
                             'to prevent them overwriting each other, e.g. '
                             '--save_dir experiments/fvgb_seed42')
    parser.add_argument('--morph_backbone', default='custom',
                        choices=['custom', 'gaitbase'],
                        help="Morphology branch backbone. 'gaitbase' "
                             "integration is a later stage of this rewrite "
                             "and will raise NotImplementedError until "
                             "then -- use 'custom' (default) in the "
                             "meantime.")
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Override sequence_length T from the dataset '
                             "yaml's default. Applied identically "
                             'regardless of which dataset is active.')
    return parser.parse_args()


def main():
    args = parse_args()

    from datasets.registry import get_dataset_entry
    dataset_entry = get_dataset_entry(args.dataset)
    cfg = load_configs(dataset_entry.config_files)

    if args.seq_len is not None:
        cfg['dataset']['sequence_length'] = args.seq_len

    if args.save_dir is not None:
        cfg['training']['checkpoint']['save_dir'] = args.save_dir

    if args.no_barlow:
        cfg['training']['loss_weights']['orthogonality'] = 0.0
        print("Barlow Twins loss DISABLED (w_orthogonality=0.0)")

    # -- Reproducibility --------------------------------------------------
    seed = args.seed
    set_seed(seed)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {seed}")
    print(f"Graph: {'DISABLED (ablation)' if args.no_graph else 'enabled'}")
    print(f"Morphology backbone: {args.morph_backbone}")

    # -- Device -------------------------------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -- Dataloaders ----------------------------------------------------------
    loaders = dataset_entry.builder(cfg)
    meta    = loaders['meta']

    print(f"\nDatasetMeta: name={meta.name}  has_gender={meta.has_gender}  "
          f"has_age={meta.has_age}")
    print(f"Training identities: {meta.num_identities}")
    print(f"Train batches:       {len(loaders['train'])}")
    print(f"Val batches:         {len(loaders['val'])}")
    if 'test_ids' in loaders:
        print(f"Test subjects:       {len(loaders['test_ids'])}")
    for pname, pdata in loaders.get('protocols', {}).items():
        if pdata is None:
            print(f"  {pname}: SKIPPED (not applicable to this test split)")
        else:
            print(f"  {pname}: "
                  f"gallery={len(pdata['gallery'].dataset)}  "
                  f"probe={len(pdata['probe'].dataset)}")

    # -- Model -----------------------------------------------------------------
    from models.factory import build_model_config
    from models.biokinematic_net import BioKinematicNet

    model_cfg = build_model_config(
        cfg['model'], cfg['heads'], meta,
        use_graph=not args.no_graph,
        morph_backbone=args.morph_backbone,
    )
    model = BioKinematicNet(model_cfg).to(device)

    breakdown = model.count_parameters()
    print("\nParameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:<20} {v:>10,}")

    # -- Loss --------------------------------------------------------------------
    from losses.combined_loss import CombinedLoss
    loss_w  = cfg['training']['loss_weights']
    loss_fn = CombinedLoss(
        w_identity=loss_w['identity'],
        w_triplet=loss_w['triplet'],
        w_gender=loss_w.get('gender', 0.5),
        w_orthogonality=loss_w.get('orthogonality', 0.05),
        w_age_cls=loss_w.get('age_cls', 0.3),
        w_age_reg=loss_w.get('age_reg', 0.3),
        triplet_margin=cfg['training']['triplet']['margin'],
        num_classes=meta.num_identities,
        lambda_off_diag=cfg['training'].get('barlow_lambda_off_diag', 0.005),
    )

    # -- Optimizer & Scheduler ----------------------------------------------------
    opt_cfg = cfg['training']['optimizer']
    sch_cfg = cfg['training']['scheduler']

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sch_cfg['T_max'],
        eta_min=sch_cfg['eta_min'],
    )

    # -- Trainer -----------------------------------------------------------------
    from trainers.trainer import Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        cfg=cfg,
        device=device,
    )

    # -- Resume --------------------------------------------------------------------
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    # -- Training loop ---------------------------------------------------------------
    epochs = cfg['training']['epochs']
    print(f"\nStarting training from epoch {start_epoch} to {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        train_losses = trainer.train_epoch(epoch)
        val_losses   = trainer.val_epoch(epoch)
        scheduler.step()

        # Conditional epoch summary -- only prints keys that are actually
        # present for this dataset (gender/age terms vary by dataset; see
        # trainers/trainer.py for how these keys get populated)
        train_parts = [f"total={train_losses['total']:.4f}",
                        f"id={train_losses['identity']:.4f}",
                        f"tri={train_losses['triplet']:.4f}",
                        f"orth={train_losses['adversarial']:.6f}"]
        if 'gender' in train_losses:
            train_parts.append(f"gen={train_losses['gender']:.4f}")
        if 'age_cls' in train_losses:
            train_parts.append(f"age_c={train_losses['age_cls']:.4f}")
        if 'age_reg' in train_losses:
            train_parts.append(f"age_r={train_losses['age_reg']:.4f}")

        val_parts = [f"total={val_losses['total']:.4f}",
                      f"id={val_losses['identity']:.4f}",
                      f"tri={val_losses['triplet']:.4f}"]
        if 'gender' in val_losses:
            val_parts.append(f"gen={val_losses['gender']:.4f}")
        if 'gender_acc' in val_losses:
            val_parts.append(f"gen_acc={val_losses['gender_acc']:.3f}")
        if 'age_cls_acc' in val_losses:
            val_parts.append(f"age_acc={val_losses['age_cls_acc']:.3f}")
        if 'age_mae' in val_losses:
            val_parts.append(f"age_mae={val_losses['age_mae']:.2f}yr")

        print(
            f"\nEpoch {epoch:03d}/{epochs} Summary | "
            f"LR={scheduler.get_last_lr()[0]:.6f}\n"
            f"  Train — " + "  ".join(train_parts) + "\n"
            f"  Val   — " + "  ".join(val_parts) + "\n"
        )

        # -- Rank-1 evaluation every 10 epochs --------------------------------
        # Save best checkpoint by primary-protocol Rank-1 -- the metric
        # that matters. Val loss is noisy and unreliable for this task.
        # The "primary protocol" is the first protocol name reported by
        # this dataset's DatasetMeta (FVG-B: 'WS'; OU-LP-Bag's protocol
        # list is dataset-specific, registered the same way).
        rank1 = 0.0
        if (epoch % 10 == 0 or epoch == epochs) and meta.protocols:
            from evaluators.gait_eval import (
                extract_embeddings, aggregate_gallery_by_subject,
            )
            from utils.metrics import cosine_distance_matrix, compute_rank_k

            primary_protocol = meta.protocols[0]
            protocol_data = loaders.get('protocols', {}).get(primary_protocol)
            if protocol_data is not None:
                model.eval()
                with torch.no_grad():
                    gal_emb, gal_ids = extract_embeddings(
                        model, protocol_data['gallery'], device
                    )
                    prb_emb, prb_ids = extract_embeddings(
                        model, protocol_data['probe'], device
                    )
                gal_emb_agg, gal_ids_agg = aggregate_gallery_by_subject(
                    gal_emb, gal_ids
                )
                dist  = cosine_distance_matrix(prb_emb, gal_emb_agg)
                rank1 = compute_rank_k(dist, prb_ids, gal_ids_agg, k=1)
                print(f"  {primary_protocol} Rank-1 (epoch {epoch}): {rank1*100:.2f}%")
                model.train()

        is_best = rank1 > trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = rank1
            print(f"Best checkpoint saved (Rank-1={rank1*100:.2f}%)")
        trainer.save_checkpoint(epoch, val_losses, is_best=is_best)

    print("Training complete.")


if __name__ == '__main__':
    main()
