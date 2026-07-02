"""
trainer.py — Training Loop (V2: dataset-agnostic, conditional gender/age)

Responsibilities:
    - One epoch of training (forward + loss + backward + step)
    - One epoch of validation (forward only, compute losses)
    - Checkpoint saving (best model + periodic)
    - Loss logging per batch and per epoch

What it does NOT do:
    - Define the model  (biokinematic_net.py)
    - Define the loss   (combined_loss.py)
    - Compute retrieval metrics (evaluators/gait_eval.py)
    - Load the dataset  (datasets/fvg_b.py, datasets/oulp_mvlp.py)

Changes from the original (FVG-B-only) trainer:
    - Batches are now dicts (from datasets/base.py gait_collate_fn), not
      bare (frames, id_labels, gender_labels) tuples. Every field access
      goes through explicit batch['key'] lookups.
    - loss_fn is now called as loss_fn(output, batch) rather than
      loss_fn(output, id_labels, gender_labels) -- see combined_loss.py.
    - Per-epoch accumulation and logging are now driven by whichever keys
      CombinedLoss actually returned for a given batch (gender/age terms
      may or may not be present depending on dataset), rather than a
      hardcoded fixed set of loss names. This is what lets the SAME
      trainer serve FVG-B (gender only) and OU-LP-Bag (gender + age)
      without a single if/else on dataset name anywhere in this file.
    - The val-epoch gender-CE recomputation trick (collecting all val
      predictions before computing CE, to avoid single-class-batch CE
      spikes) is preserved, and now ALSO applied to age classification
      for the same reason, when age is present.

Logged per batch and per epoch: whatever subset of
    {total, identity, triplet, adversarial, gender, age_cls, age_reg}
the active dataset's CombinedLoss call actually produced.
"""

import os
import time
import torch


# Canonical ordering for logging -- not every key is present every run,
# but when present, always logged in this order for readability.
_LOSS_KEY_ORDER = [
    'total', 'identity', 'triplet', 'adversarial',
    'gender', 'age_cls', 'age_reg',
]

# Short names for compact per-batch logging
_SHORT_NAME = {
    'identity':    'id',
    'triplet':     'tri',
    'adversarial': 'orth',
    'gender':      'gen',
    'age_cls':     'age_c',
    'age_reg':     'age_r',
}


class Trainer:

    def __init__(self, model, loss_fn, optimizer, scheduler,
                 train_loader, val_loader, cfg, device):
        """
        Args:
            model:        BioKinematicNet instance. May or may not have
                          gender_head / age_head depending on the dataset
                          it was built for (see biokinematic_net.py).
            loss_fn:      CombinedLoss instance
            optimizer:    torch optimizer
            scheduler:    torch lr scheduler
            train_loader: DataLoader yielding dict batches (see
                          datasets/base.py gait_collate_fn)
            val_loader:   DataLoader, same dict format
            cfg:          full config dict -- uses cfg['training'] keys
            device:       torch.device
        """
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        self.save_dir   = cfg['training']['checkpoint']['save_dir']
        self.save_every = cfg['training']['checkpoint']['save_every']
        self.log_every  = cfg['training']['log_every']

        os.makedirs(self.save_dir, exist_ok=True)

        # Tracks best WS Rank-1 (higher is better) -- initialised to 0.
        # Name kept as best_val_loss for checkpoint-dict backward
        # compatibility with existing analysis scripts that read this key.
        self.best_val_loss = 0.0

    # -- Helpers --------------------------------------------------------

    @staticmethod
    def _move_batch_to_device(batch, device):
        """
        Move every tensor field of a batch dict to device, leaving None
        fields (e.g. age_label on a dataset with no age annotation) as
        None rather than erroring on .to(device).
        """
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    # -- Training epoch ---------------------------------------------------

    def train_epoch(self, epoch):
        """
        Run one full training epoch.

        Returns:
            dict of average losses over the epoch. Keys present depend
            on the active dataset (gender/age terms only appear if the
            model + dataset support them) -- callers must not assume a
            fixed key set.
        """
        self.model.train()

        accum     = {}   # built lazily from whichever keys appear
        n_batches = 0
        t_start   = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch, self.device)

            # Forward pass -- training mode returns full output dict.
            # The model itself decides which heads to run based on what
            # it was constructed with (see biokinematic_net.py) -- the
            # trainer does not need to know.
            output = self.model(batch['frames'], mode='train')

            losses = self.loss_fn(output, batch)

            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping -- prevents exploding gradients from
            # the 3D CNN early in training when features are noisy.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            for k, v in losses.items():
                val = v.item() if hasattr(v, 'item') else v
                accum[k] = accum.get(k, 0.0) + val
            n_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                elapsed = time.time() - t_start
                parts = [f"Loss {losses['total'].item():.4f}"]
                inner = []
                for key in _LOSS_KEY_ORDER[1:]:   # skip 'total', already shown
                    if key in losses:
                        inner.append(f"{_SHORT_NAME[key]}={losses[key].item():.4f}")
                if inner:
                    parts.append("(" + " ".join(inner) + ")")
                parts.append(
                    f"pos_dist={losses['mean_pos_dist']:.3f} "
                    f"neg_dist={losses['mean_neg_dist']:.3f}"
                )
                print(
                    f"Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{len(self.train_loader):04d} | "
                    + " | ".join(parts) + f" | {elapsed:.1f}s"
                )

        avg = {k: v / n_batches for k, v in accum.items()}
        return avg

    # -- Validation epoch ---------------------------------------------------

    def val_epoch(self, epoch):
        """
        Run one full validation epoch (no gradients, no augmentation).

        Gender and age classification CE are recomputed over the FULL
        concatenated val set rather than per-batch-then-averaged, for the
        same reason as the original implementation: a batch that happens
        to contain only one class causes CE to spike to an arbitrarily
        large value unrelated to model quality. This recomputation is
        applied identically to gender (if present) and age classification
        (if present, and masked to only age-labeled val samples).

        Returns:
            dict of average losses, plus 'gender_acc' and/or
            'age_cls_acc'/'age_mae' when the corresponding labels exist
            for this dataset.
        """
        self.model.eval()

        all_id_logits  = []
        all_id_labels  = []
        all_embeddings = []
        all_gender_logits = []
        all_gender_labels = []
        all_age_bin_logits = []
        all_age_bin_labels = []
        all_age_values      = []
        all_age_labels      = []
        all_age_masks       = []

        has_gender = False
        has_age    = False

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch, self.device)
                output = self.model(batch['frames'], mode='train')

                all_id_logits.append(output['id_logits'])
                all_id_labels.append(batch['id_label'])
                all_embeddings.append(output['embedding'])

                if 'gender_logits' in output and batch.get('gender_label') is not None:
                    has_gender = True
                    all_gender_logits.append(output['gender_logits'])
                    all_gender_labels.append(batch['gender_label'])

                if 'age_bin_logits' in output:
                    has_age = True
                    # age_mask is ALWAYS a tensor from gait_collate_fn (even
                    # if all-False for this batch) -- always safe to append.
                    # age_bin/age_value/age_label are only non-None when
                    # this SPECIFIC batch has at least one age-labeled
                    # sample (see datasets/base.py gait_collate_fn) --
                    # appending None here would break the later torch.cat,
                    # so only append when the batch actually has them, and
                    # use a zero-shape placeholder otherwise to keep all
                    # five lists aligned with the same number of entries
                    # as the number of val batches consumed.
                    B = output['age_bin_logits'].shape[0]
                    all_age_bin_logits.append(output['age_bin_logits'])
                    all_age_values.append(output['age_value'])
                    if batch.get('age_mask') is not None:
                        all_age_masks.append(batch['age_mask'])
                    else:
                        all_age_masks.append(
                            torch.zeros(B, dtype=torch.bool, device=self.device)
                        )
                    if batch.get('age_bin') is not None:
                        all_age_bin_labels.append(batch['age_bin'])
                        all_age_labels.append(batch['age_label'])
                    else:
                        # This batch had zero age-labeled samples -- use
                        # placeholder values that will be correctly
                        # excluded later via the age_mask filter (the
                        # actual numeric value here is never used in any
                        # loss/metric computation since age_mask is False
                        # for every entry in this batch).
                        all_age_bin_labels.append(
                            torch.full((B,), -1, dtype=torch.long, device=self.device)
                        )
                        all_age_labels.append(
                            torch.full((B,), float('nan'), device=self.device)
                        )

        all_id_logits  = torch.cat(all_id_logits,  dim=0)
        all_id_labels  = torch.cat(all_id_labels,  dim=0)
        all_embeddings = torch.cat(all_embeddings, dim=0)

        l_identity = self.loss_fn.ce_identity(all_id_logits, all_id_labels)
        l_triplet, triplet_stats = self.loss_fn.triplet(all_embeddings, all_id_labels)

        avg = {
            'total':         None,   # filled below
            'identity':      l_identity.item(),
            'triplet':       l_triplet.item(),
            'adversarial':   0.0,    # Barlow Twins not computed on val
                                      # (no augmentation pairs to decorrelate
                                      # meaningfully against; matches original
                                      # design choice from the V1 codebase)
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }
        total = (self.loss_fn.w_identity * l_identity
               + self.loss_fn.w_triplet  * l_triplet)

        if has_gender:
            all_gender_logits = torch.cat(all_gender_logits, dim=0)
            all_gender_labels = torch.cat(all_gender_labels, dim=0)
            gender_w = self.loss_fn.gender_weights.to(all_gender_logits.device)
            l_gender = torch.nn.functional.cross_entropy(
                all_gender_logits, all_gender_labels, weight=gender_w
            )
            total = total + self.loss_fn.w_gender * l_gender
            avg['gender'] = l_gender.item()
            avg['gender_acc'] = (
                all_gender_logits.argmax(dim=1) == all_gender_labels
            ).float().mean().item()

        if has_age:
            all_age_bin_logits = torch.cat(all_age_bin_logits, dim=0)
            all_age_bin_labels = torch.cat(all_age_bin_labels, dim=0)
            all_age_values     = torch.cat(all_age_values,     dim=0)
            all_age_labels     = torch.cat(all_age_labels,     dim=0)
            all_age_masks      = torch.cat(all_age_masks,      dim=0)

            if all_age_masks.any():
                m = all_age_masks
                l_age_cls = self.loss_fn.ce_age(
                    all_age_bin_logits[m], all_age_bin_labels[m]
                )
                l_age_reg = torch.nn.functional.l1_loss(
                    all_age_values[m], all_age_labels[m]
                )
                total = total + self.loss_fn.w_age_cls * l_age_cls \
                              + self.loss_fn.w_age_reg * l_age_reg
                avg['age_cls'] = l_age_cls.item()
                avg['age_reg'] = l_age_reg.item()
                avg['age_cls_acc'] = (
                    all_age_bin_logits[m].argmax(dim=1) == all_age_bin_labels[m]
                ).float().mean().item()
                avg['age_mae'] = l_age_reg.item()   # L1 loss IS the MAE in years

        avg['total'] = total.item()
        return avg

    # -- Checkpointing ------------------------------------------------------

    def save_checkpoint(self, epoch, val_losses, is_best=False):
        """
        Save model + optimizer + scheduler state.

        Args:
            epoch:      current epoch number
            val_losses: dict of validation losses (whatever val_epoch
                        returned -- key set varies by dataset)
            is_best:    if True, also save as best.pth
        """
        state = {
            'epoch':           epoch,
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_losses':      val_losses,
            'best_val_loss':   self.best_val_loss,
        }

        if epoch % self.save_every == 0:
            path = os.path.join(self.save_dir, f'epoch_{epoch:03d}.pth')
            torch.save(state, path)
            print(f"Checkpoint saved: {path}")

        if is_best:
            path = os.path.join(self.save_dir, 'best.pth')
            torch.save(state, path)
            print(f"Best checkpoint saved: {path}  (val_loss={val_losses['total']:.4f})")

    def load_checkpoint(self, path):
        """
        Load checkpoint from path. Restores model, optimizer, scheduler.

        Returns:
            epoch: the epoch at which this checkpoint was saved
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.best_val_loss = state.get('best_val_loss', 0.0)
        print(f"Resumed from {path}  (epoch {state['epoch']})")
        return state['epoch']
