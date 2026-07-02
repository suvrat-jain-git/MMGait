"""
combined_loss.py — Combined Training Loss (V2 codebase: conditional gender/age)

L_total = w_id    * L_identity
        + w_tri   * L_triplet
        + w_bt    * L_barlow_twins(Fm, Fk)          [always active]
        + w_gen   * L_gender_Fm                      [only if dataset has gender]
        + w_age   * (L_age_cls + L_age_reg)          [only if dataset has age,
                                                        masked per-sample]

This is a rewrite of the original CombinedLoss to be dataset-agnostic per
the V2 architecture plan:
    - Disentanglement loss restored to V5's Barlow Twins formulation
      (/D normalised, w=0.05) -- the best-performing variant found across
      every loss tried (cosine orthogonality variants, un-normalised BT,
      per-sample cosine, HSIC). See losses/barlow_twins.py for the full
      multi-seed comparison that justifies this choice.
    - Gender term is now CONDITIONAL: only computed if the model has a
      gender_head AND model_output contains 'gender_logits'. This lets
      the same CombinedLoss class serve both FVG-B (gender, no age) and
      OU-LP-Bag (gender AND age) without any flag-checking by the caller.
    - Age term is NEW: classification (CE over 7 bins) + regression (L1
      in years), each masked to only the samples in the batch that
      actually have an age label (see datasets/base.py age_mask).
      If a batch has zero age-labeled samples, the age loss contributes
      0 with no gradient for that batch -- it does not crash, and it
      does not silently train against a fabricated label.

The class auto-detects which terms to compute by inspecting which keys
are present in model_output -- it never needs an explicit dataset name
or a passed-in flag. This mirrors the "DatasetMeta determines what gets
built" principle from datasets/base.py: the MODEL decides which heads
exist based on dataset metadata (see biokinematic_net.py), and the LOSS
simply reacts to what the model actually produced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.triplet import TripletLoss
from losses.barlow_twins import barlow_twins_loss


class CombinedLoss:

    def __init__(self, w_identity=0.5, w_triplet=1.0, w_gender=0.5,
                 w_orthogonality=0.05, w_age_cls=0.3, w_age_reg=0.3,
                 triplet_margin=0.5, num_classes=None, **kwargs):
        """
        Args:
            w_identity:      identity CE weight
            w_triplet:       triplet loss weight
            w_gender:        gender CE weight (ignored if no gender head)
            w_orthogonality: Barlow Twins weight (V5 calibration: 0.05)
            w_age_cls:       age-bin classification CE weight
                              (ignored if no age head)
            w_age_reg:       age regression L1 weight
                              (ignored if no age head)
            triplet_margin:  margin for triplet loss
            num_classes:     identity class count (kept for API
                              compatibility with callers that pass it,
                              unused internally -- CrossEntropyLoss
                              infers class count from logits shape)
        """
        self.w_identity      = w_identity
        self.w_triplet       = w_triplet
        self.w_gender        = w_gender
        self.w_orthogonality = w_orthogonality
        self.w_age_cls       = w_age_cls
        self.w_age_reg       = w_age_reg

        self.ce_identity    = nn.CrossEntropyLoss()
        self.triplet        = TripletLoss(margin=triplet_margin)
        self.gender_weights = torch.tensor([1.0, 2.0])

        # No fixed class weighting for age bins by default -- unlike
        # gender (consistently ~62/38 imbalanced in FVG-B/OU-LP-Bag),
        # age bin distribution is dataset-dependent and not assumed here.
        # If a specific dataset's age distribution proves skewed enough
        # to need it, pass class weights via batch['age_bin'] statistics
        # at the call site rather than hardcoding here.
        self.ce_age = nn.CrossEntropyLoss()

    def __call__(self, model_output, batch):
        """
        Args:
            model_output: dict from BioKinematicNet.forward(mode='train').
                          Always contains: id_logits, embedding, Fm, Fk.
                          Conditionally contains: gender_logits (if model
                          has a gender head), age_bin_logits + age_value
                          (if model has an age head).
            batch:        dict from datasets/base.py gait_collate_fn.
                          Always contains: id_label. Conditionally:
                          gender_label (None if dataset has no gender),
                          age_label/age_bin/age_mask (None/all-False if
                          dataset has no age, or per-sample if partial).

        Returns:
            dict with keys: total, identity, triplet, adversarial,
            mean_pos_dist, mean_neg_dist, and CONDITIONALLY: gender,
            age_cls, age_reg -- only present if the corresponding term
            was actually computed. Callers (trainer, logger) must check
            for key presence rather than assuming a fixed key set, since
            the active terms differ by dataset.
        """
        id_labels = batch['id_label']

        # ── Identity CE ────────────────────────────────────────────────────
        l_identity = self.ce_identity(model_output['id_logits'], id_labels)

        # ── Triplet ────────────────────────────────────────────────────────
        l_triplet, triplet_stats = self.triplet(
            model_output['embedding'], id_labels
        )

        # ── Barlow Twins on pre-graph Fm and Fk (always active) ───────────
        l_bt = barlow_twins_loss(model_output['Fm'], model_output['Fk'])

        total = (self.w_identity      * l_identity
               + self.w_triplet       * l_triplet
               + self.w_orthogonality * l_bt)

        result = {
            'total':         None,   # filled in at the end, after all
                                      # conditional terms are added
            'identity':      l_identity,
            'triplet':       l_triplet,
            'adversarial':   l_bt,
            'mean_pos_dist': triplet_stats['mean_pos_dist'],
            'mean_neg_dist': triplet_stats['mean_neg_dist'],
        }

        # ── Gender CE on Fm' (conditional) ─────────────────────────────────
        if 'gender_logits' in model_output and batch.get('gender_label') is not None:
            gender_w = self.gender_weights.to(model_output['gender_logits'].device)
            l_gender = F.cross_entropy(
                model_output['gender_logits'], batch['gender_label'],
                weight=gender_w,
            )
            total = total + self.w_gender * l_gender
            result['gender'] = l_gender

        # ── Age classification + regression on Fm' (conditional, masked) ──
        if 'age_bin_logits' in model_output and batch.get('age_mask') is not None \
                and batch['age_mask'].any():
            mask = batch['age_mask']

            # Only the masked subset of the batch contributes -- a sample
            # with age_bin=-1/age_label=NaN must NEVER reach the loss,
            # since CrossEntropyLoss would either error on class index -1
            # or (worse) silently treat it as a valid negative class index
            # depending on PyTorch version. NaN in L1Loss would propagate
            # NaN gradients through the WHOLE batch, not just the
            # unlabeled samples, since they share the same backward graph
            # via the shared trunk -- masking before the loss call, not
            # after, is what prevents this.
            masked_age_bin_logits = model_output['age_bin_logits'][mask]
            masked_age_bin_labels = batch['age_bin'][mask]
            masked_age_value      = model_output['age_value'][mask]
            masked_age_label      = batch['age_label'][mask]

            l_age_cls = self.ce_age(masked_age_bin_logits, masked_age_bin_labels)
            l_age_reg = F.l1_loss(masked_age_value, masked_age_label)

            total = total + self.w_age_cls * l_age_cls + self.w_age_reg * l_age_reg
            result['age_cls'] = l_age_cls
            result['age_reg'] = l_age_reg

        result['total'] = total
        return result
