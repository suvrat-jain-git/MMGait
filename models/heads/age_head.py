"""
age_head.py — Age Estimation Head (Classification + Regression)

What this module does:
    Takes Fm' [B, 512] — the morphology feature AFTER graph interaction —
    and produces BOTH:
        - age_bin_logits: [B, NUM_AGE_BINS]  classification over 7 bins
        - age_value:      [B]                regression, raw years

    Both heads from a SHARED trunk (512 -> hidden_dim), branching only at
    the final layer. This is deliberate: classification and regression on
    the same underlying attribute should share most of their representation
    — splitting earlier would double the parameters for no clear benefit
    and risks the two outputs drifting into inconsistent predictions
    (e.g. classifying "child" while regressing to age 45).

Why Fm' and NOT the fused feature (same rationale as gender_head.py):
    Age, like gender, is treated in this architecture as a property of
    body shape/morphology rather than dynamic gait pattern. Routing only
    the morphology branch to the age head keeps the disentanglement story
    consistent: Fm' = static person attributes (shape, gender, age),
    Fk' = dynamic gait pattern feeding identity.

    Note: gait dynamics genuinely do correlate with age (cadence, stride
    length change with age) — supervising age only on Fm' does not erase
    that correlation from Fk, it simply means the age LOSS does not
    directly optimise Fk to encode it. This is an acknowledged simplification,
    consistent with how gender supervision is handled, and should be stated
    plainly in the paper rather than overclaimed.

Conditional instantiation:
    This head is only ever constructed when DatasetMeta.has_age is True
    (see model factory in biokinematic_net.py). It does not exist in the
    model graph at all for datasets without age labels (e.g. FVG-B) —
    no dead parameters, nothing to explain away in an ablation table.

Partial-label handling (OU-LP-Bag specific):
    Even when has_age=True for a dataset, INDIVIDUAL samples within a
    batch may still have age_label=None (only the OULP-Age intersection
    subset of OU-LP-Bag subjects is age-labeled). This head always
    produces a prediction for every sample in the batch — masking out
    the unlabeled samples for loss computation is the responsibility of
    the loss function (losses/combined_loss.py), not this head. The head
    itself has no concept of "missing labels" — it simply predicts.

Architecture:
    Fm' [B, 512]
      |
      FC 512->256
      BN1D
      ReLU
      Dropout(0.3)
      |
      +---- FC 256->NUM_AGE_BINS  ----> age_bin_logits [B, NUM_AGE_BINS]
      |
      +---- FC 256->1 (+ softplus) ---> age_value [B]  (raw years, >=0)

Note on the regression output:
    A softplus activation is applied to the final regression scalar to
    guarantee non-negative age predictions (age cannot be negative; a raw
    linear output could predict e.g. -3.2 years early in training, which
    is nonsensical and can destabilise the L1 loss gradient).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeHead(nn.Module):
    """
    Shared-trunk MLP producing both age-bin classification logits and a
    raw-years regression value from the morphology feature.
    """

    def __init__(self, in_dim=512, hidden_dim=256, num_bins=7):
        """
        Args:
            in_dim:     input dimension (must match Fm' = 512)
            hidden_dim: shared trunk size
            num_bins:   number of age classification bins
                        (default 7, matches datasets.base.AGE_BINS)
        """
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(hidden_dim, num_bins)
        self.regressor  = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """
        Fixed-seed Xavier uniform init, matching gender_head.py's approach.

        Applied here proactively rather than reactively — the gender head
        only got this fix after observing seed-dependent collapse in
        multi-seed evaluation. Age estimation has the same architecture
        (small MLP head, CrossEntropy-flavoured classification term) and
        is liable to the same failure mode, so the fix is applied from
        the start rather than waiting to rediscover the same bug.
        """
        with torch.random.fork_rng():
            torch.manual_seed(0)
            for module in [self.trunk, self.classifier, self.regressor]:
                for m in module.modules() if hasattr(module, 'modules') else [module]:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

    def forward(self, Fm_prime):
        """
        Args:
            Fm_prime: [B, 512] — morphology feature after graph interaction

        Returns:
            age_bin_logits: [B, num_bins] — raw class scores (no softmax)
            age_value:      [B]           — predicted age in years, >= 0
        """
        h = self.trunk(Fm_prime)                  # [B, hidden_dim]
        age_bin_logits = self.classifier(h)        # [B, num_bins]
        # softplus guarantees non-negative age; squeeze removes trailing dim 1
        age_value = F.softplus(self.regressor(h)).squeeze(-1)  # [B]
        return age_bin_logits, age_value
