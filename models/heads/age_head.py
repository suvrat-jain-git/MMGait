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
        with torch.random.fork_rng(devices=[]):
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
