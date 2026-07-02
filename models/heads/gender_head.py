"""
gender_head.py — Gender Classification Head

What this module does:
    Takes Fm' [B, 512] — the morphology feature AFTER graph interaction —
    and classifies the subject's gender (Male=0, Female=1).

    FC: 512 → 128 → 2
    Output: gender logits [B, 2]
    Loss:   CrossEntropyLoss

Why Fm' and NOT the fused feature:
    This is one of the two core scientific claims of the paper:

        Gender is a property of body shape, not of how you walk.

    By routing only the morphology branch to the gender head, we force
    the model to verify this claim structurally. If gender accuracy is high,
    it confirms that morphology encodes gender-relevant body structure.
    If accuracy is low, it tells us the hypothesis needs revision.

    Using the fused feature would let motion information bleed into
    gender prediction, which would make the result uninterpretable.

Why Fm' (post-graph) rather than Fm (pre-graph):
    Fm' has received a small message from the motion branch (via alpha * Wm(Fk)).
    Using Fm' rather than Fm means gender prediction is allowed to see
    the very small motion context, but morphology is still the dominant signal.
    This is a minor distinction — in practice, alpha is small and Fm' ≈ Fm.

Architecture:
    Fm' [B, 512]
      │
      FC 512→128
      BN1D
      ReLU
      │
      FC 128→2
      │
    Logits [B, 2]

Note: No softmax here. CrossEntropyLoss in PyTorch expects raw logits.
"""

import torch.nn as nn


class GenderHead(nn.Module):
    """
    Small MLP that predicts gender from the morphology feature.
    """

    def __init__(self, in_dim=512, hidden_dim=128, num_classes=2):
        """
        Args:
            in_dim:      input dimension (must match Fm' = 512)
            hidden_dim:  intermediate layer size
            num_classes: 2 (Male / Female)
        """
        super().__init__()

        self.head = nn.Sequential(
            # First layer: compress morphology feature to hidden_dim.
            # bias=True here because BatchNorm1d follows and will
            # re-center anyway — but we keep it explicit for clarity.
            nn.Linear(in_dim, hidden_dim),

            # BatchNorm1d stabilizes the activations going into ReLU.
            # Especially useful here because Fm' magnitudes may vary
            # across subjects with different body sizes.
            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(inplace=True),

            # Final layer: project to number of gender classes (2).
            # No BN here — CrossEntropyLoss operates on raw logits.
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        """
        Fixed-seed Xavier uniform init — prevents gender head collapse.

        The gender head collapsed to 50% balanced accuracy in one of three
        seeds during multi-seed evaluation. The root cause was unlucky random
        weight initialisation pushing the decision boundary to always predict
        the majority class. Using a fixed seed for init ensures the starting
        point is consistent regardless of the global training seed.
        """
        import torch
        with torch.random.fork_rng():
            torch.manual_seed(0)   # fixed seed for gender head only
            for m in self.head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, Fm_prime):
        """
        Args:
            Fm_prime: [B, 512] — morphology feature after graph interaction

        Returns:
            logits: [B, 2] — raw class scores (no softmax)
        """
        # Straight pass through the two-layer MLP.
        # [B, 512] -> [B, 128] -> [B, 2]
        return self.head(Fm_prime)
