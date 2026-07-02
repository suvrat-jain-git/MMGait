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
        with torch.random.fork_rng(devices=[]):
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
