"""
grl.py — Gradient Reversal Layer

The Gradient Reversal Layer (GRL) is a domain adaptation technique.

Forward pass:  identity function  f(x) = x
Backward pass: negates gradients  ∂L/∂x → -λ * ∂L/∂x

How it works in our context:
    We attach a gender classifier to Fk via a GRL:

        Fk → GRL → gender_adversary → gender_logits_fk

    During forward: Fk flows through unchanged
    During backward: gradients from gender_adversary are NEGATED before
                     reaching Fk

    Effect: the gender_adversary tries to predict gender from Fk.
            The GRL makes Fk try to FOOL the adversary.
            Result: Fk becomes uninformative about gender.

    Meanwhile, Fm → gender_head is trained POSITIVELY — Fm keeps gender.
    So the two branches diverge: Fm learns gender, Fk unlearns it.

Why this directly attacks Fk probe = 70.45%:
    The adversarial loss on Fk creates a gradient signal that pushes
    Fk features away from being gender-discriminative.
    Expected result: Fk probe drops from 70% toward 55-60%.

λ (lambda):
    Controls the adversarial strength.
    λ=0: GRL has no effect (equivalent to no adversary)
    λ=1: full gradient reversal
    We start with λ=0.1 and can increase if needed.
    Alternatively, use a schedule: λ increases from 0 to 1 during training.

Reference:
    Ganin & Lempitsky. "Unsupervised Domain Adaptation by Backpropagation."
    ICML 2015.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Custom autograd function implementing the gradient reversal.
    The forward pass is identity; backward pass negates and scales gradients.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        # Save lambda for use in backward
        ctx.save_for_backward(torch.tensor(lambda_))
        # Forward is identity
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        # Reverse and scale gradients
        # Return None for lambda_ since it has no gradient
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.

    Args:
        lambda_: reversal strength (default 0.1, increase over training)
    """

    def __init__(self, lambda_=0.1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """Update reversal strength during training."""
        self.lambda_ = lambda_


class GenderAdversary(nn.Module):
    """
    Small gender classifier attached to Fk via GRL.

    Fk [B, 512] → GRL → FC 512→128 → FC 128→2 → gender_logits_fk

    The GRL makes Fk try to fool this classifier.
    The classifier tries to predict gender from Fk.
    The adversarial game pushes Fk to remove gender information.

    Architecture mirrors the gender head on Fm for symmetry.
    """

    def __init__(self, in_dim=512, hidden_dim=128, num_classes=2,
                 lambda_=0.1):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, Fk):
        """
        Args:
            Fk: [B, 512] — motion features

        Returns:
            logits: [B, 2] — gender logits from adversary
                    (used to compute adversarial gender loss)
        """
        # GRL reverses gradients on backward pass
        x = self.grl(Fk)
        return self.classifier(x)

    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)
