"""
barlow_twins.py — Barlow Twins Decorrelation Loss (V5 final formulation)

Forces every dimension of Fm to be uncorrelated with every dimension of Fk
by minimising the D×D cross-correlation matrix between the two branches.

Formula:
    C_ij = (1/B) * sum_b [ norm(Fm)_bi * norm(Fk)_bj ]

    L = ( sum_i(C_ii^2) + lambda * sum_{i!=j}(C_ij^2) ) / D

    where norm() = zero-mean + unit-std across the batch dimension.

This is the V5 formulation, confirmed via 3-seed multi-seed evaluation to
be the best-performing variant across all attempted disentanglement losses
(cosine orthogonality, this Barlow Twins, an un-normalised Barlow Twins
variant, per-sample cosine orthogonality, and HSIC):

    WS Rank-1:          59.11 +/- 2.75%   (best)
    Gender bal. acc.:   80.81 +/- 0.52%   (most stable -- other variants
                                            ranged 5.7%-14% std across seeds)
    Linear probe gap:   19.09 +/- 2.60%   (statistically tied with every
                                            other variant tried -- all
                                            landed in the 18-20% band)

The /D normalisation here is DELIBERATE -- without it (raw magnitude ~500
for D=512), the loss requires an awkward w~0.0002 to stay calibrated
against the identity loss, and a later attempt at that un-normalised
variant with "fixed" calibration was tried and produced WORSE retrieval
(-5% WS Rank-1) with no improvement to disentanglement metrics. The /D
version below, paired with w=0.05, is the one to keep going forward.

Reference:
    Zbontar et al., "Barlow Twins: Self-Supervised Learning via
    Redundancy Reduction", ICML 2021.
"""

import torch


def barlow_twins_loss(
    Fm: torch.Tensor,
    Fk: torch.Tensor,
    lambda_off_diag: float = 0.005,
) -> torch.Tensor:
    """
    Barlow Twins decorrelation loss (V5 formulation, /D normalised).

    Args:
        Fm:              [B, D] morphology features (pre-graph)
        Fk:              [B, D] motion features (pre-graph)
        lambda_off_diag: off-diagonal penalty weight (default 0.005)

    Returns:
        loss: scalar, magnitude ~1.0 for D=512, B=32 on correlated
              features. Use w_orthogonality=0.05 in combined loss.
    """
    B, D = Fm.shape

    # Standardise each dimension: zero mean, unit std across batch
    Fm_norm = (Fm - Fm.mean(dim=0)) / (Fm.std(dim=0) + 1e-8)
    Fk_norm = (Fk - Fk.mean(dim=0)) / (Fk.std(dim=0) + 1e-8)

    # D x D cross-correlation matrix -- each entry is Pearson r in [-1, 1]
    C = torch.mm(Fm_norm.t(), Fk_norm) / B   # [D, D]

    # Push all entries toward 0 (Fm orthogonal to Fk in every dimension)
    on_diag  = C.diagonal().pow(2).sum()
    off_diag = _off_diagonal(C).pow(2).sum()

    # /D normalisation -- keeps magnitude in a range where w=0.05 is
    # well-calibrated against identity loss (~0.9 at convergence)
    return (on_diag + lambda_off_diag * off_diag) / D


def _off_diagonal(C: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal elements of a square matrix as a 1D tensor."""
    D = C.shape[0]
    return C.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
