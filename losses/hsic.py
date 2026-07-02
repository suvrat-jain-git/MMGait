"""
hsic.py — HSIC Independence Loss (RBF kernel variant)

HSIC directly measures statistical dependence between two representations.
Unlike Barlow Twins which decorrelates feature dimensions linearly,
HSIC with RBF kernel can detect non-linear dependencies.

NOTE on high-dimensional regime (B=32, D=512):
    When B << D, kernel-based estimators suffer from high variance due to
    concentration of measure — all pairwise distances become similar.
    In practice for this architecture, HSIC provides a comparable signal
    to Barlow Twins. The primary disentanglement improvement comes from
    the static suppression in the motion branch (biokinematic_net.py).
    HSIC is retained as the theoretically stronger independence criterion.

Formula:
    HSIC(Fm, Fk) = trace(Km_c @ Kk_c) / (B-1)^2

    Km, Kk: RBF kernel matrices of L2-normalised Fm, Fk
    Km_c, Kk_c: double-centred kernel matrices

Reference:
    Gretton et al., "Measuring Statistical Dependence with
    Hilbert-Schmidt Norms", ALT 2005.
"""

import torch
import torch.nn.functional as F


def rbf_kernel(X: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF kernel matrix. K_ij = exp(-||Xi - Xj||^2 / 2sigma^2)"""
    sq_norm = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = (sq_norm + sq_norm.t() - 2.0 * torch.mm(X, X.t())).clamp(min=0.0)
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def centre_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-centre a kernel matrix: Kc = K - 1K - K1 + 1K1"""
    B = K.shape[0]
    ones = torch.ones(B, B, device=K.device, dtype=K.dtype) / B
    return K - ones @ K - K @ ones + ones @ K @ ones


def hsic_loss(
    Fm: torch.Tensor,
    Fk: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    HSIC independence loss between morphology and motion features.

    Args:
        Fm:    [B, D] morphology features
        Fk:    [B, D] motion features
        sigma: RBF kernel bandwidth (default 1.0 for L2-normalised features)

    Returns:
        loss: scalar — minimise to push Fm and Fk toward independence
    """
    B = Fm.shape[0]

    # L2-normalise so sigma=1.0 is meaningful across feature scales
    Fm_n = F.normalize(Fm, dim=1)
    Fk_n = F.normalize(Fk, dim=1)

    Km = rbf_kernel(Fm_n, sigma)
    Kk = rbf_kernel(Fk_n, sigma)

    Km_c = centre_kernel(Km)
    Kk_c = centre_kernel(Kk)

    return (Km_c * Kk_c).sum() / ((B - 1) ** 2)
