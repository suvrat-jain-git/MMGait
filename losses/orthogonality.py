"""
orthogonality.py — Per-Sample Cosine Orthogonality Loss

Forces each sample's morphology feature Fm_i to be orthogonal to
its corresponding motion feature Fk_i.

Formula:
    L_orth = mean_i( cosine_similarity(Fm_i, Fk_i)^2 )

Previous (incorrect) implementation:
    L_orth = mean_{i,j}( (Fm_i · Fk_j)^2 )  — B×B matrix

    This computes cosine similarity between ALL pairs (including i≠j).
    Cross-sample pairs (Fm_i vs Fk_j) are meaningless — there is no
    reason why person i's morphology should be orthogonal to person j's
    motion. These 992 irrelevant pairs (for B=32) diluted the 32 meaningful
    diagonal pairs by 30x, making the loss too weak to enforce orthogonality.

Correct implementation (this file):
    Only penalise the diagonal: cosine_similarity(Fm_i, Fk_i) for each i.
    This is the per-sample constraint — each person's body shape feature
    should be orthogonal to their own gait dynamics feature.
    Computed as element-wise dot product (no matmul), 30x stronger signal.

Why per-sample makes sense:
    The disentanglement claim is: for each person, Fm encodes their
    body shape and Fk encodes their gait dynamics. These should be
    orthogonal because body shape and gait dynamics are different
    attributes of the same person. Cross-sample pairs (Fm_i vs Fk_j)
    have no meaningful interpretation.
"""

import torch
import torch.nn.functional as F


def orthogonality_loss(Fm: torch.Tensor, Fk: torch.Tensor) -> torch.Tensor:
    """
    Per-sample cosine orthogonality loss.

    Penalises the cosine similarity between Fm_i and Fk_i for each
    sample i in the batch. Correct and 30x stronger than the previous
    B×B matrix formulation.

    Args:
        Fm: [B, D] morphology features (pre-graph)
        Fk: [B, D] motion features (pre-graph)

    Returns:
        loss: scalar in [0, 1]
              0 = Fm and Fk are perfectly orthogonal for every sample
              1 = Fm and Fk are perfectly aligned for every sample
    """
    # L2 normalise so the dot product equals cosine similarity
    Fm_norm = F.normalize(Fm, dim=1)   # [B, D]
    Fk_norm = F.normalize(Fk, dim=1)   # [B, D]

    # Per-sample cosine similarity: element-wise dot product, sum over D
    # cos_sim[i] = Fm_norm_i · Fk_norm_i = cosine_similarity(Fm_i, Fk_i)
    # Shape: [B]
    cos_sim = (Fm_norm * Fk_norm).sum(dim=1)

    # Squared and averaged — loss is 0 when all cos_sim = 0
    loss = cos_sim.pow(2).mean()

    return loss
