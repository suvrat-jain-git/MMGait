"""
triplet.py — Batch Hard Triplet Loss

What this module does:
    Given a batch of embeddings and their identity labels, computes the
    triplet loss using hard negative mining within the batch.

    For each anchor i:
        Hardest positive:  the sample with the SAME label as i that is
                           FURTHEST from i in embedding space.
        Hardest negative:  the sample with a DIFFERENT label from i that is
                           CLOSEST to i in embedding space.

    Loss per anchor:
        max(d(anchor, hard_pos) - d(anchor, hard_neg) + margin, 0)

    Where d() is Euclidean distance.

Why triplet loss for gait:
    CrossEntropy alone trains the model to classify identities in the training set.
    It says nothing about the geometry of the embedding space.
    Triplet loss explicitly enforces that:
        same identity → embeddings cluster together
        different identity → embeddings push apart
    This is essential for retrieval: at test time, we match a probe sequence
    to a gallery by nearest-neighbor search in the embedding space.
    CE loss does not guarantee a good metric space; triplet loss does.

Why hard mining:
    Easy triplets (where the positive is already close and negative is already far)
    contribute near-zero gradient. Hard mining focuses training on the difficult
    cases where the model is still making mistakes.

How the distance matrix works:
    We compute ALL pairwise Euclidean distances in one vectorised operation.
    For a batch of B embeddings, this gives a [B, B] matrix where entry (i, j)
    is the distance between embedding i and embedding j.

    We then build two masks from the labels:
        positive_mask[i, j] = True if labels[i] == labels[j] AND i != j
        negative_mask[i, j] = True if labels[i] != labels[j]

    Hard positive for anchor i: max distance where positive_mask[i] is True
    Hard negative for anchor i: min distance where negative_mask[i] is True

Parameters:
    margin: minimum required distance gap between positive and negative pairs.
            Default 0.3 (from train.yaml). Larger margin → harder constraint.
"""

import torch
import torch.nn as nn


class TripletLoss:
    """
    Batch hard triplet loss with Euclidean distance.
    """

    def __init__(self, margin=0.3):
        """
        Args:
            margin: triplet margin
        """
        self.margin = margin

    def _pairwise_distances(self, embeddings):
        """
        Compute the full [B, B] matrix of pairwise Euclidean distances.

        Uses the identity:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b

        This is more numerically stable than computing (a-b)^2 directly
        for high-dimensional vectors, and avoids a slow explicit loop.

        Args:
            embeddings: [B, D]

        Returns:
            dist_matrix: [B, B] — entry (i,j) = ||emb_i - emb_j||_2
        """
        # Squared L2 norm of each embedding: [B]
        sq_norm = (embeddings ** 2).sum(dim=1)

        # Expand to [B, B] for broadcasting:
        # sq_norm_row[i, j] = ||emb_i||^2
        # sq_norm_col[i, j] = ||emb_j||^2
        sq_norm_row = sq_norm.unsqueeze(1)  # [B, 1]
        sq_norm_col = sq_norm.unsqueeze(0)  # [1, B]

        # Dot product matrix: [B, B]
        # dot[i, j] = emb_i · emb_j
        dot = torch.mm(embeddings, embeddings.t())

        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
        # Clamp to zero to avoid negative values from floating point errors
        dist_sq = (sq_norm_row + sq_norm_col - 2.0 * dot).clamp(min=0.0)

        # Take sqrt for Euclidean distance.
        # Add small epsilon inside sqrt to avoid zero-gradient at dist=0.
        dist = (dist_sq + 1e-12).sqrt()

        return dist

    def __call__(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] — raw identity embeddings (pre-BNNeck)
            labels:     [B]    — integer identity labels

        Returns:
            loss:  scalar triplet loss
            stats: dict with 'mean_pos_dist', 'mean_neg_dist' for logging
        """
        B = embeddings.size(0)

        # ── Full pairwise distance matrix ──────────────────────────────────
        # dist[i, j] = Euclidean distance between embedding i and j
        dist = self._pairwise_distances(embeddings)  # [B, B]

        # ── Build positive and negative masks ──────────────────────────────
        # labels_equal[i, j] = True if subject i and j are the same person
        labels_col = labels.unsqueeze(0)  # [1, B]
        labels_row = labels.unsqueeze(1)  # [B, 1]
        labels_equal = labels_row == labels_col  # [B, B]

        # Positive mask: same identity, but not the anchor itself
        # (diagonal must be excluded — distance to self is always 0)
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = labels_equal & ~eye   # [B, B]

        # Negative mask: different identity
        negative_mask = ~labels_equal         # [B, B]

        # ── Hard positive: furthest same-identity sample ───────────────────
        # For each anchor i, find the hardest positive:
        # the same-identity sample that is already furthest away.
        # We mask out non-positives with 0 (safe because we take max).
        pos_dist = dist * positive_mask.float()   # [B, B]
        hardest_pos_dist = pos_dist.max(dim=1).values  # [B]

        # ── Hard negative: closest different-identity sample ───────────────
        # For each anchor i, find the hardest negative:
        # the different-identity sample that is already closest.
        # We mask out non-negatives by setting them to a large value
        # so they are never selected by min.
        neg_dist = dist + (~negative_mask).float() * 1e9  # [B, B]
        hardest_neg_dist = neg_dist.min(dim=1).values     # [B]

        # ── Triplet loss ───────────────────────────────────────────────────
        # For each anchor: max(d_pos - d_neg + margin, 0)
        # The max with 0 means easy triplets (d_neg > d_pos + margin already)
        # contribute zero loss — training focuses on the hard cases.
        triplet_loss = (hardest_pos_dist - hardest_neg_dist + self.margin)
        triplet_loss = triplet_loss.clamp(min=0.0).mean()

        # ── Stats for logging ──────────────────────────────────────────────
        # These tell you how well-separated the embedding space is.
        # Healthy training: mean_neg_dist > mean_pos_dist + margin.
        # If mean_pos_dist grows or mean_neg_dist shrinks, the embedding
        # space is collapsing.
        with torch.no_grad():
            # Only average over valid (non-masked) pairs
            mean_pos = (dist * positive_mask.float()).sum() / positive_mask.float().sum().clamp(min=1)
            mean_neg = (dist * negative_mask.float()).sum() / negative_mask.float().sum().clamp(min=1)

        stats = {
            'mean_pos_dist': mean_pos.item(),
            'mean_neg_dist': mean_neg.item(),
        }

        return triplet_loss, stats
