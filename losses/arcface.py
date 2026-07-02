"""
arcface.py — ArcFace Loss for Identity Recognition

ArcFace adds an additive angular margin to the softmax loss.
Instead of:
    CE( W·f / ||W||·||f||, y )

ArcFace computes:
    CE( s · cos(θ_y + m), y )

where:
    θ_y = angle between feature f and class centre W_y
    m   = additive angular margin (default 0.5 radians)
    s   = feature scale (default 64)

Why ArcFace over plain CrossEntropy:
    Plain CE: maximises cosine similarity between feature and class weight.
    ArcFace:  maximises cosine similarity WITH an additional angular penalty.
    The margin m forces intra-class compactness and inter-class separability
    in the angular (hyperspherical) space — exactly what retrieval needs.

    For gait recognition with 117 training subjects, ArcFace typically
    improves Rank-1 by 2-8% over plain CE because the embedding space
    is better structured for nearest-neighbour retrieval.

Parameters:
    in_features:  embedding dimension (512)
    num_classes:  number of training identities
    s:            feature scale — radius of the hypersphere (default 64)
    m:            angular margin in radians (default 0.5 ≈ 28.6°)

Reference:
    Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition." CVPR 2019.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss module.

    Acts as a replacement for nn.Linear + nn.CrossEntropyLoss
    in the identity head. The weight matrix W serves as class centres
    on the unit hypersphere.
    """

    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        """
        Args:
            in_features: input embedding dimension (512)
            num_classes: number of training identities
            s:           feature scale (hypersphere radius)
            m:           angular margin in radians
        """
        super().__init__()
        self.in_features  = in_features
        self.num_classes  = num_classes
        self.s            = s
        self.m            = m

        # Class weight matrix — each row is a class centre on unit hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values for efficiency
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Threshold: cos(π - m) — used to handle edge cases where
        # θ + m > π (feature already past the decision boundary)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

        self.ce = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        """
        Args:
            features: [B, in_features] — L2-normalised identity embeddings
                      (pre-BNNeck embeddings work best here)
            labels:   [B] — integer identity labels

        Returns:
            loss: scalar ArcFace loss
        """
        # Normalise features and weights to unit vectors
        # This maps everything onto the unit hypersphere
        feat_norm   = F.normalize(features, dim=1)           # [B, D]
        weight_norm = F.normalize(self.weight, dim=1)        # [C, D]

        # Cosine similarity between each feature and each class centre
        # cos_theta[i, j] = cos(angle between feature i and class j)
        cos_theta = F.linear(feat_norm, weight_norm)         # [B, C]
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(theta) from cos(theta) via identity: sin²+cos²=1
        sin_theta = (1.0 - cos_theta ** 2).sqrt()

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        # This is the angular margin applied to the target class
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Handle the edge case where theta + m > pi:
        # if cos_theta < threshold, use a linear approximation
        # to avoid the gradient becoming unstable
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm
        )

        # Build the final logit matrix:
        # For the target class j: use cos(theta_j + m)
        # For all other classes:  use cos(theta_j) unchanged
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale by s (hypersphere radius) and compute CE
        logits = logits * self.s
        loss   = self.ce(logits, labels)

        return loss, logits
