"""
motion_generator.py — Frame Difference Motion Sequence Generator

What this module does:
    Takes a sequence of silhouette frames X of shape [B, T, 1, 224, 224]
    and produces a motion sequence of shape [B, 1, T-1, 224, 224]
    by computing absolute frame differences.

    Motion[t] = abs(X[t] - X[t-1])   for t in 1..T-1

    Output shape: [B, 1, T-1, 224, 224]

Input format:
    [B, T, 1, 224, 224]
     B = batch size
     T = number of frames
     1 = channel (grayscale silhouette)
     224, 224 = spatial dimensions

Why frame differences capture motion:
    Pixels that don't move contribute zero (or near-zero) to the difference.
    Pixels where the body is actively moving — swinging arms, stepping legs —
    produce high activation.

    This means the motion tensor encodes WHERE and HOW MUCH the body is
    moving at each spatial location over time. The static torso contributes
    almost nothing; the moving limbs contribute strongly.

Why abs():
    Without abs(), positive and negative differences (body moving into
    vs. out of a pixel) would partially cancel in the 3D CNN's convolutions.
    abs() preserves the magnitude of movement regardless of direction.

Why [B, 1, T-1, 224, 224]:
    Conv3D expects [B, C, D, H, W] where D is the temporal depth.
    The channel dim (C=1) comes from the input's channel dim — no unsqueeze needed.
    D = T-1 because we lose one frame computing the difference.

What this module does NOT do:
    - Optical flow (explicit velocity vectors)
    - Any learned transformation (that happens in motion_encoder.py)
"""


import torch


def generate_motion(x):
    """
    Args:
        x: Tensor of shape [B, T, 1, 224, 224]
           B = batch size
           T = number of frames
           1 = channel dimension (grayscale)

    Returns:
        motion: Tensor of shape [B, 1, T-1, 224, 224]
                Ready to feed directly into the Motion 3D CNN.
    """
    # Compute absolute differences between consecutive frames.
    # x[:, 1:] selects frames 1..T-1  — shape [B, T-1, 1, 224, 224]
    # x[:, :-1] selects frames 0..T-2 — shape [B, T-1, 1, 224, 224]
    # Their difference is the per-pixel change between each consecutive pair.
    diff = torch.abs(x[:, 1:] - x[:, :-1])
    # diff: [B, T-1, 1, 224, 224]

    # Conv3D expects [B, C, D, H, W].
    # Currently we have [B, T-1, 1, H, W] — time and channel are swapped.
    # Permute to move channel (dim=2) before time (dim=1):
    # [B, T-1, 1, 224, 224] -> [B, 1, T-1, 224, 224]
    motion = diff.permute(0, 2, 1, 3, 4).contiguous()

    return motion
