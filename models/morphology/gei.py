"""
gei.py — Gait Energy Image Generator

What this module does:
    Takes a sequence of silhouette frames X of shape [B, T, 1, 224, 224]
    and produces the Gait Energy Image (GEI) by averaging across time.

    GEI = mean(X, dim=1)

    Output shape: [B, 1, 224, 224]

Input format:
    [B, T, 1, 224, 224]
     B = batch size
     T = number of frames
     1 = channel (grayscale silhouette)
     224, 224 = spatial dimensions

Why GEI captures morphology:
    Because averaging over all frames collapses motion (which cancels out
    over a full gait cycle) and retains the time-stable body structure:
    shoulder width, torso shape, leg length proportions, overall silhouette.

    This is the core disentanglement step for the morphology branch.
    The GEI does not know anything about how fast the person moves —
    only what their body looks like on average.

What this module does NOT do:
    - Any learned transformation (that happens in morphology_encoder.py)
    - Any normalization beyond the mean
"""


def generate_gei(x):
    """
    Args:
        x: Tensor of shape [B, T, 1, 224, 224]
           B = batch size
           T = number of frames in the sequence
           1 = channel dimension (grayscale)

    Returns:
        gei: Tensor of shape [B, 1, 224, 224]
             Ready for the 2D CNN which expects [B, C, H, W].
    """
    # Mean over the time dimension (dim=1).
    # [B, T, 1, 224, 224] -> [B, 1, 224, 224]
    #
    # The channel dim (1) is already present in the input,
    # so after averaging over T we get the correct output shape directly.
    # No unsqueeze needed unlike the [B, T, H, W] case.
    gei = x.mean(dim=1)

    return gei
