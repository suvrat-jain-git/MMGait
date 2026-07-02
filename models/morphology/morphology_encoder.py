"""
morphology_encoder.py — 2D CNN that encodes the GEI into Fm

What this module does:
    Takes the GEI of shape [B, 1, 224, 224] and produces a fixed-size
    feature vector Fm of shape [B, 512] representing body morphology.

Architecture:
    Input:  [B, 1,   224, 224]
    Block1: [B, 32,  224, 224]  Conv2D(1→32,   stride=1) + BN + ReLU
    Block2: [B, 64,  112, 112]  Conv2D(32→64,  stride=2) + BN + ReLU
    Block3: [B, 128, 112, 112]  Conv2D(64→128, stride=1) + BN + ReLU
    Block4: [B, 256, 56,  56)   Conv2D(128→256,stride=2) + BN + ReLU
    Block5: [B, 512, 28,  28]   Conv2D(256→512,stride=2) + BN + ReLU
    GAP:    [B, 512]            AdaptiveAvgPool2d(1) + flatten

Design decisions and why:
    - Stride convolutions instead of MaxPool: the network learns what
      spatial information to discard, rather than always keeping the max.
    - All kernel sizes 3×3 with padding 1: preserves spatial resolution
      within each block, only stride changes the spatial footprint.
    - Conv + BN + ReLU: maximally stable when training from scratch.
      If training fails, we know it's not due to normalization choice.
    - Global Average Pool at the end: produces a spatial-position-
      invariant representation. The body silhouette may be slightly
      shifted across sequences; GAP handles this gracefully.

Output:
    Fm: [B, 512]
    This is the morphology feature that feeds into the graph module.
"""

import torch.nn as nn


class MorphologyEncoder(nn.Module):
    """
    2D CNN encoder for the Gait Energy Image.

    Produces Fm [B, 512] — a representation of static body structure.
    """

    def __init__(self, in_channels=1, channels=None):
        """
        Args:
            in_channels: number of input channels (1 for grayscale GEI)
            channels: list of output channels per block
                      default: [32, 64, 128, 256, 512]
        """
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        # strides[i] controls the spatial downsampling at block i.
        # stride=1 means spatial size is preserved.
        # stride=2 means spatial size is halved (learned downsampling).
        #
        # Trace through the spatial dimensions with input [B, 1, 224, 224]:
        #   Block1: stride=1 -> [B,  32, 224, 224]  (no change)
        #   Block2: stride=2 -> [B,  64, 112, 112]  (224/2)
        #   Block3: stride=1 -> [B, 128, 112, 112]  (no change)
        #   Block4: stride=2 -> [B, 256,  56,  56]  (112/2)
        #   Block5: stride=2 -> [B, 512,  28,  28]  (56/2)
        #   GAP             -> [B, 512]
        strides = [1, 2, 1, 2, 2]

        # FOOTGUN GUARD: zip(in_dims, out_dims, strides) below silently
        # truncates to the SHORTEST of the three lists with no error if
        # `channels` has the wrong length -- this previously produced a
        # confusing matmul shape error several layers downstream (in the
        # Bio-Kinematic Graph's Wm/Wk projection) rather than a clear
        # error at the point of the actual mistake. Fail loudly here
        # instead, at model-construction time.
        assert len(channels) == len(strides), (
            f"MorphologyEncoder: channels has {len(channels)} entries but "
            f"strides has {len(strides)}. `channels` must be exactly the "
            f"per-block OUTPUT channel sizes (5 entries) -- do NOT prepend "
            f"in_channels to this list; it is supplied separately via the "
            f"in_channels argument. Got channels={channels}."
        )

        # Build the channel pairs for each conv:
        # e.g. (in_channels, 32), (32, 64), (64, 128), ...
        in_dims  = [in_channels] + channels[:-1]
        out_dims = channels

        blocks = []
        for in_c, out_c, stride in zip(in_dims, out_dims, strides):
            blocks.append(
                nn.Sequential(
                    # kernel_size=3, padding=1 keeps spatial size when stride=1,
                    # and halves it cleanly when stride=2.
                    # bias=False because BatchNorm has its own bias term (beta).
                    # Adding a conv bias on top of BN's beta is redundant.
                    nn.Conv2d(in_c, out_c,
                              kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            )

        self.encoder = nn.Sequential(*blocks)

        # AdaptiveAvgPool2d(1) collapses any spatial size down to [B, C, 1, 1].
        # "Adaptive" means we specify the OUTPUT size (1×1), not the kernel size.
        # This makes the encoder robust to input sizes other than 224×224.
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, gei):
        """
        Args:
            gei: [B, 1, 224, 224]  — Gait Energy Image

        Returns:
            Fm: [B, 512]           — morphology feature vector
        """
        # Pass through the 5 conv blocks
        # [B, 1, 224, 224] -> [B, 512, 28, 28]
        x = self.encoder(gei)

        # Global Average Pool: average over all 28×28 spatial positions
        # [B, 512, 28, 28] -> [B, 512, 1, 1]
        x = self.pool(x)

        # Remove the trailing spatial dimensions
        # [B, 512, 1, 1] -> [B, 512]
        Fm = x.flatten(start_dim=1)

        return Fm
