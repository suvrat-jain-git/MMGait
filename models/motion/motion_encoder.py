"""
motion_encoder.py — 3D CNN that encodes the motion sequence into Fk

What this module does:
    Takes the motion volume of shape [B, 1, T-1, 224, 224] and produces
    a fixed-size feature vector Fk of shape [B, 512] representing
    the person's dynamic gait motion.

Architecture:
    Input:  [B, 1,   T-1, 224, 224]
    Block1: [B, 32,  T-1, 224, 224]  Conv3D(1→32,    stride=(1,1,1)) + BN3D + ReLU
    Block2: [B, 64,  T/2, 112, 112]  Conv3D(32→64,   stride=(2,2,2)) + BN3D + ReLU
    Block3: [B, 128, T/2, 112, 112]  Conv3D(64→128,  stride=(1,1,1)) + BN3D + ReLU
    Block4: [B, 256, T/4, 56,  56]   Conv3D(128→256, stride=(2,2,2)) + BN3D + ReLU
    Block5: [B, 512, T/4, 56,  56]   Conv3D(256→512, stride=(1,1,1)) + BN3D + ReLU
    GAP:    [B, 512]                  AdaptiveAvgPool3d(1) + flatten

Note on temporal dimensions:
    With T=30 frames, the motion sequence has T-1=29 frames.
    After stride=(2,2,2) at Block2: ~14 temporal frames
    After stride=(2,2,2) at Block4: ~7 temporal frames
    AdaptiveAvgPool3d(1) collapses all remaining (T, H, W) to a single value
    per channel — a global summary of the full spatiotemporal volume.

Design decisions and why:
    - 3D kernels (3,3,3) with padding (1,1,1): each conv looks at a 3-frame
      temporal window AND a 3x3 spatial window simultaneously. This is key —
      gait dynamics are not purely temporal. A leg swing is a temporal change
      AND a spatial displacement. Both matter.
    - Stride (2,2,2) at layers 2 and 4: controlled joint spatiotemporal
      downsampling. Avoids the information loss of temporal-only pooling.
    - Mirrors the morphology branch exactly in channel progression and number
      of blocks. This symmetry makes ablation studies cleaner.
    - BatchNorm3D + ReLU: identical reasoning to morphology branch — maximally
      stable for training from scratch.

Output:
    Fk: [B, 512]
    This is the motion feature that feeds into the graph module.
"""

import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    3D CNN encoder for the frame-difference motion volume.

    Produces Fk [B, 512] — a representation of dynamic gait motion.
    """

    def __init__(self, in_channels=1, channels=None):
        """
        Args:
            in_channels: number of input channels (1 for single-channel motion)
            channels: list of output channels per block
                      default: [32, 64, 128, 256, 512]
        """
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 512]

        # strides[i] controls spatiotemporal downsampling at block i.
        # (1,1,1) = no downsampling in any dimension.
        # (2,2,2) = halve temporal depth AND halve spatial H and W.
        #
        # Trace through dimensions with input [B, 1, 29, 224, 224]:
        #   Block1: stride=(1,1,1) -> [B,  32, 29, 224, 224]
        #   Block2: stride=(2,2,2) -> [B,  64, 14, 112, 112]
        #   Block3: stride=(1,1,1) -> [B, 128, 14, 112, 112]
        #   Block4: stride=(2,2,2) -> [B, 256,  7,  56,  56]
        #   Block5: stride=(1,1,1) -> [B, 512,  7,  56,  56]
        #   GAP                    -> [B, 512]
        strides = [
            (1, 1, 1),
            (2, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
            (1, 1, 1),
        ]

        # FOOTGUN GUARD -- see the identical assertion in
        # models/morphology/morphology_encoder.py for the full
        # explanation. Same failure mode applies here.
        assert len(channels) == len(strides), (
            f"MotionEncoder: channels has {len(channels)} entries but "
            f"strides has {len(strides)}. `channels` must be exactly the "
            f"per-block OUTPUT channel sizes (5 entries) -- do NOT prepend "
            f"in_channels to this list; it is supplied separately via the "
            f"in_channels argument. Got channels={channels}."
        )

        # Build channel pairs: (in, out) for each block
        in_dims  = [in_channels] + channels[:-1]
        out_dims = channels

        blocks = []
        for in_c, out_c, stride in zip(in_dims, out_dims, strides):
            blocks.append(
                nn.Sequential(
                    # kernel_size=(3,3,3): attends to a 3-frame temporal window
                    # AND a 3x3 spatial neighbourhood simultaneously.
                    # This is the key difference from treating time and space
                    # separately — the conv sees spatiotemporal motion patterns.
                    #
                    # padding=(1,1,1): preserves all dimensions when stride=(1,1,1),
                    # and halves them cleanly when stride=(2,2,2).
                    #
                    # bias=False: BatchNorm3d has its own learnable shift (beta),
                    # making a conv bias redundant.
                    nn.Conv3d(in_c, out_c,
                              kernel_size=(3, 3, 3),
                              stride=stride,
                              padding=(1, 1, 1),
                              bias=False),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                )
            )

        self.encoder = nn.Sequential(*blocks)

        # AdaptiveAvgPool3d(1) collapses any (D, H, W) to (1, 1, 1).
        # After the two stride-2 blocks, remaining dims are ~(7, 56, 56) —
        # all collapsed to a single average per channel.
        # "Adaptive" means we specify output size, not kernel size,
        # so this works regardless of input T.
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, motion):
        """
        Args:
            motion: [B, 1, T-1, 224, 224] — frame difference volume

        Returns:
            Fk: [B, 512] — motion feature vector
        """
        # Pass through the 5 conv blocks
        # [B, 1, 29, 224, 224] -> [B, 512, 7, 56, 56]
        x = self.encoder(motion)

        # Global Average Pool over all spatiotemporal positions
        # [B, 512, 7, 56, 56] -> [B, 512, 1, 1, 1]
        x = self.pool(x)

        # Remove the trailing spatial and temporal dimensions
        # [B, 512, 1, 1, 1] -> [B, 512]
        Fk = x.flatten(start_dim=1)

        return Fk
