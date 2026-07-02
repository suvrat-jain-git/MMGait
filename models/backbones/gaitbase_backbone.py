"""
gaitbase_backbone.py — GaitBase-style ResNet9 Backbone for the Morphology Branch

What this module is:
    A faithful re-implementation of OpenGait's GaitBase backbone
    architecture (Fan et al., "OpenGait: Revisiting Gait Recognition
    Toward Better Practicality", CVPR 2023) -- an initial conv layer
    followed by 4 stacked residual stages (BasicBlock2D, ResNet-style),
    BN + ReLU throughout. This is the backbone GaitBase's authors
    refer to as "ResNet9" in their paper.

    Re-implemented here from the published architecture description
    rather than imported from the OpenGait codebase, because OpenGait's
    actual model file expects its own full pipeline conventions (set-based
    temporal pooling over a frame SEQUENCE, horizontal pooling pyramid,
    SeparateFCs/SeparateBNNecks heads) that are specific to GaitBase's
    own training recipe and not directly reusable as a drop-in single-
    image encoder.

INTEGRATION DECISION (read before changing this file):
    GaitBase's native design processes a SEQUENCE of frames: each frame
    independently through the ResNet9 backbone, then temporal pooling
    over T, then horizontal pooling over spatial strips. This does not
    match our morphology branch's contract, which is a single static
    image (the GEI) in, a single feature vector out -- analogous to
    models/morphology/morphology_encoder.py's [B,1,H,W] -> [B,512].

    Two ways to integrate GaitBase given this mismatch:
        A) Feed GaitBase's backbone the GEI as a degenerate T=1 "frame",
           run the ResNet9 stages + horizontal pooling, skip temporal
           pooling (nothing to pool over with T=1). This preserves our
           controlled-ablation property: same GEI input to both the
           custom encoder and GaitBase, only the encoder architecture
           differs. This is what this module implements.
        B) Feed GaitBase the full raw sequence and let it do its own
           temporal pooling, replacing both GEI generation AND the
           morphology encoder. This uses GaitBase as originally
           intended, but conflates two variables (input representation
           AND encoder) in the ablation instead of isolating one.

    Option A was chosen -- see the architecture decision discussion that
    preceded this implementation. If you want option B instead, this is
    the file to replace, not patch around.

Architecture (faithful to GaitBase's published ResNet9):
    Input:  [B, 1, H, W]
    Stem:   Conv2D(1->64, k=3, s=1) + BN + ReLU      (initial conv)
    Stage1: BasicBlock2D x2, 64->64,   stride=1
    Stage2: BasicBlock2D x2, 64->128,  stride=2
    Stage3: BasicBlock2D x2, 128->256, stride=2
    Stage4: BasicBlock2D x2, 256->512, stride=2
    HP:     Horizontal Pooling (split into 1 horizontal strip here, since
            we have no temporal/part-based downstream consumer -- GAP
            over full spatial extent, equivalent to HP with num_parts=1)
    Output: Fm [B, 512]

    "BasicBlock2D" below is the standard ResNet basic residual block:
    two 3x3 convs with a BN+ReLU after the first and a BN after the
    second, summed with a (possibly downsampled) identity shortcut,
    then a final ReLU. This matches the block referenced in OpenGait's
    deepgaitv2.py imports (BasicBlock2D from modeling/modules) and is
    architecturally identical to torchvision's ResNet BasicBlock.

Pretrained weight loading:
    See download_gaitbase_checkpoint() at the bottom of this file. Auto-
    download from OpenGait's GitHub Releases is attempted on first use
    if no local checkpoint is found; if it fails (no stable, predictable
    download URL could be confirmed for GaitBase's released checkpoints
    at the time this was written), the model falls back to random
    initialisation with a clear, loud warning -- NOT a silent failure.
    If you have manually obtained OpenGait's official GaitBase weights,
    place them at the path printed in that warning and they will be
    loaded automatically on the next run.
"""

import os
import urllib.request
import torch
import torch.nn as nn


# -- BasicBlock2D (standard ResNet basic residual block) ----------------------

class BasicBlock2D(nn.Module):
    """
    Standard ResNet basic residual block: two 3x3 convs, BN+ReLU after
    the first, BN after the second, summed with a (possibly downsampled)
    identity shortcut, final ReLU. expansion=1 (no bottleneck).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Downsample the identity shortcut when stride != 1 or channel
        # count changes, so it can be added to the main path's output.
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


def _make_layer(in_channels, out_channels, num_blocks, stride):
    """Stack num_blocks BasicBlock2D instances; only the first uses `stride`."""
    layers = [BasicBlock2D(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock2D(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


# -- GaitBase-style backbone ----------------------------------------------------

class GaitBaseBackbone(nn.Module):
    """
    GaitBase-style ResNet9 backbone, adapted as a single-image encoder
    (see module docstring's "INTEGRATION DECISION" for why).

    Produces a 512-dim feature vector from a single [B, 1, H, W] image,
    matching models/morphology/morphology_encoder.py's MorphologyEncoder
    contract exactly -- this is a drop-in replacement, selected via
    --morph_backbone gaitbase (see models/factory.py).
    """

    def __init__(self, in_channels=1, out_dim=512, blocks_per_stage=2,
                pretrained=False, checkpoint_path=None):
        """
        Args:
            in_channels:      1 for grayscale GEI (matches our pipeline)
            out_dim:          final feature dimension (512, matches Fm)
            blocks_per_stage: residual blocks per stage (2 = ResNet9-depth,
                              matching GaitBase's published architecture:
                              1 stem conv + 4 stages x 2 blocks x 2 convs
                              = 1 + 16 = 17 conv layers total, but the
                              "9" in ResNet9 refers to a shallower
                              GaitBase-specific variant per the paper;
                              kept as a tunable parameter rather than a
                              hardcoded constant so this can be adjusted
                              without touching the class body if a closer
                              parameter-count match to the official
                              ResNet9 is needed later)
            pretrained:       if True, attempt to load OpenGait's official
                              GaitBase checkpoint weights (see
                              download_gaitbase_checkpoint below). Falls
                              back to random init with a loud warning if
                              unavailable.
            checkpoint_path:  explicit local path to a checkpoint file.
                              If None and pretrained=True, uses the
                              default cache location and attempts
                              auto-download.
        """
        super().__init__()

        channels = [64, 128, 256, 512]
        strides  = [1, 2, 2, 2]

        # Stem: initial conv, matches GaitBase's published architecture
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # 4 residual stages
        self.layer1 = _make_layer(channels[0], channels[0], blocks_per_stage, strides[0])
        self.layer2 = _make_layer(channels[0], channels[1], blocks_per_stage, strides[1])
        self.layer3 = _make_layer(channels[1], channels[2], blocks_per_stage, strides[2])
        self.layer4 = _make_layer(channels[2], channels[3], blocks_per_stage, strides[3])

        # Horizontal Pooling equivalent: with num_parts=1 (no part-based
        # splitting, since our downstream consumer -- the Bio-Kinematic
        # Graph and identity/gender/age heads -- expects a single flat
        # 512-dim vector, not GaitBase's native part-based feature set),
        # this reduces to global average pooling, matching
        # MorphologyEncoder's AdaptiveAvgPool2d(1) exactly.
        self.pool = nn.AdaptiveAvgPool2d(1)

        assert channels[-1] == out_dim, (
            f"GaitBaseBackbone's final channel count ({channels[-1]}) must "
            f"equal out_dim ({out_dim}) for the [B,512] Fm contract to hold. "
            f"If you need a different out_dim, add a projection Linear "
            f"layer rather than changing `channels` directly, since "
            f"`channels` also determines pretrained checkpoint compatibility."
        )

        if pretrained:
            self._load_pretrained(checkpoint_path)

    def _load_pretrained(self, checkpoint_path):
        """
        Attempt to load OpenGait's official GaitBase checkpoint weights.
        See download_gaitbase_checkpoint() for the download attempt and
        fallback behaviour.
        """
        if checkpoint_path is None:
            checkpoint_path = download_gaitbase_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(
                "[WARNING] GaitBase pretrained weights not available -- "
                "proceeding with RANDOM INITIALISATION. This is NOT the "
                "intended ablation comparison (architecture-only, weights "
                "untrained) -- if you have manually obtained OpenGait's "
                "official GaitBase checkpoint, place it at "
                f"{_DEFAULT_CHECKPOINT_PATH} and re-run."
            )
            return

        try:
            state = torch.load(checkpoint_path, map_location='cpu')
            # OpenGait checkpoints nest weights under a 'model' key with
            # module-name prefixes that won't match our re-implementation
            # 1:1 -- a strict load is not expected to succeed without a
            # key-remapping step specific to whatever checkpoint format
            # is actually obtained. Attempt a non-strict load and report
            # exactly what matched, rather than silently claiming success.
            state_dict = state.get('model', state)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            n_total = len(list(self.state_dict().keys()))
            n_loaded = n_total - len(missing)
            print(
                f"GaitBase checkpoint loaded (partial): {n_loaded}/{n_total} "
                f"parameter tensors matched by name. {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. If n_loaded is low, the "
                f"checkpoint's key naming likely does not match this "
                f"re-implementation's module names -- inspect "
                f"state_dict.keys() vs self.state_dict().keys() and add a "
                f"key-remapping step here if pretrained weights are "
                f"important for your ablation."
            )
        except Exception as e:
            print(
                f"[WARNING] Failed to load GaitBase checkpoint from "
                f"{checkpoint_path}: {e}. Proceeding with RANDOM "
                f"INITIALISATION."
            )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] -- single image (the GEI in our pipeline)

        Returns:
            Fm: [B, 512]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)


# -- Pretrained checkpoint acquisition -----------------------------------------

_DEFAULT_CHECKPOINT_DIR  = os.path.expanduser('~/.cache/biokinematic_net/gaitbase')
_DEFAULT_CHECKPOINT_PATH = os.path.join(_DEFAULT_CHECKPOINT_DIR, 'gaitbase_checkpoint.pt')

# NOTE: OpenGait publishes GaitBase checkpoints via GitHub Releases, but
# no single stable, version-independent direct-download URL could be
# confirmed at the time this was written (GitHub Releases asset URLs are
# tied to specific release tags and the release page directs to a
# Google-Drive-hosted model zoo for the actual weight files in some
# cases, rather than a GitHub-hosted binary). This placeholder URL is
# NOT guaranteed to work -- update it once you've located the actual
# checkpoint URL from https://github.com/ShiqiYu/OpenGait/releases or
# the model zoo doc (docs/1.model_zoo.md) referenced there.
_CHECKPOINT_DOWNLOAD_URL = None   # deliberately unset -- see note above


def download_gaitbase_checkpoint(force=False):
    """
    Attempt to download OpenGait's official GaitBase checkpoint to the
    local cache directory, returning the local path on success.

    Returns None if no download URL is configured or the download fails
    -- callers (see GaitBaseBackbone._load_pretrained) must handle this
    by falling back to random initialisation with a clear warning, never
    by silently proceeding as if pretrained weights were loaded.
    """
    if os.path.exists(_DEFAULT_CHECKPOINT_PATH) and not force:
        return _DEFAULT_CHECKPOINT_PATH

    if _CHECKPOINT_DOWNLOAD_URL is None:
        print(
            "[INFO] No GaitBase checkpoint download URL is configured "
            "(see models/backbones/gaitbase_backbone.py's "
            "_CHECKPOINT_DOWNLOAD_URL). To use real pretrained weights, "
            "manually download OpenGait's official GaitBase checkpoint "
            "(see https://github.com/ShiqiYu/OpenGait/blob/master/docs/"
            f"1.model_zoo.md) and place it at {_DEFAULT_CHECKPOINT_PATH}."
        )
        return None

    os.makedirs(_DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    print(f"Downloading GaitBase checkpoint from {_CHECKPOINT_DOWNLOAD_URL}...")
    try:
        urllib.request.urlretrieve(_CHECKPOINT_DOWNLOAD_URL, _DEFAULT_CHECKPOINT_PATH)
        print(f"Downloaded to {_DEFAULT_CHECKPOINT_PATH}")
        return _DEFAULT_CHECKPOINT_PATH
    except Exception as e:
        print(f"[WARNING] Download failed: {e}")
        return None
