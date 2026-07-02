"""
test_gaitbase_backbone.py — GaitBase Backbone Ablation Tests

Covers models/backbones/gaitbase_backbone.py: the contract match against
MorphologyEncoder, the fallback behaviour when pretrained weights are
unavailable, and the BasicBlock2D residual block in isolation.
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.backbones.gaitbase_backbone import (
    GaitBaseBackbone, BasicBlock2D, download_gaitbase_checkpoint,
)
from models.morphology.morphology_encoder import MorphologyEncoder


B, H, W = 4, 64, 64


class TestBasicBlock2D:

    def test_output_shape_no_downsample(self):
        block = BasicBlock2D(64, 64, stride=1)
        x = torch.rand(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_shape_with_downsample(self):
        """Stride=2 and channel change both require the identity
        shortcut's downsample path to be active."""
        block = BasicBlock2D(64, 128, stride=2)
        assert block.downsample is not None
        x = torch.rand(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)

    def test_gradient_flows(self):
        block = BasicBlock2D(32, 32, stride=1)
        x = torch.rand(2, 32, 8, 8, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestGaitBaseBackbone:

    def test_output_contract_matches_morphology_encoder(self):
        """
        The core integration property: GaitBaseBackbone must be an
        exact drop-in replacement for MorphologyEncoder -- identical
        [B,1,H,W] -> [B,512] contract, since models/factory.py and
        models/biokinematic_net.py swap between them purely based on
        the --morph_backbone flag with no other code changes.
        """
        custom   = MorphologyEncoder(in_channels=1, channels=[32,64,128,256,512])
        gaitbase = GaitBaseBackbone(in_channels=1, out_dim=512)

        x = torch.rand(B, 1, H, W)
        out_custom   = custom(x)
        out_gaitbase = gaitbase(x)

        assert out_custom.shape == out_gaitbase.shape == (B, 512)

    def test_gradient_flow(self):
        backbone = GaitBaseBackbone(in_channels=1, out_dim=512)
        x = torch.rand(2, 1, H, W, requires_grad=True)
        out = backbone(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_multiple_resolutions(self):
        """AdaptiveAvgPool2d means this must work at any spatial size,
        not just the 64x64 default -- same robustness property as
        MorphologyEncoder."""
        backbone = GaitBaseBackbone(in_channels=1, out_dim=512)
        for res in [32, 64, 128]:
            x = torch.rand(2, 1, res, res)
            out = backbone(x)
            assert out.shape == (2, 512)

    def test_out_dim_mismatch_rejected(self):
        """The assertion guarding channels[-1] == out_dim must fire if
        someone changes one without the other."""
        with pytest.raises(AssertionError):
            GaitBaseBackbone(in_channels=1, out_dim=256)   # channels[-1]
                                                              # is hardcoded 512

    def test_pretrained_fallback_no_crash(self, capsys):
        """
        With no download URL configured (the honest default state --
        see module docstring), pretrained=True must fall back to
        random initialisation with a clear printed warning, never
        silently claim success or crash.
        """
        backbone = GaitBaseBackbone(in_channels=1, out_dim=512, pretrained=True)
        captured = capsys.readouterr()
        assert 'RANDOM INITIALISATION' in captured.out
        # Model must still be fully usable despite the fallback
        x = torch.rand(2, 1, H, W)
        out = backbone(x)
        assert out.shape == (2, 512)
        assert torch.isfinite(out).all()

    def test_download_checkpoint_returns_none_when_unconfigured(self):
        """
        With _CHECKPOINT_DOWNLOAD_URL deliberately unset (see module
        docstring's honest explanation of why), this must return None
        rather than raise or silently produce a fake path.
        """
        result = download_gaitbase_checkpoint()
        assert result is None or isinstance(result, str)
        # If a real checkpoint happens to be cached locally from a
        # previous manual download, result may be a valid path --
        # either outcome is acceptable, but it must never be anything
        # other than None or a real existing path.
        if result is not None:
            import os
            assert os.path.exists(result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
