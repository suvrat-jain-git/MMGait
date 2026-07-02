"""
test_morphology.py — Smoke tests for morphology branch

Tests:
    - GEI generator: [B, T, 1, H, W] → [B, 1, H, W]
    - MorphologyEncoder: [B, 1, H, W] → [B, 512]
    - Output is finite and non-zero
    - Gradient flows back to input
"""

import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.morphology.gei import generate_gei
from models.morphology.morphology_encoder import MorphologyEncoder


# ── Fixtures ────────────────────────────────────────────────────────────────

B, T, C, H, W = 4, 30, 1, 64, 64  # small spatial size for speed


@pytest.fixture
def sequence():
    """Random silhouette sequence [B, T, 1, H, W]."""
    return torch.rand(B, T, C, H, W)


@pytest.fixture
def gei(sequence):
    """GEI computed from the sequence."""
    return generate_gei(sequence)


@pytest.fixture
def encoder():
    """MorphologyEncoder with default config."""
    return MorphologyEncoder()


# ── GEI tests ───────────────────────────────────────────────────────────────

class TestGEI:

    def test_output_shape(self, sequence):
        gei = generate_gei(sequence)
        assert gei.shape == (B, 1, H, W), \
            f"Expected [{B}, 1, {H}, {W}], got {list(gei.shape)}"

    def test_output_is_mean(self, sequence):
        """GEI should equal the temporal mean of the sequence."""
        gei      = generate_gei(sequence)
        expected = sequence.mean(dim=1)
        assert torch.allclose(gei, expected, atol=1e-6), \
            "GEI is not the temporal mean of the sequence"

    def test_output_finite(self, sequence):
        gei = generate_gei(sequence)
        assert torch.isfinite(gei).all(), "GEI contains NaN or Inf"

    def test_output_range(self, sequence):
        """With input in [0, 1], GEI should also be in [0, 1]."""
        gei = generate_gei(sequence)
        assert gei.min() >= 0.0 and gei.max() <= 1.0, \
            f"GEI out of [0,1] range: min={gei.min():.4f} max={gei.max():.4f}"

    def test_static_sequence_equals_frame(self):
        """Repeated frame → GEI == that frame."""
        frame = torch.rand(1, 1, H, W)
        seq   = frame.unsqueeze(1).expand(1, T, 1, H, W)
        gei   = generate_gei(seq)
        assert torch.allclose(gei, frame, atol=1e-6)

    def test_gradient_flows(self, sequence):
        seq = sequence.clone().requires_grad_(True)
        gei = generate_gei(seq)
        gei.sum().backward()
        assert seq.grad is not None
        assert torch.isfinite(seq.grad).all()


# ── MorphologyEncoder tests ──────────────────────────────────────────────────

class TestMorphologyEncoder:

    def test_output_shape(self, encoder, gei):
        out = encoder(gei)
        assert out.shape == (B, 512), \
            f"Expected [{B}, 512], got {list(out.shape)}"

    def test_channels_length_mismatch_rejected(self):
        """
        Regression test for a real bug found during V2 integration
        testing: passing a `channels` list with the wrong length (e.g.
        accidentally prepending in_channels to the list) used to
        silently truncate via zip() to a shallower network with the
        wrong final output dimension, producing a confusing matmul
        shape error several layers downstream in the Bio-Kinematic
        Graph rather than a clear error at the point of the mistake.
        See models/morphology/morphology_encoder.py's assertion.
        """
        # The exact malformed input that caused the original bug:
        # in_channels (1) incorrectly prepended to the 5-entry channels list
        with pytest.raises(AssertionError, match="channels has 6 entries"):
            MorphologyEncoder(in_channels=1, channels=[1, 32, 64, 128, 256, 512])

        # Too few entries should also be rejected
        with pytest.raises(AssertionError):
            MorphologyEncoder(in_channels=1, channels=[32, 64, 128])

        # Correct usage (exactly 5 entries) must NOT raise
        MorphologyEncoder(in_channels=1, channels=[32, 64, 128, 256, 512])

    def test_output_finite(self, encoder, gei):
        out = encoder(gei)
        assert torch.isfinite(out).all(), "Encoder output contains NaN or Inf"

    def test_output_nonzero(self, encoder, gei):
        out = encoder(gei)
        assert out.abs().mean() > 0, "Encoder output is all zeros"

    def test_gradient_flows(self, encoder, gei):
        inp = gei.clone().requires_grad_(True)
        out = encoder(inp)
        out.sum().backward()
        assert inp.grad is not None
        assert torch.isfinite(inp.grad).all()

    def test_batch_independence(self, encoder):
        """Batch independence holds in eval mode (BN uses running stats)."""
        encoder.eval()
        with torch.no_grad():
            x1 = torch.rand(1, 1, H, W)
            x2 = torch.rand(B, 1, H, W)
            x2[0] = x1[0]
            out1 = encoder(x1)
            out2 = encoder(x2)
        encoder.train()
        assert torch.allclose(out1, out2[:1], atol=1e-5), \
            "Encoder output depends on batch elements in eval mode"
    def test_parameter_count(self, encoder):
        n = sum(p.numel() for p in encoder.parameters())
        # Should be approximately 1.57M
        assert 1_000_000 < n < 3_000_000, \
            f"Unexpected parameter count: {n:,}"

    def test_eval_vs_train_mode(self, encoder, gei):
        """Ensure model can switch between train and eval without crashing."""
        encoder.train()
        out_train = encoder(gei)
        encoder.eval()
        with torch.no_grad():
            out_eval = encoder(gei)
        assert out_train.shape == out_eval.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
