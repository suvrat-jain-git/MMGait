"""
test_motion.py — Smoke tests for motion branch

Tests:
    - MotionGenerator: [B, T, 1, H, W] → [B, 1, T-1, H, W]
    - MotionEncoder:   [B, 1, T-1, H, W] → [B, 512]
    - Static sequence (repeated frame) → near-zero motion
    - Gradient flows back to input
"""

import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.motion.motion_generator import generate_motion
from models.motion.motion_encoder import MotionEncoder


B, T, C, H, W = 4, 30, 1, 64, 64


@pytest.fixture
def sequence():
    return torch.rand(B, T, C, H, W)


@pytest.fixture
def motion(sequence):
    return generate_motion(sequence)


@pytest.fixture
def encoder():
    return MotionEncoder()


# ── MotionGenerator tests ────────────────────────────────────────────────────

class TestMotionGenerator:

    def test_output_shape(self, sequence):
        mot = generate_motion(sequence)
        assert mot.shape == (B, 1, T - 1, H, W), \
            f"Expected [{B}, 1, {T-1}, {H}, {W}], got {list(mot.shape)}"

    def test_output_finite(self, sequence):
        mot = generate_motion(sequence)
        assert torch.isfinite(mot).all()

    def test_output_nonnegative(self, sequence):
        """abs(X[t] - X[t-1]) is always >= 0."""
        mot = generate_motion(sequence)
        assert mot.min() >= 0.0, \
            f"Motion map has negative values: min={mot.min():.4f}"

    def test_static_sequence_zero_motion(self):
        """Repeated frame → motion should be exactly zero."""
        frame  = torch.rand(1, 1, H, W)
        seq    = frame.unsqueeze(1).expand(1, T, 1, H, W).clone()
        motion = generate_motion(seq)
        assert motion.abs().max() < 1e-6, \
            f"Static sequence should produce zero motion, got max={motion.abs().max():.6f}"

    def test_motion_is_frame_difference(self, sequence):
        """Motion should equal abs(X[t+1] - X[t]) for each t."""
        mot = generate_motion(sequence)
        # mot: [B, 1, T-1, H, W], sequence: [B, T, 1, H, W]
        for t in range(T - 1):
            expected = (sequence[:, t + 1] - sequence[:, t]).abs()  # [B, 1, H, W]
            actual = mot[:, :, t, :, :]  # [B, 1, H, W]
            assert torch.allclose(actual, expected, atol=1e-6), \
                f"Frame difference mismatch at t={t}"
    def test_gradient_flows(self, sequence):
        seq = sequence.clone().requires_grad_(True)
        mot = generate_motion(seq)
        mot.sum().backward()
        assert seq.grad is not None
        assert torch.isfinite(seq.grad).all()


# ── MotionEncoder tests ──────────────────────────────────────────────────────

class TestMotionEncoder:

    def test_output_shape(self, encoder, motion):
        out = encoder(motion)
        assert out.shape == (B, 512), \
            f"Expected [{B}, 512], got {list(out.shape)}"

    def test_channels_length_mismatch_rejected(self):
        """
        Regression test mirroring TestMorphologyEncoder's identical
        test in tests/test_morphology.py -- same footgun, same fix,
        applied independently to MotionEncoder (see
        models/motion/motion_encoder.py's assertion).
        """
        with pytest.raises(AssertionError, match="channels has 6 entries"):
            MotionEncoder(in_channels=1, channels=[1, 32, 64, 128, 256, 512])

        with pytest.raises(AssertionError):
            MotionEncoder(in_channels=1, channels=[32, 64, 128])

        MotionEncoder(in_channels=1, channels=[32, 64, 128, 256, 512])

    def test_output_finite(self, encoder, motion):
        out = encoder(motion)
        assert torch.isfinite(out).all()

    def test_output_nonzero(self, encoder, motion):
        out = encoder(motion)
        assert out.abs().mean() > 0

    def test_static_input_produces_output(self, encoder):
        """Even zero motion should produce a valid (non-crash) output."""
        zero_motion = torch.zeros(B, 1, T - 1, H, W)
        out = encoder(zero_motion)
        assert out.shape == (B, 512)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, encoder, motion):
        inp = motion.clone().requires_grad_(True)
        out = encoder(inp)
        out.sum().backward()
        assert inp.grad is not None
        assert torch.isfinite(inp.grad).all()

    def test_parameter_count(self, encoder):
        n = sum(p.numel() for p in encoder.parameters())
        # Should be approximately 4.7M
        assert 3_000_000 < n < 7_000_000, \
            f"Unexpected parameter count: {n:,}"

    def test_eval_mode(self, encoder, motion):
        encoder.eval()
        with torch.no_grad():
            out = encoder(motion)
        assert out.shape == (B, 512)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
