"""
test_graph.py — Smoke tests for Bio-Kinematic Graph

Tests:
    - Input: Fm [B, 512], Fk [B, 512]
    - Output: Fm' [B, 512], Fk' [B, 512]
    - Alpha (learnable mixing weight) starts at correct value
    - Residual connection: output ≈ input when alpha=0
    - Gradient flows to both Fm and Fk
    - Cross-branch interaction: Fm' depends on Fk (and vice versa)
"""

import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.graph.bio_kinematic_graph import BioKinematicGraph


B, D = 4, 512


@pytest.fixture
def graph():
    return BioKinematicGraph(node_dim=D)


@pytest.fixture
def Fm():
    return torch.rand(B, D)


@pytest.fixture
def Fk():
    return torch.rand(B, D)


class TestBioKinematicGraph:

    def test_output_shapes(self, graph, Fm, Fk):
        Fm_prime, Fk_prime = graph(Fm, Fk)
        assert Fm_prime.shape == (B, D), \
            f"Fm' shape: expected [{B}, {D}], got {list(Fm_prime.shape)}"
        assert Fk_prime.shape == (B, D), \
            f"Fk' shape: expected [{B}, {D}], got {list(Fk_prime.shape)}"

    def test_output_finite(self, graph, Fm, Fk):
        Fm_prime, Fk_prime = graph(Fm, Fk)
        assert torch.isfinite(Fm_prime).all(), "Fm' contains NaN or Inf"
        assert torch.isfinite(Fk_prime).all(), "Fk' contains NaN or Inf"

    def test_alpha_initial_value(self, graph):
        """Alpha should initialise to 0.1."""
        assert abs(graph.alpha.item() - 0.1) < 1e-6, \
            f"Alpha should be 0.1, got {graph.alpha.item()}"

    def test_residual_when_alpha_zero(self, graph, Fm, Fk):
        """With alpha=0, output should equal input (pure residual)."""
        with torch.no_grad():
            graph.alpha.fill_(0.0)
            Fm_prime, Fk_prime = graph(Fm, Fk)
        assert torch.allclose(Fm_prime, Fm, atol=1e-5), \
            "Fm' != Fm when alpha=0 (residual connection broken)"
        assert torch.allclose(Fk_prime, Fk, atol=1e-5), \
            "Fk' != Fk when alpha=0 (residual connection broken)"

    def test_cross_branch_interaction(self, graph, Fk):
        """Fm' should depend on Fk — different Fm inputs should give different Fm'."""
        Fm1 = torch.rand(B, D)
        Fm2 = torch.rand(B, D)
        with torch.no_grad():
            Fm1_prime, _ = graph(Fm1, Fk)
            Fm2_prime, _ = graph(Fm2, Fk)
        assert not torch.allclose(Fm1_prime, Fm2_prime), \
            "Fm' is identical for different Fm inputs"

    def test_fk_influences_fm_prime(self, graph, Fm):
        """Fm' should change when Fk changes (cross-branch message passing)."""
        Fk1 = torch.rand(B, D)
        Fk2 = torch.rand(B, D)
        with torch.no_grad():
            graph.alpha.fill_(0.5)   # ensure alpha is non-zero
            Fm_prime1, _ = graph(Fm, Fk1)
            Fm_prime2, _ = graph(Fm, Fk2)
        assert not torch.allclose(Fm_prime1, Fm_prime2, atol=1e-4), \
            "Fm' does not depend on Fk — cross-branch interaction missing"

    def test_gradient_flows_to_both_inputs(self, graph):
        Fm = torch.rand(B, D, requires_grad=True)
        Fk = torch.rand(B, D, requires_grad=True)
        Fm_prime, Fk_prime = graph(Fm, Fk)
        (Fm_prime.sum() + Fk_prime.sum()).backward()
        assert Fm.grad is not None and torch.isfinite(Fm.grad).all()
        assert Fk.grad is not None and torch.isfinite(Fk.grad).all()

    def test_alpha_is_learnable(self, graph):
        """Alpha must be a learnable parameter."""
        param_names = [n for n, _ in graph.named_parameters()]
        assert 'alpha' in param_names, \
            f"alpha not in named_parameters: {param_names}"

    def test_parameter_count(self, graph):
        n = sum(p.numel() for p in graph.parameters())
        # Should be approximately 525K
        assert 400_000 < n < 700_000, \
            f"Unexpected parameter count: {n:,}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
