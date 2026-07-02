"""
bio_kinematic_graph.py — Two-Node Bio-Kinematic Interaction Graph

What this module does:
    Takes Fm [B, 512] and Fk [B, 512] and allows each branch to
    incorporate information from the other via a single round of
    residual message passing.

    Fm' = Fm + alpha * Wm(Fk)
    Fk' = Fk + alpha * Wk(Fm)

    Where:
        Wm: Linear(512, 512) — projects motion features into morphology space
        Wk: Linear(512, 512) — projects morphology features into motion space
        alpha: learnable scalar, initialized to 0.1

    Output:
        Fm': [B, 512]
        Fk': [B, 512]

Why two nodes, not a full GNN:
    There are exactly two entities in this system: morphology and motion.
    There are no long-range relationships to model across many nodes.
    A full GNN with stacked layers would add optimization complexity
    with no scientific justification. One message-passing round is enough
    to test the hypothesis: does morphology benefit from knowing about motion
    and vice versa?

Why residual (Fm + message) not replacement (message only):
    The residual connection preserves the original morphology and motion
    signals. The message is an additive correction — a small piece of
    context from the other branch. This is easy to interpret:
    if alpha * Wm(Fk) is small, the branch is mostly self-sufficient.
    if it is large, the branches are tightly coupled.

Why alpha initialized to 0.1:
    At initialization, Fm and Fk are random. Without scaling, the message
    Wm(Fk) has the same magnitude as Fm itself, which can destabilize early
    training by causing the branches to immediately contaminate each other
    before they've learned anything meaningful.
    alpha=0.1 means the message starts as a gentle nudge, not a shout.
    The network can increase alpha if it learns that cross-branch
    interaction is useful.

Why alpha is learnable:
    It allows the network to decide how much morphology and motion should
    interact. If the disentanglement hypothesis is correct, we expect alpha
    to remain small. If alpha grows large, the branches are coupling —
    which is itself a useful scientific observation.

What you can inspect after training:
    self.alpha.item()               — overall interaction strength
    norm(Wm(Fk)) / norm(Fm)        — relative message magnitude for morphology
    norm(Wk(Fm)) / norm(Fk)        — relative message magnitude for motion
"""

import torch
import torch.nn as nn


class BioKinematicGraph(nn.Module):
    """
    Single-round bidirectional residual message passing between
    morphology node (Fm) and motion node (Fk).

    This is the component that connects the two disentangled branches.
    """

    def __init__(self, node_dim=512, alpha_init=0.1):
        """
        Args:
            node_dim:   dimensionality of both Fm and Fk (must match)
            alpha_init: initial value of the learnable interaction scale
        """
        super().__init__()

        # Wm: projects Fk (motion) into morphology space.
        # Produces the message that motion sends to morphology.
        # "Given what motion has learned, what is relevant for morphology to know?"
        # Linear has no bias by default — we add one here because there is no
        # BatchNorm after the graph, so the bias provides the learned offset.
        self.Wm = nn.Linear(node_dim, node_dim)

        # Wk: projects Fm (morphology) into motion space.
        # Produces the message that morphology sends to motion.
        # "Given what morphology has learned, what is relevant for motion to know?"
        self.Wk = nn.Linear(node_dim, node_dim)

        # Learnable scalar interaction strength.
        # nn.Parameter makes it part of the model's parameter set —
        # it gets updated by the optimizer like any weight.
        # torch.tensor(alpha_init) wraps the float as a 0-dim tensor.
        # Initialized to 0.1 so messages start as small corrections,
        # not full-magnitude perturbations.
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, Fm, Fk):
        """
        Args:
            Fm: [B, 512] — morphology features from MorphologyEncoder
            Fk: [B, 512] — motion features from MotionEncoder

        Returns:
            Fm_prime: [B, 512] — morphology updated with motion context
            Fk_prime: [B, 512] — motion updated with morphology context
        """
        # Motion → Morphology message
        # Wm(Fk): what motion wants to tell morphology — [B, 512]
        # Fm + alpha * Wm(Fk): morphology keeps its own signal,
        # receives a scaled additive correction from motion.
        Fm_prime = Fm + self.alpha * self.Wm(Fk)

        # Morphology → Motion message
        # Wk(Fm): what morphology wants to tell motion — [B, 512]
        # Fk + alpha * Wk(Fm): motion keeps its own signal,
        # receives a scaled additive correction from morphology.
        #
        # Important: we use the ORIGINAL Fm here, not Fm_prime.
        # Using Fm_prime would create an implicit ordering dependency —
        # morphology would be updated before motion, breaking symmetry.
        # Using the original Fm means both messages are computed from
        # the pre-interaction state, which is the correct interpretation
        # of a single synchronous message-passing round.
        Fk_prime = Fk + self.alpha * self.Wk(Fm)

        return Fm_prime, Fk_prime

    def message_stats(self, Fm, Fk):
        """
        Diagnostic method for analysis scripts.
        Returns the magnitude of each message relative to the original feature.

        Use this after training to assess how much the branches depend on
        each other. Small ratios confirm disentanglement is working.
        Large ratios suggest the branches are tightly coupled.

        Args:
            Fm: [B, 512]
            Fk: [B, 512]

        Returns:
            dict with keys:
                'alpha':            current interaction strength (scalar)
                'motion_to_morph':  mean ||alpha * Wm(Fk)|| / ||Fm|| per sample
                'morph_to_motion':  mean ||alpha * Wk(Fm)|| / ||Fk|| per sample
        """
        with torch.no_grad():
            msg_to_morph = self.alpha * self.Wm(Fk)   # [B, 512]
            msg_to_motion = self.alpha * self.Wk(Fm)  # [B, 512]

            # Compute per-sample L2 norms, then average over batch
            ratio_m = (msg_to_morph.norm(dim=1) / Fm.norm(dim=1)).mean().item()
            ratio_k = (msg_to_motion.norm(dim=1) / Fk.norm(dim=1)).mean().item()

        return {
            'alpha':           self.alpha.item(),
            'motion_to_morph': ratio_m,
            'morph_to_motion': ratio_k,
        }
