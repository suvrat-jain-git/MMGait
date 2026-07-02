"""
identity_head.py — Identity Recognition Head with BNNeck

What this module does:
    Takes Fm' and Fk' (after graph interaction), projects each to 256 dims,
    concatenates them into a 512-dim fused feature, passes through FC + BNNeck,
    and produces:
        1. An identity embedding for triplet loss   (before BNNeck)
        2. Classification logits for CE loss         (after BNNeck)

Full flow:
    Fm' [B, 512]  →  Linear(512, 256)  →  Fm_proj [B, 256]
    Fk' [B, 512]  →  Linear(512, 256)  →  Fk_proj [B, 256]
                                               │
                                         concat [B, 512]
                                               │
                                         FC 512 → 512
                                               │
                                         embedding [B, 512]   ← triplet loss applied here
                                               │
                                           BNNeck
                                           (BN1d)
                                               │
                                         bn_feat [B, 512]
                                               │
                                         FC 512 → num_classes
                                               │
                                         logits [B, num_classes]  ← CE loss applied here

Why the projection layers (512 → 256 each):
    Without projection, the identity head receives 1024 dimensions.
    For FVG-B (which may have ~200 subjects in the training split),
    1024 dimensions is large relative to the task. The projection:
    1. Forces each branch to summarize itself more compactly.
    2. Reduces the chance that one branch (e.g. morphology) dominates
       the fused representation simply because it has higher-magnitude features.
    3. Makes the fusion symmetric: both branches contribute equally in size.

Why BNNeck (the ReID trick):
    Without BNNeck, there is a tension between the triplet loss and CE loss:
    - Triplet loss wants features spread across a metric space (no constraints on norm).
    - CE loss works best with normalized, bounded features.
    BNNeck resolves this by:
    - Computing triplet loss on the raw embedding (before BN) — preserves metric geometry.
    - Computing CE loss on the BN-normalized feature — stable softmax gradients.
    This is a well-validated trick from the person re-identification literature.

What is exposed for analysis:
    self.get_embedding(Fm_prime, Fk_prime)
        Returns the pre-BNNeck embedding — use this for t-SNE, feature similarity,
        and nearest-neighbor retrieval at test time.
"""

import torch
import torch.nn as nn


class IdentityHead(nn.Module):
    """
    Fused identity head with projection, BNNeck, and dual-loss outputs.
    """

    def __init__(self, node_dim=512, proj_dim=256, hidden_dim=512, num_classes=None):
        """
        Args:
            node_dim:    input dim of Fm' and Fk' (both 512)
            proj_dim:    output dim of each projection layer (256)
            hidden_dim:  FC layer size after concat (512 = proj_dim * 2)
            num_classes: number of training identities — MUST be set at init
        """
        super().__init__()
        assert num_classes is not None, (
            "num_classes must be specified — it equals the number of "
            "unique identities in the training split."
        )

        # Project Fk only: 512 → 256
        # Identity head uses Fk exclusively — Fm is not used here.
        # Fm is supervised by the gender head only.
        self.fm_proj = nn.Linear(node_dim, proj_dim)
        self.fk_proj = nn.Linear(node_dim, proj_dim)

        # bias=False: BNNeck immediately follows.
        # FC maps concat(Fm, Fk) → embedding
        fused_dim = proj_dim * 2
        self.fc = nn.Linear(fused_dim, hidden_dim, bias=False)

        # BNNeck: a single BatchNorm1d layer with no affine transformation
        # disabled... actually we KEEP affine=True (default) here.
        # The BN learns to re-scale and re-center the embedding for the
        # classifier. The key is that triplet loss sees the pre-BN embedding
        # while CE loss sees the post-BN feature.
        self.bnneck = nn.BatchNorm1d(hidden_dim)

        # Classifier: maps BN-normalised feature to identity logits.
        # bias=False: BNNeck already provides a learned shift via beta.
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def _embed(self, Fm_prime, Fk_prime):
        """
        Internal helper: returns the pre-BNNeck embedding [B, 512].

        Uses concat(Fm_prime, Fk_prime) — both branches contribute to identity.
        Orthogonality loss (not the identity head) enforces disentanglement,
        so there is no need to restrict the identity head to Fk only.

        Both branches projected to 256 dims then concatenated → 512 dim fused
        feature passed through FC to produce the identity embedding.
        """
        # Project each branch: [B, 512] → [B, 256]
        fm_proj = self.fm_proj(Fm_prime)
        fk_proj = self.fk_proj(Fk_prime)

        # Concat: [B, 512], then FC → [B, 512] embedding
        fused = torch.cat([fm_proj, fk_proj], dim=1)
        embedding = self.fc(fused)

        return embedding

    def forward(self, Fm_prime, Fk_prime):
        """
        Args:
            Fm_prime: [B, 512] — morphology after graph
            Fk_prime: [B, 512] — motion after graph

        Returns:
            embedding: [B, 512] — pre-BNNeck, use for triplet loss and retrieval
            logits:    [B, num_classes] — post-BNNeck, use for CE loss
        """
        # Shared path: projection + concat + FC
        embedding = self._embed(Fm_prime, Fk_prime)

        # BNNeck: normalise for classifier, preserve metric geometry for triplet.
        bn_feat = self.bnneck(embedding)

        # Classifier: [B, 512] -> [B, num_classes]
        logits = self.classifier(bn_feat)

        return embedding, logits

    def get_embedding(self, Fm_prime, Fk_prime):
        """
        Returns the pre-BNNeck embedding only.
        Use at test time for nearest-neighbour retrieval and analysis.

        At test time we do NOT want BNNeck normalization — we want the
        raw metric-space embedding that triplet loss shaped during training.

        Args:
            Fm_prime: [B, 512]
            Fk_prime: [B, 512]

        Returns:
            embedding: [B, 512]
        """
        return self._embed(Fm_prime, Fk_prime)
