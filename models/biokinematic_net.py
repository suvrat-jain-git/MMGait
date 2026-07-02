"""
biokinematic_net.py — BioKinematicNet: Full Model Assembly (V2)

Top-level module wiring together morphology branch, motion branch, the
optional Bio-Kinematic Graph, and conditionally-instantiated task heads.

Full forward pass:

    Input X: [B, T, 1, H, W]
        |
        +-- GEI generator -----------------------> [B, 1, H, W]
        |        |
        |        v (beta-scaled subtraction)
        |   MorphologyEncoder
        |        |
        |   Fm [B, 512]
        |
        +-- Motion generator ---(minus beta*GEI)--> [B, 1, T-1, H, W]
                                                          |
                                                   MotionEncoder
                                                          |
                                                    Fk [B, 512]

    [use_graph flag]
      True:  Fm' = Fm + alpha*Wm(Fk)
             Fk' = Fk + alpha*Wk(Fm)         (BioKinematicGraph)
      False: Fm' = Fm, Fk' = Fk              (passthrough -- ablation)

      +-----------------+-------------------+
      |                 |                   |
  GenderHead         AgeHead          IdentityHead
  (Fm' only)        (Fm' only)        (Fm' + Fk', fused)
      |                 |                   |
  gender_logits   age_bin_logits,    embedding, id_logits
   [B,2]          age_value [B]      [B,512] [B,num_classes]

CONDITIONAL HEAD INSTANTIATION -- the key change from V1:
    gender_head is only built if cfg['gender'] is not None (the caller,
    typically a build_model() factory reading DatasetMeta, omits this
    key entirely when the active dataset has no gender labels).
    age_head is only built if cfg['age'] is not None, same principle.
    identity_head is always built -- every dataset in this project
    supports identity (it is the core task).

    This means a model built for FVG-B (no age labels) literally has no
    age_head submodule -- not a head with frozen/unused weights, not a
    head trained against dummy labels, but no nn.Module for it at all.
    model.named_parameters() for an FVG-B model will never list any
    age_head.* parameters.

GRAPH ABLATION -- use_graph flag:
    cfg['graph']['enabled'] (default True) controls whether the
    BioKinematicGraph module is constructed and used. When False, Fm'=Fm
    and Fk'=Fk are passed straight through with no graph interaction at
    all -- not alpha forced to 0 (which would still run the (now-wasted)
    matrix multiplies through Wm/Wk), but the graph module is not even
    instantiated. This was added because alpha consistently converged to
    ~0.1 (its init value) across all V1 multi-seed experiments, raising
    the open question of whether the graph contributes anything beyond a
    direct concat -- this flag makes that an explicit, reportable ablation.

Outputs (mode='train'):
    Always present:
        id_logits, embedding, Fm, Fk, Fm_prime, Fk_prime
    Conditionally present (only if the corresponding head was built):
        gender_logits  [B, 2]
        age_bin_logits [B, num_age_bins]
        age_value      [B]

Outputs (mode='inference'):
    embedding: [B, 512]  -- for gallery matching, fastest path

Outputs (mode='eval'):
    dict with embedding, and gender_logits/age_bin_logits/age_value
    IF the corresponding head exists on this model instance.
"""

import torch.nn as nn

from models.morphology.gei import generate_gei
from models.morphology.morphology_encoder import MorphologyEncoder
from models.backbones.gaitbase_backbone import GaitBaseBackbone
from models.motion.motion_generator import generate_motion
from models.motion.motion_encoder import MotionEncoder
from models.graph.bio_kinematic_graph import BioKinematicGraph
from models.heads.gender_head import GenderHead
from models.heads.age_head import AgeHead
from models.heads.identity_head import IdentityHead


class BioKinematicNet(nn.Module):
    """
    Full BioKinematicNet model with conditional gender/age heads and an
    optional (flag-controlled) Bio-Kinematic Graph.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: model config dict. Expected structure:

                cfg['morphology']['in_channels']     # int, default 1
                cfg['morphology']['channels']        # list, e.g. [1,32,64,128,256,512]

                cfg['motion']['in_channels']          # int, default 1
                cfg['motion']['channels']             # list, same shape as above
                cfg['motion'].get('static_suppression_beta', 0.5)
                                                       # float in [0,1]; how much
                                                       # of GEI to subtract from
                                                       # the motion input before
                                                       # the motion encoder

                cfg['graph'].get('enabled', True)     # bool -- the ablation flag
                cfg['graph']['node_dim']              # int, default 512
                cfg['graph']['alpha_init']            # float, default 0.1
                                                       # (only read if enabled=True)

                cfg['projection']['in_dim']           # int, default 512
                cfg['projection']['out_dim']          # int, default 256
                cfg['identity']['hidden_dim']         # int, default 512
                cfg['identity']['num_classes']        # int, REQUIRED

                cfg.get('gender')                     # dict OR None/absent.
                                                       # If None/absent: no
                                                       # gender_head is built.
                                                       # If present:
                                                       #   ['in_dim'] default 512
                                                       #   ['hidden_dim'] default 128
                                                       #   ['num_classes'] default 2

                cfg.get('age')                        # dict OR None/absent.
                                                       # If None/absent: no
                                                       # age_head is built.
                                                       # If present:
                                                       #   ['in_dim'] default 512
                                                       #   ['hidden_dim'] default 256
                                                       #   ['num_bins'] default 7
        """
        super().__init__()

        # -- Morphology branch (backbone selectable: 'custom' or 'gaitbase') --
        # cfg['morphology']['backbone'] is set by models/factory.py from
        # the --morph_backbone CLI flag. Defaults to 'custom' if absent,
        # so existing configs/checkpoints built before this flag existed
        # still construct identically (backward compatible).
        morph_backbone = cfg['morphology'].get('backbone', 'custom')
        if morph_backbone == 'custom':
            self.morph_encoder = MorphologyEncoder(
                in_channels=cfg['morphology']['in_channels'],
                channels=cfg['morphology']['channels'],
            )
        elif morph_backbone == 'gaitbase':
            self.morph_encoder = GaitBaseBackbone(
                in_channels=cfg['morphology']['in_channels'],
                out_dim=cfg['morphology']['channels'][-1],
                pretrained=cfg['morphology'].get('gaitbase_pretrained', True),
                checkpoint_path=cfg['morphology'].get('gaitbase_checkpoint_path'),
            )
        else:
            raise ValueError(
                f"Unknown morphology backbone '{morph_backbone}'. "
                f"Valid options: 'custom', 'gaitbase'."
            )
        self.morph_backbone_name = morph_backbone   # for introspection /
                                                      # count_parameters labeling

        # -- Motion branch ------------------------------------------------
        self.motion_encoder = MotionEncoder(
            in_channels=cfg['motion']['in_channels'],
            channels=cfg['motion']['channels'],
        )
        self.static_suppression_beta = cfg['motion'].get(
            'static_suppression_beta', 0.5
        )

        # -- Bio-kinematic graph (ablation flag) ---------------------------
        self.use_graph = cfg['graph'].get('enabled', True)
        if self.use_graph:
            self.graph = BioKinematicGraph(
                node_dim=cfg['graph']['node_dim'],
                alpha_init=cfg['graph']['alpha_init'],
            )
        else:
            self.graph = None   # explicit -- no graph module exists at all

        # -- Identity head (always present, fused Fm'+Fk') -----------------
        self.identity_head = IdentityHead(
            node_dim=cfg['projection']['in_dim'],
            proj_dim=cfg['projection']['out_dim'],
            hidden_dim=cfg['identity']['hidden_dim'],
            num_classes=cfg['identity']['num_classes'],
        )

        # -- Gender head (CONDITIONAL) --------------------------------------
        gender_cfg = cfg.get('gender')
        if gender_cfg is not None:
            self.gender_head = GenderHead(
                in_dim=gender_cfg.get('in_dim', 512),
                hidden_dim=gender_cfg.get('hidden_dim', 128),
                num_classes=gender_cfg.get('num_classes', 2),
            )
        else:
            self.gender_head = None   # explicit -- no params, no submodule

        # -- Age head (CONDITIONAL) ------------------------------------------
        age_cfg = cfg.get('age')
        if age_cfg is not None:
            self.age_head = AgeHead(
                in_dim=age_cfg.get('in_dim', 512),
                hidden_dim=age_cfg.get('hidden_dim', 256),
                num_bins=age_cfg.get('num_bins', 7),
            )
        else:
            self.age_head = None

    def forward(self, x, mode='train'):
        """
        Args:
            x:    [B, T, 1, H, W] -- raw silhouette sequence
            mode: 'train' | 'inference' | 'eval'

        Returns:
            mode='train':     dict, always has id_logits/embedding/Fm/Fk/
                              Fm_prime/Fk_prime, conditionally has
                              gender_logits / age_bin_logits+age_value
                              depending on which heads this model
                              instance was built with.
            mode='inference': embedding [B, 512] only
            mode='eval':      dict with embedding, plus conditional
                              gender_logits / age_bin_logits+age_value
        """
        assert mode in ('train', 'inference', 'eval'), \
            f"mode must be 'train', 'inference', or 'eval', got '{mode}'"

        # -- Step 1: Generate GEI and motion volume in parallel -------------
        gei    = generate_gei(x)        # [B, 1, H, W]
        motion = generate_motion(x)     # [B, 1, T-1, H, W]

        # -- Step 2: Encode each branch independently -----------------------
        Fm = self.morph_encoder(gei)    # [B, 512]

        # Static suppression: remove GEI component from motion input,
        # forcing the motion encoder to explain only dynamic content.
        motion_suppressed = motion - self.static_suppression_beta * gei.unsqueeze(2)
        Fk = self.motion_encoder(motion_suppressed)   # [B, 512]

        # -- Step 3: Bio-kinematic graph interaction (ablation flag) --------
        if self.use_graph:
            Fm_prime, Fk_prime = self.graph(Fm, Fk)
        else:
            # Passthrough -- no cross-branch interaction at all.
            Fm_prime, Fk_prime = Fm, Fk

        # -- Step 4: Identity head (always) ----------------------------------
        embedding, id_logits = self.identity_head(Fm_prime, Fk_prime)

        # -- Step 5: Conditional heads -----------------------------------------
        gender_logits = self.gender_head(Fm_prime) if self.gender_head is not None else None

        if self.age_head is not None:
            age_bin_logits, age_value = self.age_head(Fm_prime)
        else:
            age_bin_logits, age_value = None, None

        # -- Output assembly ---------------------------------------------------
        if mode == 'inference':
            return embedding

        if mode == 'eval':
            out = {'embedding': embedding}
            if gender_logits is not None:
                out['gender_logits'] = gender_logits
            if age_bin_logits is not None:
                out['age_bin_logits'] = age_bin_logits
                out['age_value']      = age_value
            return out

        # mode == 'train'
        out = {
            'id_logits':  id_logits,
            'embedding':  embedding,
            'Fm':         Fm,
            'Fk':         Fk,
            'Fm_prime':   Fm_prime,
            'Fk_prime':   Fk_prime,
        }
        if gender_logits is not None:
            out['gender_logits'] = gender_logits
        if age_bin_logits is not None:
            out['age_bin_logits'] = age_bin_logits
            out['age_value']      = age_value
        return out

    def get_graph_stats(self, x):
        """
        Convenience method for analysis scripts. Returns graph interaction
        statistics (alpha, cross-branch message magnitudes).

        Raises RuntimeError if use_graph=False for this model instance --
        there is no graph to report stats on.
        """
        if not self.use_graph:
            raise RuntimeError(
                "get_graph_stats() called on a model built with "
                "graph.enabled=False -- there is no graph module to "
                "report statistics for. This is expected for the "
                "no-graph ablation; do not call this method for those runs."
            )
        gei    = generate_gei(x)
        motion = generate_motion(x)
        motion_suppressed = motion - self.static_suppression_beta * gei.unsqueeze(2)
        Fm     = self.morph_encoder(gei)
        Fk     = self.motion_encoder(motion_suppressed)
        return self.graph.message_stats(Fm, Fk)

    def count_parameters(self):
        """
        Returns parameter count broken down by component. Components
        that were not instantiated (gender_head/age_head/graph when
        conditionally disabled) are simply absent from the breakdown
        dict rather than reported as 0 -- this makes it immediately
        visible in logs which heads a given run actually has.
        """
        def count(module):
            return sum(p.numel() for p in module.parameters())

        breakdown = {
            f'morph_encoder ({self.morph_backbone_name})': count(self.morph_encoder),
            'motion_encoder': count(self.motion_encoder),
            'identity_head':  count(self.identity_head),
        }
        if self.graph is not None:
            breakdown['graph'] = count(self.graph)
        if self.gender_head is not None:
            breakdown['gender_head'] = count(self.gender_head)
        if self.age_head is not None:
            breakdown['age_head'] = count(self.age_head)

        breakdown['total'] = sum(breakdown.values())
        return breakdown
