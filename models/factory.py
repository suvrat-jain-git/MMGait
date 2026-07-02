"""
factory.py — Model Construction From DatasetMeta + CLI Flags

build_model() is the single place that decides:
    - whether gender_head/age_head get built (from DatasetMeta)
    - whether the Bio-Kinematic Graph is enabled (from --no_graph flag)
    - which morphology backbone is used (from --morph_backbone flag --
      'custom' or 'gaitbase'; see models/backbones/gaitbase_backbone.py
      for the GaitBase integration and the GEI-as-single-frame design
      decision behind it)

Keeping this logic OUT of train.py keeps train.py focused on orchestration
(parse args -> load config -> get dataset -> build model -> train) rather
than knowing the details of how a model config dict gets assembled from
several yaml files plus runtime metadata. Every other entry point that
needs a model (evaluators, analysis scripts, the multi-seed runner) goes
through this same function, so model construction can never silently
diverge between training and evaluation.
"""

import yaml


def build_model_config(model_yaml_cfg, heads_yaml_cfg, dataset_meta,
                        use_graph=True, morph_backbone='custom'):
    """
    Assemble the full model config dict consumed by BioKinematicNet.

    Args:
        model_yaml_cfg:  parsed configs/model.yaml content
                         (i.e. the dict UNDER the 'model' key)
        heads_yaml_cfg:  parsed configs/heads.yaml content
                         (has 'gender' and 'age' keys)
        dataset_meta:    a datasets.base.DatasetMeta instance
        use_graph:       bool -- overrides model_yaml_cfg['graph']['enabled']
                         if explicitly passed (CLI --no_graph sets this False)
        morph_backbone:  'custom' or 'gaitbase' -- which morphology encoder
                         implementation to use. 'gaitbase' wiring is added
                         in a later stage; passing it now raises a clear
                         NotImplementedError rather than silently falling
                         back to 'custom'.

    Returns:
        dict -- ready to pass to BioKinematicNet(cfg)
    """
    cfg = dict(model_yaml_cfg)   # shallow copy, we only replace top keys

    # Graph ablation flag -- CLI takes precedence over yaml default
    cfg['graph'] = dict(cfg['graph'])
    cfg['graph']['enabled'] = use_graph

    # Identity head class count comes from the dataset, not the yaml
    cfg['identity'] = dict(cfg['identity'])
    cfg['identity']['num_classes'] = dataset_meta.num_identities

    # Conditional gender/age head injection -- the core of the
    # "dataset metadata determines what gets built" principle
    if dataset_meta.has_gender:
        cfg['gender'] = dict(heads_yaml_cfg['gender'])
    if dataset_meta.has_age:
        cfg['age'] = dict(heads_yaml_cfg['age'])

    # Morphology backbone selection -- sets cfg['morphology']['backbone'],
    # which models/biokinematic_net.py reads to decide whether to
    # instantiate MorphologyEncoder (custom) or GaitBaseBackbone
    # (gaitbase). See models/backbones/gaitbase_backbone.py for the
    # integration-decision discussion (GEI-as-single-frame, not raw
    # sequence) and the pretrained-weight fallback behaviour.
    cfg['morphology'] = dict(cfg['morphology'])
    if morph_backbone in ('custom', 'gaitbase'):
        cfg['morphology']['backbone'] = morph_backbone
    else:
        raise ValueError(
            f"Unknown morph_backbone '{morph_backbone}'. "
            f"Valid options: 'custom', 'gaitbase'."
        )

    return cfg


def build_model(model_yaml_path, heads_yaml_path, dataset_meta,
                 use_graph=True, morph_backbone='custom', device='cpu'):
    """
    Convenience wrapper: load the yaml files, assemble the config, and
    construct the model in one call. Most callers (train.py, evaluators)
    should use this rather than calling build_model_config() directly.

    Returns:
        BioKinematicNet instance, moved to `device`
    """
    from models.biokinematic_net import BioKinematicNet

    with open(model_yaml_path) as f:
        model_yaml_cfg = yaml.safe_load(f)['model']
    with open(heads_yaml_path) as f:
        heads_yaml_cfg = yaml.safe_load(f)

    cfg = build_model_config(
        model_yaml_cfg, heads_yaml_cfg, dataset_meta,
        use_graph=use_graph, morph_backbone=morph_backbone,
    )
    model = BioKinematicNet(cfg)
    return model.to(device)
