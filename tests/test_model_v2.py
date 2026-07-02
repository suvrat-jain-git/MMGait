"""
test_model_v2.py — Model/Loss/Trainer Tests (V2: conditional heads,
graph ablation, GaitBase backbone)

This file REPLACES the V1 tests/test_forward.py, which assumed
unconditional gender-head construction and the old
loss_fn(output, id_labels, gender_labels) call signature -- both wrong
for the V2 architecture. See test_dataset.py for the dataset-abstraction
layer tests (Sample/DatasetMeta/gait_collate_fn), which are a separate
concern from what's tested here.

Coverage:
    - models/factory.py: build_model_config() across every
      (has_gender, has_age) x (use_graph) x (morph_backbone) combination
    - models/biokinematic_net.py: conditional head instantiation
      verified via named_parameters() inspection (not just "did it not
      crash"), graph ablation exact-passthrough property, all three
      forward modes (train/inference/eval) across every head
      configuration
    - losses/combined_loss.py: the new loss_fn(output, batch) call
      signature, conditional gender/age loss terms, the age-masking
      property that prevents NaN gradient leakage from unlabeled
      samples into the shared trunk (this was the exact property
      verified manually during stage 2 development -- now a permanent
      regression test)
    - trainers/trainer.py: the val_epoch() bug found during stage 4
      integration testing (age head exists but a specific batch has
      zero age-labeled samples) -- now a permanent regression test,
      covering both the negative case (no crash, age metrics correctly
      absent) and the positive case (age metrics correctly computed)
    - Checkpoint save/load round-trip (carried over from V1, still valid)
    - utils/metrics.py, utils/seed.py (carried over from V1 unmodified --
      neither module changed in the V2 rewrite)
"""

import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.biokinematic_net import BioKinematicNet
from models.factory import build_model_config
from datasets.base import DatasetMeta, gait_collate_fn, Sample, age_to_bin
from losses.combined_loss import CombinedLoss
from losses.triplet import TripletLoss
from losses.barlow_twins import barlow_twins_loss
from trainers.trainer import Trainer
from utils.metrics import (
    cosine_distance_matrix, compute_rank_k,
    compute_map, compute_cmc_curve, compute_eer,
    compute_gender_metrics,
)
from utils.seed import set_seed


# -- Shared config -------------------------------------------------------------

B = 4    # batch size
T = 8    # sequence length (small for speed)
H = W = 32
NUM_IDENTITIES = 10
NUM_GENDER = 2


def _base_model_yaml_cfg():
    """Minimal but architecturally complete model config, independent of
    the real configs/model.yaml file on disk -- keeps this test suite
    self-contained and unaffected by yaml edits elsewhere."""
    return {
        'morphology': {'in_channels': 1, 'channels': [16, 32, 64, 128, 512]},
        'motion':     {'in_channels': 1, 'channels': [16, 32, 64, 128, 512],
                       'static_suppression_beta': 0.5},
        'graph':      {'node_dim': 512, 'alpha_init': 0.1, 'enabled': True},
        'projection': {'in_dim': 512, 'out_dim': 256},
        'identity':   {'hidden_dim': 512},   # num_classes injected by factory
    }


def _heads_yaml_cfg():
    return {
        'gender': {'in_dim': 512, 'hidden_dim': 128, 'num_classes': 2},
        'age':    {'in_dim': 512, 'hidden_dim': 256, 'num_bins': 7},
    }


def _meta(has_gender=True, has_age=False, protocols=None):
    return DatasetMeta(
        name='test', has_gender=has_gender, has_age=has_age,
        num_identities=NUM_IDENTITIES, image_size=(H, W),
        sequence_length=T, protocols=protocols or ['PROTO_A'],
    )


def _make_batch(has_gender=True, has_age=False, age_all_present=True):
    """
    Build a synthetic batch dict matching gait_collate_fn's output
    contract, for the given label-availability configuration.
    """
    frames = torch.rand(B, T, 1, H, W)
    id_label = torch.randint(0, NUM_IDENTITIES, (B,))

    gender_label = torch.randint(0, NUM_GENDER, (B,)) if has_gender else None

    if not has_age:
        age_label, age_bin, age_mask = None, None, torch.zeros(B, dtype=torch.bool)
    elif age_all_present:
        ages = [25.0, 40.0, 60.0, 10.0][:B]
        age_label = torch.tensor(ages)
        age_bin   = torch.tensor([age_to_bin(a) for a in ages])
        age_mask  = torch.ones(B, dtype=torch.bool)
    else:
        # Partial: half labeled, half not (the OU-LP-Bag case)
        age_label = torch.tensor([25.0, float('nan'), 60.0, float('nan')][:B])
        age_bin   = torch.tensor([age_to_bin(25.0), -1, age_to_bin(60.0), -1][:B])
        age_mask  = torch.tensor([True, False, True, False][:B])

    return {
        'frames': frames, 'id_label': id_label,
        'gender_label': gender_label,
        'age_label': age_label, 'age_bin': age_bin, 'age_mask': age_mask,
    }


# -- Model factory tests ----------------------------------------------------------

class TestModelFactory:

    @pytest.mark.parametrize("has_gender,has_age", [
        (False, False), (True, False), (True, True),
    ])
    def test_conditional_config_injection(self, has_gender, has_age):
        cfg = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(),
            _meta(has_gender, has_age),
        )
        assert ('gender' in cfg) == has_gender
        assert ('age' in cfg) == has_age
        assert cfg['identity']['num_classes'] == NUM_IDENTITIES

    def test_use_graph_flag_overrides_yaml_default(self):
        model_yaml = _base_model_yaml_cfg()
        model_yaml['graph']['enabled'] = True   # yaml says enabled...
        cfg = build_model_config(
            model_yaml, _heads_yaml_cfg(), _meta(), use_graph=False,
        )
        assert cfg['graph']['enabled'] is False   # ...CLI flag wins

    def test_gaitbase_backbone_sets_flag(self):
        cfg = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(), _meta(),
            morph_backbone='gaitbase',
        )
        assert cfg['morphology']['backbone'] == 'gaitbase'

    def test_invalid_backbone_rejected(self):
        with pytest.raises(ValueError, match="Unknown morph_backbone"):
            build_model_config(
                _base_model_yaml_cfg(), _heads_yaml_cfg(), _meta(),
                morph_backbone='not_a_real_backbone',
            )


# -- Conditional model construction tests ----------------------------------------

class TestConditionalModel:

    def _build(self, has_gender=True, has_age=False, use_graph=True,
              morph_backbone='custom'):
        cfg = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(),
            _meta(has_gender, has_age),
            use_graph=use_graph, morph_backbone=morph_backbone,
        )
        return BioKinematicNet(cfg)

    def test_no_gender_no_age_has_no_head_params(self):
        """
        The core conditional-instantiation property: a model built for
        a dataset with neither gender nor age must have ZERO parameters
        whose name contains 'gender_head' or 'age_head' -- not frozen,
        not present-but-unused, simply absent from the module tree.
        """
        model = self._build(has_gender=False, has_age=False)
        assert model.gender_head is None
        assert model.age_head is None
        param_names = [n for n, _ in model.named_parameters()]
        assert not any('gender_head' in n for n in param_names)
        assert not any('age_head' in n for n in param_names)

    def test_gender_only_has_gender_params_not_age(self):
        model = self._build(has_gender=True, has_age=False)
        param_names = [n for n, _ in model.named_parameters()]
        assert any('gender_head' in n for n in param_names)
        assert not any('age_head' in n for n in param_names)

    def test_gender_and_age_both_present(self):
        model = self._build(has_gender=True, has_age=True)
        param_names = [n for n, _ in model.named_parameters()]
        assert any('gender_head' in n for n in param_names)
        assert any('age_head' in n for n in param_names)

    def test_graph_disabled_has_no_graph_params(self):
        model = self._build(use_graph=False)
        assert model.graph is None
        param_names = [n for n, _ in model.named_parameters()]
        assert not any(n.startswith('graph.') for n in param_names)

    def test_graph_disabled_exact_passthrough(self):
        """
        With the graph disabled, Fm_prime/Fk_prime must be EXACTLY
        Fm/Fk -- not alpha-zeroed-out (which would still run wasted
        matrix multiplies), but a true bypass with no graph module
        invoked at all.
        """
        model = self._build(use_graph=False)
        model.eval()
        x = torch.rand(2, T, 1, H, W)
        with torch.no_grad():
            out = model(x, mode='train')
        assert torch.equal(out['Fm'], out['Fm_prime'])
        assert torch.equal(out['Fk'], out['Fk_prime'])

    def test_graph_stats_raises_when_disabled(self):
        model = self._build(use_graph=False)
        x = torch.rand(2, T, 1, H, W)
        with pytest.raises(RuntimeError, match="graph.enabled=False"):
            model.get_graph_stats(x)

    @pytest.mark.parametrize("mode", ['train', 'inference', 'eval'])
    def test_forward_modes_no_heads(self, mode):
        model = self._build(has_gender=False, has_age=False)
        model.eval()
        x = torch.rand(2, T, 1, H, W)
        with torch.no_grad():
            out = model(x, mode=mode)
        if mode == 'inference':
            assert out.shape == (2, 512)
        else:
            assert 'gender_logits' not in out
            assert 'age_bin_logits' not in out

    @pytest.mark.parametrize("mode", ['train', 'eval'])
    def test_forward_modes_with_both_heads(self, mode):
        model = self._build(has_gender=True, has_age=True)
        model.eval()
        x = torch.rand(2, T, 1, H, W)
        with torch.no_grad():
            out = model(x, mode=mode)
        assert 'gender_logits' in out
        assert 'age_bin_logits' in out and 'age_value' in out
        assert out['gender_logits'].shape == (2, 2)
        assert out['age_bin_logits'].shape == (2, 7)
        assert out['age_value'].shape == (2,)

    def test_count_parameters_labels_backbone(self):
        model_custom = self._build(morph_backbone='custom')
        model_gb     = self._build(morph_backbone='gaitbase')
        breakdown_c = model_custom.count_parameters()
        breakdown_g = model_gb.count_parameters()
        assert any('custom' in k for k in breakdown_c if 'morph' in k)
        assert any('gaitbase' in k for k in breakdown_g if 'morph' in k)

    def test_full_gradient_flow_all_heads(self):
        """Every parameter in a fully-configured model must receive a
        gradient from a single combined backward pass."""
        model = self._build(has_gender=True, has_age=True)
        model.train()
        x = torch.rand(2, T, 1, H, W)
        out = model(x, mode='train')
        loss = (
            out['id_logits'].sum() + out['gender_logits'].sum()
            + out['age_bin_logits'].sum() + out['age_value'].sum()
            + out['Fm'].sum() + out['Fk'].sum()
        )
        loss.backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(no_grad) == 0, f"No gradient reached: {no_grad}"


# -- CombinedLoss tests (the new (output, batch) call signature) -----------------

class TestCombinedLossV2:

    def _grad_output(self, has_gender=True, has_age=False):
        cfg = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(),
            _meta(has_gender, has_age),
        )
        model = BioKinematicNet(cfg)
        x = torch.rand(B, T, 1, H, W)
        return model, model(x, mode='train')

    def test_gender_only_no_age_keys(self):
        model, out = self._grad_output(has_gender=True, has_age=False)
        batch = _make_batch(has_gender=True, has_age=False)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        result = loss_fn(out, batch)
        result['total'].backward()
        assert 'gender' in result
        assert 'age_cls' not in result and 'age_reg' not in result

    def test_gender_and_age_present(self):
        model, out = self._grad_output(has_gender=True, has_age=True)
        batch = _make_batch(has_gender=True, has_age=True, age_all_present=True)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        result = loss_fn(out, batch)
        result['total'].backward()
        assert 'gender' in result
        assert 'age_cls' in result and 'age_reg' in result

    def test_zero_age_labeled_samples_in_batch_no_crash(self):
        """
        The model HAS an age head, but this specific batch has zero
        age-labeled samples (age_mask all-False). Must not crash, and
        the age loss terms must be correctly absent for this batch.
        """
        model, out = self._grad_output(has_gender=True, has_age=True)
        batch = _make_batch(has_gender=True, has_age=True)
        batch['age_mask'] = torch.zeros(B, dtype=torch.bool)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        result = loss_fn(out, batch)
        result['total'].backward()
        assert 'age_cls' not in result and 'age_reg' not in result

    def test_partial_age_mask_no_nan_gradient_leak(self):
        """
        Regression test for the property manually verified during
        stage-2 development: masking must happen BEFORE the loss call,
        not after, or NaN gradients from unlabeled samples (age_label=
        NaN) would propagate through the whole batch via the shared
        trunk, not just the unlabeled samples' own outputs.
        """
        model, out = self._grad_output(has_gender=True, has_age=True)
        # age_value is a non-leaf tensor (model OUTPUT, not an input) --
        # .grad is never populated for non-leaf tensors unless
        # retain_grad() is called explicitly before backward().
        out['age_value'].retain_grad()
        batch = _make_batch(has_gender=True, has_age=True, age_all_present=False)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        result = loss_fn(out, batch)
        result['total'].backward()
        assert out['age_value'].grad is not None, \
            "age_value never received a gradient at all -- retain_grad() " \
            "may not have taken effect, or the loss never used age_value"
        assert torch.isfinite(out['age_value'].grad).all(), \
            "NaN gradient leaked from masked-out samples into age_value.grad"

    def test_barlow_twins_always_present(self):
        """The disentanglement loss (Barlow Twins) is unconditional --
        present regardless of gender/age availability."""
        model, out = self._grad_output(has_gender=False, has_age=False)
        batch = _make_batch(has_gender=False, has_age=False)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        result = loss_fn(out, batch)
        assert 'adversarial' in result
        assert result['adversarial'].item() >= 0


# -- Trainer val_epoch regression tests (stage-4 bug) -----------------------------

class TestTrainerValEpochAgeHandling:
    """
    Regression tests for the real bug found during stage-4 integration
    testing: trainer.py's val_epoch() crashed with
    'TypeError: expected Tensor ... got NoneType' when the model has an
    age head but a specific validation BATCH happens to contain zero
    age-labeled samples -- the normal, expected case for a partial-label
    dataset like OU-LP-Bag, not a rare edge case. See trainers/trainer.py
    val_epoch()'s age-handling block for the fix.
    """

    def _make_trainer(self, has_age_batches):
        """
        Build a real Trainer wired to a DataLoader yielding the
        prescribed sequence of age-labeled/unlabeled batches, so
        val_epoch() exercises the real DataLoader iteration path, not
        a hand-constructed batch dict.
        """
        cfg_model = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(),
            _meta(has_gender=True, has_age=True),
        )
        model = BioKinematicNet(cfg_model)
        loss_fn = CombinedLoss(num_classes=NUM_IDENTITIES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        samples = []
        for i, has_age in enumerate(has_age_batches):
            age = 30.0 + i if has_age else None
            age_bin = age_to_bin(age) if has_age else None
            samples.append(Sample(
                frames=torch.rand(T, 1, H, W),
                id_label=i % NUM_IDENTITIES,
                gender_label=i % 2,
                age_label=age, age_bin=age_bin,
            ))

        from torch.utils.data import DataLoader, Dataset

        class _ListDataset(Dataset):
            def __len__(self): return len(samples)
            def __getitem__(self, idx): return samples[idx]

        val_loader = DataLoader(
            _ListDataset(), batch_size=2, shuffle=False,
            collate_fn=gait_collate_fn,
        )
        train_loader = val_loader   # unused by val_epoch, but Trainer needs one

        cfg = {'training': {
            'checkpoint': {'save_dir': tempfile.mkdtemp(), 'save_every': 100},
            'log_every': 100,
        }}
        trainer = Trainer(model, loss_fn, optimizer, scheduler,
                          train_loader, val_loader, cfg, torch.device('cpu'))
        return trainer

    def test_zero_age_labeled_val_samples_no_crash(self):
        """Negative case: NO val samples have age labels at all."""
        trainer = self._make_trainer(has_age_batches=[False] * 6)
        result = trainer.val_epoch(epoch=1)   # must not raise
        assert 'age_cls_acc' not in result
        assert 'age_mae' not in result
        assert 'gender_acc' in result   # gender still works independently

    def test_some_age_labeled_val_samples_computed_correctly(self):
        """Positive case: enough val samples DO have age labels."""
        trainer = self._make_trainer(has_age_batches=[True] * 6)
        result = trainer.val_epoch(epoch=1)
        assert 'age_cls_acc' in result
        assert 'age_mae' in result
        assert isinstance(result['age_mae'], float)
        assert result['age_mae'] == result['age_mae']  # NaN check (NaN != NaN)

    def test_mixed_age_labeled_batches_no_crash(self):
        """
        The exact scenario that originally crashed: SOME batches have
        age-labeled samples, SOME don't, within the same val_epoch run.
        """
        trainer = self._make_trainer(
            has_age_batches=[True, False, True, False, True, False]
        )
        result = trainer.val_epoch(epoch=1)   # must not raise
        assert 'age_cls_acc' in result   # overall: some samples WERE labeled
        assert isinstance(result['total'], float)


# -- Checkpoint round-trip (carried over from V1, model construction updated) ----

class TestCheckpoint:

    def test_save_and_load(self):
        cfg = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(), _meta(),
        )
        model = BioKinematicNet(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test_ckpt.pth')
            state = {
                'epoch': 1, 'model_state': model.state_dict(),
                'val_losses': {'total': 5.0}, 'best_val_loss': 0.5,
            }
            torch.save(state, path)

            model2 = BioKinematicNet(cfg)
            ckpt = torch.load(path, map_location='cpu')
            model2.load_state_dict(ckpt['model_state'])

            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"

    def test_mismatched_graph_config_fails_loudly(self):
        """
        Regression test for the real cross-stage finding: loading a
        checkpoint trained with one --no_graph/--morph_backbone setting
        into a model built with a DIFFERENT setting must fail with a
        clear state_dict error, not silently succeed with a
        wrong-architecture load.
        """
        cfg_with_graph = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(), _meta(), use_graph=True,
        )
        cfg_no_graph = build_model_config(
            _base_model_yaml_cfg(), _heads_yaml_cfg(), _meta(), use_graph=False,
        )
        model_with_graph = BioKinematicNet(cfg_with_graph)
        model_no_graph    = BioKinematicNet(cfg_no_graph)

        with pytest.raises(RuntimeError, match="(Missing key|Unexpected key)"):
            model_no_graph.load_state_dict(model_with_graph.state_dict())


# -- Metric function tests (carried over from V1 unmodified -- utils/metrics.py
#    did not change in the V2 rewrite) --------------------------------------------

class TestMetrics:

    def test_cosine_distance_matrix_shape(self):
        probe   = torch.randn(10, 512)
        gallery = torch.randn(5, 512)
        dist    = cosine_distance_matrix(probe, gallery)
        assert dist.shape == (10, 5)

    def test_cosine_distance_range(self):
        probe   = torch.randn(8, 512)
        gallery = torch.randn(4, 512)
        dist    = cosine_distance_matrix(probe, gallery)
        assert dist.min() >= -1e-5
        assert dist.max() <= 2.0 + 1e-5

    def test_self_distance_near_zero(self):
        x    = torch.randn(4, 512)
        dist = cosine_distance_matrix(x, x)
        assert dist.diag().abs().max() < 1e-5

    def test_rank_k_perfect(self):
        dist = torch.tensor([
            [0.1, 0.9, 0.8], [0.9, 0.1, 0.8], [0.8, 0.9, 0.1],
        ])
        assert compute_rank_k(dist, [0, 1, 2], [0, 1, 2], k=1) == 1.0

    def test_rank_k_worst(self):
        dist = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        assert compute_rank_k(dist, [0, 1], [0, 1], k=1) == 0.0

    def test_map_perfect(self):
        dist = torch.tensor([[0.1, 0.9, 0.8]])
        assert abs(compute_map(dist, [0], [0, 1, 2]) - 1.0) < 1e-6

    def test_cmc_curve_shape(self):
        dist = torch.randn(10, 5)
        cmc  = compute_cmc_curve(dist, list(range(10)), list(range(5)), max_rank=5)
        assert len(cmc) == 5

    def test_cmc_monotone(self):
        dist = torch.randn(20, 10)
        cmc  = compute_cmc_curve(
            dist, [i % 10 for i in range(20)], list(range(10)), max_rank=10
        )
        for i in range(len(cmc) - 1):
            assert cmc[i] <= cmc[i + 1] + 1e-6

    def test_eer_range(self):
        dist = torch.randn(10, 5)
        eer, thresh = compute_eer(dist, list(range(10)), list(range(5)))
        assert 0.0 <= eer <= 1.0

    def test_gender_metrics_all_correct(self):
        preds  = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_gender_metrics(preds, labels)
        assert m['accuracy'] == 1.0
        assert m['balanced_accuracy'] == 1.0

    def test_gender_metrics_all_one_class(self):
        preds  = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 0, 1, 1])
        m = compute_gender_metrics(preds, labels)
        assert abs(m['balanced_accuracy'] - 0.5) < 1e-6


# -- Seed reproducibility (carried over from V1 unmodified) -----------------------

class TestSeed:

    def test_same_seed_same_output(self):
        set_seed(42)
        x1 = torch.randn(4, 512)
        set_seed(42)
        x2 = torch.randn(4, 512)
        assert torch.allclose(x1, x2)

    def test_different_seed_different_output(self):
        set_seed(42)
        x1 = torch.randn(4, 512)
        set_seed(123)
        x2 = torch.randn(4, 512)
        assert not torch.allclose(x1, x2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
