"""v3frozen: group freezing, recipe-sized accumulation, name-aligned rates and the generalised serve gate."""
import pytest, torch
from torch import nn


class _Named(nn.Module):
    def __init__(self, names):
        super().__init__()
        self._names = names
        for i, _ in enumerate(names):
            setattr(self, f'p{i}', nn.Parameter(torch.zeros(2)))

    def named_parameters(self, *a, **k):
        return [(n, getattr(self, f'p{i}')) for i, n in enumerate(self._names)]


NAMES = ['base.paligemma.language_model.layers.0.w', 'base.vision_tower.encoder.w', 'base.multi_modal_projector.w',
         'base.gemma_expert.layers.0.w', 'base.action_in_proj.w', 'decoder.memory.weight_ih', 'global_encoder.w', 'base.gemma_expert.lm_head.w']


def test_freezing_touches_only_vision_and_vlm_and_never_memory():
    from tgcc.full_distribution_joint import apply_frozen_policy_groups
    m = _Named(NAMES)
    frozen = apply_frozen_policy_groups(m, ['vision', 'vlm'])
    assert set(frozen) == {NAMES[0], NAMES[1], NAMES[2]}
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {NAMES[3], NAMES[4], NAMES[5], NAMES[6], NAMES[7]}


def test_freezing_refuses_other_groups_and_empty_matches():
    from tgcc.full_distribution_joint import apply_frozen_policy_groups
    with pytest.raises(ValueError):
        apply_frozen_policy_groups(_Named(NAMES), ['expert'])
    with pytest.raises(ValueError):
        apply_frozen_policy_groups(_Named(['decoder.memory.w']), ['vlm'])
    assert apply_frozen_policy_groups(_Named(NAMES), ()) == []


def test_accumulation_follows_the_recipe_effective_batch():
    from tgcc.full_distribution_joint import full_accumulation
    cfg = dict(gradient_accumulation_by_world={'3': 48, '4': 36}, max_gpus=4, effective_batch=144)
    assert full_accumulation(4, 6, cfg) == 6 and full_accumulation(4, 12, cfg) == 3 and full_accumulation(3, 8, cfg) == 6 and full_accumulation(3, 16, cfg) == 3
    for world, micro in ((2, 1), (4, 5), (4, 7), (5, 1)):
        with pytest.raises(ValueError):
            full_accumulation(world, micro, cfg)
    with pytest.raises(ValueError):  # table inconsistent with the effective batch
        full_accumulation(4, 6, dict(cfg, effective_batch=96))


def test_rates_by_name_match_the_positional_schedule():
    from tgcc.full_distribution_joint import full_rates, full_rates_by_name
    recipe = dict(warmup=10, min_lr_ratio=0.1, new_updates=100, critic_warmup_updates=0, joint_updates=100, memory_unroll_blocks=8,
                  critic_lr=1e-4, retrieval_adapter_lr=1e-4, retrieval_gate_lr=3e-4, vision_lr=1e-6, vlm_lr=1e-6, expert_lr=4e-6,
                  action_io_lr=4e-6, graphs_lr=2e-5, memory_lr=1e-4, correction_lr=1e-4)
    by_name = full_rates_by_name(50, recipe); positional = full_rates(50, recipe)
    assert list(by_name) == ['critic', 'retrieval', 'retrieval_gates', 'vision', 'vlm', 'expert', 'action_io', 'graphs', 'memory', 'correction']
    assert list(by_name.values()) == positional
    assert by_name['expert'] / by_name['vision'] == pytest.approx(recipe['expert_lr'] / recipe['vision_lr'])
    assert by_name['graphs'] / by_name['expert'] == pytest.approx(recipe['graphs_lr'] / recipe['expert_lr'])


def test_audit_requirements_exclude_frozen_groups_only_in_joint_phase():
    from tgcc.full_distribution_joint import required_audit_gradients
    keys = ('critic', 'action', 'vision', 'vlm', 'expert', 'graphs', 'memory', 'correction')
    assert required_audit_gradients('warmup', keys, dict(frozen_policy_groups=['vision', 'vlm'])) == ('critic', 'action', 'execution', 'planning')
    assert required_audit_gradients('joint', keys, dict(frozen_policy_groups=['vision', 'vlm'])) == ('critic', 'action', 'expert', 'graphs', 'memory', 'correction')
    assert required_audit_gradients('joint', keys, {}) == keys


def _md(cfg, **counters):
    base = dict(schema='tgcc.distribution_joint.v1', training_complete=True, audit_disposable=False, config=dict(cfg, schema='tgcc.corrected_full_success_distribution.recipe.v1'))
    base.update(counters); return base


def test_serve_gate_accepts_old_and_frozen_recipes_and_rejects_mismatches():
    from tgcc.distribution_serve import artifact_recipe_consistent
    old = dict(critic_warmup_updates=1000, joint_updates=4000, new_updates=5000, critic_initialization_actual_updates=0)
    assert artifact_recipe_consistent(_md(old, new_updates=5000, full_model_updates=4000, fresh_critic_only_updates=1000, critic_actual_total_updates=5000, policy_new_updates=4000, policy_cumulative_updates=9000), 5000)
    frozen = dict(critic_warmup_updates=0, joint_updates=2000, new_updates=2000, critic_initialization_actual_updates=2656)
    ok = _md(frozen, new_updates=2000, full_model_updates=2000, fresh_critic_only_updates=0, critic_actual_total_updates=4656, policy_new_updates=2000, policy_cumulative_updates=12000)
    assert artifact_recipe_consistent(ok, 10000)
    assert not artifact_recipe_consistent(dict(ok, training_complete=False), 10000)
    assert not artifact_recipe_consistent(dict(ok, audit_disposable=True), 10000)
    assert not artifact_recipe_consistent(dict(ok, critic_actual_total_updates=2000), 10000)
    assert not artifact_recipe_consistent(dict(ok, full_model_updates=1900), 10000)
    assert not artifact_recipe_consistent(_md(dict(frozen, joint_updates=0), new_updates=0), 10000)
