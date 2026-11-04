"""Regression checks for causal memory, certified negatives and recovery ownership.

These tests use structured poses and real controller semantics, not token-derived
geometry or synthetic labels pretending to be executed failure trajectories.
"""

import copy
import dataclasses
import hashlib
import inspect
import json

import numpy as np
import pytest
import torch

from test_core import observation, spec
from tgcc.contracts import (
    ContractEvaluator,
    Evidence,
    NegativeValidityChecker,
    Predicate,
    TaskContract,
    Truth,
)
from tgcc.data import contract_tensors, transition_labels
from tgcc.losses import critic_loss
from tgcc.memory import ThreeGraphMemory
from tgcc.network import TGCC
from tgcc.specs import ExecutedTransition
from tgcc.train_system import TrainSystem


def pose(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:2, 3] = [x, y]
    return out.tolist()


def snapshot(timestamp=1.0, *, static=True, visible=True, entities=True):
    node = dict(
        entity_id="object",
        instance_id="target",
        pose=pose(2),
        validity=True,
        reference_frame="world",
        timestamp=timestamp,
        last_seen=timestamp,
        confidence=0.99,
        semantic_type="microwave.n.02" if static else "hotdog.n.02",
        attributes=dict(visible=visible, assumed_static=static),
    )
    return dict(
        episode_id="ep",
        timestamp=timestamp,
        entities=[node] if entities else [],
        edges=[],
    )


def memory(static=True):
    m = ThreeGraphMemory()
    m.update(
        snapshot(static=static),
        {("observed", "target", ""): Evidence(True, 1, 0.99, True)},
        robot_pose=pose(),
    )
    return m




def test_static_memory_reprojects_after_base_rotation_without_new_exposure():
    m = memory()
    m.update(snapshot(2, entities=False), {}, robot_pose=pose(yaw=np.pi / 2))
    estimate = m.estimate("target", 2)
    assert estimate is not None
    np.testing.assert_allclose(
        np.asarray(estimate["relative_pose"])[:3, 3], [0, -2, 0], atol=1e-10
    )
    assert estimate["information_source"] == "MEMORY_INFERRED"
    assert estimate["last_observed"] == 1
    assert not estimate["visibility"]
    np.testing.assert_allclose(estimate["world_pose"], pose(2))


def test_covariance_grows_with_age_and_robot_localization_uncertainty():
    m = memory()
    young = m.estimate("target", 1)
    old = m.estimate("target", 10)
    assert np.trace(old["relative_covariance"]) > np.trace(young["relative_covariance"])
    uncertain = m.estimate("target", 1, frame_cov=np.eye(6) * 0.001)
    assert np.trace(uncertain["relative_covariance"]) > np.trace(
        young["relative_covariance"]
    )
    assert m.estimate("target", 1, frame_cov=np.eye(6)) is None


def test_movable_history_expires_earlier_than_static_history():
    assert memory(static=False).estimate("target", 1.1) is not None
    assert memory(static=False).estimate("target", 4) is None
    assert memory(static=True).estimate("target", 4) is not None


def test_supplemented_snapshot_does_not_refresh_observation_timestamp():
    m = memory()
    out = m.supplement_snapshot(snapshot(2, entities=False), 2)
    n = out["entities"][0]
    assert n["last_seen"] == n["timestamp"] == 1
    assert n["attributes"]["information_source"] == "MEMORY_INFERRED"
    m.update(out, {}, robot_pose=pose(1))
    assert m.entities["target"].last_observed == 1


def test_current_valid_evidence_overrides_old_memory_and_conflict_is_not_hidden():
    m = memory()
    p = Predicate("observed", "target", maximum_observation_age=600)
    direct = Evidence(False, 2, 0.99, True)
    assert m.resolve(p, {p.key: direct}, 2) is direct
    weak_conflict = Evidence(False, 2, 0.1, True)
    assert m.resolve(p, {p.key: weak_conflict}, 2).truth_for(p, 2) == Truth.UNKNOWN


def test_memory_can_resolve_nonvisible_static_target_but_not_fresh_contact():
    m = memory()
    localization = Predicate("observed", "target", maximum_observation_age=600)
    ev = m.resolve(localization, {}, 2)
    assert ev.truth_for(localization, 2) == Truth.TRUE
    assert ev.information_source == "MEMORY_INFERRED"
    fresh = dataclasses.replace(localization, requires_fresh_observation=True)
    assert m.resolve(fresh, {}, 2).truth_for(fresh, 2) == Truth.UNKNOWN
    assert ev.truth_for(fresh, 2) == Truth.UNKNOWN


def test_irrelevant_unknown_does_not_block_stage_or_produce_residual():
    needed = Predicate("observed", "target", category="pre")
    unrelated = Predicate(
        "held", "other", category="invariant", required_at_current_stage=False
    )
    c = TaskContract(1, "navigate", (needed, unrelated))
    result = ContractEvaluator().evaluate(
        "ep", c, {needed.key: Evidence(True, 1, 1, True)}, 1
    )
    assert result["states"] == [Truth.TRUE, Truth.UNKNOWN]
    assert result["required_unknown"] == []
    assert result["violated"] == []
    assert result["contract_residual"] == 0


def test_stage_inactive_known_clause_remains_masked_for_network_validation():
    p = Predicate("held", "target", required_at_current_stage=False)
    tensors = contract_tensors(
        TaskContract(2, "navigate", (p,)), {p.key: Evidence(True, 1, 1, True)}, 1
    )
    assert not tensors["contract_mask"].any()
    assert not tensors["contract_valid"].any()
    o = observation(1)
    for key, value in tensors.items():
        setattr(o, key, value)
    o.validate()












def test_certified_negatives_require_preserve_invariant_and_closed_positive():
    checker = NegativeValidityChecker(spec())
    native = torch.zeros(3, 32, 23)
    native[:, :, 14] = -1
    native[:, :, 22] = -1
    valid = torch.ones(3, 32, dtype=torch.bool)
    preserve = torch.tensor([[True, False], [False, False], [True, False]])
    native[2, :, 14] = 1  # a legitimate release is not a preserved-grasp positive
    changed, proof, provenance = checker.task_negative(
        native, valid, preserve, lambda x: x
    )
    assert proof.tolist() == [True, False, False]
    assert changed[0, :, 14].eq(1).all()
    assert not checker.violations(changed, valid).any()
    assert torch.equal(changed[1:], native[1:])
    assert provenance["kind"] == "explicit_preserve_grasp_invariant"
    assert provenance["not_a_failure_trajectory"] is True
    physical = checker.generate(native, valid, lambda x: x, max_count=1)
    assert len(physical) == 1
    assert physical[0][1].all()
    assert len(physical) + int(proof.any()) == 2


def test_no_valid_action_cannot_prove_task_negative():
    native = torch.full((1, 32, 23), -1.0)
    _, proof, _ = NegativeValidityChecker(spec()).task_negative(
        native,
        torch.zeros(1, 32, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        lambda x: x,
    )
    assert not proof.any()


def test_arbitrary_in_bounds_perturbation_is_not_automatically_negative():
    torch.manual_seed(11)
    native = torch.rand(4, 32, 23) * 1.8 - 0.9
    valid = torch.ones(4, 32, dtype=torch.bool)
    checker = NegativeValidityChecker(spec())
    assert not checker.violations(native, valid).any()
    _, proof, _ = checker.task_negative(native, valid, None, lambda x: x)
    assert not proof.any()


def test_training_builds_at_most_one_of_each_negative_in_same_context(monkeypatch):
    parent = torch.nn.Module()
    parent.base = torch.nn.Module()
    parent.base.action_in_proj = torch.nn.Linear(1, 1024)

    class IdentityNormalizer:
        def transform(self, x):
            return x

    config = dict(
        adapter_bottleneck=16,
        margin=0.2,
        loss_weights=dict(
            rank=1.0,
            transition=1.0,
            predicate=1.0,
            progress=0.5,
            handoff=0.5,
        ),
    )
    system = TrainSystem(parent, spec(), IdentityNormalizer(), config).eval()
    before, after = observation(1), observation(1)
    after.timestamp += 32 / 30
    before.candidate_actions.zero_()
    before.candidate_actions[:, :, 14] = -1
    before.preserve_gripper_mask = torch.tensor([[True, False]])
    native = before.candidate_actions.clone()
    contexts, transitions = [], []
    original_forward = system.critic.forward_action
    original_retrospective = system.critic.retrospective

    def forward(o):
        contexts.append(o)
        return original_forward(o)

    def retrospective(tr):
        transitions.append(tr)
        return original_retrospective(tr)

    monkeypatch.setattr(system.critic, "forward_action", forward)
    monkeypatch.setattr(system.critic, "retrospective", retrospective)
    _, _, counts = system(before, after, native, 100)
    assert len(contexts) == 3  # one positive, one physical, one certified task negative
    assert counts["rank_physical"] == counts["rank_task"] == 1
    for context in contexts:
        assert context.graph_tokens is before.graph_tokens
        assert context.semantic_context is before.semantic_context
        assert context.contract_values is before.contract_values
        assert context.timestamp is before.timestamp
    assert len(transitions) == 1
    assert transitions[0].before is before and transitions[0].after is after
    assert torch.equal(transitions[0].executed_actions, native)
    assert torch.equal(before.candidate_actions, native)


def test_unknown_loss_masks_remove_values_and_gradients_instead_of_labeling_false():
    positive = {"action_energy_total": torch.zeros(2, requires_grad=True)}
    state = {
        k: torch.randn(*shape, requires_grad=True)
        for k, shape in {
            "transition_logit": (2,),
            "predicate_logits": (2, 3),
            "handoff_logit": (2,),
            "progress": (2,),
        }.items()
    }
    labels = {}
    for key, shape in (
        ("transition", (2,)),
        ("predicate", (2, 3)),
        ("handoff", (2,)),
        ("progress", (2,)),
    ):
        labels[key] = torch.ones(shape)
        labels[f"{key}_valid"] = torch.zeros(shape, dtype=torch.bool)
    loss, _, counts = critic_loss(
        positive, [], torch.zeros(2, 0, dtype=torch.bool), state, labels
    )
    assert loss.item() == 0 and not any(counts.values())
    loss.backward()
    assert all(x.grad is not None and not x.grad.any() for x in state.values())


def test_progress_and_handoff_unknown_masks_are_not_failed_targets():
    before, after = observation(1), observation(1)
    before.contract_category[:] = after.contract_category[:] = torch.tensor([[0, 1, 3]])
    before.contract_valid[:, 1:] = after.contract_valid[:, 1:] = False
    labels = transition_labels(before, after)
    assert not labels["progress_valid"].any()
    assert not labels["handoff_valid"].any()
    assert not labels["transition_valid"].any()
    assert not labels["predicate_valid"][:, 1:].any()


def test_retrospective_uses_only_actual_prefix_not_unexecuted_or_next_candidates():
    model = TGCC(spec(), 240).eval()
    before, after = observation(1), observation(1)
    after.timestamp += 16 / 30
    executed = before.candidate_actions.clone()
    mask = torch.arange(32)[None] < 16
    transition = ExecutedTransition(
        before, after, executed, mask, before.timestamp, after.timestamp
    )
    first = model.retrospective(transition)
    executed[:, 16:] = 1e6
    after.candidate_actions[:] = -1e6
    second = model.retrospective(transition)
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert list(inspect.signature(model.forward_action).parameters) == ["o"]


def test_legacy_critic_state_dict_strict_compatibility_without_active_ood_head():
    torch.manual_seed(12)
    model = TGCC(spec(), 240).eval()
    legacy = copy.deepcopy(model.state_dict())
    # Names/shapes independently read from the pre-revision afc897f network.
    # Retaining this immutable interface protects the actual step100 weights,
    # rather than merely checking that a new model can reload its own state.
    shape_identity = {k: list(v.shape) for k, v in legacy.items()}
    assert hashlib.sha256(
        json.dumps(shape_identity, sort_keys=True).encode()
    ).hexdigest() == (
        "166c9a921d4bd0782aaa1ad29e92243c710c8bcaf6f026a1912cd79dbfc74f04"
    )
    assert "ood.weight" in legacy and "ood.bias" in legacy
    clone = TGCC(spec(), 240).eval()
    clone.load_state_dict(legacy, strict=True)
    assert not any(p.requires_grad for p in clone.ood.parameters())
    o = observation(1)
    before = clone.forward_action(o)
    with torch.no_grad():
        clone.ood.weight.fill_(1e8)
        clone.ood.bias.fill_(-1e8)
    after = clone.forward_action(o)
    assert "action_ood_score" not in after
    for key in before:
        assert torch.equal(before[key], after[key]), key
