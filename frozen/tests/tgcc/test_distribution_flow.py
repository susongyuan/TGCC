import copy
import sys
from types import SimpleNamespace

import torch
from torch import nn

from tgcc.distribution_critic import DistributionConfig
from tgcc.distribution_joint import DistributionJointSystem
from tgcc.retrieval_adapter import RetrievalAdapter, RetrievalContext


class TinyExpertPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.action_in_proj = nn.Linear(32, 16)
        self.base.action_out_proj = nn.Linear(16, 32)
        self.base.paligemma_with_expert = nn.Module()
        self.base.paligemma_with_expert.gemma_expert = nn.Linear(16, 16)
        self.base.paligemma_with_expert.paligemma = nn.Linear(2, 16)
        self.global_encoder = nn.Linear(2, 16)
        self.base.sample_noise = lambda shape, device: torch.ones(shape, device=device)
        self.base.sample_time = lambda batch, device: torch.full((batch,), .5, device=device)
        self.prefix_calls = []

    def prefix(self, obs, conditions, train):
        self.prefix_calls.append(obs)
        return torch.zeros(2, 32), torch.ones(2, 1, dtype=torch.bool), None, [
            (torch.ones(2, 2, 16), torch.ones(2, 2, dtype=torch.bool))] * 2, {}

    def velocity(self, state, valid, cache, tokens, x, time):
        value = self.base.action_in_proj(x)
        value = value + tokens[0][0].mean(1)[:, None]
        value = self.base.paligemma_with_expert.gemma_expert(value)
        return self.base.action_out_proj(value)


def setup(monkeypatch, dropout=0.):
    monkeypatch.setitem(sys.modules, "tgr.data", SimpleNamespace(collate_sequences=lambda samples: samples[0]))
    monkeypatch.setitem(sys.modules, "openpi.models.model", SimpleNamespace(
        Observation=SimpleNamespace(from_dict=lambda obs: obs)))
    import tgcc.data
    monkeypatch.setattr(tgcc.data, "conditions_from_dict", lambda raw: raw)
    torch.manual_seed(42)
    policy = TinyExpertPolicy()
    model = DistributionJointSystem(policy, RetrievalAdapter(expert_width=16, context_width=8, after_width=6),
        DistributionConfig(context_width=8, after_width=6, planning_classes=3, hidden=16),
        dict(critic_warmup_updates=1000, retrieval_conditioning_dropout=dropout,
             loss_weights=dict(action=1., execution=1., planning=.2, flow=1.)))
    actions = torch.zeros(2, 32, 32)
    sample = dict(sequence=[dict(observation={"exposure": "before"}, conditions={}, actions=actions),
        dict(observation={"exposure": "actual_after"}, conditions={}, actions=actions)],
        policy_imitation_allowed=True)
    reference = RetrievalContext(torch.ones(2, 32, 23), torch.ones(2, 32, dtype=torch.bool),
        torch.randn(2, 8), torch.randn(2, 6), torch.ones(2, dtype=torch.bool), torch.ones(2, 6, dtype=torch.bool))
    batch = dict(density=dict(context=torch.randn(2, 8), actions=actions[..., :23],
        action_mask=torch.ones(2, 32, dtype=torch.bool), after_features=torch.randn(2, 6),
        next_index=torch.tensor([0, 1])), policy_samples=[sample] * 2,
        references=[reference], query_episode_ids=["1", "2"], reference_episode_ids=[["3", "4"]])
    return model, batch


def test_one_real_prefix_updates_actionexpert_and_zero_gate_without_using_after_as_policy_input(monkeypatch):
    model, batch = setup(monkeypatch)
    model.train()
    loss, parts, counts = model(batch, 1000)
    loss.backward()
    assert model.student.prefix_calls == [{"exposure": "before"}]
    assert counts["flow_sequences"] == 2 and counts["retrieval_conditioned"] == 2
    assert "flow" in parts
    assert model.student.base.paligemma_with_expert.gemma_expert.weight.grad.abs().sum() > 0
    assert model.retrieval.gates["local"].grad.abs().sum() > 0
    assert model.student.global_encoder.weight.grad is None
    assert model.student.base.paligemma_with_expert.paligemma.weight.grad is None


def test_reference_dropout_keeps_module_path_but_trains_original_unassisted_actionexpert(monkeypatch):
    model, batch = setup(monkeypatch, dropout=1.)
    model.train()
    loss, _, counts = model(batch, 1000)
    loss.backward()
    assert counts["retrieval_conditioned"] == 0 and counts["retrieval_unassisted"] == 2
    assert model.student.base.action_in_proj.weight.grad.abs().sum() > 0
    assert torch.equal(model.retrieval.gates["local"].grad, torch.zeros(1))


def test_joint_training_rejects_same_episode_reference_and_collected_imitation_targets(monkeypatch):
    import pytest
    model, batch = setup(monkeypatch)
    bad = copy.deepcopy(batch)
    bad["reference_episode_ids"] = [["1", "4"]]
    with pytest.raises(ValueError, match="whole source episode"):
        model(bad, 1000)
    bad = copy.deepcopy(batch)
    bad["policy_samples"][0]["source_kind"] = "collection"
    with pytest.raises(ValueError, match="successful demo"):
        model(bad, 1000)


def test_effective_batch_masked_gradient_matches_one_batch_with_rank_and_microbatch_imbalance():
    from tgcc.distribution_joint import effective_distribution_loss
    parameter = nn.Parameter(torch.tensor(2.))
    # Two ranks, two microbatches: 0, 1, 2, and 3 known planning labels.
    # The reference is the single six-label effective batch mean.
    targets = [torch.tensor([]), torch.tensor([1.]), torch.tensor([2., 4.]), torch.tensor([3., 5., 7.])]
    reference = torch.cat(targets)
    ((parameter - reference).square().mean()).backward()
    expected = parameter.grad.clone()
    parameter.grad = None
    for values in targets:
        n = len(values)
        loss = (parameter - values).square().sum() / max(n, 1)
        zero = parameter * 0
        # FSDP's world-size average is represented explicitly by /2 here.
        weighted = effective_distribution_loss(dict(action=zero, execution=zero, planning=loss),
            dict(action=0, execution=0, planning=n), dict(action=0, execution=0, planning=6),
            world=2, accumulation=2, weights=dict(action=1., execution=1., planning=1.))
        (weighted / 2).backward()
    torch.testing.assert_close(parameter.grad, expected)
