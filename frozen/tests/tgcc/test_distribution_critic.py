import copy
import math

import pytest
import torch

from tgcc.distribution_critic import (ConditionalMixture, DistributionConfig,
                                      DistributionCritic, distribution_loss)
from tgcc.distribution_calibration import calibrate_success_scores, distribution_readiness


def tiny():
    torch.manual_seed(42)
    model = DistributionCritic(DistributionConfig(context_width=8, after_width=6,
        planning_classes=3, hidden=16, horizon=4, action_width=2))
    batch = dict(context=torch.randn(3, 8), actions=torch.randn(3, 4, 2) * .2,
        action_mask=torch.ones(3, 4, dtype=torch.bool), after_features=torch.randn(3, 6) * .2,
        next_index=torch.tensor([0, 1, 2]), planning_valid=torch.tensor([True, False, True]))
    return model, batch


def test_after_targets_do_not_change_action_or_planning_and_are_not_trainable_features():
    model, batch = tiny()
    batch["after_features"].requires_grad_()
    first = model(batch)
    changed = dict(batch, after_features=batch["after_features"] + 100)
    second = model(changed)
    assert torch.equal(first["action_nll"], second["action_nll"])
    assert torch.equal(first["planning_logits"], second["planning_logits"])
    assert not torch.equal(first["execution_nll"], second["execution_nll"])
    loss, _, counts = distribution_loss(first, batch)
    loss.backward()
    assert batch["after_features"].grad is None
    assert counts == {"action": 3, "execution": 3, "planning": 2}


def test_unexecuted_action_suffix_nan_is_ignored_in_both_density_paths():
    model, batch = tiny()
    batch["action_mask"][:, 2:] = False
    clean = model(batch)
    batch["actions"][:, 2:] = torch.nan
    masked = model(batch)
    for name in ("action_nll", "execution_nll", "planning_nll"):
        assert torch.equal(clean[name], masked[name])
    batch["actions"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        model(batch)


def test_unknown_after_value_dimensions_are_masked_while_available_flags_remain_targets():
    model, batch = tiny()
    batch["after_feature_mask"] = torch.ones(3, 6, dtype=torch.bool)
    batch["after_feature_mask"][:, :2] = False
    clean = model(batch)["execution_nll"]
    batch["after_features"][:, :2] = torch.nan
    torch.testing.assert_close(model(batch)["execution_nll"], clean)
    batch["after_features"][0, 3] = torch.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        model(batch)


def test_planning_likelihood_uses_the_same_full_vocabulary_even_when_dispatch_eligibility_changes():
    model, batch = tiny()
    plain = model.score_planning(batch["context"], batch["next_index"])
    eligible = torch.tensor([[True, False, False]]).expand(3, -1)
    gated = model.score_planning(batch["context"], batch["next_index"], eligible)
    assert torch.equal(plain["planning_nll"], gated["planning_nll"])


def test_mixture_is_normalized_and_variance_floor_prevents_unbounded_positive_likelihood():
    head = ConditionalMixture(3, 2, 4, .05, 5.)
    hidden = torch.zeros(5, 3)
    with torch.no_grad():
        head.means.weight.zero_()
        head.means.bias.zero_()
        head.scales.weight.zero_()
        head.scales.bias.fill_(-1000)
    mean, logs, weights = head.parameters_for(hidden)
    torch.testing.assert_close(weights.exp().sum(-1), torch.ones(5))
    assert logs.exp().min() >= .05 - 1e-7
    score = head.nll(hidden, torch.zeros(5, 2))
    torch.testing.assert_close(score, torch.full_like(score, math.log(.05) + .5 * math.log(2 * math.pi)))
    assert (head.nll(hidden, torch.ones(5, 2)) > score).all()


def test_success_only_heads_train_with_finite_gradients_without_energy_collapse():
    model, batch = tiny()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    first = float(distribution_loss(model(batch), batch)[0].detach())
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = distribution_loss(model(batch), batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    last = float(distribution_loss(model(batch), batch)[0].detach())
    assert last < first - .5
    # This verifies likelihood optimization on a toy success-only batch;
    # no failed robot examples or failure-recall claim are introduced.


def calibration_rows(n=16):
    rows = []
    for ep in range(n):
        for step in (0, 32):
            rows.append(dict(episode_id=str(1000 + ep), split="validation",
                sequence_steps=[step, step + 32], executed_steps=32, action_nll=float(ep + step / 32),
                execution_nll=float(ep), planning_nll=float(ep / 10)))
            rows[-1]["executed_steps"] = 32
    return rows


def test_episode_grouping_does_not_count_chunks_as_independent_calibration_situations():
    result = calibrate_success_scores(calibration_rows(), ["1", "2"])
    action = result["heads"]["action"]
    assert action["enabled"] is True
    assert action["calibration_episode_count"] == 16
    assert action["calibration_sample_count"] == 32
    assert action["threshold"] == 16
    assert action["failure_recall"] is None
    assert action["task_success_probability"] is False


def test_sixteen_episodes_cannot_supply_finite_five_percent_conformal_threshold():
    result = calibrate_success_scores(calibration_rows(), ["1"], alpha=.05)
    assert result["heads"]["action"]["enabled"] is False
    assert result["heads"]["action"]["threshold"] is None


def test_calibration_rejects_indexed_episodes_and_repeated_chunks():
    rows = calibration_rows()
    with pytest.raises(ValueError, match="indexed"):
        calibrate_success_scores(rows, ["1000"])
    with pytest.raises(ValueError, match="Repeated"):
        calibrate_success_scores(rows + [rows[0]], ["1"])


def ready_metadata():
    identities = {k: "a" * 64 for k in ("feature_codec_sha256", "retrieval_index_sha256",
        "demonstrations_manifest_sha256", "normalization_sha256", "encoder_model_sha256")}
    md = dict(identities, schema="tgcc.distribution_joint.v1")
    features = dict(identities, index_episodes=["1", "2"], validation_episodes=["3"],
        retrieval_excludes_query_episode=True, encoder_frozen=True,
        failed_or_synthetic_targets_used=False,
        valid_training_targets=dict(action=20, execution=20, planning=12),
        actual_prefix_time_alignment_checked=True, future_targets_as_context_inputs=False)
    return md, features


def test_readiness_binds_success_data_and_prevents_episode_future_or_target_leakage():
    md, features = ready_metadata()
    assert distribution_readiness(md, features)["status"] == "passed"
    bad = copy.deepcopy(features)
    bad["encoder_model_sha256"] = "b" * 64
    bad["validation_episodes"].append("1")
    bad["future_targets_as_context_inputs"] = True
    bad["failed_or_synthetic_targets_used"] = True
    gate = distribution_readiness(md, bad)
    assert gate["status"] == "failed" and len(gate["failures"]) == 4


def test_new_warmup_freezes_encoder_and_graphs_and_creates_no_actionexpert_or_adapter_moments():
    from test_joint_training import TinyParent
    from tgcc.distribution_joint import DistributionJointSystem, distribution_optimizer_groups
    from tgcc.retrieval_adapter import RetrievalAdapter

    _, batch = tiny()
    parent = TinyParent()
    recipe = dict(critic_warmup_updates=1000, critic_lr=1e-4,
                  loss_weights=dict(action=1., execution=1., planning=.2))
    model = DistributionJointSystem(parent, RetrievalAdapter(expert_width=16, context_width=8, after_width=6),
        DistributionConfig(context_width=8, after_width=6, planning_classes=3, hidden=16, horizon=4, action_width=2), recipe)
    groups = distribution_optimizer_groups(model, recipe)
    opt = torch.optim.AdamW(groups)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    loss, _, _ = model({"density": batch}, 0)
    loss.backward()
    opt.step()
    assert all(torch.equal(v, before[k]) for k, v in parent.state_dict().items())
    assert all(p not in opt.state and p.grad is None for p in parent.parameters())
    assert all(p not in opt.state and p.grad is None for p in model.retrieval.parameters())
    assert all(not p.requires_grad for n, p in parent.named_parameters() if "vision_tower" in n or not n.startswith("base."))
    assert any(p.requires_grad for n, p in parent.named_parameters() if "gemma_expert" in n)


def test_execution_calibration_requires_true_heldout_prefix_length():
    rows = calibration_rows()
    assert calibrate_success_scores(rows, ["1"])["heads"]["execution"]["calibrated_prefix_counts"] == [32]
    rows[0]["executed_steps"] = 16
    with pytest.raises(ValueError, match="32-step"):
        calibrate_success_scores(rows, ["1"])
