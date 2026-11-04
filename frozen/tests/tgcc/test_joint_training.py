"""CPU contract tests for the two-stage full-policy continuation recipe."""

import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from test_core import spec, observation
from tgcc.checkpoint import digest, save_checkpoint
from tgcc.joint import JointSystem, learning_rates, optimizer_groups, policy_parameter_group
from tgcc.joint_checkpoint import load_student, migrate_legacy
from tgcc.train_system import TrainSystem


@pytest.fixture
def config():
    return json.loads(
        (Path(__file__).parents[2] / "configs/tgcc/default.json").read_text()
    )


class TinyParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.action_in_proj = torch.nn.Linear(3, 16)
        self.base.transformer = torch.nn.Linear(16, 16)
        self.base.paligemma_with_expert = torch.nn.Module()
        self.base.paligemma_with_expert.paligemma = torch.nn.Module()
        self.base.paligemma_with_expert.paligemma.vision_tower = torch.nn.Linear(4, 4)
        self.base.paligemma_with_expert.paligemma.multi_modal_projector = torch.nn.Linear(4, 4)
        self.base.paligemma_with_expert.paligemma.language_model = torch.nn.Linear(4, 4)
        self.base.paligemma_with_expert.gemma_expert = torch.nn.Linear(4, 4)
        self.global_encoder = torch.nn.Linear(4, 4)
        self.task_encoder = torch.nn.Linear(4, 4)
        self.local_encoder = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.GRU(4, 4)


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.ood = torch.nn.Linear(2, 1)


class TinyJoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student = TinyParent()
        self.tgcc = torch.nn.Module()
        self.tgcc.critic = TinyCritic()
        self.tgcc.adapters = torch.nn.Linear(2, 2)


class IdentityNormalizer:
    def transform(self, x):
        return x


@pytest.mark.parametrize(
    "step,expected",
    [
        (99, [1e-4, 0.0, 0.0, 0.0]),
        (100, [1e-4, 0.0, 0.0, 0.0]),
        (999, [9.494188609197971e-5, 0.0, 0.0, 0.0]),
        (1000, [9.493084337754573e-5, 2e-8, 2e-8, 1e-7]),
        (
            5999,
            [
                1.000000637937644e-5,
                2.00000184978002e-7,
                2.00000184978002e-7,
                1.00000092489001e-6,
            ],
        ),
    ],
)
def test_seven_group_learning_rates_at_phase_boundaries(config, step, expected):
    assert config["critic_warmup_updates"] == 1000
    assert config["joint_updates"] == 5000
    assert config["new_updates"] == 6000
    expected = [expected[0], expected[1], expected[2] / 2, expected[2] / 2,
        expected[2], expected[2], expected[3]]
    assert learning_rates(step, config) == pytest.approx(expected, rel=1e-12)


def test_optimizer_groups_partition_all_student_and_new_parameters_once(config):
    model = TinyJoint()
    model.tgcc.critic.ood.requires_grad_(False)
    groups = optimizer_groups(model, config)
    assert [g["name"] for g in groups] == ["critic", "adapters", "vision", "vlm", "expert", "action_io", "graphs"]
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in model.parameters()}
    for group in groups[2:]:
        assert {id(p) for p in group["params"]} == {
            id(p) for n, p in model.student.named_parameters()
            if policy_parameter_group(n) == group["name"]
        }
        assert group["params"]
    assert all(g["lr"] == 0 for g in groups[1:])


def test_zero_lr_in_warmup_is_not_used_as_substitute_for_none_gradients(config):
    """The script must clear inactive grads to avoid silently advancing Adam.

    This checks the intended optimizer contract: no parent moments are created
    during the remaining critic warmup; the first joint update starts at t=1.
    """
    model = TinyJoint()
    model.tgcc.critic.ood.requires_grad_(False)
    opt = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=0.01)
    parent_before = {k: v.clone() for k, v in model.student.state_dict().items()}
    for step in (100, 999):
        opt.zero_grad(set_to_none=True)
        for group, lr in zip(opt.param_groups, learning_rates(step, config)):
            group["lr"] = lr
        sum(
            p.square().sum() for p in model.tgcc.critic.parameters() if p.requires_grad
        ).backward()
        opt.step()
    assert all(p not in opt.state for p in model.student.parameters())
    assert all(p not in opt.state for p in model.tgcc.adapters.parameters())
    assert all(
        torch.equal(v, parent_before[k]) for k, v in model.student.state_dict().items()
    )
    opt.zero_grad(set_to_none=True)
    for group, lr in zip(opt.param_groups, learning_rates(1000, config)):
        group["lr"] = lr
    sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
    opt.step()
    assert all(int(opt.state[p]["step"]) == 1 for p in model.student.parameters())
    assert all(
        p.grad is None and p not in opt.state
        for p in model.tgcc.critic.ood.parameters()
    )


def test_joint_teacher_is_independent_frozen_and_excluded_from_optimizer(config):
    student = TinyParent()
    teacher = copy.deepcopy(student)
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    model = JointSystem(student, teacher, spec(), IdentityNormalizer(), config, {})
    try:
        student_ids = {id(p) for p in student.parameters()}
        teacher_ids = {id(p) for p in teacher.parameters()}
        assert student_ids.isdisjoint(teacher_ids)
        assert not any(p.requires_grad for p in teacher.parameters())
        assert not teacher.training
        assert all(p.requires_grad for p in student.parameters())
        assert not any(p.requires_grad for p in model.tgcc.critic.ood.parameters())
        groups = optimizer_groups(model, config)
        assert teacher_ids.isdisjoint({id(p) for g in groups for p in g["params"]})
        assert not any(k.startswith("teacher.") for k in model.state_dict())
        opt = torch.optim.AdamW(groups)
        for group, lr in zip(opt.param_groups, learning_rates(1000, config)):
            group["lr"] = lr
        sum(p.square().sum() for p in student.parameters()).backward()
        opt.step()
        assert all(
            torch.equal(v, teacher_before[k]) for k, v in teacher.state_dict().items()
        )
    finally:
        model.tap.close()


def test_joint_rejects_aliased_teacher_before_freezing_student(config):
    parent = TinyParent()
    with pytest.raises(
        (ValueError, AssertionError), match="[Tt]eacher|[Ii]ndependent|[Ss]har"
    ):
        JointSystem(parent, parent, spec(), IdentityNormalizer(), config, {})
    assert all(p.requires_grad for p in parent.parameters())


def test_joint_rejects_different_parameters_with_shared_teacher_storage(config):
    student = TinyParent()
    teacher = copy.deepcopy(student)
    teacher.global_encoder.weight = torch.nn.Parameter(
        student.global_encoder.weight.data
    )
    assert teacher.global_encoder.weight is not student.global_encoder.weight
    with pytest.raises(
        (ValueError, AssertionError), match="[Tt]eacher|[Ii]ndependent|[Ss]har"
    ):
        JointSystem(student, teacher, spec(), IdentityNormalizer(), config, {})


def write_legacy_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(31)
    np.random.seed(32)
    random.seed(33)
    old = TinyJoint()
    old_optimizer = torch.optim.AdamW(
        [
            {"params": list(old.tgcc.critic.parameters()), "lr": 1e-4},
            {"params": list(old.tgcc.adapters.parameters()), "lr": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    # Match the real step100 checkpoint: critic has Adam moments; adapters have
    # no optimizer state because their grads were None throughout warmup.
    for _ in range(100):
        old_optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in old.tgcc.critic.parameters()).backward()
        old_optimizer.step()
    saved = save_checkpoint(
        tmp_path,
        old.tgcc,
        old_optimizer,
        100,
        4800,
        dict(parent_model_sha256="audited-parent", source_sha256={"fixture": "legacy"}),
    )
    sha = digest(saved / "tgcc.safetensors")
    return old, old_optimizer, saved, sha


def test_legacy_migration_preserves_adam_rng_cursor_without_resetting_new_schedule(
    tmp_path, monkeypatch, config
):
    old, old_opt, saved, sha = write_legacy_checkpoint(tmp_path, monkeypatch)
    expected_random = (random.random(), np.random.rand(3), torch.rand(3))
    new = TinyJoint()
    new.tgcc.load_state_dict(load_file(str(saved / "tgcc.safetensors")), strict=True)
    new.tgcc.critic.ood.requires_grad_(False)
    opt = torch.optim.AdamW(
        optimizer_groups(new, config), betas=(0.8, 0.99), eps=1e-6, weight_decay=0.2
    )
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    step, cursor, migration = migrate_legacy(saved, new, opt, 0, "audited-parent", sha)
    assert (step, cursor) == (100, 4800)
    assert migration["bitwise_old_recipe_continuation"] is False
    assert random.random() == expected_random[0]
    np.testing.assert_array_equal(np.random.rand(3), expected_random[1])
    assert torch.equal(torch.rand(3), expected_random[2])
    for old_p, new_p in zip(old.tgcc.critic.parameters(), new.tgcc.critic.parameters()):
        assert torch.equal(old_p, new_p)
        assert int(opt.state[new_p]["step"]) == 100
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(opt.state[new_p][key], old_opt.state[old_p][key])
    assert all(p not in opt.state for p in new.tgcc.adapters.parameters())
    assert all(p not in opt.state for p in new.student.parameters())
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)
    assert opt.param_groups[0]["weight_decay"] == 0.01
    assert [g["lr"] for g in opt.param_groups] == [config["critic_lr"]] + [0.0] * 6
    assert not any(p.requires_grad for p in new.tgcc.critic.ood.parameters())


@pytest.mark.parametrize("field,value", [("new_updates", 99), ("data_cursor", 4752)])
def test_legacy_migration_rejects_invalid_step_or_cursor(
    tmp_path, monkeypatch, config, field, value
):
    _, _, saved, sha = write_legacy_checkpoint(tmp_path, monkeypatch)
    resume = torch.load(saved / "resume.pt", weights_only=False)
    resume[field] = value
    torch.save(resume, saved / "resume.pt")
    new = TinyJoint()
    with pytest.raises(ValueError, match="counter|cursor"):
        migrate_legacy(
            saved,
            new,
            torch.optim.AdamW(optimizer_groups(new, config)),
            0,
            "audited-parent",
            sha,
        )


def test_legacy_migration_rejects_wrong_parent_and_corrupt_weights(
    tmp_path, monkeypatch, config
):
    _, _, saved, sha = write_legacy_checkpoint(tmp_path, monkeypatch)
    new = TinyJoint()
    opt = torch.optim.AdamW(optimizer_groups(new, config))
    with pytest.raises(ValueError, match="audited"):
        migrate_legacy(saved, new, opt, 0, "different-parent", sha)
    with (saved / "tgcc.safetensors").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        migrate_legacy(saved, new, opt, 0, "audited-parent", sha)


def test_full_student_export_load_restores_tied_alias_and_checks_identity(tmp_path):
    model = torch.nn.Module()
    model.embedding = torch.nn.Linear(3, 4, bias=False)
    model.alias = torch.nn.Linear(3, 4, bias=False)
    model.alias.weight = model.embedding.weight
    weights = {"embedding.weight": torch.arange(12.0).reshape(4, 3)}
    save_file(weights, str(tmp_path / "student.safetensors"))
    md = dict(
        schema="tgcc.joint.v2",
        student_sha256=digest(tmp_path / "student.safetensors"),
        parent_aliases={"alias.weight": "embedding.weight"},
    )
    assert load_student(model, tmp_path, md) is model
    assert model.alias.weight is model.embedding.weight
    assert torch.equal(model.embedding.weight, weights["embedding.weight"])
    with (tmp_path / "student.safetensors").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="identity"):
        load_student(model, tmp_path, md)


def test_teacher_free_joint_registers_only_student_critic_and_adapters(config):
    student = TinyParent()
    model = JointSystem(student, None, spec(), IdentityNormalizer(), config, {})
    try:
        assert model.teacher is None
        assert all(p.requires_grad for p in student.parameters())
        assert not any(k.startswith("teacher.") for k in model.state_dict())
        groups = optimizer_groups(model, config)
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids))
        assert set(ids) == {id(p) for p in model.parameters()}
    finally:
        model.tap.close()


def test_teacher_none_trains_native_flow_and_skill_without_reference_velocity(config):
    class FlowParent(TinyParent):
        def __init__(self):
            super().__init__()
            self.velocity_calls = 0
            self.velocity_weight = torch.nn.Parameter(torch.tensor(0.3))

        def velocity(self, state, valid, cache, tokens, x, time):
            self.velocity_calls += 1
            return x * self.velocity_weight + tokens[0][0].mean()

    parent = FlowParent()
    system = TrainSystem(parent, spec(), IdentityNormalizer(), config).eval()
    before = observation(1)
    from tgcc.data import HISTORY_WIDTH

    before.history = torch.zeros(1, 16, HISTORY_WIDTH)
    after = copy.deepcopy(before)
    after.timestamp += 32 / 30
    actions = torch.zeros(1, 32, 23)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.5])
    token = torch.randn(1, 2, 16, requires_grad=True)
    logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
    skill = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    context = [(torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), None,
        [(token, torch.ones(1, 2, dtype=torch.bool))] * 2,
        actions, noise, time, None, skill)]
    loss, parts, _ = system(before, after, actions, 1000, context)
    assert parent.velocity_calls == 1
    assert "teacher_consistency" not in parts
    assert parts["skill"].item() > 0 and parts["flow"].item() > 0
    loss.backward()
    assert parent.velocity_weight.grad.abs().item() > 0
    assert token.grad.abs().sum().item() > 0
    assert logits.grad.abs().sum().item() > 0
