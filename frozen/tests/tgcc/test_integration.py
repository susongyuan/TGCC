import copy
import numpy as np
import pytest
import torch
from tgcc.checkpoint import save_checkpoint, load_checkpoint
from tgcc.contracts import build_contract, ContractEvaluator, Evidence
from tgcc.data import transition_labels
from tgcc.transport import ExecutedClient, native_hold
from test_core import observation, spec
from tgcc.network import TGCC


def test_exact_resume_restores_optimizer_rng_and_data_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def make():
        torch.manual_seed(91)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.Dropout(0.2),
            torch.nn.GELU(),
            torch.nn.Linear(8, 1),
        )
        return model, torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.95))

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(3, 5)
        y = torch.randn(3, 1)
        loss = (model(x) - y).square().mean()
        loss.backward()
        opt.step()

    model, opt = make()
    for _ in range(5):
        step(model, opt)
    expected = copy.deepcopy(model.state_dict())
    expected_random = torch.rand(4)
    model, opt = make()
    for _ in range(2):
        step(model, opt)
    path = save_checkpoint(
        tmp_path, model, opt, 2, 96, dict(source_sha256={"fixture": "unchanged"})
    )
    for _ in range(17):
        torch.rand(4)
    resumed, resumed_opt = make()
    count, cursor = load_checkpoint(
        path, resumed, resumed_opt, {"fixture": "unchanged"}
    )
    assert (count, cursor) == (2, 96)
    for _ in range(3):
        step(resumed, resumed_opt)
    for name, p in resumed.state_dict().items():
        assert torch.equal(p, expected[name]), name
    assert torch.equal(torch.rand(4), expected_random)
    assert all(int(s["step"]) == 5 for s in resumed_opt.state.values())
    with pytest.raises(ValueError):
        load_checkpoint(path, resumed, resumed_opt, {"fixture": "changed"})


def test_cooking_contract_requires_food_inside_even_when_switch_is_on():
    c = build_contract(
        {"id": "on", "skill_ids": [69], "targets": [["microwave.n.02_1"]]},
        ("hotdog.n.02_1", "hotdog.n.02_2"),
    )
    ev = ContractEvaluator()
    for t in range(4):
        evidence = {
            ("observed", "microwave.n.02_1", ""): Evidence(True, t, 1.0, True),
            ("open", "microwave.n.02_1", ""): Evidence(False, t, 1.0, True),
            ("toggled_on", "microwave.n.02_1", ""): Evidence(True, t, 1.0, True),
        }
        result = ev.evaluate("ep", c, evidence, t)
        assert not result["completed"]
    for t in range(4, 7):
        evidence = {
            ("observed", "microwave.n.02_1", ""): Evidence(True, t, 1.0, True),
            ("open", "microwave.n.02_1", ""): Evidence(False, t, 1.0, True),
            ("toggled_on", "microwave.n.02_1", ""): Evidence(True, t, 1.0, True),
            ("inside", "hotdog.n.02_1", "microwave.n.02_1"): Evidence(
                True, t, 1.0, True
            ),
            ("inside", "hotdog.n.02_2", "microwave.n.02_1"): Evidence(
                True, t, 1.0, True
            ),
        }
        result = ev.evaluate("ep", c, evidence, t)
    assert result["completed"]


def test_actual_pick_transition_produces_positive_progress_without_time_labels():
    before = observation(1)
    after = observation(1)
    before.contract_category[:] = torch.tensor([[0, 1, 3]])
    after.contract_category = before.contract_category
    before.contract_values[:] = torch.tensor([[1.0, 0.0, 0.0]])
    after.contract_values[:] = torch.tensor([[0.0, 1.0, 1.0]])
    label = transition_labels(before, after)
    assert (
        label["transition"].item() == 1.0
        and label["progress"].item() == 1.0
        and label["handoff"].item() == 1.0
    )
    after.contract_valid[:, 1:] = False
    label = transition_labels(before, after)
    assert not label["progress_valid"].item() and not label["handoff_valid"].item()


def test_nan_padding_cannot_pollute_valid_forward_values():
    o = observation(1)
    m = TGCC(spec(), 240).eval()
    o.graph_masks[0][:, :2] = False
    o.history_mask[:, :4] = False
    o.candidate_action_mask[:, 20:] = False
    clean = m.forward_action(o)["action_energy_total"]
    o.graph_tokens[0][:, :2] = float("nan")
    o.graph_timestamps[0][:, :2] = float("nan")
    o.history[:, :4] = float("nan")
    o.candidate_actions[:, 20:] = float("nan")
    assert torch.equal(clean, m.forward_action(o)["action_energy_total"])


def test_execution_ack_checks_actual_controller_permutation():
    c = ExecutedClient(lambda: None, "fixture")
    native = np.arange(23, dtype=np.float32)
    # A different robot-wide ordering is legal only through the audited mapping.
    names = {
        "base": range(0, 3),
        "trunk": range(3, 7),
        "arm_left": range(7, 14),
        "gripper_left": [14],
        "arm_right": range(15, 22),
        "gripper_right": [22],
    }
    permutation = np.arange(23)[::-1]
    indices = {k: permutation[list(v)] for k, v in names.items()}
    actual = np.zeros(23, np.float32)
    actual[permutation] = native
    c.last_native = native.copy()
    c.acknowledge_step(actual, indices)
    assert np.array_equal(c.executed[0], native)
    c.last_native = native.copy()
    actual[0] += 1
    with pytest.raises(ValueError):
        c.acknowledge_step(actual, indices)


def test_measured_hold_inverts_real_gripper_output_scaling():
    raw = {
        "state": dict(
            trunk_qpos=np.zeros(4),
            left_arm=np.zeros(7),
            right_arm=np.zeros(7),
            left_gripper=[0.02, 0.02],
            right_gripper=[0.05, 0.05],
        ),
        "control_contract": {
            "controllers": {
                k: dict(
                    command_input_limits=[[-1.0], [1.0]],
                    command_output_limits=[[0.0], [0.05]],
                )
                for k in ("gripper_left", "gripper_right")
            }
        },
        "graph_snapshot": {"entities": [], "edges": []},
    }
    a = native_hold(raw)
    assert a.shape == (23,) and abs(a[14] + 0.2) < 1e-6 and abs(a[22] - 1.0) < 1e-6
    assert not a[:3].any()
