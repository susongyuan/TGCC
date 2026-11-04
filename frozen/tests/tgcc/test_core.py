import dataclasses
import pytest
import torch
from tgcc.specs import RobotActionSpec, CriticObservation, ExecutedTransition
from tgcc.network import TGCC
from tgcc.policy import ResidualAdapter
from tgcc.contracts import (
    Evidence,
    Truth,
    Predicate,
    TaskContract,
    ContractEvaluator,
    build_contract,
    NegativeValidityChecker,
)


def spec():
    return RobotActionSpec(
        tuple(map(str, range(23))),
        {
            "base": (0, 1, 2),
            "left": tuple(range(7, 14)),
            "right": tuple(range(15, 22)),
            "coupled": (3, 4, 5, 6, 14, 22),
        },
        ("native",) * 23,
        (-1.0,) * 23,
        (1.0,) * 23,
        (True,) * 23,
    )


def observation(b=2):
    ts = torch.full((b,), 20.0)
    return CriticObservation(
        tuple(torch.randn(b, n, w) for n, w in [(8, 2048), (4, 2048), (8, 1024)]),
        tuple(torch.ones(b, n, dtype=torch.bool) for n in (8, 4, 8)),
        tuple(ts[:, None].expand(b, n).clone() for n in (8, 4, 8)),
        tuple(torch.ones(b, n) for n in (8, 4, 8)),
        ts,
        torch.randn(b, 256),
        torch.zeros(b, 3, dtype=torch.long),
        torch.zeros(b, 3, dtype=torch.long),
        torch.randn(b, 3, 8),
        torch.zeros(b, 3, dtype=torch.long),
        torch.ones(b, 3, dtype=torch.bool),
        torch.zeros(b, 3),
        torch.ones(b, 3, dtype=torch.bool),
        torch.randn(b, 16, 240),
        torch.ones(b, 16, dtype=torch.bool),
        torch.arange(4.0, 20.0).expand(b, 16),
        torch.randn(b, 32, 23),
        torch.ones(b, 32, dtype=torch.bool),
    )


def test_action_groups_partition_trunk():
    s = spec()
    assert torch.equal(s.masks().sum(0), torch.ones(23))
    assert s.masks()[3, 3:7].all()


def test_bad_partition_rejected():
    with pytest.raises(ValueError):
        dataclasses.replace(spec(), groups={"base": (0, 1, 2)})


def test_forward_shapes_and_budget():
    m = TGCC(spec(), 240).eval()
    for b in (1, 2):
        y = m.forward_action(observation(b))
        assert y["group_energy"].shape == (b, 4)
        assert torch.isfinite(y["action_energy_total"]).all()
    assert sum(p.numel() for p in m.parameters()) < 8_000_000


def test_future_graph_is_rejected():
    o = observation()
    o.graph_timestamps[0][0, 0] = 21
    with pytest.raises(ValueError):
        o.validate()


def test_history_current_exposure_rejected():
    o = observation()
    o.history_timestamps[0, 0] = 20
    with pytest.raises(ValueError):
        o.validate()


def test_padding_cannot_change_score():
    torch.manual_seed(42)
    m = TGCC(spec(), 240).eval()
    o = observation()
    o.history_mask[:, :4] = False
    o.candidate_action_mask[:, 20:] = False
    o.graph_masks[0][:, :2] = False
    a = m.forward_action(o)["action_energy_total"]
    o.history[:, :4] = 100000
    o.candidate_actions[:, 20:] = 100000
    o.graph_tokens[0][:, :2] = 100000
    b = m.forward_action(o)["action_energy_total"]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_forward_cannot_read_post_state():
    m = TGCC(spec(), 240).eval()
    o = observation()
    after = observation()
    a = m.forward_action(o)["action_energy_total"]
    after.graph_tokens[0].add_(1e4)
    b = m.forward_action(o)["action_energy_total"]
    assert torch.equal(a, b)


def test_retrospective_time_alignment():
    o = observation()
    a = observation()
    a.timestamp += 32 / 30
    a.history_timestamps = a.history_timestamps - 1
    tr = ExecutedTransition(
        o, a, o.candidate_actions, o.candidate_action_mask, o.timestamp, a.timestamp
    )
    y = TGCC(spec(), 240).eval().retrospective(tr)
    assert y["predicate_logits"].shape == (2, 3)
    tr.end_time = tr.end_time + 1
    with pytest.raises(ValueError):
        tr.validate()


def test_unknown_is_not_false_completion():
    c = build_contract({"id": "open", "skill_ids": [10], "targets": [["fridge"]]})
    e = ContractEvaluator()
    for t in (0.0, 1.0, 2.0, 3.0):
        assert not e.evaluate("ep", c, {}, t)["completed"]


def test_completion_three_distinct_valid_exposures():
    c = TaskContract(10, "open", (Predicate("open", "fridge"),))
    e = ContractEvaluator()
    for t, done in [(0.0, False), (0.0, False), (1.0, False), (2.0, True)]:
        assert (
            e.evaluate(
                "ep", c, {("open", "fridge", ""): Evidence(True, t, 1.0, True)}, t
            )["completed"]
            == done
        )
    assert not e.evaluate("ep", c, {}, 3.0)["completed"]


def test_future_and_stale_evidence_unknown():
    assert Evidence(True, 5.0, 1.0, True).truth(4.0) == Truth.UNKNOWN
    assert Evidence(True, 0.0, 1.0, True).truth(4.0) == Truth.UNKNOWN


def test_negative_only_with_known_limit_proof():
    s = spec()
    check = NegativeValidityChecker(s)
    a = torch.zeros(2, 32, 23)
    mask = torch.ones(2, 32, dtype=torch.bool)
    assert not check.violations(a, mask).any()
    neg = check.generate(a, mask, lambda x: x)
    assert len(neg) == 3 and all(p.all() for _, p, _ in neg)
    assert not NegativeValidityChecker(
        dataclasses.replace(s, bounds_valid=(False,) * 23)
    ).generate(a, mask, lambda x: x)






def test_zero_adapter_exact_bypass_and_gate_grad():
    a = ResidualAdapter(1024)
    x = torch.randn(2, 8, 1024)
    assert torch.equal(a(x), x)
    a(x).sum().backward()
    assert a.gate.grad is not None and torch.isfinite(a.gate.grad)
