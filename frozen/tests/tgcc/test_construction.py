import numpy as np
import pytest
from tgcc.construction import (
    construct_pairs,
    CommandConstraint,
    preserved_grasp_constraints,
)
from tgcc.contracts import Evidence
from tgcc.specs import RobotActionSpec


def spec():
    return RobotActionSpec(
        tuple("a" + str(i) for i in range(23)),
        {"all": tuple(range(23))},
        ("native",) * 23,
        (-1.0,) * 23,
        (1.0,) * 23,
        (True,) * 23,
    )


def test_certified_bound_rotation_and_prefix_only():
    s = spec()
    x = np.zeros((32, 23), np.float32)
    mask = np.arange(32) < 13
    seen = set()
    for i in range(92):
        pairs, rejected = construct_pairs(x, mask, s, "same", i)
        assert len(pairs) == 1 and pairs[0].kind == "physical"
        pair = pairs[0]
        assert np.array_equal(pair.negative_native[~mask], x[~mask])
        assert pair.record()["state_transition_supervision"] is False
        assert (abs(pair.negative_native[mask]) > 1).any()
        seen.add((pair.certificate["dimension"], pair.certificate["bound_side"]))
        assert "task" in rejected
    assert len(seen) == 46


def test_task_requires_evidence_and_same_context():
    s = spec()
    x = np.zeros((32, 23), np.float32)
    mask = np.ones(32, bool)
    c = CommandConstraint(
        "no_base_motion", "a", 0, -0.1, 0.1, ("base_braced", "robot", "")
    )
    assert len(construct_pairs(x, mask, s, "a", constraints=[c])[0]) == 1
    ev = {c.evidence_key: Evidence(True, 10, 1, True, source="actual_feedback")}
    pairs, _ = construct_pairs(x, mask, s, "a", constraints=[c], evidence=ev, now=10)
    assert [p.kind for p in pairs] == ["physical", "task"]
    assert abs(pairs[1].negative_native[0, 0]) == 1
    assert not (abs(pairs[1].negative_native) > 1).any()
    assert (
        len(construct_pairs(x, mask, s, "a", constraints=[c], evidence=ev, now=20)[0])
        == 1
    )
    with pytest.raises(ValueError, match="another"):
        construct_pairs(x, mask, s, "different", constraints=[c], evidence=ev, now=10)


def test_grasp_needs_ownership_and_positive_closed_prefix():
    s = spec()
    x = np.zeros((32, 23), np.float32)
    x[:, 14] = -1
    mask = np.ones(32, bool)
    e = {("held", "hotdog", ""): Evidence(True, 1, 1, True)}
    assert not preserved_grasp_constraints("x", e, s, 1)
    e[("held", "hotdog", "left")] = Evidence(
        True, 1, 1, True, source="verified_ownership"
    )
    assert not preserved_grasp_constraints(
        "x", e, s, 1
    )  # placement may intentionally release
    cs = preserved_grasp_constraints("x", e, s, 1, preserved_objects=("hotdog",))
    pairs, _ = construct_pairs(x, mask, s, "x", constraints=cs, evidence=e, now=1)
    assert len(pairs) == 2 and pairs[1].negative_native[0, 14] == 1
    x[:, 14] = 0
    assert (
        len(construct_pairs(x, mask, s, "x", constraints=cs, evidence=e, now=1)[0]) == 1
    )


def test_invalid_positive_and_empty_prefix_never_become_rank_pairs():
    x = np.zeros((32, 23), np.float32)
    mask = np.zeros(32, bool)
    assert construct_pairs(x, mask, spec(), "x")[0] == []
    x[0, 0] = 2
    mask[:] = True
    assert construct_pairs(x, mask, spec(), "x")[0] == []
