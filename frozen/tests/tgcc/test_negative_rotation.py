"""Training negatives cover native limits without corrupting masked steps."""

import torch

from test_core import spec
from tgcc.contracts import NegativeValidityChecker


def test_training_stride_covers_subsystems_sides_and_durations():
    checker = NegativeValidityChecker(spec())
    native = torch.zeros(2, 32, 23)
    valid = torch.zeros(2, 32, dtype=torch.bool)
    valid[0, 5:20] = True
    seen = set()
    for step in range(92):
        [(negative, proof, certificate)] = checker.generate(
            native, valid, lambda x: x, max_count=1, offset=step * 3
        )
        assert proof.tolist() == [True, False]
        assert torch.equal(negative[~valid], native[~valid])
        delta = (negative != native).any(-1)
        expected = 1 if certificate["duration"] == "impulse" else 15
        assert delta[0].sum() == expected
        seen.add(
            (
                certificate["dimension"],
                certificate["limit_side"],
                certificate["duration"],
            )
        )
    assert len(seen) == 23 * 2 * 2


def test_same_precision_bound_is_legal_and_real_excess_is_not():
    checker = NegativeValidityChecker(spec())
    native = torch.ones(1, 32, 23, dtype=torch.float64)
    valid = torch.ones(1, 32, dtype=torch.bool)
    assert not checker.violations(native, valid).any()
    native[0, 3, 4] += 1e-8
    assert checker.violations(native, valid).all()
