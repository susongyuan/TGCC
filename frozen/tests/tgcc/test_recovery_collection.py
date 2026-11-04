import math

import pytest

from tgcc.recovery_collection import (
    body_velocity_to_world_target,
    certify_position_return,
    timing_summary,
)


def test_world_target_remains_correct_after_base_rotation():
    q = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    command = body_velocity_to_world_target([0, 0, 0], q, [0.01, 0])
    assert command == pytest.approx([0, -0.02, 0], abs=1e-12)
    command = body_velocity_to_world_target([0, 0, 0], q, [10, 0])
    assert math.hypot(*command[:2]) == pytest.approx(0.04)


@pytest.mark.parametrize(
    "change",
    [
        {"restore_verified": False},
        {"terminal": True},
        {"executed_steps": 0},
        {"initial_error_m": 0.001},
        {"final_error_m": 0.03},
        {"maximum_unrelated_object_displacement_m": 0.003},
    ],
)
def test_geometry_success_requires_execution_and_no_collateral_changes(change):
    values = dict(
        initial_error_m=0.02,
        final_error_m=0.002,
        executed_steps=30,
        maximum_unrelated_object_displacement_m=0.001,
        restore_verified=True,
        terminal=False,
    )
    assert certify_position_return(**values).verified
    values.update(change)
    assert not certify_position_return(**values).verified


def test_zero_yield_does_not_report_success_throughput():
    summary = timing_summary([dict(wall_seconds=2, certificate={"verified": False})])
    assert summary["seconds_per_accepted_amortized"] is None
    assert summary["acceptance_rate"] == 0
