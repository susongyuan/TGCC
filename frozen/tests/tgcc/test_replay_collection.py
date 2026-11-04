import numpy as np

from tgcc.replay_collection import reexecute


def test_recovery_label_requires_measured_loss_then_real_regain():
    actions = np.zeros((100, 23), np.float32)
    actions[:, 14] = -1
    state = {"held": True}
    exposures = []

    def step(action):
        state["held"] = bool(action[14] < 0)
        return False

    result = reexecute(
        actions,
        step,
        lambda: {("food", "left"): state["held"]},
        max_injections=1,
        on_state_change=lambda: exposures.append(state["held"]),
    )
    assert result["verified_failures"] == 1
    assert result["verified_recoveries"] == 1
    assert exposures == [False, True]
    assert np.all(actions[:, 14] == -1)  # source actions never mutated


def test_command_corruption_without_actual_loss_is_not_recovery_success():
    actions = np.zeros((100, 23), np.float32)
    actions[:, 14] = -1
    result = reexecute(
        actions, lambda a: False, lambda: {("food", "left"): True}, max_injections=1
    )
    assert result["verified_failures"] == 0
    assert result["verified_recoveries"] == 0
    assert result["events"][0]["status"] == "incomplete_verification"


def test_unknown_grasp_and_planned_release_do_not_trigger_preserve_grasp_corruption():
    actions = np.zeros((100, 23), np.float32)
    for feedback in ({("food", "left"): None}, {("food", "left"): True}):
        result = reexecute(actions, lambda a: False, lambda: feedback, max_injections=1)
        assert result["events"] == []
