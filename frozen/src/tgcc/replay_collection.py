"""New simulator executions of demonstration commands with measured grasp retries.

The source action sequence is not evidence that its replay succeeded. All
failure/recovery labels require actual feedback after native physics steps.
"""

import time
import numpy as np


def reexecute(
    actions,
    step_native,
    held_feedback,
    *,
    progress=None,
    max_injections=12,
    release_steps=4,
    retry_window=128,
    on_state_change=None,
    on_before_action=None,
):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 23 or not np.isfinite(actions).all():
        raise ValueError("finite native23 demonstration commands required")
    if not 1 <= release_steps <= 8 or retry_window <= release_steps:
        raise ValueError("bounded release and verification window required")
    start = time.monotonic()
    events = []
    active = None
    cooldown_until = 64
    executed = 0
    for index, original in enumerate(actions):
        feedback = held_feedback()
        if active is None and index >= cooldown_until and len(events) < max_injections:
            for (obj, side), value in feedback.items():
                dimension = {"left": 14, "right": 22}[side]
                planned = actions[index : index + 32, dimension]
                if value is True and len(planned) == 32 and np.all(planned < -0.5):
                    active = dict(
                        object_id=obj,
                        side=side,
                        dimension=dimension,
                        start_step=index,
                        observed_loss=False,
                        verified_recovered=False,
                        source="controlled_native_release_then_demonstration_retry",
                        stage_constraint="preserve_verified_current_grasp",
                        original_plan_keeps_gripper_closed=True,
                        states=[],
                        status="executing",
                    )
                    events.append(active)
                    break
        action = original.copy()
        if active is not None and index - active["start_step"] < release_steps:
            action[active["dimension"]] = 1.0
        annotation = dict(source_step=index, original_native=original.tolist(),
                          is_controlled_release=not np.array_equal(action, original),
                          stage_constraint=active["stage_constraint"] if active else None,
                          intervention_object=active["object_id"] if active else None,
                          intervention_side=active["side"] if active else None,
                          first_intervention_step=bool(active and index == active["start_step"]))
        if on_before_action is not None:
            # The exposure immediately BEFORE a controlled intervention must
            # exist, rather than an earlier 32-step sample.
            on_before_action(annotation)
        terminal = step_native(action)
        executed += 1
        if active is not None:
            measured = held_feedback().get((active["object_id"], active["side"]))
            active["states"].append(dict(step=executed, held=measured))
            newly_lost = measured is False and not active["observed_loss"]
            active["observed_loss"] |= measured is False
            if newly_lost and on_state_change is not None:
                on_state_change()
            if (
                active["observed_loss"]
                and measured is True
                and index - active["start_step"] >= release_steps
            ):
                active.update(
                    verified_recovered=True,
                    status="verified_recovered",
                    end_step=executed,
                )
                if on_state_change is not None:
                    on_state_change()
                cooldown_until = executed + retry_window
                active = None
            elif index - active["start_step"] >= retry_window or terminal:
                active.update(
                    status="recovery_failed"
                    if active["observed_loss"]
                    else "no_verified_failure",
                    end_step=executed,
                )
                cooldown_until = executed + retry_window
                active = None
        if progress is not None and (executed % 32 == 0 or terminal):
            progress(
                dict(
                    executed_steps=executed,
                    source_steps=len(actions),
                    elapsed_seconds=time.monotonic() - start,
                    events=events,
                )
            )
        if terminal:
            break
    if active is not None:
        active.update(status="incomplete_verification", end_step=executed)
    return dict(
        executed_steps=executed,
        source_steps=len(actions),
        elapsed_seconds=time.monotonic() - start,
        events=events,
        verified_recoveries=sum(e["verified_recovered"] for e in events),
        verified_failures=sum(e["observed_loss"] for e in events),
        source_actions_are_not_assumed_successful=True,
        no_human_recovery_demonstrations=True,
    )
