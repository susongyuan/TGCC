"""Causal task writes shared by BPTT training and inference.

Only current skill selection is supervised in the demonstrations. The rule
manager writes PENDING/ACTIVE; unsupported physical completion stays masked.
"""

from dataclasses import replace
import hashlib
import torch

REVISION = "procedural_pending_feedback_v2"


def constrained_plan_states(decision):
    """Untrained procedural manager: selected legal skill ACTIVE, others PENDING."""
    selected = decision["selected_index"].detach()
    b, k = decision["next"].shape
    states = torch.zeros((b, k), dtype=torch.long, device=selected.device)
    row = torch.arange(b, device=selected.device)
    active = torch.ones(b, dtype=torch.bool, device=selected.device)
    if "subgoal_valid" in decision:
        active = decision["subgoal_valid"].any(-1)
    states[row, selected] = active.long()
    return states


def roll_forward_task_graph(graph, decision, slots, elapsed):
    """Insert only the preceding model write; never consult current/future labels."""
    if slots is None:
        raise ValueError("explicit fixed candidate-to-task-node slots required")
    states = constrained_plan_states(decision)
    if slots.shape != states.shape or bool((slots < 0).any()):
        raise ValueError("candidate task slot mapping mismatch")
    if elapsed.shape != (len(states),) or bool((elapsed <= 0).any()):
        raise ValueError("positive causal observation interval required")
    features = graph.features.clone()
    row = torch.arange(len(states), device=features.device)[:, None]
    features[row, -1, slots, 17:23] = torch.nn.functional.one_hot(
        states, num_classes=6
    ).to(features.dtype)
    features[row, -1, slots, 29] = 0.0  # no ML VERIFIED_DONE write
    if graph.events is None:
        events = features.new_zeros(len(states), 64, 32)
        valid = torch.zeros(len(states), 64, dtype=torch.bool, device=features.device)
    else:
        events = graph.events.clone()
        valid = graph.event_valid.clone()
    selected = decision["selected_index"].detach()
    for index in range(len(states)):
        if int(states[index, selected[index]]) != 1:
            continue
        count = int(valid[index].sum())
        position = min(count, events.shape[1] - 1)
        if count == events.shape[1]:
            events[index, :-1] = events[index, 1:].clone()
            valid[index, :-1] = valid[index, 1:].clone()
        event = features.new_zeros(32)
        event[0] = (
            0.0  # same non-physical prediction confidence convention as TaskManager events
        )
        event[1] = torch.log1p(elapsed[index]).to(features.dtype) / 4
        event[2] = (
            int.from_bytes(hashlib.sha256(b"task_state").digest()[:4], "big")
            % 256
            / 255
        )
        event[4] = 0.0
        event[5] = 1 / 5
        events[index, position] = event
        valid[index, position] = True
    return replace(graph, features=features, events=events, event_valid=valid)
