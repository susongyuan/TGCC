"""Full TC-ESGR + pi fine-tuning losses; labels without evidence are masked."""

import torch
from torch.nn import functional as F


def full_parameter_groups(model, base_lr=1e-5, graph_lr=1e-4):
    model.requires_grad_(True)
    base = []
    graph = []
    names = []
    for name, p in model.named_parameters():
        names.append(name)
        (base if name.startswith("base.") else graph).append(p)
    return [{"params": base, "lr": base_lr}, {"params": graph, "lr": graph_lr}], names


def per_sample_ce(logits, target, mask):
    values = F.cross_entropy(
        logits.flatten(0, 1), target.flatten(), reduction="none"
    ).reshape_as(target)
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def objective_vector(
    result, labels, state_weight=0.01, next_weight=0.01, goal_weight=0.01
):
    flow = result["action_loss"][..., :23].float().mean((1, 2))
    decision = result["decision"]
    # TaskManager is procedural. There is no learned task-state/goal classifier.
    state = torch.zeros_like(flow)
    nxt = (
        F.cross_entropy(decision["next"].float(), labels["next"], reduction="none")
        * labels["next_mask"]
    )
    goal = torch.zeros_like(flow)
    return flow + next_weight * nxt, dict(flow=flow, state=state, next=nxt, goal=goal)
