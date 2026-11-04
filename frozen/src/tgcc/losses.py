import torch
from contextlib import contextmanager
from contextvars import ContextVar
from torch.nn import functional as F

_local_reduction = ContextVar("tgcc_local_loss_reduction", default=False)


@contextmanager
def local_loss_reduction():
    """Use per-sample means without hidden distributed collectives.

    Isolated supplemental samples have different known clauses and certificate
    counts on each rank. A collective inside their loss can deadlock. The joint
    wrapper gives every training unit equal weight, averaging known clauses
    within that unit; FSDP then averages those gradients across ranks.
    """
    token = _local_reduction.set(True)
    try:
        yield
    finally:
        _local_reduction.reset(token)


def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    count = mask.sum().detach()
    import torch.distributed as dist

    world = 1
    if dist.is_initialized() and not _local_reduction.get():
        dist.all_reduce(count)
        world = dist.get_world_size()
    return (x * mask).sum() * world / count.clamp_min(1)


def critic_loss(
    action_positive,
    negative_outputs,
    negative_valid,
    state,
    labels,
    margin=0.2,
    weights=None,
):
    zero = action_positive["action_energy_total"].sum() * 0
    rank = zero
    if negative_outputs:
        terms = torch.stack(
            [
                F.relu(
                    margin
                    + action_positive["action_energy_total"]
                    - x["action_energy_total"]
                )
                for x in negative_outputs
            ],
            -1,
        )
        # Balance proof types, not their dataset frequency. Unknown task
        # negatives stay absent; they are never filled with unproven failures.
        per_type = [
            masked_mean(terms[:, i], negative_valid[:, i])
            for i in range(terms.shape[1])
        ]
        active = negative_valid.sum(0).detach()
        import torch.distributed as dist

        if dist.is_initialized() and not _local_reduction.get():
            dist.all_reduce(active)
        rank = sum(per_type) / (active > 0).sum().clamp_min(1)
    parts = {"rank": rank}
    counts = {"rank": int(negative_valid.sum())}
    for key, logit, label, valid in [
        ("transition", "transition_logit", "transition", "transition_valid"),
        ("predicate", "predicate_logits", "predicate", "predicate_valid"),
        ("handoff", "handoff_logit", "handoff", "handoff_valid"),
    ]:
        mask = labels[valid]
        target = labels[label]
        parts[key] = masked_mean(
            F.binary_cross_entropy_with_logits(
                state[logit], target.float(), reduction="none"
            ),
            mask,
        )
        counts[key] = int(mask.sum())
    parts["progress"] = masked_mean(
        F.smooth_l1_loss(state["progress"], labels["progress"], reduction="none"),
        labels["progress_valid"],
    )
    counts["progress"] = int(labels["progress_valid"].sum())
    weights = weights or dict(
        rank=1.0, transition=1.0, predicate=1.0, progress=0.5, handoff=0.5
    )
    total = sum(weights[k] * v for k, v in parts.items())
    if negative_outputs:
        for name, value in zip(("rank_physical", "rank_task"), per_type):
            parts[name] = value.detach()
    return total, parts, counts
