"""CPU gradient proof for unequal local groups with one final rank reduction.

This proves effective-batch weighting, not NCCL/FSDP state-machine behavior.
Run against the eventual candidate source to exercise its loss implementation.
"""
import torch
import pytest


@pytest.mark.parametrize('micro', [1, 2, 3, 4, 6, 12])
def test_variable_local_groups_equal_global_batch48_gradient(micro):
    from tgcc.distribution_joint import effective_distribution_loss
    rng = torch.Generator().manual_seed(321)
    x = torch.randn(48, 3, generator=rng, dtype=torch.float64)
    targets = torch.randn(48, 3, generator=rng, dtype=torch.float64)
    # Genuine partially observed labels, including all-UNKNOWN local groups.
    masks = torch.rand(48, 3, generator=rng) > .45
    masks[:8, 2] = False
    flow_target = torch.randn(48, generator=rng, dtype=torch.float64)
    hinge_target = torch.randn(48, generator=rng, dtype=torch.float64)
    skill_mask = torch.rand(48, generator=rng) > .35
    world, accumulation = 4, 12 // micro
    weights = dict(action=1., execution=1., planning=.2, flow=1., reference_improvement=.2)
    heads = ('action', 'execution', 'planning')
    global_counts = {name: int(masks[:, k].sum()) for k, name in enumerate(heads)}
    global_skills = int(skill_mask.sum())
    initial = torch.tensor([.4, -.3, .2], dtype=torch.float64)
    w_global = initial.clone().requires_grad_()
    prediction = x * w_global
    density_error = (prediction - targets).square()
    global_loss = sum(weights[name] * density_error[masks[:, k], k].mean()
                      for k, name in enumerate(heads))
    prediction_sum = prediction.sum(-1)
    global_loss = global_loss + (prediction_sum-flow_target).square().mean()
    global_loss = global_loss + .2 * torch.relu(prediction_sum-hinge_target).mean()
    global_loss = global_loss + .01 * prediction_sum[skill_mask].square().mean()
    global_loss.backward()
    rank_gradients, rank_losses, group_counts = [], [], []
    for rank in range(world):
        w_local = initial.clone().requires_grad_()
        local_loss, local_groups = 0., 0
        rank_indices = list(range(rank, 48, world))
        for start in range(0, 12, micro):
            indices = rank_indices[start:start+micro]
            # The ranks deliberately have different forward/backward counts.
            groups = {}
            for i in indices:
                groups.setdefault((i // 4) % (rank+1), []).append(i)
            for group in groups.values():
                idx = torch.tensor(group)
                p = x[idx] * w_local
                errors = (p-targets[idx]).square()
                counts = {name: int(masks[idx, k].sum()) for k, name in enumerate(heads)}
                parts = {name: torch.where(masks[idx, k], errors[:, k], torch.zeros_like(errors[:, k])).sum()
                         / max(counts[name], 1) for k, name in enumerate(heads)}
                p_sum = p.sum(-1)
                parts['flow'] = (p_sum-flow_target[idx]).square().mean()
                parts['reference_improvement'] = torch.relu(p_sum-hinge_target[idx]).mean()
                scale = accumulation * micro / len(group)
                loss = effective_distribution_loss(parts, counts, global_counts, world, scale, weights)
                loss += .01 * (p_sum.square()*skill_mask[idx]).sum() * world / global_skills
                # All local backwards accumulate before the sole rank average.
                loss.backward()
                local_loss += float(loss.detach())
                local_groups += 1
        rank_gradients.append(w_local.grad)
        rank_losses.append(local_loss)
        group_counts.append(local_groups)
    assert torch.allclose(torch.stack(rank_gradients).mean(0), w_global.grad, atol=1e-12, rtol=1e-12)
    assert abs(sum(rank_losses)/world-float(global_loss.detach())) < 1e-12
    if micro > 1:
        assert len(set(group_counts)) > 1
