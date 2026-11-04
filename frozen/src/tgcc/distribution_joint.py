"""Success-only likelihoods plus retrieval-conditioned Action Expert flow.

The index encoder, vision tower, VLM and three-graph/semantic modules stay
frozen. The original native action targets are the only policy imitation
targets; reference demonstrations are conditioning, never replacement labels.
"""

import math
from dataclasses import replace

import torch
from torch import nn

from .distribution_critic import DistributionCritic, distribution_loss


def trainable_policy_name(name):
    if not name.startswith("base."):
        return False
    # The expert returns hidden states to action_out_proj; its unused language
    # vocabulary head is never part of the suffix flow and must not be gathered
    # as if it were a trainable 257k-token actuator output.
    if "gemma_expert.lm_head." in name:
        return False
    return ("gemma_expert" in name or name.split(".")[1] in (
        "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
        "state_proj", "action_time_mlp_in", "action_time_mlp_out"))


def freeze_index_policy(student):
    student.requires_grad_(False)
    allowed = []
    for name, parameter in student.named_parameters():
        if trainable_policy_name(name):
            parameter.requires_grad_(True)
            allowed.append(name)
    if not allowed or not any("gemma_expert" in n for n in allowed):
        raise ValueError("No actual Action Expert parameters found in tested policy")
    return allowed


class DistributionJointSystem(nn.Module):
    def __init__(self, student, retrieval_adapter, distribution_config, recipe):
        super().__init__()
        self.student = student
        self.policy_trainable_names = freeze_index_policy(student)
        self.critic = DistributionCritic(distribution_config)
        self.retrieval = retrieval_adapter
        self.recipe = recipe

    def forward(self, batch, new_step):
        density = batch["density"]
        output = self.critic(density)
        total, parts, counts = distribution_loss(output, density, self.recipe["loss_weights"])
        if new_step < self.recipe["critic_warmup_updates"]:
            return total, parts, counts
        references = batch["references"]
        samples = batch["policy_samples"]
        if len(samples) != len(density["context"]) or len(references) != 1:
            raise ValueError("Policy/density/reference training unit mismatch")
        query_episodes = batch.get("query_episode_ids")
        reference_episodes = batch.get("reference_episode_ids")
        if (query_episodes is None or reference_episodes is None or
                len(query_episodes) != len(samples) or len(reference_episodes) != 1 or
                any(len(ids) != len(samples) for ids in reference_episodes) or
                any(str(q) == str(r) for ids in reference_episodes for q, r in zip(query_episodes, ids))):
            raise ValueError("Retrieval must exclude the query's whole source episode")
        for sample in samples:
            if (sample.get("policy_imitation_allowed", True) is not True or
                    sample.get("source_kind", "demonstration") != "demonstration" or
                    "critic_supervision" in sample):
                raise ValueError("Only original successful demo targets may train Action Expert")
        from tgr.data import collate_sequences
        from openpi.models.model import Observation
        from .data import conditions_from_dict, move_tree

        device = next(self.student.parameters()).device
        policy_batch = move_tree(collate_sequences(samples), device)
        self.student.eval()  # Prefix KV is required, HF checkpoint mode is disabled.
        flows = []
        # The second exposure is the observed after-target, not a second
        # Action Expert target or a future feature for this prefix.
        for k, item in enumerate(policy_batch["sequence"][:1]):
            conditions = conditions_from_dict(item["conditions"])
            with torch.no_grad():
                state, valid, cache, tokens, decision = self.student.prefix(
                    Observation.from_dict(item["observation"]), conditions, False)
            reference = references[k]
            if reference is None:
                raise ValueError("Joint training requires a source-isolated retrieved demo")
            dropout = self.recipe.get("retrieval_conditioning_dropout", 0.25)
            if not 0 <= dropout <= 1:
                raise ValueError("Invalid retrieval-conditioning dropout")
            if self.training and dropout:
                keep = torch.rand(len(reference.valid), device=reference.valid.device) >= dropout
                reference = replace(reference, valid=reference.valid & keep)
            counts["retrieval_conditioned"] = int(reference.valid.sum())
            counts["retrieval_unassisted"] = int((~reference.valid).sum())
            tokens = self.retrieval(tokens, reference)
            actions = item["actions"]
            noise = self.student.base.sample_noise(actions.shape, device)
            time = self.student.base.sample_time(len(actions), device)
            x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            velocity = self.student.velocity(state, valid, cache, tokens, x, time)
            flows.append((velocity[..., :23] - (noise - actions)[..., :23]).square().mean())
        flow = torch.stack(flows).mean()
        parts["flow"] = flow
        counts["flow_sequences"] = len(samples)
        total = total + self.recipe["loss_weights"]["flow"] * flow
        return total, parts, counts


def distribution_optimizer_groups(model, recipe):
    return [dict(name="critic", params=list(model.critic.parameters()), lr=recipe["critic_lr"]),
        dict(name="retrieval", params=list(model.retrieval.parameters()), lr=0.0),
        dict(name="action_expert", params=[p for p in model.student.parameters() if p.requires_grad], lr=0.0)]


def effective_distribution_loss(parts, counts, global_counts, world, accumulation, weights):
    """An exact effective-batch mean despite differing UNKNOWN masks.

    FSDP averages gradients across ranks. Each local sum therefore receives
    world/global-known-count; averaging already normalized local means would
    overweight microbatches with only one known planning label.
    """
    if world < 1 or accumulation < 1:
        raise ValueError("Invalid distributed effective-batch dimensions")
    total = sum(weights[head] * parts[head] * counts[head] * world /
                max(global_counts[head], 1) for head in ("action", "execution", "planning"))
    if "flow" in parts:
        total = total + weights["flow"] * parts["flow"] / accumulation
    if 'reference_improvement' in parts:
        total += weights.get('reference_improvement',0)*parts['reference_improvement']/accumulation
    return total


def distribution_learning_rates(step, recipe):
    def scale(position, total):
        if position < 0:
            return 0.0
        warm = recipe["warmup"]
        if position < warm:
            return (position + 1) / max(warm, 1)
        decay = 0.5 * (1 + math.cos(math.pi * min(1, (position - warm) / max(1, total - warm))))
        return recipe["min_lr_ratio"] + (1 - recipe["min_lr_ratio"]) * decay
    joint = step - recipe["critic_warmup_updates"]
    return [recipe["critic_lr"] * scale(step, recipe["new_updates"]),
        recipe["retrieval_adapter_lr"] * scale(joint, recipe["joint_updates"]),
        recipe["action_expert_lr"] * scale(joint, recipe["joint_updates"])]
