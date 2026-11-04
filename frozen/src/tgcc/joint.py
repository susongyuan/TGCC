"""Full-policy training wrapper; all trainable forwards occur inside FSDP.

Critic auxiliary gradients are deliberately stopped at policy features. The
policy learns directly from native flow and annotated skill CE. A separately
supplied immutable teacher is optional and absent from the primary recipe.
This prevents a bound-violation ranking shortcut from distorting perception.
"""

from torch import nn
from .features import extract_pair
from .policy import PolicyFeatures
from .feature_cache import slice_observation
from .train_system import TrainSystem


class JointSystem(nn.Module):
    def __init__(self, student, teacher, spec, normalizer, config, plan):
        super().__init__()

        def storage_ids(module):
            return {
                (str(p.device), p.untyped_storage().data_ptr())
                for p in module.parameters()
            }

        if teacher is not None and (
            student is teacher or storage_ids(student) & storage_ids(teacher)
        ):
            raise ValueError("Teacher must have independent parameter storage")
        self.student = student
        self.tgcc = TrainSystem(student, spec, normalizer, config)
        object.__setattr__(self, "teacher", teacher)
        if self.teacher is not None:
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        self.tap = PolicyFeatures(student)
        self.plan = plan
        self.config = config

    def forward(self, samples, new_step):
        from .losses import local_loss_reduction

        # Each rank executes one root FSDP forward per fixed-size training unit.
        # Losses have no implicit collectives: UNKNOWN clause counts and proof
        # types can differ between ranks without changing collective ordering.
        with local_loss_reduction():
            if all("policy_sample" in s and "evaluation_sample" in s for s in samples):
                parts, counts, losses = {}, {}, []
                weight = self.config.get("evaluation_loss_weight", 1.0)
                for unit in samples:
                    for prefix, sample, scale in (
                        ("demonstration", unit["policy_sample"], 1.0),
                        ("evaluation", unit["evaluation_sample"], weight),
                    ):
                        loss, ps, cs = self._forward([sample], new_step)
                        losses.append(scale * loss / len(samples))
                        for key, value in ps.items():
                            key = prefix + "/" + key
                            parts[key] = parts.get(key, 0) + scale * value / len(samples)
                        for key, value in cs.items():
                            key = prefix + "/" + key
                            counts[key] = counts.get(key, 0) + value
                return sum(losses), parts, counts
            if any("policy_sample" in s for s in samples):
                raise ValueError("Incomplete mixed training unit")
            if any("critic_supervision" in sample for sample in samples):
                parts, counts, losses = {}, {}, []
                for sample in samples:
                    loss, ps, cs = self._forward([sample], new_step)
                    losses.append(loss)
                    for key, value in ps.items():
                        parts[key] = parts.get(key, 0) + value / len(samples)
                    for key, value in cs.items():
                        counts[key] = counts.get(key, 0) + value
                return sum(losses) / len(losses), parts, counts
            return self._forward(samples, new_step)

    def _forward(self, samples, new_step):
        joint = new_step >= self.config["critic_warmup_updates"]
        imitation = all(s.get("policy_imitation_allowed", True) for s in samples)
        self.student.eval()  # KV required; eval does not disable gradients
        if self.teacher is not None:
            self.teacher.eval()
        before, after, native, flow = extract_pair(
            self.student,
            samples,
            self.tap,
            next(self.student.parameters()).device,
            self.plan,
            with_flow=joint and imitation,
            trainable=joint and imitation,
            teacher=self.teacher if joint else None,
        )
        # Preserve the previous compact critic representation on both train and
        # inference paths; this is a stop-gradient, not a frozen policy forward.
        from .features import concatenate_observations

        device = native.device
        before = concatenate_observations(
            [slice_observation(before, i, device, True) for i in range(len(native))]
        )
        after = concatenate_observations(
            [slice_observation(after, i, device, True) for i in range(len(native))]
        )
        supervision = [s.get("critic_supervision") for s in samples]
        return self.tgcc(before, after, native, new_step, flow,
                         supervision if any(s is not None for s in supervision) else None)


def policy_parameter_group(name):
    """Partition actual pi0.5 module names without losing shared weights."""
    if not name.startswith("base."):
        return "graphs"
    if "vision_tower" in name or "multi_modal_projector" in name:
        return "vision"
    if "gemma_expert" in name:
        return "expert"
    if "paligemma" in name:
        return "vlm"
    return "action_io"


def optimizer_groups(model, config):
    policy = {name: [] for name in ("vision", "vlm", "expert", "action_io", "graphs")}
    for name, parameter in model.student.named_parameters():
        policy[policy_parameter_group(name)].append(parameter)
    return [
        dict(
            name="critic",
            params=list(model.tgcc.critic.parameters()),
            lr=config["critic_lr"],
        ),
        dict(name="adapters", params=list(model.tgcc.adapters.parameters()), lr=0.0),
    ] + [dict(name=name, params=parameters, lr=0.0) for name, parameters in policy.items()]


def learning_rates(step, config):
    import math

    def scale(s, total):
        warm = config["warmup"]
        if s < 0:
            return 0.0
        if s < warm:
            return (s + 1) / warm
        return config["min_lr_ratio"] + (1 - config["min_lr_ratio"]) * 0.5 * (
            1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))
        )

    start = config["critic_warmup_updates"]
    return [config["critic_lr"] * scale(step, config["new_updates"])] + [
        config[k] * scale(step - start, config["joint_updates"])
        for k in ("adapter_lr", "vision_lr", "vlm_lr", "expert_lr", "action_io_lr", "graph_lr")
    ]
