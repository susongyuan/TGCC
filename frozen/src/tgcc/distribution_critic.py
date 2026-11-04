"""Success-only conditional likelihoods; a low density is not a failure label.

Targets are fixed, checkpoint-owned normalized actions, actually observed
after-features, and annotated subtask selection. No synthetic actions, recovery
classes, task-completion labels or counterfactual after states are used.
"""

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DistributionConfig:
    context_width: int
    after_width: int
    planning_classes: int
    hidden: int = 256
    horizon: int = 32
    action_width: int = 23
    action_mixtures: int = 4
    execution_mixtures: int = 2
    minimum_scale: float = 0.05
    maximum_scale: float = 5.0

    def __post_init__(self):
        if (min(self.context_width, self.after_width, self.planning_classes,
                self.hidden, self.horizon, self.action_width,
                self.action_mixtures, self.execution_mixtures) < 1 or
                not 0 < self.minimum_scale < self.maximum_scale):
            raise ValueError("Invalid conditional-density dimensions/scale bounds")

    def record(self):
        return asdict(self)


class ConditionalMixture(nn.Module):
    """Normalized diagonal Gaussian mixture with bounded nonzero variance.

    Mixture density includes its normalization and mixture weights. It cannot
    minimize positive-only energy merely by subtracting an arbitrary constant.
    The positive scale floor prevents unbounded likelihood on memorized points.
    NLL is reported per observed target dimension to compare partial prefixes.
    """

    def __init__(self, hidden, width, components, minimum_scale, maximum_scale):
        super().__init__()
        self.width, self.components = width, components
        self.minimum_log_scale = math.log(minimum_scale)
        self.maximum_log_scale = math.log(maximum_scale)
        self.means = nn.Linear(hidden, components * width)
        self.scales = nn.Linear(hidden, components * width)
        self.weights = nn.Linear(hidden, components)
        # Start near sigma=.5 instead of near either bounding saturation.
        fraction = (math.log(0.5) - self.minimum_log_scale) / (
            self.maximum_log_scale - self.minimum_log_scale)
        fraction = min(max(fraction, 0.01), 0.99)
        nn.init.constant_(self.scales.bias, math.log(fraction / (1 - fraction)))

    def parameters_for(self, hidden):
        shape = (len(hidden), self.components, self.width)
        mean = self.means(hidden).reshape(shape)
        raw = self.scales(hidden).reshape(shape)
        log_scale = self.minimum_log_scale + raw.sigmoid() * (
            self.maximum_log_scale - self.minimum_log_scale)
        log_weights = self.weights(hidden).log_softmax(-1)
        return mean, log_scale, log_weights

    def nll(self, hidden, target, valid=None):
        if target.shape != (len(hidden), self.width):
            raise ValueError("Conditional density target shape mismatch")
        valid = torch.ones_like(target, dtype=torch.bool) if valid is None else valid
        if valid.shape != target.shape or valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError("Density requires at least one observed target dimension")
        if not torch.isfinite(target[valid]).all():
            raise ValueError("Nonfinite observed density target")
        # NaN in a masked, unexecuted suffix cannot contaminate arithmetic.
        target = torch.where(valid, target, torch.zeros_like(target)).float()
        mean, log_scale, weights = self.parameters_for(hidden.float())
        component_logp = -0.5 * ((target[:, None] - mean) * (-log_scale).exp()).square()
        component_logp = component_logp - log_scale - 0.5 * math.log(2 * math.pi)
        component_logp = torch.where(valid[:, None], component_logp,
                                     torch.zeros_like(component_logp)).sum(-1)
        logp = torch.logsumexp(component_logp + weights, -1)
        return -logp / valid.sum(-1)


class DistributionCritic(nn.Module):
    schema = "tgcc.success_distribution.v1"

    def __init__(self, config):
        super().__init__()
        self.config = config if isinstance(config, DistributionConfig) else DistributionConfig(**config)
        c = self.config
        self.context = nn.Sequential(nn.LayerNorm(c.context_width),
            nn.Linear(c.context_width, c.hidden), nn.GELU(),
            nn.Linear(c.hidden, c.hidden), nn.GELU())
        self.executed_action = nn.Sequential(
            nn.Linear(c.horizon * (c.action_width + 1), c.hidden), nn.GELU())
        self.execution_context = nn.Sequential(nn.Linear(2 * c.hidden, c.hidden), nn.GELU())
        self.action_density = ConditionalMixture(c.hidden, c.horizon * c.action_width,
            c.action_mixtures, c.minimum_scale, c.maximum_scale)
        self.execution_density = ConditionalMixture(c.hidden, c.after_width,
            c.execution_mixtures, c.minimum_scale, c.maximum_scale)
        self.planning = nn.Linear(c.hidden, c.planning_classes)

    def encode_context(self, context):
        if context.ndim != 2 or context.shape[-1] != self.config.context_width or not torch.isfinite(context).all():
            raise ValueError("Nonfinite/wrong-width causal context")
        return self.context(context.float())

    def action_inputs(self, actions, mask):
        c = self.config
        if (actions.shape != (len(actions), c.horizon, c.action_width) or
                mask.shape != actions.shape[:2] or mask.dtype != torch.bool or
                not mask.any(-1).all() or not torch.isfinite(actions[mask]).all()):
            raise ValueError("Wrong shape/mask or nonfinite actual action prefix")
        actions = torch.where(mask[..., None], actions, torch.zeros_like(actions)).float()
        return actions, mask[..., None].expand_as(actions).reshape(len(actions), -1)

    def forward_action(self, context, actions, mask):
        hidden = self.encode_context(context)
        actions, valid = self.action_inputs(actions, mask)
        return {"action_nll": self.action_density.nll(hidden, actions.flatten(1), valid)}

    def score_execution(self, context, actual_actions, after_features, mask, after_mask=None):
        hidden = self.encode_context(context)
        actions, _ = self.action_inputs(actual_actions, mask)
        action = torch.cat((actions, mask[..., None].to(actions.dtype)), -1).flatten(1)
        hidden = self.execution_context(torch.cat((hidden, self.executed_action(action)), -1))
        return {"execution_nll": self.execution_density.nll(hidden, after_features, after_mask)}

    def score_planning(self, context, next_index, eligible=None):
        logits = self.planning(self.encode_context(context))
        if eligible is not None:
            if eligible.shape != logits.shape or eligible.dtype != torch.bool or not eligible.any(-1).all():
                raise ValueError("Invalid planning candidate mask")
            # Eligibility controls dispatch separately. Renormalizing the
            # likelihood after a changing legal mask would invalidate the
            # same-vocabulary train/calibration/runtime threshold.
        if next_index.shape != (len(context),) or next_index.dtype != torch.long:
            raise ValueError("Invalid planning candidate index")
        if ((next_index < 0) | (next_index >= logits.shape[-1])).any():
            raise ValueError("Planning index outside frozen vocabulary")
        return {"planning_nll": -logits.log_softmax(-1).gather(1, next_index[:, None])[:, 0],
                "planning_logits": logits}

    def forward(self, batch):
        # This method has no after->action/planning dependence. The after-target
        # is detached even if an upstream caller accidentally enables its grad.
        output = self.forward_action(batch["context"], batch["actions"], batch["action_mask"])
        output.update(self.score_execution(batch["context"], batch["actions"],
                                          batch["after_features"].detach(), batch["action_mask"],
                                          batch.get("after_feature_mask")))
        output.update(self.score_planning(batch["context"], batch["next_index"], batch.get("eligible")))
        return output


def distribution_loss(output, batch, weights=None):
    """Local means only: no label-dependent distributed collectives."""
    weights = weights or {"action": 1.0, "execution": 1.0, "planning": 0.2}
    parts, counts = {}, {}
    for name in ("action", "execution", "planning"):
        nll = output[name + "_nll"]
        mask = batch.get(name + "_valid", torch.ones_like(nll, dtype=torch.bool))
        if mask.shape != nll.shape or mask.dtype != torch.bool:
            raise ValueError("Invalid density supervision mask")
        if not torch.isfinite(nll[mask]).all():
            raise FloatingPointError("Nonfinite valid conditional likelihood")
        safe = torch.where(mask, nll, torch.zeros_like(nll))
        parts[name] = safe.sum() / mask.sum().clamp_min(1)
        counts[name] = int(mask.sum())
    return sum(weights[k] * v for k, v in parts.items()), parts, counts
