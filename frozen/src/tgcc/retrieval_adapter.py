"""A retrieved successful segment conditions the existing ActionExpert tokens.

Reference outcomes belong to an earlier, independently owned demonstration.
The current query's future observations never enter this adapter. Reference
absolute commands are features, not actuator commands or imitation targets.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RetrievalContext:
    normalized_actions: torch.Tensor
    action_mask: torch.Tensor
    before_features: torch.Tensor
    after_features: torch.Tensor
    valid: torch.Tensor
    after_feature_mask: torch.Tensor

    def validate(self, context_width, after_width, horizon=32):
        batch = len(self.valid)
        if (self.valid.shape != (batch,) or self.valid.dtype != torch.bool
                or self.normalized_actions.shape != (batch, horizon, 23)
                or self.action_mask.shape != (batch, horizon)
                or self.action_mask.dtype != torch.bool
                or self.before_features.shape != (batch, context_width)
                or self.after_features.shape != (batch, after_width)
                or self.after_feature_mask.shape != (batch, after_width)
                or self.after_feature_mask.dtype != torch.bool):
            raise ValueError("Retrieved segment shape/mask mismatch")
        tensors = (self.normalized_actions, self.action_mask,
                   self.before_features, self.after_features, self.after_feature_mask)
        if any(t.device != self.valid.device for t in tensors):
            raise ValueError("Retrieved segment device mismatch")
        if bool((self.valid & ~self.action_mask.any(-1)).any()):
            raise ValueError("A valid reference requires actual demonstration actions")
        if (not torch.isfinite(self.normalized_actions[self.action_mask & self.valid[:, None]]).all()
                or not torch.isfinite(self.before_features[self.valid]).all()
                or not torch.isfinite(self.after_features[self.valid[:, None] & self.after_feature_mask]).all()):
            raise ValueError("Nonfinite valid reference segment")

    def to(self, device):
        return RetrievalContext(**{name: value.to(device) for name, value in vars(self).items()})


class RetrievalAdapter(nn.Module):
    def __init__(self, expert_width=1024, context_width=32, after_width=32,
                 hidden=128, heads=4, horizon=32, initial_gate=0.0):
        super().__init__()
        if expert_width % heads or hidden < 1 or horizon != 32:
            raise ValueError("Unsupported retrieval adapter dimensions")
        self.config = dict(expert_width=expert_width, context_width=context_width,
                           after_width=after_width, hidden=hidden, heads=heads, horizon=horizon,
                           initial_gate=float(initial_gate))
        if not -0.1 <= initial_gate <= 0.1:
            raise ValueError('Reference gate initialization must be bounded')
        self.reference_action = nn.Linear(23, hidden)
        self.reference_before = nn.Linear(context_width, hidden)
        self.reference_after = nn.Linear(2 * after_width, hidden)
        self.position = nn.Embedding(horizon, hidden)
        self.endpoint_type = nn.Embedding(2, hidden)
        self.reference_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, expert_width))
        self.channels = nn.ModuleDict({name: nn.MultiheadAttention(expert_width, heads, batch_first=True)
                                       for name in ("local", "subgoal")})
        # Non-scalar gates retain exact zero bypass and support FSDP2 sharding.
        self.gates = nn.ParameterDict({name: nn.Parameter(torch.full((1,),float(initial_gate)))
                                       for name in ("local", "subgoal")})
        self.last_metrics = {}

    def record(self):
        return dict(self.config)

    def encode_reference(self, reference):
        reference.validate(self.config["context_width"], self.config["after_width"], self.config["horizon"])
        real_actions = reference.action_mask & reference.valid[:, None]
        actions = torch.where(real_actions[..., None], reference.normalized_actions,
                              torch.zeros_like(reference.normalized_actions))
        before = torch.where(reference.valid[:, None], reference.before_features,
                             torch.zeros_like(reference.before_features))
        after_known = reference.valid[:, None] & reference.after_feature_mask
        after = torch.where(after_known, reference.after_features,
                            torch.zeros_like(reference.after_features))
        after = torch.cat((after, after_known.to(after.dtype)), -1)
        dtype = self.reference_action.weight.dtype
        a = self.reference_action(actions.to(dtype)) + self.position.weight[None]
        before = self.reference_before(before.to(dtype)) + self.endpoint_type.weight[0]
        after = self.reference_after(after.to(dtype)) + self.endpoint_type.weight[1]
        tokens = self.reference_out(torch.cat((before[:, None], after[:, None], a), 1))
        mask = torch.cat((reference.valid[:, None].expand(-1, 2), real_actions), 1)
        # MultiheadAttention cannot handle entirely masked rows. Its dummy token
        # has no effect: the reference-valid mask zeros the residual afterwards.
        no_reference = ~mask.any(-1)
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        mask = mask.clone()
        mask[no_reference, 0] = True
        return tokens, mask

    def forward(self, tokens, reference=None):
        self.last_metrics = {}
        if len(tokens) != 2:
            raise ValueError("Existing local/subgoal token slots are required")
        if reference is None:
            return tokens
        reference.validate(self.config["context_width"], self.config["after_width"], self.config["horizon"])
        # Evaluation at initial zero gates reproduces the original model without
        # changed suffix lengths, extra attention, random draws or rounding.
        if not self.training and all(bool(g.detach().eq(0).all()) for g in self.gates.values()):
            return tokens
        reference_tokens, reference_mask = self.encode_reference(reference)
        outputs = []
        for name, (value, mask) in zip(("local", "subgoal"), tokens):
            if (value.ndim != 3 or mask.shape != value.shape[:2]
                    or value.shape[-1] != self.config["expert_width"]
                    or len(value) != len(reference.valid)):
                raise ValueError("ActionExpert token shape mismatch")
            if not self.training and bool(self.gates[name].detach().eq(0).all()):
                outputs.append((value, mask))
                continue
            dtype = self.channels[name].in_proj_weight.dtype
            attended = self.channels[name](value.to(dtype), reference_tokens.to(dtype),
                                           reference_tokens.to(dtype),
                                           key_padding_mask=~reference_mask, need_weights=False)[0]
            residual = attended * reference.valid[:, None, None] * mask[..., None]
            delta = (self.gates[name].tanh() * residual).to(value.dtype)
            with torch.no_grad():
                input_rms=value.float().square().mean().sqrt()
                delta_rms=delta.float().square().mean().sqrt()
                self.last_metrics[name]=dict(gate=self.gates[name].detach().tanh(),
                    input_rms=input_rms,delta_rms=delta_rms,relative_rms=delta_rms/input_rms.clamp_min(1e-8))
            outputs.append((value + delta, mask))
        return outputs
