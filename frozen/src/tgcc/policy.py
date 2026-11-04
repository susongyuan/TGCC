import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, width, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        residual = self.up(
            torch.nn.functional.gelu(self.down(x.to(self.down.weight.dtype)))
        )
        return x + (self.gate.tanh() * residual).to(x.dtype)


class PolicyFeatures:
    """Read-only feature taps; hooks return None and never replace parent outputs."""

    def __init__(self, policy):
        self.values = {}
        self.hooks = []
        for name in ("global_encoder", "task_encoder", "local_encoder"):

            def tap(module, args, output, key=name):
                self.values[key] = output

            self.hooks.append(getattr(policy, name).register_forward_hook(tap))

    def clear(self):
        self.values.clear()

    def close(self):
        for h in self.hooks:
            h.remove()


class AdapterPolicy(nn.Module):
    """Zero-gated suffix adapters; teacher reuses original tokens/velocity at identical noise/time."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.parent.requires_grad_(False)
        self.local = ResidualAdapter(parent.base.action_in_proj.out_features)
        self.subgoal = ResidualAdapter(parent.base.action_in_proj.out_features)
        device = next(parent.parameters()).device
        self.local.to(device)
        self.subgoal.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def train_pair(self, observation, conditions, actions, noise, time):
        self.parent.eval()
        with torch.no_grad():
            state, valid, cache, tokens, decision = self.parent.prefix(
                observation, conditions, False
            )
        modified = [
            (adapter(t), m)
            for adapter, (t, m) in zip((self.local, self.subgoal), tokens)
        ]
        x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        with torch.no_grad():
            teacher = self.parent.velocity(state, valid, cache, tokens, x, time)
        student = self.parent.velocity(state, valid, cache, modified, x, time)
        return {
            "flow": (student[..., :23] - (noise - actions)[..., :23]).square().mean(),
            "teacher_consistency": (student[..., :23] - teacher[..., :23])
            .square()
            .mean(),
            "decision": decision,
        }

    def adapter_tokens(self, tokens):
        return [(a(t), m) for a, (t, m) in zip((self.local, self.subgoal), tokens)]

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        conditions=None,
        return_decision=False,
    ):
        self.parent.validate(conditions)
        state, valid, cache, tokens, decision = self.parent.prefix(
            observation, conditions, False
        )
        tokens = self.adapter_tokens(tokens)
        x = (
            self.parent.base.sample_noise((len(state), 32, 32), device)
            if noise is None
            else noise
        )
        if num_steps < 1:
            raise ValueError("Positive Euler steps required")
        dt = -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((len(x),), 1 + i * dt, device=device, dtype=torch.float32)
            x = x + dt * self.parent.velocity(state, valid, cache, tokens, x, t)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite adapted Euler output")
        self.parent.last_decision = decision
        return dict(actions=x, decision=decision) if return_decision else x
