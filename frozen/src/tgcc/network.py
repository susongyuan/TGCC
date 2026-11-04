import torch
from torch import nn
from .specs import GraphFeatureSpec


class FusionBlock(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.cross = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.self_attention = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, d)
        )

    def forward(self, q, m, mask):
        x = self.norms[0](q)
        q = (
            q
            + self.cross(
                x,
                self.norms[1](m),
                self.norms[1](m),
                key_padding_mask=~mask,
                need_weights=False,
            )[0]
        )
        x = self.norms[2](q)
        q = q + self.self_attention(x, x, x, need_weights=False)[0]
        return q + self.ff(self.norms[3](q))


class ActionEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.register_buffer("groups", spec.masks())
        self.proj = nn.Linear(spec.width, 128)
        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Linear(128, 256)
        self.group = nn.Embedding(4, 256)
        self.position = nn.Embedding(8, 256)

    def forward(self, a, valid):
        b, h, d = a.shape
        g = len(self.groups)
        a = torch.where(valid[..., None], a, torch.zeros_like(a))
        x = a[:, None] * self.groups[None, :, None]
        x = self.proj(x).reshape(b * g, h, 128).transpose(1, 2)
        # Mask after each conv so padded positions cannot create observations.
        mask = valid[:, None].expand(b, g, h).reshape(b * g, 1, h)
        x = x * mask
        for layer in self.conv:
            x = layer(x) * mask
        pooled = nn.functional.adaptive_avg_pool1d(x, 8)
        counts = nn.functional.adaptive_avg_pool1d(mask.float(), 8)
        pooled = pooled / counts.clamp_min(1e-6)
        x = self.out(pooled.transpose(1, 2)).reshape(b, g, 8, 256)
        x = x + self.group.weight[None, :, None] + self.position.weight[None, None]
        return x.reshape(b, g * 8, 256), counts.reshape(b, g * 8) > 0


class TGCC(nn.Module):
    def __init__(
        self, action_spec, history_width, spec=GraphFeatureSpec(), predicate_types=128
    ):
        super().__init__()
        self.action_spec = action_spec
        self.spec = spec
        self.graph = nn.ModuleList(
            [
                nn.Linear(w, 256)
                for w in (spec.global_width, spec.task_width, spec.local_width)
            ]
        )
        self.graph_type = nn.Embedding(3, 256)
        self.graph_meta = nn.Linear(2, 256)
        self.history_input = nn.Linear(history_width, 128)
        self.history_gru = nn.GRU(128, 128, batch_first=True)
        self.history_out = nn.Linear(128, 256)
        self.action = ActionEncoder(action_spec)
        self.predicate_type = nn.Embedding(predicate_types, 64)
        self.entities = nn.Embedding(512, 64)
        self.numeric = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64))
        self.category = nn.Embedding(16, 64)
        self.contract_out = nn.Linear(256, 256)
        self.semantic = nn.Linear(spec.semantic_width, 256)
        self.queries = nn.Parameter(torch.randn(4, 256) * 0.02)
        self.robot = nn.Linear(64, 256)
        self.task_identity = nn.Embedding(64, 256, padding_idx=0)
        self.skill_identity = nn.Embedding(128, 256, padding_idx=0)
        self.fusion = nn.ModuleList([FusionBlock(), FusionBlock()])
        self.energy = nn.Linear(256, 1)
        self.weights = nn.Parameter(torch.zeros(4))
        self.ood = nn.Linear(256, 1)
        # Legacy checkpoint storage only. No forward use, loss or recovery gate.
        self.ood.requires_grad_(False)
        self.transition_before = nn.Linear(256, 256)
        self.transition_after = nn.Linear(256, 256)
        self.endpoint_type = nn.Embedding(2, 256)
        self.state_predicate = nn.Linear(256, 1)
        self.state_match = nn.Linear(1024, 1)
        self.progress = nn.Linear(1024, 1)
        self.handoff = nn.Linear(1024, 1)
        n = sum(p.numel() for p in self.parameters())
        if n > 8_000_000:
            raise ValueError(f"Critic parameter budget exceeded: {n}")

    def encode(self, o, actions=None, action_mask=None):
        o.validate()
        tokens = []
        masks = []
        for i, (x, m, t, c) in enumerate(
            zip(o.graph_tokens, o.graph_masks, o.graph_timestamps, o.graph_confidences)
        ):
            age = (o.timestamp[:, None] - t).clamp_min(0).float()
            valid = m & (age <= self.spec.max_age_seconds) & (c > 0)
            age = torch.where(valid, age, torch.zeros_like(age))
            x = torch.where(valid[..., None], x, torch.zeros_like(x))
            v = (
                self.graph[i](x.float())
                + self.graph_type.weight[i]
                + self.graph_meta(torch.stack([torch.log1p(age), c.float()], -1))
            )
            tokens.append(v)
            masks.append(valid)
        history = torch.where(
            o.history_mask[..., None], o.history, torch.zeros_like(o.history)
        )
        x = self.history_input(history.float())
        out = []
        h = x.new_zeros(1, x.shape[0], 128)
        for i in range(x.shape[1]):
            y, proposed = self.history_gru(x[:, i : i + 1], h)
            h = torch.where(o.history_mask[:, i][None, :, None], proposed, h)
            out.append(h[0])
        tokens.append(self.history_out(torch.stack(out, 1)))
        masks.append(o.history_mask)
        a, m = self.action(
            o.candidate_actions if actions is None else actions,
            o.candidate_action_mask if action_mask is None else action_mask,
        )
        tokens.append(a)
        masks.append(m)
        c = torch.cat(
            [
                self.predicate_type(o.contract_ids),
                self.entities(o.contract_entities),
                self.numeric(o.contract_numeric.float()),
                self.category(o.contract_category),
            ],
            -1,
        )
        contract = self.contract_out(c)
        tokens.append(contract)
        masks.append(o.contract_mask)
        semantic = self.semantic(o.semantic_context.float())
        if o.task_ids is not None:
            semantic = semantic + self.task_identity(o.task_ids)
        if o.skill_ids is not None:
            semantic = semantic + self.skill_identity(o.skill_ids)
        robot = (
            semantic.new_zeros(len(semantic), 64)
            if o.robot_current is None
            else o.robot_current.float()
        )
        tokens.append(self.robot(robot)[:, None])
        masks.append(
            torch.full(
                (len(semantic), 1),
                o.robot_current is not None,
                dtype=torch.bool,
                device=semantic.device,
            )
        )
        tokens.append(semantic[:, None])
        masks.append(
            torch.ones(len(semantic), 1, dtype=torch.bool, device=semantic.device)
        )
        return torch.cat(tokens, 1), torch.cat(masks, 1), semantic, contract

    def fuse(self, m, mask, context):
        q = self.queries[None] + context[:, None]
        for block in self.fusion:
            q = block(q, m, mask)
        return q

    def forward_action(self, o):
        m, mask, c, contract = self.encode(o)
        q = self.fuse(m, mask, c)
        energy = self.energy(q).squeeze(-1)
        return dict(
            group_energy=energy,
            action_energy_total=(energy * self.weights.softmax(-1)).sum(-1),
            precondition_logits=self.state_predicate(contract + q[:, 0, None]).squeeze(
                -1
            ),
        )

    def retrospective(self, tr):
        tr.validate()
        b, bm, bc, contract = self.encode(
            tr.before, tr.executed_actions, tr.executed_mask
        )
        # after.candidate_actions are not outcomes: explicitly replace with actual executed prefix.
        a, am, ac, _ = self.encode(tr.after, tr.executed_actions, tr.executed_mask)
        b = self.transition_before(b) + self.endpoint_type.weight[0]
        a = self.transition_after(a) + self.endpoint_type.weight[1]
        q = self.fuse(torch.cat([b, a], 1), torch.cat([bm, am], 1), bc)
        flat = q.flatten(1)
        logits = self.state_predicate(contract + q[:, 0, None]).squeeze(-1)
        return dict(
            predicate_logits=logits,
            transition_logit=self.state_match(flat).squeeze(-1),
            progress=self.progress(flat).squeeze(-1),
            handoff_logit=self.handoff(flat).squeeze(-1),
        )
