"""TC-ESGR V2: typed sparse aggregation, routed queries, and one task memory."""

from dataclasses import dataclass
import torch
from torch import nn

ROLE_UNKNOWN = 0
ROLE_OBJECT = 1
ROLE_BASE = 2
ROLE_TRUNK = 3
ROLE_LEFT_ARM = 4
ROLE_RIGHT_ARM = 5
ROLE_LEFT_EE = 6
ROLE_RIGHT_EE = 7
ROLE_OBSTACLE = 8
ROLE_PREDICATE = 9
ROLE_PLAN = 10
ROLE_FEASIBILITY = 11


@dataclass
class GraphBatch:
    features: torch.Tensor
    semantics: torch.Tensor
    relations: torch.Tensor
    valid: torch.Tensor
    oracle: bool = False
    roles: torch.Tensor | None = None
    relevance: torch.Tensor | None = None
    events: torch.Tensor | None = None
    event_valid: torch.Tensor | None = None
    joints: torch.Tensor | None = None
    joint_roles: torch.Tensor | None = None
    pair_nodes: torch.Tensor | None = None


class TypedMessageAggregation(nn.Module):
    """Permutation-invariant directed typed messages; no node self-attention."""

    def __init__(self, d=256, relations=32):
        super().__init__()
        self.relation = nn.Embedding(relations, 32)
        self.message = nn.Sequential(
            nn.Linear(2 * d + 32, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x, relations, valid):
        b, n, d = x.shape
        edge = (relations != 0) & valid[:, :, None] & valid[:, None, :]
        bi, source, target = edge.nonzero(as_tuple=True)
        aggregate = torch.zeros_like(x).reshape(b * n, d)
        degree = x.new_zeros(b * n, 1)
        if bi.numel():
            message = self.message(
                torch.cat(
                    (
                        x[bi, source],
                        x[bi, target],
                        self.relation(relations[bi, source, target]),
                    ),
                    dim=-1,
                )
            )
            destination = bi * n + target
            aggregate.index_add_(0, destination, message.to(aggregate.dtype))
            degree.index_add_(0, destination, x.new_ones(len(destination), 1))
        return (
            self.norm(
                x + aggregate.reshape(b, n, d) / degree.reshape(b, n, 1).clamp_min(1)
            )
            * valid[..., None]
        )


class RoutedQueryPool(nn.Module):
    def __init__(self, d, out_dim, tokens):
        super().__init__()
        self.query = nn.Parameter(torch.randn(tokens, d) * 0.02)
        self.condition = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x, valid, context, allowed=None):
        query = self.query[None] + self.condition(context)[:, None]
        mask = valid[:, None, :] if allowed is None else allowed & valid[:, None, :]
        scores = (query @ self.key(x).transpose(-1, -2)) / x.shape[-1] ** 0.5
        weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        output_mask = mask.any(-1)
        if output_mask.shape[1] == 1:
            output_mask = output_mask.expand(len(x), len(self.query))
        return self.out(weights @ self.value(x)) * output_mask[..., None], output_mask


class StateEmbedding(nn.Module):
    def __init__(self, features, d=256):
        super().__init__()
        self.feature = nn.Linear(features, d)
        self.semantic = nn.Embedding(256, d)
        self.role = nn.Embedding(12, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, g):
        roles = g.roles if g.roles is not None else torch.zeros_like(g.semantics)
        # Graph scaling is fixed and documented; official pi normalization is untouched.
        features = g.features.clone()
        features[..., 1] = torch.log1p(features[..., 1].clamp_min(0)) / 4
        features[..., 4:7] = features[..., 4:7] * 0.2
        return (
            self.norm(
                self.feature(features) + self.semantic(g.semantics) + self.role(roles)
            )
            * g.valid[..., None]
        )


class GlobalGraphEncoder(nn.Module):
    def __init__(self, features, out_dim, tokens=8, d=256):
        super().__init__()
        self.embedding = StateEmbedding(features, d)
        self.aggregate = TypedMessageAggregation(d)
        self.pool = RoutedQueryPool(d, out_dim, tokens)

    def forward(self, g, context):
        x = self.embedding(g)[:, -1]
        valid = g.valid[:, -1]
        relations = g.relations[:, -1]
        x = self.aggregate(x, relations, valid)
        if g.relevance is not None:
            retrieved = g.relevance[:, -1] & valid
            neighbours = ((relations != 0) & retrieved[:, :, None]).any(1) | (
                (relations != 0) & retrieved[:, None, :]
            ).any(2)
            valid = valid & (retrieved | neighbours)
        return self.pool(x, valid, context)


class LocalInteractionEncoder(nn.Module):
    """Independent two-token Left/Right/Base/Target queries with compact history."""

    def __init__(self, features, out_dim, tokens=8, d=256):
        super().__init__()
        if tokens != 8:
            raise ValueError("V2 uses two queries per Left/Right/Base/Target role")
        self.embedding = StateEmbedding(features, d)
        self.aggregate = TypedMessageAggregation(d)
        self.event_gate = nn.Linear(2 * d, d)
        self.event_value = nn.Linear(d, d)
        self.joint = nn.Linear(4, d)
        self.pool = RoutedQueryPool(d, out_dim, tokens)

    def forward(self, g, context, target_mask=None):
        embedded = self.embedding(g)
        current = embedded[:, -1]
        valid = g.valid[:, -1]
        # Historical changes are gated weighted sums, not per-node recurrent networks.
        previous = embedded[:, :-1]
        if previous.shape[1]:
            present = g.valid[:, :-1]
            summary = (previous * present[..., None]).sum(1) / present.sum(1).clamp_min(
                1
            )[..., None]
            gate = torch.sigmoid(self.event_gate(torch.cat((current, summary), -1)))
            current = current + gate * self.event_value(current - summary)
        x = self.aggregate(current, g.relations[:, -1], valid)
        roles = (
            g.roles[:, -1]
            if g.roles is not None
            else torch.zeros_like(valid, dtype=torch.long)
        )
        object_mask = (
            (roles == ROLE_OBJECT)
            | (roles == ROLE_OBSTACLE)
            | (roles == ROLE_FEASIBILITY)
        )
        if target_mask is not None:
            object_mask = object_mask & target_mask
        robot_mask = (roles >= ROLE_BASE) & (roles <= ROLE_RIGHT_EE)
        if target_mask is not None and g.pair_nodes is not None:
            # Keep actual body anchors, filter derived body-to-unrelated-target pairs.
            robot_mask = robot_mask & (~g.pair_nodes[:, -1] | target_mask)
        mask = valid & (object_mask | robot_mask)
        left = ((roles == ROLE_LEFT_ARM) | (roles == ROLE_LEFT_EE) | object_mask) & mask
        right = (
            (roles == ROLE_RIGHT_ARM) | (roles == ROLE_RIGHT_EE) | object_mask
        ) & mask
        base = ((roles == ROLE_BASE) | (roles == ROLE_TRUNK) | object_mask) & mask
        target = (object_mask | robot_mask) & mask
        allowed = torch.stack((left, left, right, right, base, base, target, target), 1)
        if g.joints is not None:
            jx = self.joint(g.joints)
            jvalid = g.joints[..., 3] > 0
            jr = g.joint_roles
            jallowed = (
                torch.stack(
                    (
                        jr == 1,
                        jr == 1,
                        jr == 2,
                        jr == 2,
                        jr == 0,
                        jr == 0,
                        jvalid,
                        jvalid,
                    ),
                    1,
                )
                & jvalid[:, None]
            )
            x = torch.cat((x, jx), 1)
            valid = torch.cat((valid, jvalid), 1)
            allowed = torch.cat((allowed, jallowed), 2)
        return self.pool(x, valid, context, allowed)


class TaskGraphMemory(nn.Module):
    """Current predicate/plan references plus bounded typed event summaries."""

    def __init__(self, features, out_dim, tokens=4, d=256):
        super().__init__()
        self.embedding = StateEmbedding(features, d)
        self.aggregate = TypedMessageAggregation(d)
        self.event = nn.Sequential(nn.Linear(32, d), nn.SiLU(), nn.Linear(d, d))
        self.memory = nn.Linear(d, d)
        self.pool = RoutedQueryPool(d, out_dim, tokens)

    def forward(self, g, context, memory=None):
        x = self.embedding(g)[:, -1]
        valid = g.valid[:, -1]
        x = self.aggregate(x, g.relations[:, -1], valid)
        if g.events is not None and g.event_valid is not None:
            x = torch.cat((x, self.event(g.events)), 1)
            valid = torch.cat((valid, g.event_valid), 1)
        if memory is not None:
            context = context + self.memory(memory)
        return self.pool(x, valid, context)


class TaskDecisionDecoder(nn.Module):
    """Learn current skill/subgoal and one causal summary; statuses are procedural."""

    def __init__(self, vlm_dim, expert_dim, d=256):
        super().__init__()
        self.vlm = nn.Linear(vlm_dim, d)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True)
        self.next = nn.Linear(d, 1)
        self.subgoal = nn.Linear(d, 2 * expert_dim)
        self.expert_dim = expert_dim
        self.memory = nn.GRUCell(d, d)
        self.memory_out = nn.Linear(d, d)

    def forward(self, hidden, valid, queries, legal, memory=None, event_changed=None):
        x = self.vlm(hidden)
        b = len(x)
        if memory is None:
            memory = x.new_zeros(b, x.shape[-1])
        attended, _ = self.cross(queries, x, x, key_padding_mask=~valid)
        pooled = (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        proposed = self.memory(pooled, memory)
        new_memory = self.memory_out(proposed)
        if event_changed is not None:
            # The carried memory is already projected. A no-event update is
            # an exact hold, rather than applying the projection again.
            new_memory = torch.where(event_changed[:, None], new_memory, memory)
        # The current semantic decision reads the updated memory after prefix;
        # only the next chunk may place it back into VLM task tokens.
        nodes = queries + attended + new_memory[:, None]
        logits = self.next(nodes).squeeze(-1).masked_fill(~legal, -1e4)
        weights = logits.softmax(-1) * legal
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        context = (nodes * weights[..., None]).sum(1)
        decision = dict(
            next=logits,
            context=context,
            selected_index=logits.argmax(-1),
            subgoal=self.subgoal(context).reshape(b, 2, self.expert_dim),
            subgoal_valid=legal.any(-1)[:, None].expand(b, 2),
            memory=new_memory,
            memory_tokens=None,
        )
        from .task_feedback import constrained_plan_states

        decision["plan_states"] = constrained_plan_states(decision)
        return decision
