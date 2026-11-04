"""TC-ESGR V2 composition; pristine baseline files and original flow/control semantics."""

from dataclasses import dataclass
import torch
from torch import nn
from .encoders import (
    GlobalGraphEncoder,
    LocalInteractionEncoder,
    TaskGraphMemory,
    TaskDecisionDecoder,
)

MODES = ("baseline", "graph_shadow", "tc_esgr", "tri_graph")


@dataclass
class Conditions:
    global_graph: object = None
    local_graph: object = None
    task_graph: object = None
    task_condition: object = None
    queries: object = None
    legal: object = None
    memory: object = None
    memory_tokens: object = None
    oracle: bool = False
    restricted: bool = True
    candidate_mask: object = None
    event_changed: object = None
    task_candidate_slots: object = None


class TriGraphPolicy(nn.Module):
    def __init__(self, base, features=32, hidden=256, mode="baseline"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        if not base.pi05:
            raise ValueError("Only actual pi0.5 PyTorch path supported")
        self.base = base
        self.mode = "tc_esgr" if mode == "tri_graph" else mode
        self.trained = False
        self.architecture_version = "tc_esgr_v2"
        self.feature_schema = "tc_esgr.feature32.v3.approx_center_geometry"
        self.features = features
        self.hidden = hidden
        self.last_decision = None
        vd = base.paligemma_with_expert.paligemma.config.text_config.hidden_size
        ed = base.action_in_proj.out_features
        # Adding modules does not consume baseline RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.global_encoder = GlobalGraphEncoder(features, vd, 8, hidden)
            self.local_encoder = LocalInteractionEncoder(features, ed, 8, hidden)
            self.task_encoder = TaskGraphMemory(features, vd, 4, hidden)
            self.decoder = TaskDecisionDecoder(vd, ed, hidden)

    def train(self, mode=True):
        super().train(mode)
        # Eval preserves autograd; HF train/checkpoint mode disables prefill KV.
        self.base.eval()
        return self

    def take_decision(self):
        """Consume last plan/memory after policy.infer without changing its tensor API.
        Caller owns per-environment memory and MUST reset it at episode boundaries.
        """
        decision = self.last_decision
        self.last_decision = None
        return decision

    def validate(self, c, allow_untrained=False):
        if c is None:
            raise ValueError("graph conditions required")
        if c.restricted and (
            c.oracle
            or any(
                getattr(g, "oracle", False)
                for g in (c.global_graph, c.local_graph, c.task_graph)
            )
        ):
            raise ValueError("oracle leakage")
        if not self.training and not self.trained and not allow_untrained:
            raise ValueError("Untrained graph tokens forbidden in evaluation")

    def prefix(self, observation, c, train):
        from .masks import make_att_2d_masks

        images, masks, text, text_masks, state = self.base._preprocess_observation(
            observation, train=train
        )
        x, valid, ar = self.base.embed_prefix(images, masks, text, text_masks)
        context = c.task_condition if c.memory is None else c.task_condition + c.memory
        extra = [
            self.global_encoder(c.global_graph, context),
            self.task_encoder(c.task_graph, c.task_condition, c.memory),
        ]
        for token, mask in extra:
            x = torch.cat((x, token.to(x.dtype)), 1)
            valid = torch.cat((valid, mask), 1)
            ar = torch.cat((ar, torch.zeros_like(mask)), 1)
        # Prefix hidden states are the planner input, KV is the sole global context route to expert.
        blocks = make_att_2d_masks(valid, ar)
        positions = valid.long().cumsum(1) - 1
        layers = self.base.paligemma_with_expert
        dtype = layers.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        layers.paligemma.language_model.config._attn_implementation = "eager"
        (hidden, _), cache = layers.forward(
            attention_mask=self.base._prepare_attention_masks_4d(blocks),
            position_ids=positions,
            inputs_embeds=[x.to(dtype), None],
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError(
                "Prefill returned no KV cache; disable backbone gradient checkpointing/training mode"
            )
        decision = self.decoder(
            hidden.float(), valid, c.queries, c.legal, c.memory, c.event_changed
        )
        target_mask = None
        if c.candidate_mask is not None:
            target_mask = c.candidate_mask[
                torch.arange(len(hidden), device=hidden.device),
                decision["selected_index"],
            ]
        local = self.local_encoder(c.local_graph, decision["context"], target_mask)
        conditions = [local, (decision["subgoal"], decision["subgoal_valid"])]
        return state, valid, cache, conditions, decision

    def velocity(self, state, prefix_mask, cache, conditions, x, time):
        from .masks import make_att_2d_masks

        emb, valid, ar, ada = self.base.embed_suffix(state, x, time)
        if conditions:
            token = torch.cat([a for a, m in conditions], 1).to(emb.dtype)
            mask = torch.cat([m for a, m in conditions], 1)
            flags = torch.zeros(mask.shape, device=ar.device, dtype=ar.dtype)
            flags[:, 0] = 1
            emb = torch.cat((token, emb), 1)
            valid = torch.cat((mask, valid), 1)
            ar = torch.cat((flags, ar), 1)
        b, s = valid.shape
        p = prefix_mask.shape[1]
        att = torch.cat(
            (
                prefix_mask[:, None, :].expand(b, s, p) & valid[:, :, None],
                make_att_2d_masks(valid, ar),
            ),
            2,
        )
        positions = prefix_mask.long().sum(-1)[:, None] + valid.long().cumsum(-1) - 1
        layers = self.base.paligemma_with_expert
        dtype = layers.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        layers.gemma_expert.model.config._attn_implementation = "eager"
        (_, out), _ = layers.forward(
            attention_mask=self.base._prepare_attention_masks_4d(att),
            position_ids=positions,
            past_key_values=cache,
            inputs_embeds=[None, emb.to(dtype)],
            use_cache=False,
            adarms_cond=[None, ada],
        )
        return self.base.action_out_proj(
            out[:, -self.base.config.action_horizon :].float()
        )

    def forward(
        self,
        observation,
        actions,
        conditions=None,
        noise=None,
        time=None,
        compute_action=True,
        validation=False,
    ):
        if self.mode in ("baseline", "graph_shadow"):
            return {
                "action_loss": self.base(observation, actions, noise=noise, time=time),
                "decision": None,
            }
        self.validate(conditions, allow_untrained=validation)
        if noise is None:
            noise = self.base.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.base.sample_time(len(actions), actions.device)
        state, valid, cache, tokens, decision = self.prefix(
            observation, conditions, self.training and not validation
        )
        if not compute_action:
            return {"action_loss": None, "decision": decision}
        x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity = self.velocity(state, valid, cache, tokens, x, time)
        return {
            "action_loss": (velocity - (noise - actions)).square(),
            "decision": decision,
        }

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
        self.last_decision = None
        if self.mode in ("baseline", "graph_shadow"):
            return self.base.sample_actions(
                device, observation, noise=noise, num_steps=num_steps
            )
        self.validate(conditions)
        if num_steps < 1:
            raise ValueError("positive Euler step count required")
        state, valid, cache, tokens, decision = self.prefix(
            observation, conditions, False
        )
        if noise is None:
            noise = self.base.sample_noise(
                (
                    len(state),
                    self.base.config.action_horizon,
                    self.base.config.action_dim,
                ),
                device,
            )
        x = noise
        dt = -1.0 / num_steps
        for i in range(num_steps):
            time = torch.full((len(x),), 1 + i * dt, device=device, dtype=torch.float32)
            x = x + dt * self.velocity(state, valid, cache, tokens, x, time)
        if not torch.isfinite(x).all():
            raise FloatingPointError("nonfinite Euler output")
        self.last_decision = decision
        if return_decision:
            return {"actions": x, "decision": decision}
        return x
