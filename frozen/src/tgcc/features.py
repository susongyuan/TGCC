"""Causal two-chunk features with independent student and teacher rollouts."""

from dataclasses import fields
from types import SimpleNamespace
import contextlib
import torch
from .contracts import build_contract
from .data import move_tree, conditions_from_dict, critic_observation
from .specs import CriticObservation


def concatenate_observations(items):
    def cat(v):
        if all(x is None for x in v):
            return None
        if isinstance(v[0], tuple):
            return tuple(cat([x[i] for x in v]) for i in range(len(v[0])))
        return torch.cat(v, 0)

    return CriticObservation(
        **{
            f.name: cat([getattr(o, f.name) for o in items])
            for f in fields(CriticObservation)
        }
    )


def causal_conditions(raw, aux, previous, now, plan, trainable):
    """Roll a policy's own preceding memory/write into fresh conditions.

    Physical evidence is shared, but learned decisions and memory belong to
    each policy. In particular the frozen teacher must not inherit a student's
    evolving hidden state. Feedback modifies legal in-place, so isolate that
    mask from the collated batch and the other policy's rollout.
    """
    from tgr.task_feedback import roll_forward_task_graph
    from .feedback import training_feedback

    conditions = conditions_from_dict(raw)
    if conditions.legal is not None:
        conditions.legal = conditions.legal.clone()
    if previous is not None:
        decision, previous_time = previous
        conditions.memory = (
            decision["memory"] if trainable else decision["memory"].detach()
        )
        conditions.task_graph = roll_forward_task_graph(
            conditions.task_graph,
            decision,
            conditions.task_candidate_slots,
            now - previous_time,
        )
    if all(a.get("evaluation_only", False) for a in aux):
        return conditions
    return training_feedback(conditions, aux, plan)


def extract_pair(
    parent, samples, tap, device, plan, with_flow=False, trainable=False, teacher=None
):
    from tgr.data import collate_sequences
    from openpi.models.model import Observation

    if teacher is parent:
        raise ValueError("Teacher must be an independent policy")
    batch = move_tree(collate_sequences(samples), device)
    records = []
    previous = None
    teacher_previous = None
    flow = []
    for k, item in enumerate(batch["sequence"]):
        now = torch.cat([s["tgcc_aux"][k]["timestamp"] for s in samples]).to(device)
        aux = [s["tgcc_aux"][k] for s in samples]
        c = causal_conditions(
            item["conditions"], aux, previous, now, plan, trainable
        )
        tap.clear()
        with contextlib.nullcontext() if trainable else torch.no_grad():
            state, valid, cache, tokens, decision = parent.prefix(
                Observation.from_dict(item["observation"]), c, False
            )
        copied = {key: (a.detach(), m) for key, (a, m) in tap.values.items()}
        records.append((copied, decision, c))
        previous = (decision, now)
        if with_flow:
            actions = item["actions"]
            noise = parent.base.sample_noise(actions.shape, device)
            time = parent.base.sample_time(len(actions), device)
            teacher_velocity = None
            if teacher is not None:
                with torch.no_grad():
                    tc = causal_conditions(
                        item["conditions"], aux, teacher_previous, now, plan, False
                    )
                    ts, tv, tk, tt, teacher_decision = teacher.prefix(
                        Observation.from_dict(item["observation"]), tc, False
                    )
                    teacher_previous = (teacher_decision, now)
                    tx = (
                        time[:, None, None] * noise
                        + (1 - time[:, None, None]) * actions
                    )
                    teacher_velocity = teacher.velocity(
                        ts, tv, tk, tt, tx, time
                    ).detach()
                    del tk, tt
            from .losses import masked_mean

            ce = torch.nn.functional.cross_entropy(
                decision["next"].float(), item["labels"]["next"], reduction="none"
            )
            skill_loss = masked_mean(ce, item["labels"]["next_mask"])
            flow.append(
                (
                    state,
                    valid,
                    cache,
                    tokens,
                    actions,
                    noise,
                    time,
                    teacher_velocity,
                    skill_loss,
                )
            )
        del cache
    before = []
    after = []
    for b, sample in enumerate(samples):
        selected = int(records[0][1]["selected_index"][b])
        candidate = plan["candidates"][selected]
        contract = build_contract(
            candidate,
            ("hotdog.n.02_1", "hotdog.n.02_2")
            if candidate["skill_ids"] == [69]
            and "microwave" in str(candidate["targets"])
            else (),
            preserved_grasps=tuple(
                n
                for n, sides in sample["tgcc_aux"][0].get("held_sides", {}).items()
                if sides
            ),
        )
        if sample.get("critic_supervision", {}).get("contract") is not None:
            from .evaluation_data import read_contract

            contract = read_contract(sample["critic_supervision"]["contract"])
        for k, out in enumerate((before, after)):
            ts, ds, cs = records[k]
            taps = {key: (a[b : b + 1], m[b : b + 1]) for key, (a, m) in ts.items()}
            decision = {
                key: (v[b : b + 1] if isinstance(v, torch.Tensor) else v)
                for key, v in ds.items()
            }
            condition = SimpleNamespace(
                **{
                    name: SimpleNamespace(
                        features=getattr(cs, name).features[b : b + 1],
                        valid=getattr(cs, name).valid[b : b + 1],
                    )
                    for name in ("global_graph", "task_graph", "local_graph")
                }
            )
            actions = sample["sequence"][0]["actions"].to(device)
            out.append(
                critic_observation(
                    taps, decision, sample["tgcc_aux"][k], actions, contract, condition
                )
            )
    native = torch.cat([s["tgcc_aux"][0]["native_actions"] for s in samples]).to(device)
    return (
        concatenate_observations(before),
        concatenate_observations(after),
        native,
        flow,
    )
