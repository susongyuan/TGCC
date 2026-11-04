"""Causal demonstration metadata. Labels never enter parent policy conditions."""

import hashlib
import numpy as np
import torch
from .contracts import Evidence, Truth
from .specs import CriticObservation

HISTORY_WIDTH = 240
PREDICATES = {
    "observed": 1,
    "held": 2,
    "open": 3,
    "inside": 4,
    "on_top": 5,
    "toggled_on": 6,
    "operable": 7,
    "unsupported_contract": 8,
}
CATEGORIES = {"pre": 0, "post": 1, "invariant": 2, "handoff": 3}


def entity_index(name):
    return 1 + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 511


def decoded_evidence(decoded, now, *, grasp_fields_reliable=False):
    """_real means existence/pose, never visibility.

    Legacy 2025 raw states did not serialize assisted-grasp ownership. A replay
    can consequently export default -1 gripper fields while carrying an object.
    Such fields cannot prove either held or not-held. Explicit trusted provenance
    is required before using any exported grasp value as physical supervision.
    Current live simulator evidence uses snapshot_evidence, not this decoder.
    """
    out = {}
    for key, value in decoded.items():
        if not key.endswith("_real"):
            continue
        name = key[:-5]
        exists = float(np.asarray(value).reshape(-1)[0]) == 1
        out[("observed", name, "")] = Evidence(
            exists, now, 1.0, True, source="recorded_existence_pose"
        )
        vals = [
            float(np.asarray(decoded[name + "_in_gripper_" + s]).reshape(-1)[0])
            for s in ("left", "right")
            if name + "_in_gripper_" + s in decoded
        ]
        if exists and vals:
            held = (
                True
                if 1.0 in vals
                else (False if len(vals) == 2 and all(v == -1 for v in vals) else None)
            )
            if not grasp_fields_reliable:
                held = None
            out[("held", name, "")] = Evidence(
                held,
                now,
                1.0 if held is not None else 0.0,
                held is not None,
                source="recorded_grasp_state"
                if grasp_fields_reliable
                else "unverified_legacy_grasp_export",
            )
    return out


def contract_tensors(contract, evidence, now, device="cpu"):
    if len(contract.predicates) > 16:
        raise ValueError("Explicit contract partitions required")
    ids = torch.zeros(1, 16, dtype=torch.long, device=device)
    ent = ids.clone()
    cat = ids.clone()
    numeric = torch.zeros(1, 16, 8, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    valid = mask.clone()
    values = numeric[:, :, 0].clone()
    for i, p in enumerate(contract.predicates):
        ids[0, i] = PREDICATES[p.kind]
        ent[0, i] = entity_index(p.subject)
        cat[0, i] = CATEGORIES[p.category]
        mask[0, i] = p.required_at_current_stage
        # Target identity is preserved separately, not confused with a measured distance.
        numeric[0, i, :4] = torch.tensor(
            [
                p.expected,
                p.tolerance,
                entity_index(p.target) / 512.0 if p.target else 0.0,
                1.0,
            ],
            device=device,
        )
        truth = evidence.get(p.key, Evidence()).truth_for(p, now)
        if p.required_at_current_stage and truth != Truth.UNKNOWN:
            valid[0, i] = True
            values[0, i] = float((truth == Truth.TRUE) == p.expected)
    return dict(
        contract_ids=ids,
        contract_entities=ent,
        contract_numeric=numeric,
        contract_category=cat,
        contract_mask=mask,
        contract_values=values,
        contract_valid=valid,
    )


def pose9(matrix):
    t = np.asarray(matrix)
    return np.r_[t[:3, 3], t[:3, :2].T.reshape(-1)]


class DemonstrationMetadata:
    """Adds actual executed histories and timestamped facts to existing dataset.

    History slot layout: graph summaries96, action mean/last46, proprio32,
    base pose9, base velocity3, EEs18, prior-selected-skill23, dt1, flags12.
    Demonstrations do not contain prior model decisions: that slot's validity is
    FALSE rather than substituting ground-truth phase annotations as inputs.
    """

    def __init__(self, dataset, normalizer, *, grasp_fields_reliable=False):
        self.dataset = dataset
        self.normalizer = normalizer
        self.grasp_fields_reliable = grasp_fields_reliable

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = self.dataset[index]
        data = self.dataset.current_data
        row = self.dataset.rows[index]
        aux = []
        from tgr.build_batches import (
            task_fields,
            named_proprio,
            base_world_from_named,
            ee_world_from_named,
        )

        for k, item in enumerate(batch["sequence"]):
            frame = row["sequence_steps"][k]
            r = data.records[frame]
            now = float(r["timestamp"])
            decoded = task_fields(
                data.meta["task_obs_keys"], r["observation.task_info"]
            )
            previous_facts = []
            for prior in range(max(0, frame - 3 * 32), frame, 32):
                pr = data.records[prior]
                pt = float(pr["timestamp"])
                previous_facts.append(
                    (
                        pt,
                        decoded_evidence(
                            task_fields(
                                data.meta["task_obs_keys"], pr["observation.task_info"]
                            ),
                            pt,
                            grasp_fields_reliable=self.grasp_fields_reliable,
                        ),
                    )
                )
            native = np.asarray(
                data.records.columns["action"][frame : frame + 32], np.float32
            ).copy()
            current = named_proprio(r["observation.state"])
            current_ee = ee_world_from_named(current)
            robot = np.zeros(64, np.float32)
            robot[:32] = item["observation"]["state"][0].numpy()
            robot[32:41] = pose9(base_world_from_named(current))
            robot[41:44] = current["base_qvel"]
            robot[44:53] = pose9(current_ee["left"])
            robot[53:62] = pose9(current_ee["right"])
            robot[62:] = 1.0
            expected = self.normalizer.transform(native).astype(np.float32)
            if not np.allclose(
                expected, item["actions"][0, :, :23].numpy(), atol=3e-6, rtol=2e-6
            ):
                raise ValueError(
                    "Fast action transform differs from checkpoint-owned official pipeline"
                )
            history = np.zeros((16, HISTORY_WIDTH), np.float32)
            hm = np.zeros(16, bool)
            ht = np.zeros(16, np.float64)
            past = list(range(max(0, frame - 16 * 32), frame, 32))[-16:]
            # Graphs own observed history. Match exposure timestamps explicitly.
            for j, start in enumerate(past, 16 - len(past)):
                rr = data.records[start]
                end = min(start + 32, frame)
                when = float(rr["timestamp"])
                fields_ = named_proprio(rr["observation.state"])
                flags = np.zeros(12, np.float32)
                for gi, name in enumerate(
                    ("global_graph", "task_graph", "local_graph")
                ):
                    g = item["conditions"][name]
                    ts = item["provenance"]["history_timestamps"][name]
                    matches = [
                        ii for ii, tt in enumerate(ts) if abs(float(tt) - when) < 1e-5
                    ]
                    if matches:
                        n = matches[-1]
                        valid = g["valid"][0, n]
                        feat = g["features"][0, n]
                        history[j, 32 * gi : 32 * (gi + 1)] = (
                            feat[valid].mean(0).numpy() if valid.any() else 0.0
                        )
                        flags[gi] = float(valid.any())
                act = self.normalizer.transform(
                    np.asarray(data.records.columns["action"][start:end], np.float32)
                )
                history[j, 96:119] = act.mean(0)
                history[j, 119:142] = act[-1]
                flags[3] = 1
                from tgr.input_adapter import legacy_state_from_named
                from openpi.policies.b1k_policy import extract_state_from_proprio

                state = extract_state_from_proprio(legacy_state_from_named(fields_))
                history[j, 142:165] = self.normalizer.transform(state, "state")
                flags[4] = 1
                history[j, 174:183] = pose9(base_world_from_named(fields_))
                flags[5] = 1
                history[j, 183:186] = fields_["base_qvel"]
                flags[6] = 1
                ee = ee_world_from_named(fields_)
                history[j, 186:195] = pose9(ee["left"])
                history[j, 195:204] = pose9(ee["right"])
                flags[7:9] = 1
                history[j, 227] = float(data.records[end]["timestamp"]) - when
                flags[10] = 1
                history[j, 228:] = flags
                hm[j] = True
                ht[j] = when
            aux.append(
                dict(
                    timestamp=torch.tensor([now], dtype=torch.float64),
                    history=torch.from_numpy(history)[None],
                    history_mask=torch.from_numpy(hm)[None],
                    history_timestamps=torch.from_numpy(ht)[None],
                    native_actions=torch.from_numpy(native)[None],
                    evidence=decoded_evidence(
                        decoded, now, grasp_fields_reliable=self.grasp_fields_reliable
                    ),
                    episode_id=str(row["episode_id"]),
                    frame=frame,
                    robot_current=torch.from_numpy(robot)[None],
                    previous_facts=previous_facts,
                    held_sides={
                        name: tuple(
                            side
                            for side in ("left", "right")
                            if float(
                                np.asarray(
                                    decoded.get(name + "_in_gripper_" + side, [0])
                                ).reshape(-1)[0]
                            )
                            == 1.0
                        )
                        for name in (
                            key[:-5] for key in decoded if key.endswith("_real")
                        )
                        if self.grasp_fields_reliable
                        and float(np.asarray(decoded[name + "_real"]).reshape(-1)[0])
                        == 1.0
                    },
                )
            )
        batch["tgcc_aux"] = aux
        return batch


def move_tree(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_tree(v, device) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(move_tree(v, device) for v in x)
    return x


def conditions_from_dict(c):
    from tgr.model import Conditions
    from tgr.encoders import GraphBatch

    return Conditions(
        **{
            k: GraphBatch(**c[k]) for k in ("global_graph", "task_graph", "local_graph")
        },
        **{
            k: v
            for k, v in c.items()
            if k not in ("global_graph", "task_graph", "local_graph")
        },
    )


def critic_observation(taps, decision, aux, actions, contract, conditions, detach=True):
    device = actions.device
    now = aux["timestamp"].to(device)
    tokens = []
    masks = []
    times = []
    conf = []
    for enc, name in zip(
        ("global_encoder", "task_encoder", "local_encoder"),
        ("global_graph", "task_graph", "local_graph"),
    ):
        t, m = taps[enc]
        tokens.append(t.detach() if detach else t)
        masks.append(m)
        # Learned pooled tokens are materialized NOW from causal history. They
        # are not an assertion that every contributing relation was observed now.
        g = getattr(conditions, name)
        nodevalid = g.valid
        confidence = g.features[..., 0].clamp(0, 1)
        confidence = (confidence * nodevalid).sum((-2, -1)) / nodevalid.sum(
            (-2, -1)
        ).clamp_min(1)
        times.append(now[:, None].expand_as(m))
        conf.append(confidence[:, None].expand_as(m))
    return CriticObservation(
        tuple(tokens),
        tuple(masks),
        tuple(times),
        tuple(conf),
        now,
        decision["context"].detach() if detach else decision["context"],
        **contract_tensors(contract, aux["evidence"], float(now[0]), device),
        history=aux["history"].to(device),
        history_mask=aux["history_mask"].to(device),
        history_timestamps=aux["history_timestamps"].to(device),
        candidate_actions=actions[..., :23],
        candidate_action_mask=(
            aux["candidate_action_mask"].to(device)
            if "candidate_action_mask" in aux
            else torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
        ),
        robot_current=aux["robot_current"].to(device),
        task_ids=torch.ones(len(actions), dtype=torch.long, device=device),
        skill_ids=torch.full(
            (len(actions),),
            contract.skill + 1 if contract.skill >= 0 else 127,
            dtype=torch.long,
            device=device,
        ),
        preserve_gripper_mask=torch.tensor(
            [
                [
                    any(
                        p.kind == "held"
                        and p.expected
                        and p.category == "invariant"
                        and side in aux.get("held_sides", {}).get(p.subject, ())
                        for p in contract.predicates
                    )
                    for side in ("left", "right")
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),
    )


def transition_labels(before, after):
    """Same contract evaluated on two real exposures, not phase/time completion."""
    valid = before.contract_valid & after.contract_valid & before.contract_mask
    post = (before.contract_category == 1) & before.contract_mask
    required = post | ((before.contract_category == 2) & before.contract_mask)
    hand = (before.contract_category == 3) & before.contract_mask
    post_valid = (~required | valid).all(-1) & post.any(-1)
    hand_valid = (~hand | valid).all(-1) & hand.any(-1)

    def ratio(o):
        return (o.contract_values * post).sum(-1) / post.sum(-1).clamp_min(1)

    return dict(
        predicate=after.contract_values,
        predicate_valid=after.contract_valid & before.contract_mask,
        # This head measures executed transition / task-contract consistency,
        # not a learned simulator or "all demonstration chunks succeeded".
        transition=((~required) | (after.contract_values > 0.5)).all(-1).float(),
        transition_valid=post_valid,
        progress=ratio(after) - ratio(before),
        progress_valid=post_valid,
        handoff=((~hand) | (after.contract_values > 0.5)).all(-1).float(),
        handoff_valid=hand_valid,
    )
