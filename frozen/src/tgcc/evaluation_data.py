"""Owned, hash-bound action evaluation data. No recovery-category supervision.

Reference actions, actual execution, and counterfactual candidates are separate.
Every collected action is excluded from policy imitation unless independently
admitted as a demonstration; collection provenance alone never admits it.
"""

from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .contracts import Evidence, Predicate, TaskContract, Truth
from .data import HISTORY_WIDTH, pose9
from .runtime import snapshot_evidence


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exposure(directory, receipt):
    directory = Path(directory).resolve()
    path = (directory / receipt["file"]).resolve()
    if path.parent != directory or path.is_symlink() or sha256(path) != receipt["sha256"]:
        raise ValueError("Exposure path/hash mismatch")
    with gzip.open(path, "rb") as stream:
        record = pickle.load(stream)  # Only trusted files from our own recorder.
    raw = record["raw"]
    now = record["timestamp"]
    if not np.isfinite(now) or now != receipt["timestamp"]:
        raise ValueError("Exposure receipt/time mismatch")
    if raw["session"] != record["session"] or raw["sim_time"] != now:
        raise ValueError("Raw exposure session/time mismatch")
    snap = raw["graph_snapshot"]
    if snap["episode_id"] != record["session"] or snap["timestamp"] != now:
        raise ValueError("Snapshot exposure mismatch")
    if raw["robot_fields"]["timestamp"] != now:
        raise ValueError("Robot feedback exposure mismatch")
    for node in snap["entities"]:
        if node["timestamp"] > now or node["last_seen"] > now:
            raise ValueError("Future node evidence")
    for edge in snap["edges"]:
        if edge["timestamp"] > now:
            raise ValueError("Future edge evidence")
    for name in ("global_graph", "task_graph", "local_graph"):
        graph = record["conditions"][name]
        valid = graph["valid"]
        if not torch.isfinite(graph["features"][valid]).all():
            raise ValueError("Nonfinite valid graph features")
    return record


def verify_transition(directory, row):
    before = load_exposure(directory, row["before"])
    after = load_exposure(directory, row["after"])
    n = row["executed_steps"]
    actions = np.asarray(row["executed_native"], np.float32)
    if (not 1 <= n <= 32 or actions.shape != (n, 23)
            or not np.isfinite(actions).all()
            or row["action_mask"] != [True] * n + [False] * (32 - n)
            or before["session"] != after["session"]
            or abs(after["timestamp"] - before["timestamp"] - n / 30) > 2e-4):
        raise ValueError("Actual execution prefix/session/time mismatch")
    if row.get("simulated_outcome") is not False:
        raise ValueError("An invented transition cannot supervise execution")
    if any(row.get(k) is not None for k in ("task_success_label", "recovery_success_label")):
        raise ValueError("Generic collection cannot fabricate task/recovery success")
    annotations = row.get("execution_annotations")
    if annotations is not None and len(annotations) != n:
        raise ValueError("Action annotation alignment mismatch")
    return before, after, actions


def held_by_side(record):
    snap = record["raw"]["graph_snapshot"]
    now = record["timestamp"]
    nodes = {n["entity_id"]: n for n in snap["entities"]}
    out = {}
    for edge in snap["edges"]:
        if edge["relation_type"] != "held_by":
            continue
        source, target = nodes.get(edge["source_id"]), nodes.get(edge["target_id"])
        if source is None or target is None:
            continue
        side = target["instance_id"].rsplit(":", 1)[-1]
        if side not in ("left", "right"):
            continue
        attrs = edge.get("relative_attributes", {})
        value = attrs.get("value")
        valid = attrs.get("evidence_valid", edge["validity"] or
                          (value is False and edge["source"] == "oracle"))
        ev = Evidence(value, edge["timestamp"], edge["confidence"],
                      bool(valid and source["validity"] and target["validity"]),
                      source=edge["source"])
        key = (source["instance_id"], side)
        if key in out and out[key].value != ev.value:
            out[key] = Evidence()
        else:
            out[key] = ev
    return {k: v for k, v in out.items() if v.truth(now) != Truth.UNKNOWN}


def clean_conditions(conditions, *, controller_collection=False):
    """Remove simulator predicate labels before any policy/critic feature tap.

    Pose/joint inputs retain their explicit oracle protocol provenance. We do
    not rename oracle geometry as perception. Outcome bits, label-derived
    edges, and controller collection task events are never forward inputs.
    """
    from tgr.tensorize import bucket

    result = deepcopy(conditions)
    forbidden = {1 + bucket(k, 31) for k in
                 ("held_by", "inside", "on_top", "contact", "grasped_by")}
    for name in ("global_graph", "task_graph", "local_graph"):
        g = result[name]
        x = g["features"]
        x[..., 23:30] = 0
        if name == "task_graph":
            # Task-node state_index occupies 17:23 as well as VERIFIED_DONE
            # at 29. It may have been produced by privileged collection facts.
            x[..., 17:30] = 0
        if name == "local_graph" and g.get("pair_nodes") is not None:
            pairs = g["pair_nodes"].bool()
            x[..., 21:27] = torch.where(pairs[..., None],
                                       torch.zeros_like(x[..., 21:27]), x[..., 21:27])
        for relation in forbidden:
            g["relations"][g["relations"] == relation] = 0
        if g.get("events") is not None:
            g["events"].zero_()
            g["event_valid"].zero_()
    if result.get("legal") is not None:
        # A privileged TaskManager predicate filter must not tell the learned
        # precondition head which candidate is bad before it sees geometry.
        result["legal"].fill_(True)
    if result.get("event_changed") is not None:
        result["event_changed"].zero_()
    # Runtime controller collections have no learned semantic memory. Do not
    # accept a manually serialized memory containing task/terminal labels.
    if controller_collection:
        result["memory"] = None
        result["memory_tokens"] = None
    return result


def contract_record(contract):
    return asdict(contract)


def read_contract(record):
    from .data import PREDICATES, CATEGORIES

    contract = TaskContract(record["skill"], record["candidate_id"],
                            tuple(Predicate(**p) for p in record["predicates"]))
    if not -1 <= contract.skill < 127 or not 1 <= len(contract.predicates) <= 16:
        raise ValueError("Unsupported skill/empty or over-capacity contract")
    for p in contract.predicates:
        if (p.kind not in PREDICATES or p.category not in CATEGORIES or
                type(p.expected) is not bool or not p.subject):
            raise ValueError("Invalid evaluation contract clause")
    return contract


def maintain_contract(names):
    # A maintained state is a local execution contract, not completion of a
    # new pick/place subtask or proof of overall task success.
    return TaskContract(-1, "maintain_current_grasp", tuple(
        Predicate("held", name, category=category,
                  requires_fresh_observation=True, memory_acceptable=False,
                  maximum_observation_age=1.2)
        for name in sorted(set(names)) for category in ("invariant", "post")
    ))


def named_state(raw):
    state = raw["state"]
    return dict(base_qvel=state["base_qvel"], trunk_qpos=state["trunk_qpos"],
                arm_left_qpos=state["left_arm"], arm_right_qpos=state["right_arm"],
                gripper_left_qpos=state["left_gripper"],
                gripper_right_qpos=state["right_gripper"])


def validate_pair(supervision, spec, before=None, actual=None, mask=None):
    """Recompute the proof, never trust a JSON boolean saying 'bad'."""
    if not supervision.get("rank_valid", False):
        return
    reference = np.asarray(supervision["reference_native"], np.float32)
    negative = np.asarray(supervision["negative_native"], np.float32)
    mask = np.asarray(mask if mask is not None else [True] * 32, bool)
    lo, hi = np.asarray(spec.lower, np.float32), np.asarray(spec.upper, np.float32)
    known = np.asarray(spec.bounds_valid, bool)
    if (reference.shape != (32, 23) or negative.shape != reference.shape
            or mask.shape != (32,) or not mask.any()
            or not np.isfinite(reference[mask]).all() or not np.isfinite(negative[mask]).all()
            or ((((reference < lo) | (reference > hi)) & known)[mask]).any()):
        raise ValueError("Invalid ranking reference/candidate/mask")
    proof = supervision["certificate"]
    kind = supervision["negative_kind"]
    if kind == "physical":
        if proof["rule"] != "native_controller_limit":
            raise ValueError("Unsupported physical proof")
        d = int(proof["dimension"])
        if not 0 <= d < 23 or not known[d] or not ((negative[mask, d] < lo[d]) | (negative[mask, d] > hi[d])).any():
            raise ValueError("Negative has no native-limit violation")
    elif kind == "task":
        if before is None or proof["rule"] != "preserve_current_verified_grasp":
            raise ValueError("Task proof needs real pre-action grasp evidence")
        obj, side = proof["object_id"], proof["side"]
        d = 14 if side == "left" else 22 if side == "right" else -1
        ev = held_by_side(before).get((obj, side), Evidence())
        if (d < 0 or ev.truth(before["timestamp"]) != Truth.TRUE
                or not np.all(reference[mask, d] < -0.5)
                or not np.any(negative[mask, d] > 0.5)
                or (((negative < lo) | (negative > hi)) & known)[mask].any()):
            raise ValueError("No certified in-domain premature-release negative")
    elif kind == "planning":
        if before is None or proof["rule"] != "known_false_subtask_precondition":
            raise ValueError("Planning proof needs real pre-action evidence")
        evidence = snapshot_evidence(before["raw"]["graph_snapshot"], before["timestamp"])
        contract = read_contract(supervision["negative_contract"])
        false = [p for p in contract.predicates if p.category == "pre" and
                 p.required_at_current_stage and
                 (t := evidence.get(p.key, Evidence()).truth_for(p, before["timestamp"])) != Truth.UNKNOWN
                 and (t == Truth.TRUE) != p.expected]
        if not false:
            raise ValueError("UNKNOWN/unproved planning error cannot be a negative")
        if ((negative != reference)[mask]).any():
            raise ValueError("Planning pair must isolate contract change from action change")
        if (((negative < lo) | (negative > hi)) & known)[mask].any():
            raise ValueError("Planning negative cannot hide a control-limit violation")
    else:
        raise ValueError("Unknown negative certificate kind")
    if supervision.get("synthetic_after_state") is not None:
        raise ValueError("Counterfactual actions cannot receive after states")


class CollectionDataset:
    """Convert real RGB/three-graph transitions to the same causal train format.

    Uses the original checkpoint's official input transform. No skill targets,
    recovery targets, or flow targets are inferred from a replay's source.
    """

    def __init__(self, adapter, normalizer, spec):
        self.adapter, self.normalizer, self.spec = adapter, normalizer, spec

    def sample(self, descriptor):
        directory = Path(descriptor["directory"])
        row = descriptor["transition"]
        before, after, actual = verify_transition(directory, row)
        expected_plan = descriptor["plan_sha256"]
        if before["plan_sha256"] != expected_plan or after["plan_sha256"] != expected_plan:
            raise ValueError("Collection plan identity mismatch")
        padded = np.zeros((32, 23), np.float32)
        padded[:len(actual)] = actual
        mask = torch.tensor([row["action_mask"]], dtype=torch.bool)
        supervision = deepcopy(descriptor["supervision"])
        if supervision.get("policy_imitation_allowed", False):
            raise ValueError("Collection cannot self-admit a policy imitation target")
        validate_pair(supervision, self.spec, before, actual, row["action_mask"])
        if supervision.get("negative_contract"):
            supervision["before_evidence"] = snapshot_evidence(before["raw"]["graph_snapshot"], before["timestamp"])
        sequence, aux = [], []
        history = []
        # Real historical prefixes only; descriptor contains hash-bound receipts.
        for prior in descriptor.get("history", [])[-16:]:
            pb, pa, px = verify_transition(directory, prior)
            if (pb["session"] != before["session"] or
                    pa["timestamp"] > before["timestamp"]):
                raise ValueError("Future/cross-session executed history")
            if history and pb["timestamp"] < history[-1][1]["timestamp"]:
                raise ValueError("Overlapping or unsorted executed history")
            history.append((pb, pa, px))
        for record in (before, after):
            transformed = self.adapter(named_proprio=named_state(record["raw"]),
                                       rgb=record["raw"]["images"], actions=padded,
                                       task=record["raw"]["task"])
            state = transformed["observation"]["state"][0].numpy()
            robot = np.zeros(64, np.float32)
            robot[:32] = state
            robot[32:41] = pose9(record["raw"]["robot_fields"]["base_world"])
            robot[41:44] = record["raw"]["state"]["base_qvel"]
            nodes = {n["instance_id"]: n for n in record["raw"]["graph_snapshot"]["entities"]}
            for side, sl, flag in (("left", slice(44, 53), 62), ("right", slice(53, 62), 63)):
                node = nodes.get("robot:ee:" + side)
                if node and node["validity"] and node["pose"] is not None:
                    robot[sl] = pose9(node["pose"])
                    robot[flag] = 1
            h = np.zeros((16, HISTORY_WIDTH), np.float32)
            hm, ht = np.zeros(16, bool), np.zeros(16, np.float64)
            for j, (pb, pa, px) in enumerate(history[-16:], 16 - len(history[-16:])):
                if pb["timestamp"] >= record["timestamp"]:
                    raise ValueError("History must precede feature exposure")
                norm = self.normalizer.transform(px).astype(np.float32)
                h[j, 96:119], h[j, 119:142] = norm.mean(0), norm[-1]
                # Missing historical robot/graph summaries remain masked, never
                # substituted with current or after-state feedback.
                h[j, 227] = pa["timestamp"] - pb["timestamp"]
                h[j, 231] = 1
                hm[j], ht[j] = True, pb["timestamp"]
            side_evidence = held_by_side(record)
            held = {}
            for (name, side), ev in side_evidence.items():
                if ev.truth(record["timestamp"]) == Truth.TRUE:
                    held.setdefault(name, []).append(side)
            aux.append(dict(timestamp=torch.tensor([record["timestamp"]], dtype=torch.float64),
                            history=torch.from_numpy(h)[None], history_mask=torch.from_numpy(hm)[None],
                            history_timestamps=torch.from_numpy(ht)[None],
                            robot_current=torch.from_numpy(robot)[None],
                            evidence=snapshot_evidence(record["raw"]["graph_snapshot"], record["timestamp"]),
                            held_sides=held, native_actions=torch.from_numpy(padded)[None],
                            candidate_action_mask=mask, previous_facts=[],
                            episode_id=descriptor["source_episode_id"], frame=int(record["timestamp"] * 30),
                            evaluation_only=True))
            sequence.append(dict(observation=transformed["observation"],
                                 actions=transformed["actions"],
                                 conditions=clean_conditions(record["conditions"], controller_collection=True),
                                 labels={"next": torch.zeros(1, dtype=torch.long),
                                         "next_mask": torch.zeros(1, dtype=torch.bool)},
                                 normalization_sha256=transformed["normalization_sha256"],
                                 provenance={"plan_sha256": expected_plan},
                                 episode_id=before["session"], timestamp=record["timestamp"],
                                 step_id=record["timestamp"] * 30, has_graph_supervision=False))
            if record is before:
                history.append((before, after, actual))
        return dict(sequence=sequence, tgcc_aux=aux,
                    normalization_sha256=self.adapter.normalization_sha256,
                    critic_supervision=supervision,
                    provenance={"plan_sha256": expected_plan},
                    policy_imitation_allowed=False)


class EvaluationDataset:
    """Manifest entry point for demonstrations plus supplemental real execution."""

    def __init__(self, manifest, demonstrations, collection, data_root=None):
        self.manifest_path = Path(manifest).resolve()
        self.data_root = Path(data_root).resolve() if data_root else self.manifest_path.parent
        self.rows = [json.loads(s) for s in self.manifest_path.read_text().splitlines() if s.strip()]
        self.demonstrations, self.collection = demonstrations, collection
        ownership = {}
        ids = set()
        for row in self.rows:
            key = row["source_episode_id"]
            if ownership.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Source episode train/validation leakage")
            if row["id"] in ids or row["split"] not in ("train", "validation"):
                raise ValueError("Duplicate evaluation row or invalid split")
            ids.add(row["id"])
            if row["source_kind"] == "demonstration":
                original = demonstrations.dataset.rows[row["demonstration_index"]]
                if original["episode_id"] != key or original["split"] != row["split"]:
                    raise ValueError("Demonstration source/split mismatch")
            elif row["source_kind"] != "collection":
                raise ValueError("Unknown evaluation data source")
        self.train_indices = [i for i, r in enumerate(self.rows) if r["split"] == "train"]
        self.validation_indices = [i for i, r in enumerate(self.rows) if r["split"] == "validation"]
        if not self.train_indices or not self.validation_indices:
            raise ValueError("Evaluation needs source-isolated train and validation rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row["source_kind"] == "collection":
            descriptor = deepcopy(row)
            path = Path(descriptor["directory"])
            if not path.is_absolute():
                path = self.data_root / path
            descriptor["directory"] = str(path.resolve())
            return self.collection.sample(descriptor)
        sample = self.demonstrations[row["demonstration_index"]]
        sample["critic_supervision"] = row["supervision"]
        sample["policy_imitation_allowed"] = True
        # Counterfactual candidates never replace the demonstration flow target.
        actual = sample["tgcc_aux"][0]["native_actions"][0].numpy()
        reference = np.asarray(row["supervision"]["reference_native"], np.float32)
        if not np.array_equal(actual, reference):
            raise ValueError("Reference differs from hash-verified demonstration")
        validate_pair(row["supervision"], self.collection.spec)
        return sample


class MixedTrainingDataset:
    """One independent successful demo and one evaluation per training unit."""

    def __init__(self, demonstrations, evaluations):
        self.demonstrations, self.evaluations = demonstrations, evaluations

    def __getitem__(self, indices):
        demonstration_index, evaluation_index = indices
        if (self.demonstrations.dataset.rows[demonstration_index]["split"] != "train"
                or self.evaluations.rows[evaluation_index]["split"] != "train"):
            raise ValueError("Held-out data cannot enter joint training")
        policy = self.demonstrations[demonstration_index]
        policy["policy_imitation_allowed"] = True
        return dict(policy_sample=policy, evaluation_sample=self.evaluations[evaluation_index])


def mixed_order(demonstration_rows, evaluation_rows, total, seed):
    """Deterministic independent permutations; cursor counts training units.

    Repeated evaluation exposures are explicitly repetitions, never new
    independent situations. Both orders can be reproduced from one cursor.
    """
    def stream(rows, offset):
        rng = np.random.default_rng(seed + offset)
        ids = [i for i, r in enumerate(rows) if r["split"] == "train"]
        if not ids:
            raise ValueError("Empty mixed training source")
        out = []
        while len(out) < total:
            out.extend(rng.permutation(ids).tolist())
        return out[:total]

    return list(zip(stream(demonstration_rows, 0), stream(evaluation_rows, 100003)))
