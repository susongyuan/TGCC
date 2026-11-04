from dataclasses import dataclass, field, asdict, replace
from collections import deque
import json, hashlib
from typing import Protocol


@dataclass(frozen=True)
class Entity:
    entity_id: str
    semantic_type: str
    instance_id: str
    pose: list | None
    reference_frame: str
    timestamp: float
    last_seen: float
    confidence: float
    source: str
    validity: bool
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source_id: str
    target_id: str
    relation_type: str
    relative_attributes: dict
    timestamp: float
    confidence: float
    validity: bool
    source: str = "observation"


@dataclass(frozen=True)
class Event:
    timestamp: float
    episode_id: str
    entity_id: str
    predicate: str
    previous_value: object
    current_value: object
    confidence: float
    evidence_id: str
    source: str


@dataclass(frozen=True)
class ObservationSnapshot:
    episode_id: str
    timestamp: float
    entities: tuple
    edges: tuple = ()
    events: tuple = ()
    oracle: bool = False


class EntityRegistry:
    def __init__(self):
        self.ids = {}

    def reset(self):
        self.ids.clear()

    def resolve(self, instance):
        # Provider must supply tracker/instance IDs, never category-only nearest matching.
        if not instance:
            raise ValueError("stable instance identity required")
        return self.ids.setdefault(
            instance, hashlib.sha256(instance.encode()).hexdigest()[:24]
        )


class TemporalEventStore:
    def __init__(self, k=4096):
        self.events = deque(maxlen=k)

    def reset(self):
        self.events.clear()

    def update(self, event):
        self.events.append(event)

    def query(self, entity_id=None):
        return [
            asdict(x)
            for x in self.events
            if entity_id is None or x.entity_id == entity_id
        ]


class StateProvider(Protocol):
    def snapshot(
        self, observation, registry, episode_id, timestamp
    ) -> ObservationSnapshot: ...
class ObservationStateProvider:
    def __init__(self, tracker):
        self.tracker = tracker

    def snapshot(self, observation, registry, episode_id, timestamp):
        s = self.tracker(observation, registry, episode_id, timestamp)
        if contains_oracle(s):
            raise ValueError("oracle leak in observation provider")
        return s


def contains_oracle(snapshot):
    return snapshot.oracle or any(
        x.source == "oracle"
        for x in (*snapshot.entities, *snapshot.edges, *snapshot.events)
    )


class OracleStateProvider:
    """Explicit simulator adapter. Relations MUST come from legal state queries, not proximity."""

    def __init__(self, extract):
        self.extract = extract

    def snapshot(self, observation, registry, episode_id, timestamp):
        s = self.extract(observation, registry, episode_id, timestamp)
        if not s.oracle:
            raise ValueError("oracle snapshot must be labeled")
        return s


class TemporalGraph:
    def __init__(self, k=16):
        if not 1 <= k <= 16:
            raise ValueError("history length must be1..16")
        self.k = k
        self.reset()

    def reset(self):
        self.nodes = {}
        self.edges = {}
        self.history = deque(maxlen=self.k)
        self.history_base = {}
        self.oracle = False

    def update(self, snapshot):
        self.oracle = contains_oracle(snapshot)
        changed = []
        observed = {n.entity_id for n in snapshot.entities}
        for key, old in list(self.nodes.items()):
            if key not in observed and old.validity:
                unknown = replace(old, validity=False, confidence=old.confidence * 0.5)
                self.nodes[key] = unknown
                changed.append(asdict(unknown))
        for node in snapshot.entities:
            old = self.nodes.get(node.entity_id)
            if old != node:
                self.nodes[node.entity_id] = node
                changed.append(asdict(node))
        # Relations are current observations; do not silently keep stale true edges.
        self.edges = {
            (e.source_id, e.target_id, e.relation_type): e for e in snapshot.edges
        }
        if self.history and self.history[-1]["timestamp"] == snapshot.timestamp:
            old = self.history.pop()
            combined = {n["entity_id"]: n for n in old["changed_nodes"]}
            combined.update({n["entity_id"]: n for n in changed})
            changed = list(combined.values())
        if len(self.history) == self.k:
            for node in self.history[0]["changed_nodes"]:
                self.history_base[node["entity_id"]] = node
        self.history.append(
            {
                "timestamp": snapshot.timestamp,
                "changed_nodes": changed,
                "edges": [asdict(e) for e in snapshot.edges],
                "oracle": self.oracle,
            }
        )

    def query(self, entity_id):
        return self.nodes.get(entity_id)

    def snapshot(self):
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "oracle": self.oracle,
        }

    def get_history(self):
        return list(self.history)

    def serialize(self):
        return json.dumps(self.snapshot())

    def visualize(self, path):
        from .view import export_view

        export_view(self.snapshot(), path)


class LocalMotionGraph(TemporalGraph):
    """Current physical entities are references; history stores compact local motion."""

    def update(self, snapshot):
        super().update(snapshot)
        delta = self.history[-1]
        # World poses remain in the shared current authority, not K copies.
        delta["changed_nodes"] = [
            dict(
                node,
                pose=None,
                attributes={
                    key: value
                    for key, value in node.get("attributes", {}).items()
                    if key
                    in (
                        "state",
                        "state_index",
                        "feasibility",
                        "geometry9",
                        "arm",
                        "base_id",
                        "target_id",
                        "nominal_extent_m",
                        "extent_known",
                        "geometry_source",
                        "container",
                        "assumed_static",
                    )
                },
            )
            for node in delta["changed_nodes"]
        ]


class GraphRuntime:
    def __init__(self, provider, restricted=True, k=16, task_spec=None):
        self.provider = provider
        self.restricted = restricted
        self.k = k
        self.task_spec = task_spec
        self.environments = {}

    def reset(self, env, episode):
        self.environments[env] = {
            "episode": episode,
            "time": -float("inf"),
            "registry": EntityRegistry(),
            "events": TemporalEventStore(),
            "global": TemporalGraph(1),
            "local": LocalMotionGraph(self.k),
            "task": TemporalGraph(1),
            "version": 0,
            "authority": None,
        }
        if self.task_spec is not None:
            from .task import TaskManager

            self.environments[env]["task_manager"] = TaskManager.from_bddl(
                self.task_spec, self.environments[env]["registry"]
            )
            self._update_task(env, 0.0, {}, {})

    def _update_task(self, env, timestamp, predictions, evidence):
        from .task import task_snapshot

        s = self.environments[env]
        manager = s.get("task_manager")
        if manager is None:
            return
        before = manager.states.copy()
        manager.update(
            predictions, evidence, timestamp=timestamp, restricted=self.restricted
        )
        for key, value in manager.states.items():
            if before[key] != value:
                ev = evidence.get(key, {})
                source = ev.get("source", "prediction")
                eid = s["registry"].resolve("task:" + key)
                s["events"].update(
                    Event(
                        timestamp,
                        s["episode"],
                        eid,
                        "task_state",
                        before[key],
                        value,
                        ev.get("confidence", 0.0),
                        ev.get("evidence_id", f"prediction:{timestamp}:{key}"),
                        source,
                    )
                )
        task = task_snapshot(manager, s["registry"], s["episode"], timestamp)
        if s.get("feasibility_nodes"):
            nodes = {n.entity_id: n for n in task.entities}
            edges = list(task.edges)
            for n in s["feasibility_nodes"]:
                nodes[n.entity_id] = n
                for key, record in manager.records.items():
                    if record.get("action_type") == "REPOSITION_BASE":
                        edges.append(
                            Edge(
                                n.entity_id,
                                s["registry"].resolve("task:" + key),
                                "feasibility_condition",
                                {},
                                timestamp,
                                n.confidence,
                                True,
                                n.source,
                            )
                        )
            task = replace(
                task,
                entities=tuple(nodes.values()),
                edges=tuple(edges),
                oracle=task.oracle
                or any(n.source == "oracle" for n in s["feasibility_nodes"]),
            )
        s["task"].update(task)

    def update_task(self, env, predictions, evidence=None, timestamp=None):
        s = self.environments[env]
        if "task_manager" not in s:
            raise ValueError("task_spec required")
        t = s["time"] if timestamp is None else timestamp
        if t != s["time"] or not __import__("math").isfinite(t):
            raise ValueError("task update must use current observation time")
        self._update_task(env, t, predictions, evidence or {})
        return s

    def update(
        self, env, observation, timestamp, feasibility_summary=None, defer_task=False
    ):
        s = self.environments[env]
        snap = self.provider.snapshot(
            observation, s["registry"], s["episode"], timestamp
        )
        if feasibility_summary is not None:
            snap = merge_feasibility(snap, feasibility_summary, s["registry"])
        if (
            snap.episode_id != s["episode"]
            or snap.timestamp != timestamp
            or timestamp <= s["time"]
        ):
            raise ValueError("episode/time mismatch")
        if self.restricted and contains_oracle(snap):
            raise ValueError("Oracle Upper Bound forbidden in restricted evaluation")
        validate_snapshot(snap)
        ids = {n.entity_id for n in snap.entities} | set(s["global"].nodes)
        if any(e.entity_id not in ids for e in snap.events):
            raise ValueError("unknown event entity")
        if any(e.source_id not in ids or e.target_id not in ids for e in snap.edges):
            raise ValueError("unknown relation endpoint")
        if any(
            n.timestamp > timestamp or n.last_seen > timestamp for n in snap.entities
        ) or any(e.timestamp > timestamp for e in (*snap.edges, *snap.events)):
            raise ValueError("future information")
        # Generate sparse, evidence-labelled changes; timestamp refresh alone is not an event.
        import numpy as np

        for node in snap.entities:
            old = s["global"].nodes.get(node.entity_id)
            changed = (
                old is not None
                and old.pose is not None
                and node.pose is not None
                and not np.allclose(old.pose, node.pose, atol=1e-5, rtol=0.0)
            )
            if changed:
                token = hashlib.sha256(
                    f"{snap.episode_id}:{timestamp}:{node.entity_id}:pose".encode()
                ).hexdigest()[:24]
                s["events"].update(
                    Event(
                        timestamp,
                        snap.episode_id,
                        node.entity_id,
                        "pose",
                        old.pose,
                        node.pose,
                        node.confidence,
                        token,
                        node.source,
                    )
                )
        old_edges = s["global"].edges
        for edge in snap.edges:
            key = (edge.source_id, edge.target_id, edge.relation_type)
            old = old_edges.get(key)
            previous = (
                None
                if old is None
                else old.relative_attributes.get("value", old.validity)
            )
            current = edge.relative_attributes.get("value", edge.validity)
            if previous != current:
                token = hashlib.sha256(
                    f"{snap.episode_id}:{timestamp}:{key}".encode()
                ).hexdigest()[:24]
                s["events"].update(
                    Event(
                        timestamp,
                        snap.episode_id,
                        edge.source_id,
                        edge.relation_type,
                        previous,
                        current,
                        edge.confidence,
                        token,
                        edge.source,
                    )
                )
        s["global"].update(snap)
        s["authority"] = s["global"].nodes
        s["current_snapshot"] = snap
        s["version"] += 1
        from .geometry import relative

        local_entities = tuple(
            n
            for n in snap.entities
            if not n.semantic_type.startswith(("floor.", "room", "world_reference"))
        )
        by_id = {n.entity_id: n for n in local_entities}
        local = []
        for hand in local_entities:
            if (
                hand.semantic_type
                not in (
                    "left_ee",
                    "right_ee",
                    "robot_base",
                    "base",
                    "trunk",
                    "robot_trunk",
                    "left_arm",
                    "right_arm",
                )
                or hand.pose is None
                or not hand.validity
            ):
                continue
            for obj in local_entities:
                if (
                    obj.entity_id == hand.entity_id
                    or obj.pose is None
                    or not obj.validity
                ):
                    continue
                if obj.semantic_type in (
                    "left_ee",
                    "right_ee",
                    "robot_base",
                    "base",
                    "trunk",
                    "robot_trunk",
                    "left_arm",
                    "right_arm",
                ) and {hand.semantic_type, obj.semantic_type} != {
                    "left_ee",
                    "right_ee",
                }:
                    continue
                if hand.reference_frame != "world" or obj.reference_frame != "world":
                    raise ValueError("SE3 inputs must share world frame")
                x = relative(hand.pose, obj.pose)
                local.append(
                    Edge(
                        hand.entity_id,
                        obj.entity_id,
                        "relative_to",
                        local_attributes(s, hand, obj, x, timestamp),
                        timestamp,
                        min(hand.confidence, obj.confidence),
                        True,
                        "oracle" if snap.oracle else "observation",
                    )
                )
        s["local"].update(
            ObservationSnapshot(
                snap.episode_id,
                timestamp,
                local_entities,
                tuple(local),
                oracle=snap.oracle,
            )
        )
        for event in snap.events:
            s["events"].update(event)
        s["feasibility_nodes"] = [
            n for n in snap.entities if n.semantic_type == "feasibility_summary_v1"
        ]
        s["time"] = timestamp
        if "task_manager" in s and not defer_task:
            from .task import evidence_from_snapshot

            self._update_task(
                env, timestamp, {}, evidence_from_snapshot(s["task_manager"], snap)
            )
        return s


def validate_snapshot(snap):
    import math
    from .geometry import relative

    if not math.isfinite(snap.timestamp):
        raise ValueError("nonfinite observation time")
    ids = [n.entity_id for n in snap.entities]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate entity ID")
    if len({n.instance_id for n in snap.entities}) != len(ids):
        raise ValueError("duplicate instance identity")
    for n in snap.entities:
        if (
            not all(math.isfinite(x) for x in (n.timestamp, n.last_seen, n.confidence))
            or not 0 <= n.confidence <= 1
        ):
            raise ValueError("invalid node metadata")
        if n.timestamp > snap.timestamp or n.last_seen > snap.timestamp:
            raise ValueError("future entity")
        if n.pose is not None:
            if n.reference_frame != "world":
                raise ValueError("provider must transform poses to world frame")
            relative(n.pose, n.pose)
    for x in (*snap.edges, *snap.events):
        if (
            not math.isfinite(x.timestamp)
            or x.timestamp > snap.timestamp
            or not math.isfinite(x.confidence)
            or not 0 <= x.confidence <= 1
        ):
            raise ValueError("invalid relation/event metadata")
        if isinstance(x, Event) and (
            x.episode_id != snap.episode_id or not x.evidence_id
        ):
            raise ValueError("event episode/evidence mismatch")


def local_attributes(state, hand, obj, x, timestamp):
    import numpy as np

    attrs = {
        "geometry9": x.tolist(),
        "hand": hand.semantic_type,
        "distance": float(np.linalg.norm(x[:3])),
        "relative_velocity": None,
        "contact_state": None,
        "grasp_state": None,
        "reachability": None,
        "time_since_last_observed": timestamp - min(hand.last_seen, obj.last_seen),
        "reference_frame": hand.entity_id,
    }
    previous = state["local"].edges.get((hand.entity_id, obj.entity_id, "relative_to"))
    # Discrete derivative in moving EE coordinates; not inertial physical velocity.
    if previous is not None and timestamp > previous.timestamp:
        attrs["relative_velocity"] = (
            (x[:3] - np.asarray(previous.relative_attributes["geometry9"])[:3])
            / (timestamp - previous.timestamp)
        ).tolist()
    for edge in state["global"].edges.values():
        if (
            edge.source_id == obj.entity_id
            and edge.target_id == hand.entity_id
            and (edge.validity or type(edge.relative_attributes.get("value")) is bool)
        ):
            if edge.relation_type == "held_by":
                attrs["grasp_state"] = edge.relative_attributes.get("value", True)
            elif edge.relation_type == "contact":
                attrs["contact_state"] = True
    return attrs


def merge_feasibility(snapshot, summary, registry):
    """Consume reachability owner's assessment; NO IK, collision solver or ranking here.
    Schema v1: episode_id,timestamp,source,evidence_id,base_id,target_id,metrics,
    optional body_nodes (Entity/dict with verified world poses), confidence/method.
    metrics names match tensorize.FEASIBILITY_FIELDS; None is UNKNOWN, not False.
    """
    import math, numpy as np
    from .tensorize import FEASIBILITY_FIELDS
    from .geometry import relative

    if isinstance(summary, (list, tuple)):
        for item in summary:
            snapshot = merge_feasibility(snapshot, item, registry)
        return snapshot
    assessment = summary.get("feasibility", summary)
    if assessment.get("schema") == "tgr.reachability.v1":
        return merge_feasibility(
            snapshot, reachability_summaries(snapshot, assessment, registry), registry
        )
    if summary.get("schema_version", "tgr.feasibility.v1") != "tgr.feasibility.v1":
        raise ValueError("bad feasibility schema")
    if (
        summary.get("episode_id") != snapshot.episode_id
        or summary.get("timestamp") != snapshot.timestamp
    ):
        raise ValueError("feasibility episode/time mismatch")
    source = summary.get("source")
    if source not in ("oracle", "observation") or not summary.get("evidence_id"):
        raise ValueError("feasibility provenance required")
    entities = {n.entity_id: n for n in snapshot.entities}
    for body in summary.get("body_nodes", []):
        body = Entity(**body) if isinstance(body, dict) else body
        if body.semantic_type not in (
            "base",
            "robot_base",
            "left_arm",
            "right_arm",
            "left_ee",
            "right_ee",
        ):
            raise ValueError("unsupported kinematic body")
        if (
            body.entity_id != registry.resolve(body.instance_id)
            or body.timestamp != snapshot.timestamp
            or body.source != source
        ):
            raise ValueError("body identity/time/provenance mismatch")
        entities[body.entity_id] = body
    base_id = summary.get("base_id")
    target_id = summary.get("target_id")
    if base_id not in entities or target_id not in entities or base_id == target_id:
        raise ValueError("feasibility endpoints missing")
    base, target = entities[base_id], entities[target_id]
    if base.semantic_type not in ("base", "robot_base"):
        raise ValueError("base_id must identify robot base")
    metrics = summary.get("metrics", {})
    if set(metrics) - set(FEASIBILITY_FIELDS):
        raise ValueError("unknown feasibility metric")
    values = {}
    for name in FEASIBILITY_FIELDS:
        value = metrics.get(name)
        if value is not None:
            if not isinstance(value, (bool, float, int)) or not math.isfinite(value):
                raise ValueError("nonfinite feasibility")
            if (
                name not in ("workspace_margin", "manipulability")
                and not 0 <= value <= 1
            ):
                raise ValueError("probability/feasibility outside0..1")
            if name == "manipulability" and value < 0:
                raise ValueError("negative manipulability")
        values[name] = value
    attrs = {
        "feasibility": values,
        "base_id": base_id,
        "target_id": target_id,
        "evidence_id": summary["evidence_id"],
        "method": summary.get("method"),
        "known_meaning": (
            "model_internal_estimate"
            if (summary.get("assessment") or {}).get("approximate")
            else "verified_scoped_evidence"
        ),
        "physical_verified": not bool(
            (summary.get("assessment") or {}).get("approximate")
        ),
        "feature_schema": "tgr.feature32.v2.typed_feasibility",
        "geometry9": None,
    }
    if (
        base.validity
        and target.validity
        and base.pose is not None
        and target.pose is not None
    ):
        if base.reference_frame != "world" or target.reference_frame != "world":
            raise ValueError("base/target frame mismatch")
        x = relative(base.pose, target.pose)
        attrs["geometry9"] = x.tolist()
    confidence = summary.get("confidence", 1.0)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid feasibility confidence")
    side = summary.get("arm")
    if side not in (None, "left", "right"):
        raise ValueError("invalid feasibility arm identity")
    attrs["arm"] = side
    attrs["assessment"] = summary.get("assessment")
    instance = "feasibility:" + base_id + ":" + target_id + (":" + side if side else "")
    eid = registry.resolve(instance)
    node = Entity(
        eid,
        "feasibility_summary_v1",
        instance,
        None,
        "world",
        snapshot.timestamp,
        snapshot.timestamp,
        confidence,
        source,
        True,
        attrs,
    )
    entities[eid] = node
    edges = tuple(snapshot.edges) + (
        Edge(
            eid,
            base_id,
            "assesses_base",
            {},
            snapshot.timestamp,
            confidence,
            True,
            source,
        ),
        Edge(
            eid,
            target_id,
            "assesses_target",
            {},
            snapshot.timestamp,
            confidence,
            True,
            source,
        ),
    )
    return replace(
        snapshot,
        entities=tuple(entities.values()),
        edges=edges,
        oracle=snapshot.oracle or source == "oracle",
    )


def assessment_fingerprint(assessment):
    """Stable estimate evidence identity; wall-clock telemetry is not evidence."""
    import hashlib, json

    telemetry = {
        "cpu_wall_ms",
        "elapsed_seconds",
        "wall_time_seconds",
        "profile_seconds",
    }

    def content(value):
        if isinstance(value, dict):
            return {
                key: content(item)
                for key, item in value.items()
                if key not in telemetry and key != "full_assessment_sha256"
            }
        if isinstance(value, (list, tuple)):
            return [content(item) for item in value]
        return value

    return hashlib.sha256(
        json.dumps(content(assessment), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reachability_summaries(snapshot, assessment, registry):
    """Two typed nodes retain independent arm metrics; unverified feasibility UNKNOWN."""
    import hashlib, json

    if (
        assessment.get("episode_id") != snapshot.episode_id
        or assessment.get("timestamp") != snapshot.timestamp
        or assessment.get("frame_id") != "world"
    ):
        raise ValueError("reachability assessment episode/time/frame mismatch")
    base = next(
        (n for n in snapshot.entities if n.semantic_type in ("base", "robot_base")),
        None,
    )
    if base is None:
        raise ValueError("measured base endpoint missing")
    target = assessment.get("target_id")
    if target not in {n.entity_id for n in snapshot.entities}:
        raise ValueError("canonical reachability target missing")

    def certified(info):
        c = info.get("collision", {})
        if (
            not c.get("verified")
            or c.get("scope") != "full_configuration"
            or info.get("ik_status") != "SOLVED"
        ):
            return None
        return (
            True
            if c.get("status") == "FEASIBLE"
            else False if c.get("status") == "INFEASIBLE" else None
        )

    def collision(info):
        c = info.get("collision", {})
        if not c.get("verified") or c.get("scope") != "full_configuration":
            return None
        return (
            True
            if c.get("status") == "FEASIBLE"
            else False if c.get("status") == "INFEASIBLE" else None
        )

    rows = []
    for side in ("left", "right"):
        info = assessment.get("arms", {}).get(side, {})
        if assessment.get("approximate") and assessment.get("estimate_trusted"):
            values = dict(
                left_reachability=None,
                right_reachability=None,
                workspace_margin=info.get("workspace_margin"),
                manipulability=info.get("manipulability"),
                collision_feasibility=float(
                    bool(info.get("estimated_collision_feasible"))
                ),
                base_reposition_feasibility=float(
                    assessment.get("base_reposition_feasibility") == "MODEL_PASS"
                ),
            )
            values[side + "_reachability"] = float(
                bool(info.get("estimated_reachability"))
            )
        else:
            values = dict(
                left_reachability=None,
                right_reachability=None,
                workspace_margin=assessment.get("workspace_margin", {}).get(side),
                manipulability=assessment.get("manipulability", {}).get(side),
                collision_feasibility=collision(info),
                base_reposition_feasibility=None,
            )
            values[side + "_reachability"] = certified(info)
        evidence = assessment.get("full_assessment_sha256") or assessment_fingerprint(
            assessment
        )
        rows.append(
            dict(
                schema_version="tgr.feasibility.v1",
                episode_id=snapshot.episode_id,
                timestamp=snapshot.timestamp,
                source="oracle" if assessment.get("oracle") else "observation",
                evidence_id=evidence,
                base_id=base.entity_id,
                target_id=target,
                arm=side,
                metrics=values,
                method="bounded_URDF_IK_scoped_collision",
                assessment=assessment,
            )
        )
    return rows
