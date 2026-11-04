"""Structured, causal graph memory. Never reconstruct metric poses from tokens.

All times are exposure times. Reprojection changes the estimate, not last_seen.
The fallback uncertainty model is explicit: 1 cm position / 1 degree rotation
at observation, with 2 mm/s static or 20 cm/s movable process noise.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from .contracts import Evidence, Truth


class InformationSource(str, Enum):
    OBSERVED = "OBSERVED"
    MEMORY_INFERRED = "MEMORY_INFERRED"
    UNKNOWN = "UNKNOWN"


def covariance(value=None):
    if value is None:
        return np.diag([0.01**2] * 3 + [math.radians(1) ** 2] * 3)
    x = np.asarray(value, dtype=float)
    if x.shape == (3, 3):
        out = covariance()
        out[:3, :3] = x
        x = out
    if x.shape != (6, 6) or not np.isfinite(x).all():
        raise ValueError("Pose covariance must be finite 3x3 or 6x6")
    if not np.allclose(x, x.T) or np.linalg.eigvalsh(x).min() < -1e-10:
        raise ValueError("Pose covariance must be positive semidefinite")
    return x


def se3(value):
    x = np.asarray(value, dtype=float)
    if x.shape != (4, 4) or not np.isfinite(x).all():
        raise ValueError("Expected finite homogeneous world pose")
    if not np.allclose(x[3], [0, 0, 0, 1]) or not np.allclose(
        x[:3, :3].T @ x[:3, :3], np.eye(3), atol=1e-4
    ):
        raise ValueError("Invalid SE3 pose")
    if np.linalg.det(x[:3, :3]) < 0.99:
        raise ValueError("Reflected frame")
    return x


def relative_with_covariance(world_object, object_cov, world_robot, robot_cov):
    """First-order independent-error SE3 propagation, world small-angle errors."""
    obj, robot = se3(world_object), se3(world_robot)
    rot = robot[:3, :3].T
    d = obj[:3, 3] - robot[:3, 3]
    skew = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    jo = np.zeros((6, 6))
    jr = np.zeros((6, 6))
    jo[:3, :3] = rot
    jo[3:, 3:] = rot
    jr[:3, :3] = -rot
    jr[:3, 3:] = rot @ skew
    jr[3:, 3:] = -rot
    cov = jo @ covariance(object_cov) @ jo.T + jr @ covariance(robot_cov) @ jr.T
    return np.linalg.inv(robot) @ obj, cov


@dataclass
class MemoryEntity:
    object_id: str
    instance_id: str
    world_pose: list
    last_observed: float
    confidence: float
    visibility: object
    motion: str
    pose_covariance: list
    node: dict = field(default_factory=dict)
    information_source: str = InformationSource.OBSERVED.value
    coordinate_frame: str = "world"


class ThreeGraphMemory:
    def __init__(self):
        self.episode = None
        self.time = -math.inf
        self.entities = {}
        self.facts = {}  # local most recent -> persistent global semantic memory
        self.task_facts = {}
        self.version = 0
        self.last_resolution = {}
        self.robot_pose = None
        self.robot_cov = covariance()

    def reset(self, episode):
        self.__init__()
        self.episode = episode

    def update(
        self,
        snapshot,
        direct_evidence,
        robot_pose=None,
        robot_cov=None,
        task_states=None,
    ):
        episode = snapshot.get("episode_id", self.episode)
        now = float(snapshot.get("timestamp", self.time))
        if self.episode != episode:
            self.reset(episode)
        if not math.isfinite(now) or now < self.time:
            raise ValueError("Noncausal graph memory update")
        if now == self.time:
            return
        self.time = now
        self.version += 1
        for n in snapshot["entities"]:
            if not n["validity"] or n.get("pose") is None:
                continue
            if n["reference_frame"] != "world":
                raise ValueError("Metric memory requires world-frame provider output")
            attrs = n.get("attributes", {})
            last = float(n.get("last_seen", n["timestamp"]))
            if last > now or float(n["timestamp"]) > now:
                raise ValueError("Future memory exposure")
            if (
                attrs.get("information_source")
                == InformationSource.MEMORY_INFERRED.value
            ):
                continue
            se3(n["pose"])
            if (
                not math.isfinite(float(n["confidence"]))
                or not 0 <= float(n["confidence"]) <= 1
            ):
                raise ValueError("Invalid memory confidence")
            previous = self.entities.get(n["instance_id"])
            if previous is not None and last <= previous.last_observed:
                continue
            static = attrs.get("assumed_static") is True or n[
                "semantic_type"
            ].startswith(
                ("floor.", "room", "refrigerator", "microwave", "cabinet", "counter")
            )
            motion = attrs.get("motion_property", "static" if static else "movable")
            self.entities[n["instance_id"]] = MemoryEntity(
                n["entity_id"],
                n["instance_id"],
                n["pose"],
                last,
                float(n["confidence"]),
                attrs.get("visible"),
                motion,
                covariance(attrs.get("pose_covariance")).tolist(),
                deepcopy(n),
            )
        for key, ev in direct_evidence.items():
            previous = self.facts.get(key)
            if (
                ev.valid
                and math.isfinite(ev.timestamp)
                and ev.timestamp <= now
                and ev.information_source == "OBSERVED"
                and (previous is None or ev.timestamp > previous.timestamp)
            ):
                self.facts[key] = ev
        # Task evidence is kept separately; task completion cannot manufacture
        # physical held/open/inside facts through circular reasoning.
        self.task_facts = deepcopy(
            self.task_facts if task_states is None else task_states
        )
        self.robot_pose = None if robot_pose is None else se3(robot_pose)
        self.robot_cov = covariance(robot_cov)

    def estimate(self, name, now, frame_pose=None, frame_cov=None):
        record = self.entities.get(name)
        if record is None:
            return None
        if not math.isfinite(now):
            raise ValueError("Nonfinite memory query time")
        age = now - record.last_observed
        static = record.motion == "static"
        if age < 0 or age > (600 if static else 2.2):
            return None
        cov = covariance(record.pose_covariance).copy()
        cov[:3, :3] += np.eye(3) * ((0.002 if static else 0.2) * age) ** 2
        pose = se3(record.world_pose)
        base = self.robot_pose if frame_pose is None else frame_pose
        base_cov = self.robot_cov if frame_cov is None else frame_cov
        relative, relcov = (
            (pose, cov)
            if base is None
            else relative_with_covariance(pose, cov, base, base_cov)
        )
        sigma = float(np.sqrt(np.linalg.eigvalsh(relcov[:3, :3]).max()))
        conf = record.confidence * math.exp(-age / (600 if static else 3))
        if sigma > (0.25 if static else 0.12) or conf < 0.5:
            return None
        return dict(
            object_id=record.object_id,
            instance_id=name,
            world_pose=pose.tolist(),
            relative_pose=relative.tolist(),
            coordinate_frame="world" if base is None else "robot_base",
            last_observed=record.last_observed,
            age=age,
            confidence=conf,
            visibility=False,
            motion=record.motion,
            pose_covariance=cov.tolist(),
            relative_covariance=relcov.tolist(),
            uncertainty_sigma_m=sigma,
            information_source="MEMORY_INFERRED",
        )

    def resolve(self, predicate, current, now):
        direct = current.get(predicate.key, Evidence())
        if direct.truth_for(predicate, now) != Truth.UNKNOWN:
            return direct
        if predicate.requires_fresh_observation or not predicate.memory_acceptable:
            return Evidence()
        ev = self.facts.get(predicate.key)
        if ev is None:
            # Task state may establish a logical dependency, never manufacture
            # a physical held/open/inside fact from past task completion.
            if predicate.kind in ("task_completed", "dependency_satisfied"):
                state = self.task_facts.get(predicate.subject)
                if isinstance(state, dict) and state.get("verified") is True:
                    ev = Evidence(
                        state.get("value"),
                        float(state.get("timestamp", 0)),
                        float(state.get("confidence", 0)),
                        True,
                        source="verified_task_history",
                        information_source="MEMORY_INFERRED",
                    )
                    return (
                        ev
                        if ev.truth_for(predicate, now) != Truth.UNKNOWN
                        else Evidence()
                    )
            return Evidence()
        # A current contradictory but low-confidence measurement invalidates
        # blind reuse: uncertainty is resolved by observation, not old certainty.
        if direct.valid and type(direct.value) is bool and direct.value != ev.value:
            return Evidence()
        names = [predicate.subject] + ([predicate.target] if predicate.target else [])
        estimates = [self.estimate(n, now) for n in names]
        if any(x is None for x in estimates):
            return Evidence()
        conf = min([ev.confidence] + [x["confidence"] for x in estimates])
        inferred = Evidence(
            ev.value,
            ev.timestamp,
            conf,
            ev.valid,
            ev.frame,
            ev.source,
            "MEMORY_INFERRED",
        )
        return (
            inferred
            if inferred.truth_for(predicate, now) != Truth.UNKNOWN
            else Evidence()
        )

    def resolve_contracts(self, contracts, current, now):
        # Each predicate has its own age/freshness check at evaluation. Resolve
        # with the strictest semantics for duplicate physical keys.
        resolved = dict(current)
        for contract in contracts:
            for p in contract.predicates:
                ev = self.resolve(p, current, now)
                if ev.valid and (p.key not in resolved or not resolved[p.key].valid):
                    resolved[p.key] = ev
                elif (
                    ev.valid
                    and ev.information_source == "MEMORY_INFERRED"
                    and current.get(p.key, Evidence()).truth_for(p, now)
                    == Truth.UNKNOWN
                ):
                    resolved[p.key] = ev
        self.last_resolution = {
            "|".join(k): dict(
                value=e.value,
                information_source=e.information_source,
                last_observed=e.timestamp,
                confidence=e.confidence,
            )
            for k, e in resolved.items()
        }
        return resolved

    def supplement_snapshot(self, snapshot, now):
        """Feed retained geometry through the EXISTING graph encoder interface."""
        out = deepcopy(snapshot)
        present = {
            n["instance_id"]
            for n in out["entities"]
            if n["validity"] and n.get("pose") is not None
        }
        for name, record in self.entities.items():
            estimate = self.estimate(name, now)
            if name in present or estimate is None:
                continue
            node = deepcopy(record.node)
            node.update(
                pose=estimate["world_pose"],
                timestamp=record.last_observed,
                last_seen=record.last_observed,
                confidence=estimate["confidence"],
                validity=True,
            )
            node["attributes"] = dict(
                node.get("attributes", {}),
                information_source="MEMORY_INFERRED",
                visible=False,
                pose_covariance=estimate["pose_covariance"],
                motion_property=record.motion,
            )
            out["entities"] = [
                n for n in out["entities"] if n["instance_id"] != name
            ] + [node]
        return out
