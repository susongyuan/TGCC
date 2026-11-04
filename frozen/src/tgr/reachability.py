"""Base/arm geometric evidence only. Does not own or issue actuator commands.
Default collision model is explicitly approximate; clear proxies do NOT certify
free space. Unsupported / absent / stale evidence stays UNKNOWN.
"""

from dataclasses import dataclass, field, replace, asdict
from typing import Protocol
import hashlib
import json
import numpy as np
from .kinematics import rigid, rotation, rotation_log

UNKNOWN = "UNKNOWN"


def _oracle_source(source):
    return source in ("oracle", "task_info", "simulator_privileged")


@dataclass(frozen=True)
class RobotState:
    timestamp: float
    world_T_base: object
    joints: dict
    episode_id: str
    frame_id: str = "world"
    source: str = "robot_feedback"


@dataclass(frozen=True)
class TargetPose:
    entity_id: str
    timestamp: float
    world_T_target: object
    episode_id: str
    frame_id: str = "world"
    source: str = "observation"
    pose_kind: str = "object"  # An object pose is NOT an EE grasp pose.
    arm: str | None = None


@dataclass(frozen=True)
class SphereObstacle:
    entity_id: str
    center: object
    radius: float
    timestamp: float
    episode_id: str
    frame_id: str = "world"
    source: str = "observation"


@dataclass(frozen=True)
class CollisionEvidence:
    status: str = UNKNOWN  # FEASIBLE / INFEASIBLE only verified scoped backends
    scope: str = "none"
    minimum_clearance: float | None = None
    proxy_status: str = UNKNOWN
    reason: str = "no collision backend"
    verified: bool = False


def _collision_evidence(value, expected_scope):
    if not isinstance(value, CollisionEvidence):
        raise ValueError("typed collision evidence required")
    if value.status not in (UNKNOWN, "FEASIBLE", "INFEASIBLE"):
        raise ValueError("invalid collision status")
    if value.status != UNKNOWN and (
        not value.verified or value.scope != expected_scope
    ):
        return replace(
            value,
            status=UNKNOWN,
            reason="unverified or wrong-scope collision claim: " + value.reason,
            verified=False,
        )
    return value


class CollisionBackend(Protocol):
    def configuration(
        self, kinematics, state, joints, obstacles
    ) -> CollisionEvidence: ...
    def base_motion(
        self, kinematics, state, candidate_pose, obstacles
    ) -> CollisionEvidence: ...


def _segment_distance(p, a, b):
    d = b - a
    t = float(np.clip((p - a) @ d / max(d @ d, 1e-16), 0, 1))
    return float(np.linalg.norm(p - (a + t * d)))


class CapsuleSphereProxy:
    """Real CPU link capsule vs observed sphere checks. Not mesh or self collision.
    Radii are engineering heuristics, NOT from URDF collision meshes. No empty
    observation, unseen volume or clear proxy is interpreted as certified free.
    Path samples include the held full-body posture, but not carried object mesh.
    """

    def __init__(self, link_radius=0.06, base_radius=0.38, path_samples=9):
        if link_radius <= 0 or base_radius <= 0 or not 2 <= path_samples <= 64:
            raise ValueError("proxy configuration")
        self.link_radius = link_radius
        self.base_radius = base_radius
        self.path_samples = path_samples

    def configuration(self, kinematics, state, joints, obstacles):
        if not obstacles:
            return CollisionEvidence(
                scope="capsule_sphere_proxy", reason="no observed obstacle coverage"
            )
        clearances = []
        for arm in ("left", "right"):
            frames, _ = kinematics.frames(joints, arm, state.world_T_base)
            points = [x[:3, 3] for x in frames.values()]
            for a, b in zip(points[:-1], points[1:]):
                for o in obstacles:
                    clearances.append(
                        _segment_distance(np.asarray(o.center), a, b)
                        - self.link_radius
                        - o.radius
                    )
        # Base sphere centre is heuristic 0.2m above URDF base root.
        base = rigid(state.world_T_base)
        p = (base @ np.array([0.0, 0.0, 0.2, 1.0]))[:3]
        clearances.extend(
            float(np.linalg.norm(np.asarray(o.center) - p))
            - self.base_radius
            - o.radius
            for o in obstacles
        )
        clearance = float(min(clearances))
        return CollisionEvidence(
            UNKNOWN,
            "capsule_sphere_proxy",
            clearance,
            "PROXY_BLOCKED" if clearance < 0 else "PROXY_CLEAR",
            "unverified full geometry/self collision/coverage",
        )

    def base_motion(self, kinematics, state, candidate_pose, obstacles):
        start = rigid(state.world_T_base)
        end = rigid(candidate_pose)
        local = np.linalg.solve(start, end)
        yaw = float(np.arctan2(local[1, 0], local[0, 0]))
        results = []
        for t in np.linspace(0, 1, self.path_samples):
            step = np.eye(4)
            step[:3, 3] = local[:3, 3] * t
            step[:3, :3] = rotation([0, 0, 1], yaw * t)
            s = replace(state, world_T_base=start @ step)
            results.append(self.configuration(kinematics, s, s.joints, obstacles))
        values = [
            r.minimum_clearance for r in results if r.minimum_clearance is not None
        ]
        return CollisionEvidence(
            UNKNOWN,
            "sampled_full_posture_proxy",
            min(values) if values else None,
            (
                "PROXY_BLOCKED"
                if any(r.proxy_status == "PROXY_BLOCKED" for r in results)
                else ("PROXY_CLEAR" if values else UNKNOWN)
            ),
            "sampled, partial geometry; not swept-volume certification",
        )


@dataclass(frozen=True)
class ReachabilityConfig:
    max_age: float = 0.5
    max_skew: float = 0.1
    max_iterations: int = 80
    candidate_radii: tuple = (0.15, 0.3)
    candidate_angles: tuple = (
        0.0,
        1.5707963267948966,
        3.141592653589793,
        -1.5707963267948966,
    )
    candidate_yaws: tuple = (0.0, -0.25, 0.25)
    max_candidates: int = 12
    max_translation: float = 0.35
    max_rotation: float = 0.4
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.035
    minimum_joint_margin: float = 0.03
    minimum_manipulability: float = 1e-5

    def __post_init__(self):
        if not (
            0 < self.max_age <= 10
            and 0 <= self.max_skew <= self.max_age
            and 1 <= self.max_candidates <= 64
            and 1 <= self.max_iterations <= 500
        ):
            raise ValueError("invalid bounds")
        numbers = [
            *self.candidate_radii,
            *self.candidate_angles,
            *self.candidate_yaws,
            self.max_translation,
            self.max_rotation,
            self.position_tolerance,
            self.orientation_tolerance,
            self.minimum_joint_margin,
            self.minimum_manipulability,
        ]
        if not np.isfinite(numbers).all() or any(
            r <= 0 or r > self.max_translation for r in self.candidate_radii
        ):
            raise ValueError("invalid candidate geometry")
        if (
            self.max_translation <= 0
            or self.max_rotation < 0
            or self.position_tolerance <= 0
            or self.orientation_tolerance <= 0
        ):
            raise ValueError("invalid tolerances")


class ReachabilityEngine:
    def __init__(self, kinematics=None, collision_backend=None, config=None):
        self.kinematics = kinematics
        self.collision = collision_backend
        self.config = config or ReachabilityConfig()

    def _validate(self, state, target, now, obstacles):
        c = self.config
        if not np.isfinite(now):
            raise ValueError("nonfinite time")
        if state.frame_id != "world" or target.frame_id != state.frame_id:
            raise ValueError("frame mismatch")
        if not state.episode_id or target.episode_id != state.episode_id:
            raise ValueError("episode mismatch")
        if not target.entity_id:
            raise ValueError("missing canonical target identity")
        rigid(state.world_T_base)
        rigid(target.world_T_target)
        for timestamp in (state.timestamp, target.timestamp):
            if (
                not np.isfinite(timestamp)
                or timestamp > now + 1e-8
                or now - timestamp > c.max_age
            ):
                raise ValueError("stale/future state")
        if abs(state.timestamp - target.timestamp) > c.max_skew:
            raise ValueError("unsynchronized target and joints")
        for o in obstacles:
            if o.frame_id != state.frame_id or o.episode_id != state.episode_id:
                raise ValueError("obstacle provenance mismatch")
            if (
                not o.entity_id
                or np.shape(o.center) != (3,)
                or not np.isfinite(o.center).all()
                or not np.isfinite(o.radius)
                or o.radius <= 0
            ):
                raise ValueError("invalid obstacle")
            if (
                not np.isfinite(o.timestamp)
                or o.timestamp > now + 1e-8
                or now - o.timestamp > c.max_age
                or abs(o.timestamp - state.timestamp) > c.max_skew
            ):
                raise ValueError("stale/unsynchronized obstacle")
        if self.kinematics is not None:
            self.kinematics.validate(state.joints)

    def evaluate(
        self, state, target, now, obstacles=(), required_arms=("left", "right")
    ):
        if isinstance(target, dict):
            return self.evaluate_bimanual(state, target, now, obstacles, required_arms)
        if (
            not required_arms
            or len(set(required_arms)) != len(required_arms)
            or any(a not in ("left", "right") for a in required_arms)
        ):
            raise ValueError("explicit valid arm identities required")
        result = {
            "schema": "tgr.reachability.v1",
            "target_id": target.entity_id,
            "timestamp": state.timestamp,
            "episode_id": state.episode_id,
            "frame_id": state.frame_id,
            "source": "kinematics",
            "oracle": _oracle_source(state.source)
            or _oracle_source(target.source)
            or any(_oracle_source(o.source) for o in obstacles),
            "T_base_target": None,
            "left_reachability": UNKNOWN,
            "right_reachability": UNKNOWN,
            "workspace_margin": {"left": None, "right": None},
            "workspace_margin_kind": "normalized_joint_limit_margin_not_cartesian_workspace",
            "manipulability": {"left": None, "right": None},
            "collision_feasibility": UNKNOWN,
            "base_reposition_feasibility": UNKNOWN,
            "arms": {},
            "valid": False,
            "required_arms": list(required_arms),
        }
        try:
            self._validate(state, target, now, obstacles)
        except (ValueError, KeyError) as e:
            result["reason"] = str(e)
            return result
        result["valid"] = True
        result["T_base_target"] = np.linalg.solve(
            rigid(state.world_T_base), rigid(target.world_T_target)
        ).tolist()
        if self.kinematics is None:
            result["reason"] = "no FK/IK backend"
            return result
        result["urdf_sha256"] = self.kinematics.sha256
        if target.pose_kind != "end_effector_goal" or target.arm not in required_arms:
            result["reason"] = (
                "object pose is not a calibrated arm-specific grasp/operation pose"
            )
            return result
        # One explicit arm-specific target per call. Bimanual feasibility requires
        # two independent goals plus joint collision validation; never mirror a grasp.
        arm = target.arm
        ik = self.kinematics.ik(
            state.joints,
            arm,
            target.world_T_target,
            state.world_T_base,
            max_iterations=self.config.max_iterations,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
        )
        info = {
            "ik_status": ik.status,
            "position_error": ik.position_error,
            "orientation_error": ik.orientation_error,
            "iterations": ik.iterations,
            "reachability": UNKNOWN,
            "collision": asdict(CollisionEvidence()),
            "joints": ik.joints if ik.status == "SOLVED" else None,
        }
        if ik.status == "SOLVED":
            collision = (
                self.collision.configuration(
                    self.kinematics, state, ik.joints, obstacles
                )
                if self.collision
                else CollisionEvidence()
            )
            collision = _collision_evidence(collision, "full_configuration")
            info["collision"] = asdict(collision)
            info["reachability"] = (
                "REACHABLE"
                if collision.status == "FEASIBLE"
                else (
                    "COLLISION_BLOCKED"
                    if collision.status == "INFEASIBLE"
                    else "KINEMATIC_ONLY"
                )
            )
            result["workspace_margin"][arm] = ik.joint_limit_margin
            result["manipulability"][arm] = ik.manipulability
            result["collision_feasibility"] = collision.status
            info["quality_insufficient"] = (
                ik.joint_limit_margin < self.config.minimum_joint_margin
                or ik.manipulability < self.config.minimum_manipulability
            )
            info["reposition_advice"] = (
                "CONSIDER_REPOSITION"
                if info["quality_insufficient"] or collision.status == "INFEASIBLE"
                else "NO_CONFIRMED_NEED"
            )
        else:
            info["reason"] = (
                "bounded numerical IK failed; unreachability NOT established"
            )
            info["reposition_advice"] = (
                "UNRESOLVED_IK_CONSIDER_CANDIDATES_NOT_UNREACHABILITY"
            )
        result["arms"][arm] = info
        result[arm + "_reachability"] = info["reachability"]
        result["reason"] = "single-arm evidence only; other arm UNKNOWN"
        return result

    def evaluate_bimanual(
        self, state, targets, now, obstacles=(), required_arms=("left", "right")
    ):
        if (
            not required_arms
            or set(targets) != set(required_arms)
            or any(t.arm != a for a, t in targets.items())
        ):
            raise ValueError("distinct explicit goal for each required arm")
        individual = {
            a: self.evaluate(state, t, now, obstacles, (a,)) for a, t in targets.items()
        }
        result = dict(next(iter(individual.values())))
        result["target_ids"] = {a: t.entity_id for a, t in targets.items()}
        result["target_id"] = "|".join(
            a + ":" + t.entity_id for a, t in sorted(targets.items())
        )
        result["T_base_target"] = {a: r["T_base_target"] for a, r in individual.items()}
        result["required_arms"] = list(required_arms)
        result["oracle"] = any(r["oracle"] for r in individual.values())
        result["valid"] = all(r["valid"] for r in individual.values())
        result["arms"] = {
            a: r["arms"][a] for a, r in individual.items() if a in r["arms"]
        }
        result["workspace_margin"] = {
            a: r["workspace_margin"][a] for a, r in individual.items()
        }
        result["manipulability"] = {
            a: r["manipulability"][a] for a, r in individual.items()
        }
        result["collision_feasibility"] = UNKNOWN
        for arm in required_arms:
            result[arm + "_reachability"] = UNKNOWN
        if (
            result["valid"]
            and len(result["arms"]) == len(required_arms)
            and all(r["ik_status"] == "SOLVED" for r in result["arms"].values())
        ):
            merged = dict(state.joints)
            for a, r in result["arms"].items():
                for n in self.kinematics.arm_joints[a]:
                    merged[n] = r["joints"][n]
            collision = (
                self.collision.configuration(self.kinematics, state, merged, obstacles)
                if self.collision
                else CollisionEvidence()
            )
            collision = _collision_evidence(collision, "full_configuration")
            result["collision_feasibility"] = collision.status
            for a, r in result["arms"].items():
                r["joints"] = merged.copy()
                r["collision"] = asdict(collision)
                r["reachability"] = (
                    "REACHABLE"
                    if collision.status == "FEASIBLE"
                    else (
                        "COLLISION_BLOCKED"
                        if collision.status == "INFEASIBLE"
                        else "KINEMATIC_ONLY"
                    )
                )
                result[a + "_reachability"] = r["reachability"]
            result["reason"] = (
                "combined dual-arm joint configuration checked; torso held fixed"
            )
        else:
            result["reason"] = (
                "one or more arm goals missing valid IK; dual-arm feasibility UNKNOWN"
            )
        return result

    def propose(
        self, state, target, now, obstacles=(), required_arms=("left", "right")
    ):
        targets = list(target.values()) if isinstance(target, dict) else [target]
        for t in targets:
            self._validate(state, t, now, obstacles)
        primary = targets[0]
        if self.kinematics is None or any(
            t.pose_kind != "end_effector_goal" or t.arm not in required_arms
            for t in targets
        ):
            return []
        c = self.config
        candidates = []
        start = rigid(state.world_T_base)
        # Offset grid, not object-distance reachability. SE2 is relative to measured
        # base frame and composed into world, retaining world base height/tilt.
        offsets = [
            (r, a, y)
            for r in c.candidate_radii
            for a in c.candidate_angles
            for y in c.candidate_yaws
            if abs(y) <= c.max_rotation
        ]
        offsets = list(dict.fromkeys(offsets))
        # Interleave radii to avoid spending the full bounded budget on only the
        # shortest ring; cost is considered by final ranking.
        offsets.sort(key=lambda v: (abs(v[2]), v[1], v[0]))
        for r, a, yaw in offsets[: c.max_candidates]:
            delta = np.eye(4)
            delta[:3, 3] = [r * np.cos(a), r * np.sin(a), 0]
            delta[:3, :3] = rotation([0, 0, 1], yaw)
            candidate = start @ delta
            moved = replace(state, world_T_base=candidate)
            assessment = self.evaluate(moved, target, now, obstacles, required_arms)
            path = (
                self.collision.base_motion(self.kinematics, state, candidate, obstacles)
                if self.collision
                else CollisionEvidence()
            )
            path = _collision_evidence(path, "full_base_motion")
            solved = sum(
                assessment["arms"].get(arm, {}).get("ik_status") == "SOLVED"
                for arm in required_arms
            )
            # Unspecified other hand stays UNKNOWN; do not count a one-arm solve as
            # simultaneous dual-arm feasibility. Caller must supply both goals.
            clear = path.status == "FEASIBLE"
            blocked = (
                path.status == "INFEASIBLE" or path.proxy_status == "PROXY_BLOCKED"
            )
            margin = min(
                (assessment["workspace_margin"].get(a) or 0.0) for a in required_arms
            )
            manipulation = min(
                (assessment["manipulability"].get(a) or 0.0) for a in required_arms
            )
            score = (
                10 * solved
                + 2 * clear
                - 20 * blocked
                + margin
                + 0.01 * np.log1p(manipulation)
                - r
                - 0.2 * abs(yaw)
            )
            candidate_id = hashlib.sha256(
                json.dumps(
                    [
                        state.episode_id,
                        state.timestamp,
                        assessment["target_id"],
                        candidate.tolist(),
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:24]
            all_arms = all(
                assessment[a + "_reachability"] == "REACHABLE" for a in required_arms
            )
            feasible = (
                "FEASIBLE"
                if clear and all_arms
                else ("INFEASIBLE" if path.status == "INFEASIBLE" else UNKNOWN)
            )
            assessment["base_reposition_feasibility"] = feasible
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "timestamp": state.timestamp,
                    "episode_id": state.episode_id,
                    "target_id": assessment["target_id"],
                    "frame_id": state.frame_id,
                    "world_T_base": candidate.tolist(),
                    "motion_cost": r + 0.2 * abs(yaw),
                    "score": float(score),
                    "base_reposition_feasibility": feasible,
                    "path_evidence": asdict(path),
                    "assessment": assessment,
                    "execution_authorized": False,
                    "evidence_only": True,
                }
            )
        return sorted(candidates, key=lambda x: (-x["score"], x["candidate_id"]))

    @staticmethod
    def vlm_summary(assessment, candidates=()):
        # Compact evidence only, never entire local graph or joint solutions.
        keep = (
            "schema",
            "target_id",
            "timestamp",
            "episode_id",
            "frame_id",
            "oracle",
            "valid",
            "reason",
            "T_base_target",
            "left_reachability",
            "right_reachability",
            "workspace_margin",
            "workspace_margin_kind",
            "manipulability",
            "collision_feasibility",
            "base_reposition_feasibility",
        )
        return {
            "feasibility": {k: assessment[k] for k in keep if k in assessment},
            "base_candidates": [
                {
                    k: c[k]
                    for k in (
                        "candidate_id",
                        "world_T_base",
                        "score",
                        "base_reposition_feasibility",
                        "path_evidence",
                        "timestamp",
                    )
                }
                for c in candidates
            ],
            "task_completion_evidence": False,
            "control_owner": "frozen_pi_23D_action_expert",
            "requires_trained_validated_vlm_selection": True,
            "reposition_advice": {
                a: info.get("reposition_advice", UNKNOWN)
                for a, info in assessment.get("arms", {}).items()
            },
        }


class RepositionGate:
    """Optional future authorization/feedback state machine; no actuator method.
    Default training metadata denies all REPOSITION_BASE selections. Existing pi
    expert remains the ONLY control owner. Time/action issuance cannot complete.
    """

    def __init__(
        self,
        trained_validated=False,
        position_tolerance=0.03,
        yaw_tolerance=0.05,
        consecutive_samples=3,
        max_age=0.5,
        max_gap=0.25,
    ):
        if (
            consecutive_samples < 2
            or min(position_tolerance, yaw_tolerance, max_age, max_gap) <= 0
        ):
            raise ValueError("feedback bounds")
        self.trained_validated = bool(trained_validated)
        self.ptol = position_tolerance
        self.ytol = yaw_tolerance
        self.required = consecutive_samples
        self.max_age = max_age
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.last_time = None
        self.completed = False

    def select(self, decision, candidate, now, episode_id):
        self.reset()
        if not self.trained_validated:
            raise PermissionError(
                "REPOSITION_BASE head not trained and validated; evidence-only"
            )
        if (
            decision.get("source") != "vlm"
            or decision.get("subtask") != "REPOSITION_BASE"
            or decision.get("candidate_id") != candidate.get("candidate_id")
        ):
            raise PermissionError("explicit matching VLM selection required")
        if (
            candidate.get("episode_id") != episode_id
            or candidate.get("frame_id") != "world"
            or not np.isfinite(now)
            or not 0 <= now - candidate["timestamp"] <= self.max_age
        ):
            raise PermissionError("stale/mismatched selection")
        if (
            candidate.get("base_reposition_feasibility") != "FEASIBLE"
            or candidate.get("path_evidence", {}).get("status") != "FEASIBLE"
            or not candidate.get("path_evidence", {}).get("verified")
            or candidate.get("path_evidence", {}).get("scope") != "full_base_motion"
        ):
            raise PermissionError(
                "verified reachability and motion collision evidence required"
            )
        rigid(candidate["world_T_base"])
        self.candidate = json.loads(json.dumps(candidate))
        self.last_time = candidate["timestamp"]
        return {
            "authorized_subtask": "REPOSITION_BASE",
            "candidate_id": candidate["candidate_id"],
            "control_owner": "frozen_pi_23D_action_expert",
            "issue_actuator_commands": False,
        }

    def observe(self, state, now):
        if self.candidate is None:
            return {"status": "NOT_SELECTED", "completed": False}
        if self.completed:
            return {
                "status": "COMPLETED",
                "completed": True,
                "requires_graph_update_and_reassessment": True,
            }
        c = self.candidate
        fresh = (
            state.episode_id == c["episode_id"]
            and state.frame_id == c["frame_id"]
            and np.isfinite(state.timestamp)
            and np.isfinite(now)
            and 0 <= now - state.timestamp <= self.max_age
            and state.timestamp > self.last_time
            and state.source == "robot_feedback"
        )
        if not fresh:
            self.count = 0
            return {"status": "INVALID_FEEDBACK", "completed": False}
        if state.timestamp - self.last_time > self.max_gap:
            self.count = 0
        measured = rigid(state.world_T_base)
        goal = rigid(c["world_T_base"])
        # Full rotation error is conservative relative to yaw-only tolerance.
        position_error = float(np.linalg.norm(measured[:3, 3] - goal[:3, 3]))
        angle = float(np.linalg.norm(rotation_log(goal[:3, :3] @ measured[:3, :3].T)))
        self.last_time = state.timestamp
        self.count = (
            self.count + 1 if position_error <= self.ptol and angle <= self.ytol else 0
        )
        self.completed = self.count >= self.required
        return {
            "status": "COMPLETED" if self.completed else "AWAITING_FEEDBACK",
            "completed": self.completed,
            "position_error": position_error,
            "orientation_error": angle,
            "consecutive_samples": self.count,
            "requires_graph_update_and_reassessment": self.completed,
        }
