"""Mesh-free URDF FK / spatial Jacobian / bounded DLS IK. No simulator required.
URDF frames, metres, radians; arm-only IK holds measured torso fixed.
Numerical non-convergence is NOT an unreachability certificate.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np


def rigid(T):
    T = np.asarray(T, dtype=float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("invalid SE3")
    return T.copy()


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rotation_log(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return v / 2
    if np.pi - theta < 1e-5:
        _, eig = np.linalg.eigh((R + R.T) / 2)
        axis = eig[:, -1]
        if axis @ v < 0:
            axis = -axis
        return axis * theta
    return theta / (2 * np.sin(theta)) * v


def pose_error(current, target):
    return np.r_[
        target[:3, 3] - current[:3, 3], rotation_log(target[:3, :3] @ current[:3, :3].T)
    ]


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class IKResult:
    status: str
    joints: dict
    position_error: float
    orientation_error: float
    iterations: int
    joint_limit_margin: float | None = None
    manipulability: float | None = None


class URDFKinematics:
    def __init__(self, path, root="base_link", tips=None, tool_transforms=None):
        self.path = str(path)
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.root = root
        self.tips = tips or {"left": "left_gripper_link", "right": "right_gripper_link"}
        self.tool_transforms = {a: rigid(T) for a, T in (tool_transforms or {}).items()}
        if set(self.tool_transforms) - set(self.tips):
            raise ValueError("unknown explicit tool arm")
        tree = ET.parse(path).getroot()
        self.joints = {}
        self.by_child = {}
        for e in tree.findall("joint"):
            if e.find("mimic") is not None:
                raise ValueError("mimic joints unsupported")
            kind = e.attrib["type"]
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(kind)
            o = e.find("origin")
            xyz = np.fromstring(
                o.get("xyz", "0 0 0") if o is not None else "0 0 0", sep=" "
            )
            rpy = np.fromstring(
                o.get("rpy", "0 0 0") if o is not None else "0 0 0", sep=" "
            )
            T = np.eye(4)
            T[:3, 3] = xyz
            T[:3, :3] = (
                rotation([0, 0, 1], rpy[2])
                @ rotation([0, 1, 0], rpy[1])
                @ rotation([1, 0, 0], rpy[0])
            )
            a = e.find("axis")
            axis = np.fromstring(
                a.get("xyz", "1 0 0") if a is not None else "1 0 0", sep=" "
            )
            if (
                axis.shape != (3,)
                or not np.isfinite(axis).all()
                or (kind != "fixed" and np.linalg.norm(axis) == 0)
            ):
                raise ValueError("invalid URDF joint axis")
            axis = (
                axis / np.linalg.norm(axis)
                if np.linalg.norm(axis) > 0
                else np.array([0.0, 0.0, 1.0])
            )
            l = e.find("limit")
            lo = float(l.get("lower", "-inf")) if l is not None else -np.inf
            hi = float(l.get("upper", "inf")) if l is not None else np.inf
            j = Joint(
                e.get("name"),
                e.find("parent").get("link"),
                e.find("child").get("link"),
                kind,
                T,
                axis,
                lo,
                hi,
            )
            self.joints[j.name] = j
            self.by_child[j.child] = j
        self.chains = {arm: self.chain(tip) for arm, tip in self.tips.items()}
        self.arm_joints = {
            arm: [
                j.name
                for j in chain
                if j.name.startswith(arm + "_arm_joint") and j.kind != "fixed"
            ]
            for arm, chain in self.chains.items()
        }
        if any(len(x) != 7 for x in self.arm_joints.values()):
            raise ValueError("expected R1Pro 7-DOF arms")

    def chain(self, tip):
        result = []
        seen = set()
        while tip != self.root:
            if tip in seen:
                raise ValueError("URDF cycle")
            seen.add(tip)
            j = self.by_child[tip]
            result.append(j)
            tip = j.parent
        return result[::-1]

    def validate(self, joints):
        for chain in self.chains.values():
            for j in chain:
                if j.kind == "fixed":
                    continue
                if j.name not in joints or not np.isfinite(joints[j.name]):
                    raise ValueError("missing/nonfinite joint " + j.name)
                if not j.lower - 1e-8 <= joints[j.name] <= j.upper + 1e-8:
                    raise ValueError("joint limit " + j.name)

    def frames(self, joints, arm, base=None):
        if arm not in self.chains:
            raise ValueError("arm must be left or right")
        T = np.eye(4) if base is None else rigid(base)
        result = {self.root: T.copy()}
        axes = {}
        for j in self.chains[arm]:
            T = T @ j.origin
            if j.kind != "fixed":
                value = float(joints[j.name])
                if not np.isfinite(value):
                    raise ValueError("nonfinite joint")
                axes[j.name] = (T[:3, 3].copy(), T[:3, :3] @ j.axis)
                M = np.eye(4)
                if j.kind in ("revolute", "continuous"):
                    M[:3, :3] = rotation(j.axis, value)
                else:
                    M[:3, 3] = j.axis * value
                T = T @ M
            result[j.child] = T.copy()
        return result, axes

    def fk(self, joints, arm, base=None):
        return self.frames(joints, arm, base)[0][
            self.tips[arm]
        ] @ self.tool_transforms.get(arm, np.eye(4))

    def jacobian(self, joints, arm, base=None):
        frames, axes = self.frames(joints, arm, base)
        p = self.fk(joints, arm, base)[:3, 3]
        out = []
        for name in self.arm_joints[arm]:
            origin, axis = axes[name]
            j = self.joints[name]
            out.append(
                np.r_[np.cross(axis, p - origin), axis]
                if j.kind != "prismatic"
                else np.r_[axis, np.zeros(3)]
            )
        return np.stack(out, axis=1)

    def ik(
        self,
        joints,
        arm,
        target,
        base=None,
        max_iterations=80,
        position_tolerance=0.005,
        orientation_tolerance=0.035,
        damping=0.05,
        max_step=0.15,
    ):
        self.validate(joints)
        target = rigid(target)
        q = dict(joints)
        if arm not in self.arm_joints:
            raise ValueError("arm identity")
        if max_iterations < 1:
            raise ValueError("positive iterations required")
        for i in range(max_iterations + 1):
            error = pose_error(self.fk(q, arm, base), target)
            pe, oe = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
            if pe <= position_tolerance and oe <= orientation_tolerance:
                J = self.jacobian(q, arm, base)
                s = np.linalg.svd(J, compute_uv=False)
                margins = [
                    min(q[n] - self.joints[n].lower, self.joints[n].upper - q[n])
                    / (self.joints[n].upper - self.joints[n].lower)
                    for n in self.arm_joints[arm]
                ]
                return IKResult(
                    "SOLVED", q, pe, oe, i, float(min(margins)), float(np.prod(s))
                )
            if i == max_iterations:
                break
            J = self.jacobian(q, arm, base)
            # Orientation weighted in both residual and Jacobian (metre/radian tradeoff).
            W = np.diag([1.0, 1.0, 1.0, 0.35, 0.35, 0.35])
            Jw = W @ J
            ew = W @ error
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping * damping * np.eye(6), ew)
            dq *= min(1.0, max_step / max(np.linalg.norm(dq), 1e-12))
            for name, delta in zip(self.arm_joints[arm], dq):
                j = self.joints[name]
                q[name] = float(np.clip(q[name] + delta, j.lower, j.upper))
        return IKResult("NOT_CONVERGED", q, pe, oe, max_iterations)

    def ik_position(
        self,
        joints,
        arm,
        target_position,
        base=None,
        max_iterations=32,
        position_tolerance=0.01,
        damping=0.035,
        max_step=0.20,
    ):
        """Bounded arm-only position DLS; orientation is unconstrained.

        The linear Jacobian is the world-frame 3x7 Jacobian of the same
        source-declared tool endpoint as fk(). orientation_error is unused
        (zero) in the backward-compatible IKResult; no orientation is promised.
        Failure is numerical nonconvergence, not physical unreachability.
        """
        self.validate(joints)
        if arm not in self.arm_joints:
            raise ValueError("arm identity")
        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("finite world target position required")
        if (
            not 1 <= max_iterations <= 500
            or not np.isfinite([position_tolerance, damping, max_step]).all()
            or min(position_tolerance, damping, max_step) <= 0
        ):
            raise ValueError("position IK bounds")
        # Validate the measured base once. FK is assembled locally, then its
        # endpoint/joint axes are mapped to world for the same world 3x7 DLS.
        measured_base = np.eye(4) if base is None else rigid(base)
        world_R, world_p = measured_base[:3, :3], measured_base[:3, 3]
        q = dict(joints)
        names = self.arm_joints[arm]
        chain = self.chains[arm]
        first = next(i for i, joint in enumerate(chain) if joint.name in names)
        parent = chain[first].parent
        prefix_key = (arm, tuple((joint.name, float(q[joint.name]))
                                for joint in chain[:first] if joint.kind != "fixed"))
        if not hasattr(self, "_position_prefix_cache"):
            self._position_prefix_cache = {}
        prefix = self._position_prefix_cache.get(prefix_key)
        if prefix is None:
            prefix = self.frames(q, arm)[0][parent].copy()
            self._position_prefix_cache[prefix_key] = prefix
            if len(self._position_prefix_cache) > 32:
                del self._position_prefix_cache[next(iter(self._position_prefix_cache))]
        branch = chain[first:]
        rotation_data = {}
        for joint in branch:
            if joint.kind in ("revolute", "continuous"):
                axis = joint.axis / np.linalg.norm(joint.axis)
                K = skew(axis)
                rotation_data[joint.name] = (K, K @ K)
        identity3, identity4 = np.eye(3), np.eye(4)
        tool = self.tool_transforms.get(arm, identity4)
        for i in range(max_iterations + 1):
            # Torso/prefix is held fixed throughout arm-only IK. Reusing it and
            # constant Rodrigues terms preserves the exact URDF branch order.
            T = prefix.copy()
            axes = {}
            for joint in branch:
                T = T @ joint.origin
                if joint.kind != "fixed":
                    value = q[joint.name]
                    axes[joint.name] = (T[:3, 3].copy(), T[:3, :3] @ joint.axis)
                    motion = identity4.copy()
                    if joint.kind in ("revolute", "continuous"):
                        K, K2 = rotation_data[joint.name]
                        motion[:3, :3] = identity3 + np.sin(value) * K + (1 - np.cos(value)) * K2
                    else:
                        motion[:3, 3] = joint.axis * value
                    T = T @ motion
            endpoint = T @ tool
            p = world_R @ endpoint[:3, 3] + world_p
            error = target - p
            pe = float(np.linalg.norm(error))
            origins = np.array([axes[name][0] for name in names]) @ world_R.T + world_p
            directions = np.array([axes[name][1] for name in names]) @ world_R.T
            lever = p - origins
            # Explicit batched cross-product components avoid seven generic
            # np.cross/moveaxis calls per tiny 3x7 Jacobian.
            linear = np.empty_like(directions)
            linear[:, 0] = directions[:, 1] * lever[:, 2] - directions[:, 2] * lever[:, 1]
            linear[:, 1] = directions[:, 2] * lever[:, 0] - directions[:, 0] * lever[:, 2]
            linear[:, 2] = directions[:, 0] * lever[:, 1] - directions[:, 1] * lever[:, 0]
            for column, name in enumerate(names):
                if self.joints[name].kind == "prismatic":
                    linear[column] = directions[column]
            Jp = linear.T.copy()
            if pe <= position_tolerance:
                margins = []
                for name in names:
                    joint = self.joints[name]
                    width = joint.upper - joint.lower
                    if np.isfinite(width) and width > 0:
                        margins.append(
                            min(q[name] - joint.lower, joint.upper - q[name]) / width
                        )
                return IKResult(
                    "SOLVED",
                    q,
                    pe,
                    0.0,
                    i,
                    float(min(margins)) if margins else 1.0,
                    float(np.prod(np.linalg.svd(Jp, compute_uv=False))),
                )
            if i == max_iterations:
                break
            dq = Jp.T @ np.linalg.solve(
                Jp @ Jp.T + damping * damping * np.eye(3), error
            )
            dq *= min(1.0, max_step / max(float(np.linalg.norm(dq)), 1e-12))
            for name, delta in zip(names, dq):
                joint = self.joints[name]
                q[name] = float(np.clip(q[name] + delta, joint.lower, joint.upper))
        return IKResult("NOT_CONVERGED", q, pe, 0.0, max_iterations)
