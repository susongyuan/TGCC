"""Available center-position estimates with explicit, partial collision proxies.

Known means a deterministic model estimate, never physical certification. No
actuator calls, task predicates, scene resets, or implicit grasp orientation.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import time
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from .kinematics import rigid, rotation
from .reachability import RobotState, TargetPose

IDENTITY = "center_position_ik_capsule_box_v1"


@dataclass(frozen=True)
class OrientedBoxObstacle:
    entity_id: str
    world_T_box: object  # world pose of the nominal box CENTER, not root origin
    extent_m: object  # full dimensions, metres
    timestamp: float
    episode_id: str
    source: str = "observation"
    assumed_static: bool = False
    container: bool = False
    frame_id: str = "world"


def _point_box_distance(point, obstacle):
    # Callers validate each box once; do not repeat SE3 checks per link pair.
    T = np.asarray(obstacle.world_T_box, dtype=float)
    local = T[:3, :3].T @ (np.asarray(point) - T[:3, 3])
    excess = np.maximum(np.abs(local) - np.asarray(obstacle.extent_m) / 2, 0)
    return float(np.linalg.norm(excess))


def _inside_box(point, obstacle):
    # Callers validate each box once; do not repeat SE3 checks per link pair.
    T = np.asarray(obstacle.world_T_box, dtype=float)
    local = T[:3, :3].T @ (np.asarray(point) - T[:3, 3])
    return bool(np.all(np.abs(local) <= np.asarray(obstacle.extent_m) / 2 + 1e-9))


def _segment_box_distance(a, b, obstacle):
    """Exact piecewise-quadratic distance of a segment to an oriented box.

    The primitives are approximate; the segment-box minimization itself is not
    point sampling. Intersection returns zero, not penetration depth.
    """
    # Callers validate each box once; do not repeat SE3 checks per link pair.
    T = np.asarray(obstacle.world_T_box, dtype=float)
    p = T[:3, :3].T @ (np.asarray(a) - T[:3, 3])
    d = T[:3, :3].T @ (np.asarray(b) - np.asarray(a))
    half = np.asarray(obstacle.extent_m, dtype=float) / 2
    knots = [0.0, 1.0]
    for axis in range(3):
        if abs(d[axis]) > 1e-14:
            for plane in (-half[axis], half[axis]):
                t = float((plane - p[axis]) / d[axis])
                if 0 < t < 1:
                    knots.append(t)
    knots = sorted(set(knots))

    def squared(t):
        v = np.maximum(np.abs(p + t * d) - half, 0)
        return float(v @ v)

    best = min(squared(t) for t in knots)
    for lo, hi in zip(knots[:-1], knots[1:]):
        middle = p + (lo + hi) / 2 * d
        active = np.abs(middle) > half
        if not active.any():
            return 0.0
        sign = np.sign(middle[active])
        intercept = p[active] - sign * half[active]
        slope = d[active]
        denom = float(slope @ slope)
        if denom > 1e-20:
            t = float(np.clip(-(slope @ intercept) / denom, lo, hi))
            best = min(best, squared(t))
    return float(np.sqrt(max(0.0, best)))


def _segment_segment_distance(a, b, c, d):
    """Bounded convex quadratic minimization, including degenerate segments."""
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    candidates = []
    for s in (0.0, 1.0):
        t = min(1.0, max(0.0, (bb * s + ee) / cc)) if cc > 1e-20 else 0.0
        candidates.append((s, t))
    for t in (0.0, 1.0):
        s = min(1.0, max(0.0, (bb * t - dd) / aa)) if aa > 1e-20 else 0.0
        candidates.append((s, t))
    determinant = aa * cc - bb * bb
    if determinant > 1e-20:
        s = (-cc * dd + bb * ee) / determinant
        t = (-bb * dd + aa * ee) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    return min(float(np.linalg.norm(w + s * u - t * v)) for s, t in candidates)


def _points_boxes_distance(points, rotations, centers, half):
    if not len(points) or not len(centers):
        return np.empty((len(points), len(centers)))
    local = np.einsum("mkc,kcd->mkd", points[:, None, :] - centers[None, :, :], rotations)
    excess = np.maximum(np.abs(local) - half[None, :, :], 0.0)
    return np.sqrt(np.sum(excess * excess, axis=-1))


def _segments_boxes_distance(starts, ends, rotations, centers, half):
    """Vectorized same exact piecewise-quadratic segment/OBB minimization."""
    n, k = len(starts), len(centers)
    if not n or not k:
        return np.empty((n, k))
    p = np.einsum("nkc,kcd->nkd", starts[:, None, :] - centers[None, :, :], rotations)
    d = np.einsum("nc,kcd->nkd", ends - starts, rotations)
    p, d = p.reshape(-1, 3), d.reshape(-1, 3)
    h = np.broadcast_to(half, (n, k, 3)).reshape(-1, 3)
    planes = np.stack((-h - p, h - p), axis=-1)
    crossings = np.divide(
        planes, d[:, :, None], out=np.ones_like(planes),
        where=np.abs(d[:, :, None]) > 1e-14,
    )
    valid = (crossings > 0) & (crossings < 1) & (np.abs(d[:, :, None]) > 1e-14)
    knots = np.sort(np.concatenate(
        (np.zeros((n*k, 1)), np.ones((n*k, 1)), np.where(valid, crossings, 1.0).reshape(-1, 6)), axis=1
    ), axis=1)
    lo, hi = knots[:, :-1], knots[:, 1:]
    middle = p[:, None, :] + ((lo + hi) / 2)[:, :, None] * d[:, None, :]
    active = np.abs(middle) > h[:, None, :]
    intercept = np.where(active, p[:, None, :] - np.sign(middle) * h[:, None, :], 0.0)
    slope = np.where(active, d[:, None, :], 0.0)
    denom = np.sum(slope * slope, axis=-1)
    optimum = np.divide(
        -np.sum(slope * intercept, axis=-1), denom, out=lo.copy(), where=denom > 1e-20
    )
    optimum = np.minimum(hi, np.maximum(lo, optimum))
    times = np.concatenate((knots, optimum), axis=1)
    excess = np.maximum(np.abs(p[:, None, :] + times[:, :, None] * d[:, None, :]) - h[:, None, :], 0.0)
    squared = np.sum(excess * excess, axis=-1)
    return np.sqrt(np.maximum(np.min(squared, axis=1), 0.0)).reshape(n, k)


def _segments_pair_distance(starts, ends):
    """All pair distances using the same bounded convex quadratic as scalar."""
    u = ends - starts
    w = starts[:, None, :] - starts[None, :, :]
    aa = np.sum(u * u, axis=1)[:, None]
    cc = aa.T
    bb = u @ u.T
    dd = np.sum(w * u[:, None, :], axis=-1)
    ee = np.sum(w * u[None, :, :], axis=-1)
    best = np.full(bb.shape, np.inf)
    def candidate(s, t, valid=None):
        nonlocal best
        residual = w + s[:, :, None] * u[:, None, :] - t[:, :, None] * u[None, :, :]
        squared = np.sum(residual * residual, axis=-1)
        if valid is not None:
            squared = np.where(valid, squared, np.inf)
        best = np.minimum(best, squared)
    for value in (0.0, 1.0):
        s = np.full(bb.shape, value)
        t = np.clip(np.divide(bb * s + ee, cc, out=np.zeros_like(bb), where=cc > 1e-20), 0, 1)
        candidate(s, t)
    for value in (0.0, 1.0):
        t = np.full(bb.shape, value)
        s = np.clip(np.divide(bb * t - dd, aa, out=np.zeros_like(bb), where=aa > 1e-20), 0, 1)
        candidate(s, t)
    determinant = aa * cc - bb * bb
    s = np.divide(-cc * dd + bb * ee, determinant, out=np.zeros_like(bb), where=determinant > 1e-20)
    t = np.divide(-bb * dd + aa * ee, determinant, out=np.zeros_like(bb), where=determinant > 1e-20)
    valid = (determinant > 1e-20) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    candidate(s, t, valid)
    return np.sqrt(np.maximum(best, 0.0))

class CapsuleBoxProxy:
    """URDF link-origin capsules, nominal boxes and a heuristic base sphere.

    Includes actual-link capsules, source-declared gripper planning spheres,
    and finite posture/path interpolation. The visual EEF frame's translation
    is NOT a solid body. Finger spheres require measured named finger joints.
    Mesh coverage, carried-object mesh, arm-trunk self collision and articulated
    obstacle doors remain unmodeled.
    """
    def __init__(
        self, link_radius=0.025, base_radius=0.38, path_samples=5
    ):
        if (
            not np.isfinite([link_radius, base_radius]).all()
            or min(link_radius, base_radius) <= 0
            or not 2 <= path_samples <= 32
        ):
            raise ValueError("proxy bounds")
        self.link_radius = float(link_radius)
        self.base_radius = float(base_radius)
        self.path_samples = int(path_samples)
        catalog_path = Path(__file__).resolve().parents[2] / "reports/robot_gripper_proxy_source.json"
        raw = catalog_path.read_bytes()
        self._gripper_catalog = json.loads(raw)
        self._gripper_catalog_sha256 = hashlib.sha256(raw).hexdigest()
        self._box_cache = OrderedDict()
        self._base_cache = OrderedDict()
        self._frames_cache = OrderedDict()
        self._configuration_cache = OrderedDict()
        self._exclusion_cache = {}
        self.profile_counts = {"configuration_cache_hits": 0, "configuration_computations": 0}
        self._source_ignore = {
            tuple(sorted(pair))
            for pair in self._gripper_catalog["self_collision_ignore_pairs"]
        }
        self._gripper_assembly = {
            arm + suffix: arm
            for arm in ("left", "right")
            for suffix in ("_arm_link7", "_gripper_link", "_gripper_finger_link1", "_gripper_finger_link2")
        }
        self._sphere_local = {
            link: (np.asarray([sphere["center"] for sphere in rows], dtype=float),
                   tuple(float(sphere["radius"]) for sphere in rows))
            for link, rows in self._gripper_catalog["spheres_by_link"].items()
        }

    def _self_excluded(self, first, second):
        key = (first, second)
        cached = self._exclusion_cache.get(key)
        if cached is not None:
            return cached
        excluded = first == second or tuple(sorted(key)) in self._source_ignore
        if not excluded:
            group = self._gripper_assembly.get(first)
            excluded = group is not None and group == self._gripper_assembly.get(second)
        self._exclusion_cache[key] = excluded
        return excluded

    def _gripper_spheres(self, kinematics, state, joints, frames_by_arm=None):
        spheres, missing = [], []
        if getattr(kinematics, "sha256", None) != self._gripper_catalog["compatible_urdf_sha256"]:
            return spheres, ["source_gripper_catalog_URDF_not_matched"]
        for arm in ("left", "right"):
            frames = frames_by_arm[arm] if frames_by_arm is not None else kinematics.frames(joints, arm, state.world_T_base)[0]
            grip_name = arm + "_gripper_link"
            if grip_name not in frames:
                missing.append(grip_name + "_world_frame_missing")
                continue
            grip = frames[grip_name]
            for link in (grip_name, arm + "_gripper_finger_link1", arm + "_gripper_finger_link2"):
                if link == grip_name:
                    world = grip
                else:
                    joint = kinematics.by_child.get(link)
                    if joint is None or joint.parent != grip_name or joint.kind != "prismatic":
                        missing.append(link + "_source_prismatic_joint_not_supported")
                        continue
                    value = joints.get(joint.name)
                    if (
                        value is None or not np.isfinite(value)
                        or not joint.lower - 1e-8 <= value <= joint.upper + 1e-8
                    ):
                        missing.append(link + "_measured_joint_missing_or_invalid")
                        continue
                    motion = np.eye(4)
                    motion[:3, 3] = joint.axis * float(value)
                    world = grip @ joint.origin @ motion
                local, radii = self._sphere_local[link]
                world_centers = local @ world[:3, :3].T + world[:3, 3]
                spheres.extend((arm, link, center, radius) for center, radius in zip(world_centers, radii))
        return spheres, missing

    def _capsules(self, kinematics, state, joints, frames_by_arm=None):
        arms, body = {}, {}
        for arm in ("left", "right"):
            frames = frames_by_arm[arm] if frames_by_arm is not None else kinematics.frames(joints, arm, state.world_T_base)[0]
            items = list(frames.items())
            root = arm + "_arm_base_link"
            start = next((i for i, (name, _) in enumerate(items) if name == root), None)
            if start is None:
                raise ValueError("explicit URDF arm root missing")
            for (na, fa), (nb, fb) in zip(items[:start], items[1:start + 1]):
                key = (na, nb)
                if key not in body and np.linalg.norm(fb[:3, 3] - fa[:3, 3]) > 1e-9:
                    body[key] = (na + "->" + nb, fa[:3, 3], fb[:3, 3])
            points = [(n, f[:3, 3]) for n, f in items[start:]]
            # The source eef_vis_link is a coordinate frame with no URDF
            # collision body. Preserve its FK/IK offset, not invented volume.
            arms[arm] = [
                (na + "->" + nb, a, b)
                for (na, a), (nb, b) in zip(points[:-1], points[1:])
                if np.linalg.norm(b - a) > 1e-9
            ]
        return arms, list(body.values())

    @staticmethod
    def _kin_identity(kinematics):
        # Include model/tool configuration as well as the live object identity.
        # Cache entries remain numeric geometry; freshness is still validated
        # at the public evaluate entry for every current observation.
        tools = getattr(kinematics, "tool_transforms", {})
        return (
            id(kinematics), getattr(kinematics, "sha256", None),
            getattr(kinematics, "root", None),
            tuple(sorted(getattr(kinematics, "tips", {}).items())),
            tuple((name, np.asarray(T, dtype=float).tobytes()) for name,T in sorted(tools.items())),
        )

    def _prepared_boxes(self, obstacles):
        rows = []
        for obstacle in obstacles:
            T = np.asarray(obstacle.world_T_box, dtype=float)
            extent = np.asarray(obstacle.extent_m, dtype=float)
            key = (T.shape, T.tobytes(), extent.shape, extent.tobytes())
            value = self._box_cache.get(key)
            if value is None:
                T = rigid(T)
                if extent.shape != (3,) or not np.isfinite(extent).all() or (extent <= 0).any():
                    raise ValueError("positive finite box extent in metres required")
                value = (T[:3, :3].copy(), T[:3, 3].copy(), extent.copy() / 2)
                self._box_cache[key] = value
                if len(self._box_cache) > 64:
                    self._box_cache.popitem(last=False)
            rows.append(value)
        if not rows:
            return np.empty((0,3,3)), np.empty((0,3)), np.empty((0,3))
        return tuple(np.stack([row[i] for row in rows]) for i in range(3))

    def _world_frames(self, kinematics, state, joints, arm, base):
        if not hasattr(kinematics, "chains"):
            return kinematics.frames(joints, arm, base)[0]
        names = [j.name for j in kinematics.chains[arm] if j.kind != "fixed"]
        key = (self._kin_identity(kinematics), state.episode_id, arm, base.tobytes(),
               tuple((name, float(joints[name])) for name in names))
        result = self._frames_cache.get(key)
        if result is not None:
            self._frames_cache.move_to_end(key)
            return result
        # Same URDF FK with a once-validated world left product; no repeated
        # rigid-base validation within each chain calculation.
        local, _ = kinematics.frames(joints, arm)
        result = {name: base @ T for name, T in local.items()}
        self._frames_cache[key] = result
        if len(self._frames_cache) > 512:
            self._frames_cache.popitem(last=False)
        return result

    def configuration(self, kinematics, state, joints, obstacles, target_position=None):
        kinematics.validate(joints)
        raw_base = np.asarray(state.world_T_base, dtype=float)
        base_key = (raw_base.shape, raw_base.tobytes())
        base = self._base_cache.get(base_key)
        if base is None:
            base = rigid(raw_base)
            self._base_cache[base_key] = base
            if len(self._base_cache) > 32:
                self._base_cache.popitem(last=False)
        obstacles = tuple(obstacles)
        rotations, box_centers, half = self._prepared_boxes(obstacles)
        centers = np.empty((0,3))
        if target_position is not None:
            centers = np.asarray(target_position, dtype=float)
            if centers.shape == (3,):
                centers = centers[None, :]
            if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not np.isfinite(centers).all():
                raise ValueError("finite world target center(s)")
        cache_key = (
            self._kin_identity(kinematics), state.episode_id, state.source, state.timestamp,
            base_key, tuple(sorted((name, float(value)) for name, value in joints.items())),
            tuple((o.entity_id, o.episode_id, o.frame_id, o.source, o.timestamp,
                   o.assumed_static, o.container, rotations[i].tobytes(),
                   box_centers[i].tobytes(), half[i].tobytes())
                  for i,o in enumerate(obstacles)),
            centers.tobytes(), self.link_radius, self.base_radius,
        )
        cached = self._configuration_cache.get(cache_key)
        if cached is not None:
            self._configuration_cache.move_to_end(cache_key)
            self.profile_counts["configuration_cache_hits"] += 1
            return deepcopy(cached)
        self.profile_counts["configuration_computations"] += 1
        interior_mask = np.zeros(len(obstacles), dtype=bool)
        if len(centers) and len(obstacles):
            local = np.einsum("mkc,kcd->mkd", centers[:,None,:]-box_centers[None,:,:], rotations)
            inside = np.any(np.all(np.abs(local) <= half[None,:,:]+1e-9, axis=-1), axis=0)
            interior_mask = inside & np.array([o.container for o in obstacles])
        interior = [o.entity_id for i,o in enumerate(obstacles) if interior_mask[i]]
        frames = {arm:self._world_frames(kinematics,state,joints,arm,base) for arm in ("left","right")}
        arms, body = self._capsules(kinematics,state,joints,frames_by_arm=frames)
        flat = [(arm,name,a,b) for arm in ("left","right") for name,a,b in arms[arm]]
        nleft, nright = len(arms["left"]), len(arms["right"])
        arm_starts = np.array([row[2] for row in flat]).reshape(-1,3)
        arm_ends = np.array([row[3] for row in flat]).reshape(-1,3)
        all_caps = flat + [("body",name,a,b) for name,a,b in body]
        starts = np.array([row[2] for row in all_caps]).reshape(-1,3)
        ends = np.array([row[3] for row in all_caps]).reshape(-1,3)
        env_clear = _segments_boxes_distance(starts,ends,rotations,box_centers,half)-self.link_radius
        pair_clear = _segments_pair_distance(arm_starts,arm_ends)-2*self.link_radius
        minima, blocked = [], []
        def keep(values):
            if np.size(values):
                minima.append(float(np.min(values)))
        valid_arm_boxes = ~interior_mask
        keep(env_clear[:len(flat),valid_arm_boxes])
        keep(env_clear[len(flat):])
        offset = 0
        for arm in ("left","right"):
            caps = arms[arm]
            for i,(name,_,_) in enumerate(caps):
                for j in np.flatnonzero((env_clear[offset+i] < -1e-9) & valid_arm_boxes):
                    blocked.append({"kind":"arm_box","arm":arm,"link":name,
                                    "obstacle_id":obstacles[j].entity_id,"clearance_m":float(env_clear[offset+i,j])})
                for j in range(i+2,len(caps)):
                    first = caps[i][0].split("->",1)[0]; second = caps[j][0].split("->",1)[0]
                    if self._self_excluded(first,second):
                        continue
                    value = float(pair_clear[offset+i,offset+j]); minima.append(value)
                    if value < -1e-9:
                        blocked.append({"kind":"same_arm_nonadjacent","arm":arm,
                                        "segments":[i,j],"clearance_m":value})
            offset += len(caps)
        for i,(_,name,_,_) in enumerate(all_caps[len(flat):],len(flat)):
            for j in np.flatnonzero(env_clear[i] < -1e-9):
                blocked.append({"kind":"trunk_box","obstacle_id":obstacles[j].entity_id,
                                "clearance_m":float(env_clear[i,j])})
        cross = pair_clear[:nleft,nleft:nleft+nright]
        keep(cross)
        for i,j in np.argwhere(cross < -1e-9):
            blocked.append({"kind":"cross_arm","clearance_m":float(cross[i,j])})
        spheres, missing_gripper = self._gripper_spheres(kinematics,state,joints,frames_by_arm=frames)
        if spheres:
            points = np.array([row[2] for row in spheres])
            radii = np.array([row[3] for row in spheres])
            sphere_box = _points_boxes_distance(points,rotations,box_centers,half)-radii[:,None]
            keep(sphere_box[:,valid_arm_boxes])
            direction = arm_ends-arm_starts
            delta = points[:,None,:]-arm_starts[None,:,:]
            denom = np.maximum(np.sum(direction*direction,axis=1),1e-20)
            fraction = np.clip(np.sum(delta*direction[None,:,:],axis=-1)/denom[None,:],0,1)
            closest = arm_starts[None,:,:]+fraction[:,:,None]*direction[None,:,:]
            distances = np.sqrt(np.sum((points[:,None,:]-closest)**2,axis=-1))
            sphere_caps = distances-radii[:,None]-self.link_radius
            mask = np.array([
                [arm != other_arm or not self._self_excluded(link,name.split("->",1)[0])
                 for other_arm,name,_,_ in flat]
                for arm,link,_,_ in spheres
            ],dtype=bool)
            keep(sphere_caps[mask])
            local_indices = {}
            for i,(arm,link,_,_) in enumerate(spheres):
                local_index=local_indices.get(link,0); local_indices[link]=local_index+1
                for j in np.flatnonzero((sphere_box[i] < -1e-9) & valid_arm_boxes):
                    blocked.append({"kind":"gripper_sphere_box","arm":arm,"link":link,
                                    "obstacle_id":obstacles[j].entity_id,"sphere_index":local_index,
                                    "clearance_m":float(sphere_box[i,j])})
                for j in np.flatnonzero((sphere_caps[i] < -1e-9) & mask[i]):
                    other_arm,name,_,_=flat[j]
                    blocked.append({"kind":"gripper_sphere_arm_capsule","arm":arm,"link":link,
                                    "other_arm":other_arm,"segment":name,"sphere_index":local_index,
                                    "clearance_m":float(sphere_caps[i,j])})
            distances=np.sqrt(np.sum((points[:,None,:]-points[None,:,:])**2,axis=-1))
            sphere_pairs=distances-radii[:,None]-radii[None,:]
            mask=np.zeros(sphere_pairs.shape,dtype=bool)
            for i,(arm,link,_,_) in enumerate(spheres):
                for j,(other_arm,other_link,_,_) in enumerate(spheres[i+1:],i+1):
                    mask[i,j]=arm!=other_arm or not self._self_excluded(link,other_link)
            keep(sphere_pairs[mask])
            for i,j in np.argwhere((sphere_pairs < -1e-9) & mask):
                blocked.append({"kind":"gripper_sphere_pair","links":[spheres[i][1],spheres[j][1]],
                                "clearance_m":float(sphere_pairs[i,j])})
        base_center=(base@np.array([0.,0.,.2,1.]))[:3]
        base_clear=_points_boxes_distance(base_center[None,:],rotations,box_centers,half)[0]-self.base_radius
        keep(base_clear)
        for j in np.flatnonzero(base_clear < -1e-9):
            blocked.append({"kind":"base_box","obstacle_id":obstacles[j].entity_id,
                            "clearance_m":float(base_clear[j])})
        feasible=not blocked
        result={
            "estimated_collision_feasible":feasible,"status":"MODEL_CLEAR" if feasible else "MODEL_BLOCKED",
            "proxy_status":"PROXY_CLEAR" if feasible else "PROXY_BLOCKED",
            "minimum_clearance":min(minima) if minima else None,"scope":"capsule_box_partial_proxy",
            "physical_verified":False,"verified":False,"geometry_identity":IDENTITY,
            "blocked_pairs":blocked,"blocked_pair_count":len(blocked),"pair_summary_truncated":False,
            "obstacle_count":len(obstacles),"arm_capsule_counts":{a:len(v) for a,v in arms.items()},
            "container_arm_interior_access_assumption":interior,
            "base_container_envelopes_retained":[o.entity_id for o in obstacles if o.container],
            "adjacent_same_arm_capsules_excluded":True,"source_self_collision_ignore_pairs_used":True,
            "fixed_gripper_assembly_internal_pairs_excluded":True,
            "shared_trunk_excluded_from_arm_arm_tests":True,"tool_endpoint_included":False,
            "tool_endpoint_used_by_ik":True,"virtual_tool_frame_is_solid_volume":False,
            "gripper_sphere_count":len(spheres),"gripper_sphere_links":sorted({link for _,link,_,_ in spheres}),
            "missing_gripper_geometry":missing_gripper,
            "gripper_proxy_source":self._gripper_catalog["source"],
            "gripper_proxy_source_sha256":self._gripper_catalog["source_sha256"],
            "gripper_catalog_sha256":self._gripper_catalog_sha256,
            "radii_source":"arm_capsules_heuristic; palm_finger_spheres_original_CuRobo_source",
            "missing_geometry":["mesh_and_unobserved_environment","carried_object_mesh",
                                "arm_trunk_self_collision","current_articulated_doors_and_shelves"]+missing_gripper,
        }
        self._configuration_cache[cache_key]=result
        if len(self._configuration_cache)>256:
            self._configuration_cache.popitem(last=False)
        return deepcopy(result)

    @staticmethod
    def _pair_key(pair):
        identity = {k: v for k, v in pair.items() if k != "clearance_m"}
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def _initial_overlap_escape(self, samples, state):
        """First sampled quarter-path exit of existing external contact proxies.

        Clearance deficit is a primitive-model metric, not physical penetration.
        Missing between-sample collision evidence never becomes certification.
        Self/cross/base contacts are deliberately outside this arm policy.
        """
        initial = samples[0]["blocked_pairs"]
        out = {
            "policy": "model_internal_overlap_escape",
            "accepted": False, "escape_pairs": [],
            "max_clear_sample": 1, "max_clear_path_fraction": 0.25,
            "continuous_collision_certificate": False,
        }
        if not initial:
            out["reason"] = "NO_INITIAL_BLOCKERS"
            return out
        if state.source not in ("robot_feedback", "oracle", "task_info", "simulator_privileged"):
            out["reason"] = "INITIAL_POSTURE_NOT_DECLARED_MEASURED"
            return out
        if len(samples) < 5:
            out["reason"] = "INSUFFICIENT_PATH_SAMPLES_FOR_EARLY_ESCAPE"
            return out
        eligible = {"arm_box", "gripper_sphere_box"}
        if any(pair["kind"] not in eligible for pair in initial):
            out["reason"] = "INITIAL_BLOCKER_OUTSIDE_ARM_OBJECT_ESCAPE_SCOPE"
            return out
        starts = {self._pair_key(pair): pair for pair in initial}
        starts_depth = {
            key: max(0.0, -float(pair["clearance_m"])) for key, pair in starts.items()
        }
        max_depth = dict(starts_depth)
        first_clear = {key: None for key in starts}
        failure = None
        for index, sample in enumerate(samples[1:], 1):
            now = {self._pair_key(pair): pair for pair in sample["blocked_pairs"]}
            if set(now) - set(starts):
                failure = failure or "NEW_BLOCKER_APPEARED"
            for key in starts:
                if key not in now:
                    if first_clear[key] is None:
                        first_clear[key] = index
                    continue
                depth = max(0.0, -float(now[key]["clearance_m"]))
                max_depth[key] = max(max_depth[key], depth)
                if first_clear[key] is not None:
                    failure = failure or "ESCAPED_PAIR_RECONTACTED"
                if depth > starts_depth[key] + 1e-9:
                    # Prefer the specific depth failure over an early-clear
                    # failure when both describe this same starting pair.
                    failure = "INITIAL_OVERLAP_DEEPENED"
                if index >= 1:
                    failure = failure or "INITIAL_OVERLAP_NOT_CLEARED_EARLY"
        for key, pair in starts.items():
            out["escape_pairs"].append({
                "pair": pair,
                "start_clearance_m": float(pair["clearance_m"]),
                "max_penetration_proxy_m": max_depth[key],
                "penetration_kind": "negative_proxy_clearance_deficit_not_mesh_depth",
                "clear_sample": first_clear[key],
                "clear_path_fraction": (
                    first_clear[key] / (len(samples) - 1) if first_clear[key] is not None else None
                ),
            })
        out["max_penetration_proxy_m"] = max(max_depth.values(), default=0.0)
        out["accepted"] = failure is None and samples[-1]["estimated_collision_feasible"]
        out["reason"] = failure or "EXISTING_ARM_OBJECT_OVERLAP_CLEARED_FIRST_POST_INITIAL_SAMPLE"
        return out

    def arm_motion(self, kinematics, state, goal_joints, obstacles, target_position=None):
        samples = []
        for t in np.linspace(0, 1, self.path_samples):
            q = {n: float(state.joints[n] + t * (goal_joints[n] - state.joints[n]))
                 for n in state.joints}
            samples.append(self.configuration(
                kinematics, state, q, obstacles, target_position
            ))
        result = dict(samples[-1])
        result["scope"] = "sampled_arm_posture_capsule_box_proxy"
        strict = all(s["estimated_collision_feasible"] for s in samples)
        escape = self._initial_overlap_escape(samples, state)
        result["estimated_collision_feasible"] = strict or escape["accepted"]
        result["initial_overlap_escape"] = escape
        result["escape_pairs"] = escape["escape_pairs"]
        result["final_configuration_feasible"] = samples[-1]["estimated_collision_feasible"]
        result["status"] = (
            "MODEL_CLEAR_WITH_OVERLAP_ESCAPE" if escape["accepted"]
            else "MODEL_CLEAR" if strict else "MODEL_BLOCKED"
        )
        result["proxy_status"] = "PROXY_CLEAR" if result["estimated_collision_feasible"] else "PROXY_BLOCKED"
        values = [s["minimum_clearance"] for s in samples if s["minimum_clearance"] is not None]
        result["minimum_clearance"] = min(values) if values else None
        result["path_samples"] = self.path_samples
        result["blocked_sample_indices"] = [
            i for i, s in enumerate(samples) if not s["estimated_collision_feasible"]
        ]
        result["allowed_initial_blocked_sample_indices"] = [0] if escape["accepted"] else []
        result["path_is_continuous_collision_certificate"] = False
        return result

    def base_motion(self, kinematics, state, candidate_pose, obstacles, target_position=None):
        start, end = rigid(state.world_T_base), rigid(candidate_pose)
        relative = np.linalg.solve(start, end)
        yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
        samples = []
        for t in np.linspace(0, 1, self.path_samples):
            delta = np.eye(4)
            delta[:3, 3] = relative[:3, 3] * t
            delta[:3, :3] = rotation([0, 0, 1], yaw * t)
            moved = replace(state, world_T_base=start @ delta)
            samples.append(self.configuration(
                kinematics, moved, state.joints, obstacles, target_position
            ))
        result = dict(samples[-1])
        result["scope"] = "sampled_base_full_posture_capsule_box_proxy"
        result["estimated_collision_feasible"] = all(s["estimated_collision_feasible"] for s in samples)
        result["status"] = "MODEL_CLEAR" if result["estimated_collision_feasible"] else "MODEL_BLOCKED"
        result["proxy_status"] = "PROXY_CLEAR" if result["estimated_collision_feasible"] else "PROXY_BLOCKED"
        values = [s["minimum_clearance"] for s in samples if s["minimum_clearance"] is not None]
        result["minimum_clearance"] = min(values) if values else None
        result["path_samples"] = self.path_samples
        result["blocked_sample_indices"] = [i for i, s in enumerate(samples) if not s["estimated_collision_feasible"]]
        result["path_is_continuous_collision_certificate"] = False
        result["initial_overlap_escape_policy"] = "unsupported_for_base_motion_strict_all_samples"
        result["posture_scope"] = "measured_posture_during_base_motion_then_sampled_arm_motion_at_candidate"
        return result


class CenterApproxReachabilityEngine:
    def __init__(
        self, kinematics, collision=None, max_iterations=32, max_candidates=4,
        *, max_age=0.5, max_skew=0.1, position_tolerance=0.01,
    ):
        if (
            not 1 <= max_iterations <= 500 or not 1 <= max_candidates <= 4
            or not 0 < max_age <= 10 or not 0 <= max_skew <= max_age
            or not np.isfinite([max_age, max_skew, position_tolerance]).all()
            or position_tolerance <= 0
        ):
            raise ValueError("center estimate bounds")
        self.kinematics = kinematics
        self.collision = collision if collision is not None else CapsuleBoxProxy()
        self.max_iterations = int(max_iterations)
        self.max_candidates = int(max_candidates)
        self.max_age, self.max_skew = float(max_age), float(max_skew)
        self.position_tolerance = float(position_tolerance)
        self._outer_bounds = {}
        for arm in ("left", "right"):
            chain = kinematics.chains[arm]
            root = arm + "_arm_base_link"
            index = next((i for i, j in enumerate(chain) if j.child == root), None)
            if index is None:
                raise ValueError("explicit URDF arm root required")
            bound = sum(float(np.linalg.norm(j.origin[:3, 3])) for j in chain[index + 1:])
            for joint in chain[index + 1:]:
                if joint.kind == "prismatic":
                    if not np.isfinite([joint.lower, joint.upper]).all():
                        raise ValueError("bounded prismatic chain required")
                    bound += max(abs(joint.lower), abs(joint.upper))
            bound += float(np.linalg.norm(
                kinematics.tool_transforms.get(arm, np.eye(4))[:3, 3]
            ))
            self._outer_bounds[arm] = bound

    def _validate(self, state, target, now, obstacles):
        if not np.isfinite(now):
            raise ValueError("finite current time required")
        if state.frame_id != "world" or target.frame_id != "world":
            raise ValueError("world frame required")
        if not state.episode_id or target.episode_id != state.episode_id:
            raise ValueError("episode mismatch")
        if not target.entity_id:
            raise ValueError("canonical target required")
        rigid(state.world_T_base)
        rigid(target.world_T_target)
        self.kinematics.validate(state.joints)
        for timestamp in (state.timestamp, target.timestamp):
            if (
                not np.isfinite(timestamp) or timestamp > now + 1e-8
                or now - timestamp > self.max_age
            ):
                raise ValueError("stale/future current geometry")
        if abs(state.timestamp - target.timestamp) > self.max_skew:
            raise ValueError("unsynchronized current geometry")
        seen = set()
        for obstacle in obstacles:
            if not isinstance(obstacle, OrientedBoxObstacle):
                raise ValueError("typed oriented box required")
            if (
                not obstacle.entity_id or obstacle.entity_id in seen
                or obstacle.frame_id != "world" or obstacle.episode_id != state.episode_id
                or not obstacle.source
            ):
                raise ValueError("obstacle identity/frame/episode/source mismatch")
            seen.add(obstacle.entity_id)
            rigid(obstacle.world_T_box)
            extent = np.asarray(obstacle.extent_m, dtype=float)
            if extent.shape != (3,) or not np.isfinite(extent).all() or (extent <= 0).any():
                raise ValueError("positive finite metre box extent")
            if not np.isfinite(obstacle.timestamp) or obstacle.timestamp > now + 1e-8:
                raise ValueError("future/nonfinite static or observed obstacle")
            if not obstacle.assumed_static and (
                now - obstacle.timestamp > self.max_age
                or abs(obstacle.timestamp - state.timestamp) > self.max_skew
            ):
                raise ValueError("stale/unsynchronized dynamic obstacle")

    def _seeds(self, state, arm):
        names = self.kinematics.arm_joints[arm]
        current = dict(state.joints)
        middle = dict(current)
        elbow = dict(current)
        sign = 1.0 if arm == "left" else -1.0
        values = [-0.5, 0.7 * sign, 0.2, -0.9, 0.1, 0.2, -0.1]
        for name, value in zip(names, values):
            joint = self.kinematics.joints[name]
            middle[name] = float((joint.lower + joint.upper) / 2)
            elbow[name] = float(np.clip(value, joint.lower, joint.upper))
        return [("current", current), ("joint_midpoint", middle), ("bounded_elbow", elbow)]

    def _arm(self, state, target, obstacles, arm):
        center = rigid(target.world_T_target)[:3, 3]
        frames, _ = self.kinematics.frames(state.joints, arm, state.world_T_base)
        shoulder = frames[arm + "_arm_base_link"][:3, 3]
        distance = float(np.linalg.norm(center - shoulder))
        bound = self._outer_bounds[arm]
        empty = {
            "estimated_reachability": False, "estimated_collision_feasible": False,
            "ik_status": "OUTSIDE_CHAIN_BOUND", "status": "MODEL_REJECT",
            "joints": None, "position_error": max(0.0, distance - bound),
            "orientation_error": None, "orientation_constrained": False,
            "workspace_margin": None, "manipulability": None, "iterations": 0,
            "seed_attempts": 0, "selected_seed": None,
            "outer_bound_radius_m": bound, "shoulder_center_distance_m": distance,
            "physical_verified": False,
            "collision": {"estimated_collision_feasible": False, "scope": "no_solved_configuration",
                          "physical_verified": False, "minimum_clearance": None},
            "reposition_advice": "CONSIDER_REPOSITION",
            "reason": "center outside strict URDF chain outer bound; no distance-based positive reachability",
        }
        if distance > bound + self.position_tolerance:
            return empty
        tried = []
        for label, seed in self._seeds(state, arm):
            ik = self.kinematics.ik_position(
                seed, arm, center, state.world_T_base,
                max_iterations=self.max_iterations,
                position_tolerance=self.position_tolerance,
            )
            info = dict(empty)
            info.update(
                ik_status=ik.status, iterations=ik.iterations,
                position_error=ik.position_error, selected_seed=label,
                seed_attempts=len(tried) + 1,
                reason="bounded position IK nonconvergence rejects this algorithm candidate only",
            )
            if ik.status == "SOLVED":
                collision = self.collision.arm_motion(
                    self.kinematics, state, ik.joints, obstacles, center
                )
                clear = bool(collision["estimated_collision_feasible"])
                quality = (
                    ik.joint_limit_margin >= 0.03 and ik.manipulability >= 1e-5
                )
                info.update(
                    estimated_reachability=clear, estimated_collision_feasible=clear,
                    status="MODEL_PASS" if clear else "MODEL_REJECT",
                    joints=ik.joints, workspace_margin=ik.joint_limit_margin,
                    manipulability=ik.manipulability, collision=collision,
                    quality_insufficient=not quality,
                    reposition_advice="NO_MODEL_REPOSITION_NEEDED" if clear and quality else "CONSIDER_REPOSITION",
                    reason="position IK plus sampled partial collision proxy; orientation unconstrained",
                )
            tried.append(info)
            # The deterministic alternatives are attempted when the first solve
            # does not produce a clear, sufficient-quality candidate.
            if info["estimated_reachability"] and not info.get("quality_insufficient", True):
                break
        def rank(info):
            return (
                info["estimated_reachability"], info["ik_status"] == "SOLVED",
                info["workspace_margin"] or 0.0, info["manipulability"] or 0.0,
                -info["position_error"],
            )
        best = dict(max(tried, key=rank))
        best["seed_attempts"] = len(tried)
        best["total_iterations"] = sum(v["iterations"] for v in tried)
        return best

    def evaluate(self, state, target, now, obstacles=(), required_arms=("left", "right")):
        if isinstance(target, dict):
            return self.evaluate_bimanual(state, target, now, obstacles, required_arms)
        if (
            not required_arms or len(set(required_arms)) != len(required_arms)
            or any(a not in ("left", "right") for a in required_arms)
        ):
            raise ValueError("explicit valid arm identities")
        started = time.perf_counter()
        obstacles = tuple(obstacles)
        result = {
            "schema": "tgr.reachability.v1", "geometry_identity": IDENTITY,
            "approximate": True, "estimate_trusted": False, "physical_verified": False,
            "target_id": target.entity_id, "timestamp": state.timestamp,
            "episode_id": state.episode_id, "frame_id": state.frame_id,
            "source": "URDF_position_IK_partial_capsule_box_estimate",
            "oracle": state.source in ("oracle", "task_info", "simulator_privileged")
                      or target.source in ("oracle", "task_info", "simulator_privileged")
                      or any(o.source in ("oracle", "task_info", "simulator_privileged",
                                          "initial_static_scene_proxy") for o in obstacles),
            "valid": False, "known_meaning": "model_internal_estimate_not_physical_certification",
            "T_base_target": None, "left_reachability": "UNKNOWN", "right_reachability": "UNKNOWN",
            "workspace_margin": {"left": None, "right": None},
            "workspace_margin_kind": "normalized_arm_joint_limit_margin",
            "manipulability": {"left": None, "right": None},
            "manipulability_kind": "product_singular_values_position_jacobian_3x7",
            "collision_feasibility": "UNKNOWN", "base_reposition_feasibility": "UNKNOWN",
            "arms": {}, "required_arms": list(required_arms),
            "simultaneous_bimanual_assessment": False,
            "center_source": "supplied_nominal_bbox_center_world_pose",
        }
        try:
            self._validate(state, target, now, obstacles)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            result["reason"] = str(error)
            result["status"] = "INVALID_INPUT"
            return result
        used = tuple(o for o in obstacles if o.entity_id != target.entity_id)
        result["target_contact_exclusion"] = [o.entity_id for o in obstacles if o.entity_id == target.entity_id]
        result["obstacle_provenance"] = [
            {"entity_id": o.entity_id, "source": o.source,
             "source_timestamp": o.timestamp, "assumed_static": o.assumed_static}
            for o in used
        ]
        result.update(valid=True, estimate_trusted=True, base_reposition_feasibility="NOT_EVALUATED")
        result["urdf_sha256"] = self.kinematics.sha256
        result["T_base_target"] = np.linalg.solve(
            rigid(state.world_T_base), rigid(target.world_T_target)
        ).tolist()
        for arm in required_arms:
            info = self._arm(state, target, used, arm)
            result["arms"][arm] = info
            result[arm + "_reachability"] = info["status"]
            result["workspace_margin"][arm] = info["workspace_margin"]
            result["manipulability"][arm] = info["manipulability"]
        passing = [a for a, info in result["arms"].items() if info["estimated_reachability"]]
        result["status"] = "MODEL_PASS" if passing else "MODEL_REJECT"
        result["collision_feasibility"] = "MODEL_PASS" if any(
            info["estimated_collision_feasible"] for info in result["arms"].values()
        ) else "MODEL_REJECT"
        result["available_arms"] = passing
        result["reason"] = "independent arm center estimates; no simultaneous dual-arm certification"
        result["cpu_wall_ms"] = (time.perf_counter() - started) * 1000
        result["ik_seed_solves"] = sum(v["seed_attempts"] for v in result["arms"].values())
        return result

    def evaluate_bimanual(self, state, targets, now, obstacles=(), required_arms=("left", "right")):
        if (
            set(targets) != set(required_arms)
            or not required_arms
            or any(a not in ("left", "right") for a in required_arms)
            or any(t.arm != a for a, t in targets.items())
        ):
            raise ValueError("distinct explicit target identity for each arm")
        obstacles = tuple(obstacles)
        individual = {a: self.evaluate(state, t, now, obstacles, (a,)) for a, t in targets.items()}
        result = dict(next(iter(individual.values())))
        result["target_ids"] = {a: t.entity_id for a, t in targets.items()}
        result["target_id"] = "|".join(a + ":" + targets[a].entity_id for a in sorted(targets))
        result["T_base_target"] = {a: r["T_base_target"] for a, r in individual.items()}
        result["required_arms"] = list(required_arms)
        result["arms"] = {a: r["arms"].get(a, {}) for a, r in individual.items()}
        result["workspace_margin"] = {a: r["workspace_margin"][a] for a, r in individual.items()}
        result["manipulability"] = {a: r["manipulability"][a] for a, r in individual.items()}
        result["valid"] = all(r["valid"] for r in individual.values())
        result["estimate_trusted"] = result["valid"]
        result["simultaneous_bimanual_assessment"] = True
        result["physical_verified"] = False
        result["oracle"] = any(r["oracle"] for r in individual.values())
        if not result["valid"]:
            result["reason"] = "invalid independent target geometry"
            return result
        all_solved = all(v.get("ik_status") == "SOLVED" for v in result["arms"].values())
        combined_clear = False
        if all_solved:
            merged = dict(state.joints)
            for arm, info in result["arms"].items():
                for name in self.kinematics.arm_joints[arm]:
                    merged[name] = info["joints"][name]
            ids = {t.entity_id for t in targets.values()}
            used = tuple(o for o in obstacles if o.entity_id not in ids)
            centers = np.stack([
                rigid(target.world_T_target)[:3, 3] for target in targets.values()
            ])
            collision = self.collision.arm_motion(
                self.kinematics, state, merged, used, centers
            )
            combined_clear = collision["estimated_collision_feasible"]
            for arm, info in result["arms"].items():
                info.update(
                    estimated_reachability=combined_clear,
                    estimated_collision_feasible=combined_clear,
                    status="MODEL_PASS" if combined_clear else "MODEL_REJECT",
                    joints=merged.copy(), collision=collision,
                )
        for arm in required_arms:
            result[arm + "_reachability"] = result["arms"][arm].get("status", "MODEL_REJECT")
        result["status"] = "MODEL_PASS" if all_solved and combined_clear else "MODEL_REJECT"
        result["collision_feasibility"] = result["status"]
        result["reason"] = "merged dual-arm posture and finite interpolation checked in partial proxy model"
        result["ik_seed_solves"] = sum(r.get("ik_seed_solves", 0) for r in individual.values())
        return result

    def propose(self, state, target, now, obstacles=(), required_arms=("left", "right")):
        targets = list(target.values()) if isinstance(target, dict) else [target]
        obstacles = tuple(obstacles)
        for t in targets:
            self._validate(state, t, now, obstacles)
        ids = {t.entity_id for t in targets}
        used = tuple(o for o in obstacles if o.entity_id not in ids)
        center = (
            np.stack([rigid(t.world_T_target)[:3, 3] for t in targets])
            if isinstance(target, dict) else rigid(targets[0].world_T_target)[:3, 3]
        )
        start = rigid(state.world_T_base)
        offsets = [(0.15, 0.0, 0.0), (-0.15, 0.0, 0.0),
                   (0.0, 0.15, 0.15), (0.0, -0.15, -0.15)]
        candidates = []
        for x, y, yaw in offsets[:self.max_candidates]:
            delta = np.eye(4)
            delta[:3, 3] = [x, y, 0.0]
            delta[:3, :3] = rotation([0, 0, 1], yaw)
            pose = start @ delta
            moved = replace(state, world_T_base=pose)
            assessment = self.evaluate(moved, target, now, obstacles, required_arms)
            passes = sum(v.get("estimated_reachability", False) for v in assessment["arms"].values())
            solved = sum(v.get("ik_status") == "SOLVED" for v in assessment["arms"].values())
            enough = passes == len(required_arms) if isinstance(target, dict) else passes > 0
            if not enough:
                # Endpoint rejection already makes this candidate infeasible.
                # An omitted path test is not a claimed collision obstruction.
                path = {
                    "evaluated": False,
                    "reason": "no_reachable_endpoint",
                    "estimated_collision_feasible": None,
                    "status": "NOT_EVALUATED",
                    "proxy_status": "NOT_EVALUATED",
                    "minimum_clearance": None,
                    "scope": "skipped_no_reachable_endpoint",
                    "physical_verified": False,
                }
            else:
                path = self.collision.base_motion(self.kinematics, state, pose, used, center)
                path["evaluated"] = True
            feasible = bool(enough and path["estimated_collision_feasible"])
            margins = [v for v in assessment["workspace_margin"].values() if v is not None]
            margin = min(margins) if margins else 0.0
            motion_cost = float(np.hypot(x, y) + 0.2 * abs(yaw))
            path_penalty = 20 * (path["evaluated"] and not path["estimated_collision_feasible"])
            score = 20 * passes + 5 * solved + margin - motion_cost - path_penalty
            candidate_id = hashlib.sha256(json.dumps(
                [state.episode_id, state.timestamp, assessment["target_id"], pose.tolist()],
                sort_keys=True,
            ).encode()).hexdigest()[:24]
            assessment["base_reposition_feasibility"] = "MODEL_PASS" if feasible else "MODEL_REJECT"
            candidates.append({
                "candidate_id": candidate_id, "timestamp": state.timestamp,
                "episode_id": state.episode_id, "target_id": assessment["target_id"],
                "frame_id": "world", "world_T_base": pose.tolist(), "score": float(score),
                "motion_cost": motion_cost,
                "score_semantics": "omitted_path_has_no_collision_penalty; ranking_change_only_nonfeasible_candidates",
                "base_reposition_feasibility": assessment["base_reposition_feasibility"],
                "estimated_reposition_feasible": feasible, "assessment": assessment, "path_evidence": path,
                "geometry_identity": IDENTITY, "approximate": True, "physical_verified": False,
                "execution_authorized": False, "evidence_only": True, "hypothetical_base_pose": True,
                "arm_selection_scope": "combined_targets" if isinstance(target, dict) else "either_arm_independent",
            })
        return sorted(candidates, key=lambda c: (-c["score"], c["candidate_id"]))

    @staticmethod
    def vlm_summary(assessment, candidates=()):
        keep = ("schema", "geometry_identity", "approximate", "estimate_trusted",
                "physical_verified", "target_id", "timestamp", "episode_id", "frame_id",
                "oracle", "valid", "reason", "T_base_target", "left_reachability",
                "right_reachability", "workspace_margin", "workspace_margin_kind",
                "manipulability", "manipulability_kind", "collision_feasibility",
                "base_reposition_feasibility", "known_meaning")
        return {
            "feasibility": {k: assessment[k] for k in keep if k in assessment},
            "base_candidates": [{k: c[k] for k in (
                "candidate_id", "world_T_base", "score", "base_reposition_feasibility",
                "path_evidence", "timestamp", "physical_verified"
            )} for c in candidates],
            "task_completion_evidence": False,
        }
