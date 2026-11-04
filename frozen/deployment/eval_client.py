#!/usr/bin/env python3
"""TGR-owned resident paired evaluator; staged ONLY in a unique /tmp directory.
No video, no extra render, official task step/metric semantics unchanged.
Oracle snapshot extractor is an owned deployment copy of runtime oracle adapter.
"""
import argparse, json, os, sys, time, traceback, hashlib, importlib
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

ROOT = Path(os.environ.get("TGR_LOCAL_PROJECT", "/home/user/Robotic/project2"))
os.environ.setdefault("OMNIGIBSON_DATA_PATH", str(ROOT / "shared/omnigibson"))
os.environ.setdefault("OMNIGIBSON_APPDATA_PATH", str(Path(__file__).parent / "appdata"))


def load_provider(spec, env):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)(env)


def install_wrapper():
    from omnigibson.envs import EnvironmentWrapper
    from omnigibson.eval.utils.eval_utils import set_sensor_modalities

    class RGBResolutionWrapper(EnvironmentWrapper):
        def __init__(
            self, env, image_size=224, head_image_size=720, wrist_image_size=480
        ):
            super().__init__(env=env)
            for name, sensor in env.robots[0].sensors.items():
                if not hasattr(sensor, "image_height") or not hasattr(
                    sensor, "image_width"
                ):
                    continue
                set_sensor_modalities(sensor, {"rgb"})
                size = head_image_size if "zed_link" in name else wrist_image_size
                sensor.image_height = int(size)
                sensor.image_width = int(size)
                space = sensor.load_observation_space()
                if env.observation_space is not None:
                    env.observation_space.spaces[env.robots[0].name].spaces[
                        name
                    ] = space

    globals()["RGBResolutionWrapper"] = RGBResolutionWrapper
    sys.modules["eval_client"] = sys.modules[__name__]


import hashlib


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


import numpy as np


def transform(position, quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-10:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    t = np.eye(4)
    t[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    t[:3, 3] = position
    return t


"""Wire adapter for the official BEHAVIOR R1Pro joint-controller protocol.

The policy receives only onboard RGB and joint/velocity feedback. Object state,
task predicates and privileged evaluator observations never enter its input.
"""
import numpy as np

EMBODIMENT = "behavior_r1pro"
STATE_KEYS = {
    "left_arm": ("arm_left_qpos", 7),
    "right_arm": ("arm_right_qpos", 7),
    "left_gripper": ("gripper_left_qpos", 2),
    "right_gripper": ("gripper_right_qpos", 2),
    "trunk_qpos": ("trunk_qpos", 4),
    "base_qvel": ("base_qvel", 3),
}
TASKS = {
    "moving_boxes_to_storage": "Move the two storage containers from the living room to the garage. In the garage, place one container on the floor and stack the other container on top of it (either order is fine).",
    "putting_shoes_on_rack": "Put the shoes on the rack.",
    "putting_away_Halloween_decorations": "Put away the Halloween decorations.",
    "turning_on_radio": "Turn on the radio.",
    "make_microwave_popcorn": "In the kitchen, take the popcorn bag from the countertop, put it into the microwave, and heat it until the popcorn is cooked so the cooked popcorn ends up inside the bag.",
    "cook_hot_dogs": "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked.",
    "installing_a_modem": "Place the modem on the cabinet next to the television and turn on the modem.",
    "composting_waste": "Put the banana half and the pomegranate half from the kitchen countertop into the compost bin.",
    "picking_up_trash": "Pick up the trash and put it in the trash can.",
    "bringing_water": "Bring the bottles of water from the refrigerator to the coffee table and close the refrigerator.",
    "wash_a_baseball_cap": "Wash the dirty baseball caps in the washing machine.",
}


def array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def vector(x, size, name):
    x = array(x).astype(np.float32, copy=True).reshape(-1)
    if x.shape != (size,) or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected {size} finite values, got {x}")
    return x


def observation(robot, obs, camera_names, task):
    proprio = robot._get_proprioception_dict()
    state = {k: vector(proprio[src], size, k) for k, (src, size) in STATE_KEYS.items()}
    images = {}
    for role, key in [
        ("head", "head_rgb"),
        ("left_wrist", "left_wrist_rgb"),
        ("right_wrist", "right_wrist_rgb"),
    ]:
        im = array(obs[camera_names[role] + "::rgb"])
        if im.ndim != 3 or im.shape[-1] not in (3, 4) or im.dtype != np.uint8:
            raise ValueError(f"Invalid RGB {role}: {im.shape}, {im.dtype}")
        images[key] = np.ascontiguousarray(im[..., :3].transpose(2, 0, 1))
    return {
        "images": images,
        "state": state,
        "task": task,
        "embodiment_type": EMBODIMENT,
        "frequency": 30.0,
    }


def action_vector(parts, controller_indices, action_dim):
    """Postprocessor output is already absolute joints and native [-1,1] grips.

    BehaviorPerKeyTransform.backward restores lower_body to [base(3),trunk(4)].
    Controller indices come from the live robot, never from an assumed order.
    """
    lb = vector(parts["lower_body"], 7, "lower_body")
    values = {
        "base": lb[:3],
        "trunk": lb[3:],
        "arm_left": vector(parts["left_arm"], 7, "left_arm"),
        "arm_right": vector(parts["right_arm"], 7, "right_arm"),
        "gripper_left": vector(parts["left_gripper"], 1, "left_gripper"),
        "gripper_right": vector(parts["right_gripper"], 1, "right_gripper"),
    }
    result = np.zeros(action_dim, dtype=np.float32)
    used = set()
    for name, indices in controller_indices.items():
        indices = array(indices).astype(int).reshape(-1)
        if len(indices) == 0:
            continue
        if name not in values or len(indices) != len(values[name]):
            raise ValueError(f"Unsupported controller {name}: {indices}")
        if any(i in used or i < 0 or i >= action_dim for i in indices):
            raise ValueError(f"Invalid controller indices: {name}")
        result[indices] = values[name]
        used.update(indices.tolist())
    if len(used) != action_dim:
        raise ValueError(f"Only mapped {len(used)}/{action_dim} action dimensions")
    # Native base and gripper controllers clip these commands to their limits.
    # Do not silently clamp or alter model output in the adapter.
    return result


def hold_missing(parts, state, previous=None):
    """Official server omits action groups the AR model did not emit.

    Hold measured arm/trunk positions, stop the base, and preserve the last
    gripper command (initially measured width). Never synthesize task motions.
    """
    previous = previous or {}
    defaults = {
        "left_arm": state["left_arm"],
        "right_arm": state["right_arm"],
        "lower_body": np.concatenate([np.zeros(3), state["trunk_qpos"]]),
    }
    for k in ("left_gripper", "right_gripper"):
        defaults[k] = previous.get(
            k, np.array([2.0 * np.sum(state[k]) / 0.1 - 1.0], np.float32)
        )
    missing = sorted(set(defaults) - set(parts))
    return {**defaults, **parts}, missing


def audit_controller_contract(robot, controller_view=None):
    """Fail closed on an incompatible live controller schema; not dynamics proof."""
    indices = {
        k: array(v).astype(int).reshape(-1)
        for k, v in robot.controller_action_idx.items()
    }
    expected = {
        "base": ("HolonomicBaseJointController", "velocity"),
        "trunk": ("JointController", "position"),
        "arm_left": ("JointController", "position"),
        "arm_right": ("JointController", "position"),
        "gripper_left": ("MultiFingerGripperController", None),
        "gripper_right": ("MultiFingerGripperController", None),
    }
    assert robot.action_dim == 23, robot.action_dim
    assert (
        getattr(robot, "_action_normalize", None) is False
    ), "Unexpected whole-action normalization"
    receipt = {"action_normalize": False, "controllers": {}}
    for name, (kind, motor) in expected.items():
        c = robot.controllers[name]
        convert = array
        if isinstance(c, tuple):
            if controller_view is None:
                from omnigibson.controllers.controller_view import ControllerView

                controller_view = ControllerView
            from omnigibson.utils.backend_utils import _compute_backend as cb

            c = controller_view._controller_groups[c[0]]
            convert = lambda value: array(cb.to_torch(value))
        assert type(c).__name__ == kind, (name, type(c).__name__)
        if motor is not None:
            assert c._motor_type == motor, (name, c._motor_type)
            assert not c.use_delta_commands, (name, "delta commands not supported")
        receipt["controllers"][name] = {
            "type": kind,
            "motor_type": motor,
            "indices": indices[name].tolist(),
            "command_input_limits": (
                None
                if c._command_input_limits is None
                else [convert(x).tolist() for x in c._command_input_limits]
            ),
            "command_output_limits": (
                None
                if c._command_output_limits is None
                else [convert(x).tolist() for x in c._command_output_limits]
            ),
        }
    # Check every dimension is assigned once using the actual live permutation.
    action_vector(
        {
            "lower_body": np.zeros(7),
            "left_arm": np.zeros(7),
            "right_arm": np.zeros(7),
            "left_gripper": [0.0],
            "right_gripper": [0.0],
        },
        indices,
        23,
    )
    receipt["status"] = "passed"
    receipt["scope"] = (
        "live controller contract and index coverage; not physical trajectory validation"
    )
    return receipt


"""Explicit simulator truth adapter. Does not import or start Isaac at module import."""
import hashlib
import numpy as np


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class NominalGeometryAssets:
    """Tiny owned deployment formula; no import of the server TGR framework."""

    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "tgr.deployment_nominal_geometry_assets.v1" or data.get("units") != "metres":
            raise ValueError("audited metric geometry assets required")
        self.records = {(row["category"], row["model"]): row for row in data["records"]}
        self.static_names = tuple(data["static_object_names"])
        if self.static_names != ("bottom_cabinet_no_top_gjeoer_0", "bottom_cabinet_no_top_rkgjer_0"):
            raise ValueError("exact two fixed cabinet IDs required")

    def spec(self, category, model, scale):
        row = self.records.get((category, model))
        if row is None:
            return None
        scale = np.asarray(scale, dtype=float)
        if scale.shape == ():
            scale = np.repeat(scale, 3)
        if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("positive dimensionless asset scale required")
        return dict(category=category, model=model,
                    extent_m=(scale * np.asarray(row["native_bbox_extent_m"])).tolist(),
                    center_offset_m=(scale * np.asarray(row["native_bbox_center_offset_from_root_m"])).tolist(),
                    source_sha256=row["metadata_source_sha256"],
                    container=category in ("fridge", "microwave"),
                    geometry_source="nominal_original_asset_metadata", physical_verified=False)

    def object_spec(self, obj):
        category, model = getattr(obj, "category", None), getattr(obj, "model", None)
        if (category, model) not in self.records:
            return None
        if not hasattr(obj, "scale"):
            raise ValueError("loaded nominal model has no measured object scale")
        return self.spec(category, model, array(obj.scale))

    @staticmethod
    def centered(root, spec):
        root = np.asarray(root, dtype=float)
        if root.shape != (4, 4) or not np.isfinite(root).all():
            raise ValueError("finite measured root world transform required")
        center = root.copy()
        center[:3, 3] += root[:3, :3] @ np.asarray(spec["center_offset_m"])
        attrs = dict(spec, original_root_world_pose=root.tolist(),
                     canonical_pose_kind="nominal_bbox_center", nominal_extent_m=spec["extent_m"],
                     extent_known=True, world_T_box=center.tolist())
        return center, attrs


def live_robot_fields(robot, feedback, timestamp, geometry_spec=None):
    """Four actual finger coordinates share the dataset's two-value side schema."""
    position, quaternion = robot.get_position_orientation()
    joints = {}
    for target, source, size in (
        ("trunk_qpos", "trunk_qpos", 4),
        ("left_arm_qpos", "arm_left_qpos", 7),
        ("right_arm_qpos", "arm_right_qpos", 7),
        ("left_gripper_qpos", "gripper_left_qpos", 2),
        ("right_gripper_qpos", "gripper_right_qpos", 2),
    ):
        joints[target] = vector(feedback[source], size, target).tolist()
    result = dict(reference_frame="world", world_poses_verified=True, timestamp=timestamp,
                  base_world=transform(array(position), array(quaternion)).tolist(),
                  source="oracle", joints=joints)
    if geometry_spec is not None:
        result["geometry_spec"] = geometry_spec
    return result


class OmniGibsonExtractor:
    """Queries existing environment at chunk boundaries; no stepping/rendering.
    arm_mapping is mandatory if the live arm names are not literally left/right.
    Missing predicates remain unknown; proximity never establishes contact/grasp.
    """

    def __init__(self, env, arm_mapping=None, state_types=None, grasp_states=None, geometry_assets=None):
        self.env = env
        self.arm_mapping = arm_mapping
        self.state_types = state_types
        self.grasp_states = grasp_states
        self.geometry_assets = geometry_assets
        self.reset()

    def reset(self):
        self.previous = {}
        self.episode = None
        self.registry = None
        self.last_time = -float("inf")
        self.static_proxies = {}
        self.geometry_spec = {"task_objects": {}, "static_objects": []}

    def __call__(self, observation, registry, episode_id, timestamp):
        if self.state_types is None:
            from omnigibson.object_states import Inside, OnTop, Open, Cooked, ToggledOn

            states = {
                "inside": Inside,
                "on_top": OnTop,
                "open": Open,
                "cooked": Cooked,
                "toggled_on": ToggledOn,
            }
        else:
            states = self.state_types
        if self.grasp_states is None:
            from omnigibson.controllers.controller_base import IsGraspingState

            true_state = IsGraspingState.TRUE
            false_state = IsGraspingState.FALSE
        else:
            true_state, false_state = self.grasp_states
        if self.episode != episode_id or self.registry is not registry:
            self.reset()
            self.episode = episode_id
            self.registry = registry
        if timestamp <= self.last_time:
            raise ValueError("non-monotonic simulator sample")
        robots = self.env.robots
        if len(robots) != 1:
            raise ValueError(
                "explicit robot selection required for multi-robot environment"
            )
        robot = robots[0]
        arms = list(robot.arm_names)
        mapping = self.arm_mapping or {"left": "left", "right": "right"}
        if (
            set(mapping) != {"left", "right"}
            or len(set(mapping.values())) != 2
            or any(a not in arms for a in mapping.values())
        ):
            raise ValueError(
                "explicit left/right mapping to live R1Pro arm_names required"
            )
        entities = []
        edges = []
        events = []
        objects = {}
        current = {}

        def event(eid, predicate, value, source="oracle"):
            key = (eid, predicate)
            old = self.previous.get(key)
            current[key] = value
            if old != value:
                evidence = hashlib.sha256(
                    f"{episode_id}:{timestamp}:{eid}:{predicate}".encode()
                ).hexdigest()[:24]
                events.append(
                    Event(
                        timestamp,
                        episode_id,
                        eid,
                        predicate,
                        old,
                        value,
                        1.0,
                        evidence,
                        source,
                    )
                )

        def add_pose(instance, semantic, obj, attrs=None):
            pos, quat = obj.get_position_orientation()
            root_pose = transform(array(pos), array(quat))
            pose = root_pose.tolist()
            eid = registry.resolve(instance)
            info = dict(attrs or {})
            nominal = self.geometry_assets.object_spec(obj) if self.geometry_assets else None
            if nominal is not None:
                centered, nominal_attrs = self.geometry_assets.centered(root_pose, nominal)
                pose = centered.tolist()
                info.update(nominal_attrs)
                self.geometry_spec["task_objects"][instance] = nominal
            info.update(last_observed_time=timestamp, valid_from=timestamp)
            try:
                lo, hi = obj.aabb
                info["aabb"] = [array(lo).tolist(), array(hi).tolist()]
                info["aabb_source"] = "live_actual_world_aabb_diagnostic"
            except AttributeError:
                pass
            entities.append(
                Entity(
                    eid,
                    semantic,
                    instance,
                    pose,
                    "world",
                    timestamp,
                    timestamp,
                    1.0,
                    "oracle",
                    True,
                    info,
                )
            )
            event(eid, "pose", pose)
            return eid

        scope = self.env.task.object_scope
        for name, obj in scope.items():
            if (
                obj is None
                or obj is robot
                or not hasattr(obj, "get_position_orientation")
            ):
                continue
            attrs = {}
            errors = []
            for label in ("open", "cooked", "toggled_on"):
                cls = states.get(label)
                if cls is not None and cls in obj.states:
                    try:
                        value = obj.states[cls].get_value()
                        if isinstance(value, (bool, np.bool_)):
                            attrs[label] = bool(value)
                        else:
                            errors.append(label + ":non_boolean")
                    except (ValueError, RuntimeError, KeyError) as exc:
                        errors.append(label + ":" + type(exc).__name__)
            if errors:
                attrs["unknown_state_queries"] = errors
            attrs["sim_category"] = getattr(obj, "category", None)
            eid = add_pose(name, name.rsplit("_", 1)[0], obj, attrs)
            objects[eid] = obj
            for label, value in attrs.items():
                if type(value) is bool:
                    event(eid, label, value)
        if self.geometry_assets is not None:
            scene = getattr(self.env, "scene", None)
            if scene is None:
                scenes = getattr(self.env, "scenes", [])
                scene = scenes[0] if len(scenes) == 1 else None
            if scene is None or not callable(getattr(scene, "object_registry", None)):
                raise ValueError("actual loaded scene object registry required for static cabinets")
            for name in self.geometry_assets.static_names:
                if name not in self.static_proxies:
                    obj = scene.object_registry("name", name)
                    if obj is None or getattr(obj, "fixed_base", None) is not True or bool(getattr(obj, "visual_only", False)):
                        raise ValueError("required fixed cabinet absent or not a physical loaded static object: " + name)
                    nominal = self.geometry_assets.object_spec(obj)
                    if nominal is None:
                        raise ValueError("fixed cabinet model lacks audited nominal bounds: " + name)
                    pos, quat = obj.get_position_orientation()
                    root_pose = transform(array(pos), array(quat))
                    center, attrs = self.geometry_assets.centered(root_pose, nominal)
                    instance = "scene_static:" + name
                    attrs.update(instance_id=instance, root_world=root_pose.tolist(),
                                 source_timestamp=timestamp, assumed_static=True,
                                 provenance="initial_static_scene_proxy_not_fresh_observation",
                                 initial_pose_source="live_loaded_scene_root_feedback")
                    try:
                        lo, hi = obj.aabb
                        attrs["initial_actual_aabb_diagnostic"] = [array(lo).tolist(), array(hi).tolist()]
                    except AttributeError:
                        pass
                    self.static_proxies[name] = Entity(registry.resolve(instance), "local_obstacle", instance,
                        center.tolist(), "world", timestamp, timestamp, 1.0, "oracle", True, attrs)
                entities.append(self.static_proxies[name])
            self.geometry_spec["static_objects"] = [dict(node.attributes) for node in self.static_proxies.values()]
        robot_alias = next(
            (name for name, obj in scope.items() if obj is robot), "robot:base"
        )
        base = add_pose(robot_alias, "robot_base", robot)
        # The server attach_geometry adds the same collision_context edges as training.
        # Match the training torso identity with actual live link feedback.
        # Missing links stay absent/unknown; no pose is guessed from timing.
        trunk_link = getattr(robot, "links", {}).get("torso_link4")
        if trunk_link is not None:
            trunk = add_pose(
                "robot:trunk",
                "robot_trunk",
                trunk_link,
                {"pose_source": "live_link_feedback"},
            )
            edges.append(
                Edge(trunk, base, "connected_to", {}, timestamp, 1.0, True, "oracle")
            )
        ee = {}
        for side, arm in mapping.items():
            # Actual live link query; no inferred arm pose or IK truth.
            arm_link = getattr(robot, "links", {}).get(side + "_arm_base_link")
            if arm_link is not None:
                arm_id = add_pose(
                    "robot:arm:" + side,
                    side + "_arm",
                    arm_link,
                    {"arm_name": arm, "pose_source": "live_link_feedback"},
                )
                edges.append(
                    Edge(
                        arm_id, base, "connected_to", {}, timestamp, 1.0, True, "oracle"
                    )
                )
            eid = add_pose(
                "robot:ee:" + side,
                side + "_ee",
                robot.eef_links[arm],
                {"arm_name": arm},
            )
            ee[arm] = eid
            edges.append(
                Edge(eid, base, "connected_to", {}, timestamp, 1.0, True, "oracle")
            )
        for eid, obj in objects.items():
            for target_id, target in objects.items():
                if eid == target_id:
                    continue
                for label in ("inside", "on_top"):
                    cls = states.get(label)
                    if cls is None or cls not in obj.states:
                        continue
                    try:
                        value = obj.states[cls].get_value(target)
                    except (ValueError, RuntimeError, KeyError):
                        continue
                    if not isinstance(value, (bool, np.bool_)):
                        continue
                    value = bool(value)
                    event(eid, label + ":" + target_id, value)
                    # Explicit negative facts are evidence, but validity False keeps them out of adjacency.
                    edges.append(
                        Edge(
                            eid,
                            target_id,
                            label,
                            {"value": value},
                            timestamp,
                            1.0,
                            value,
                            "oracle",
                        )
                    )
            for arm, hand in ee.items():
                result = robot.is_grasping(arm=arm, candidate_obj=obj)
                # FALSE=-1 is truthy! Never bool(result).
                grasp = (
                    True
                    if result == true_state
                    else False if result == false_state else None
                )
                event(eid, "held_by:" + hand, grasp)
                edges.append(
                    Edge(
                        eid,
                        hand,
                        "held_by",
                        {"value": grasp},
                        timestamp,
                        1.0 if grasp is not None else 0.0,
                        grasp is True,
                        "oracle",
                    )
                )
        self.previous = current
        self.last_time = timestamp
        return ObservationSnapshot(
            episode_id,
            timestamp,
            tuple(entities),
            tuple(edges),
            tuple(events),
            oracle=True,
        )


def write(p, x):
    t = p.with_suffix(".tmp")
    t.write_text(
        json.dumps(
            x, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)
        )
    )
    t.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--oracle-upper-bound", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=0)
    parser.add_argument("--snapshot-provider")
    parser.add_argument("--geometry-assets", type=Path, default=Path(__file__).parent / "geometry_assets.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    baseline = json.loads(args.baseline.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    args.texture_memory_budget = 0.25
    args.image_size = 224
    args.head_image_size = 720
    args.wrist_image_size = 480
    args.max_steps = None
    args.record_video = False
    args.task = "cook_hot_dogs"
    instruction = TASKS[args.task]
    if args.preflight_steps and not 1 <= args.preflight_steps <= 320:
        raise ValueError("preflight must be bounded1..320")
    if bool(protocol["oracle_inputs"]) != args.oracle_upper_bound:
        raise ValueError("Oracle protocol mismatch")
    if not args.oracle_upper_bound and not args.snapshot_provider:
        raise ValueError(
            "restricted mode requires audited observation tracker; no Oracle fallback"
        )
    if args.geometry_assets.resolve() != (Path(__file__).parent / "geometry_assets.json").resolve():
        raise ValueError("geometry assets must be frozen with the staged client")
    geometry_assets = NominalGeometryAssets(args.geometry_assets)
    local_hashes = {
        n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
        for n in ("eval_client.py", "transport.py", "geometry_assets.json")
    }
    report = {"status": "initializing", "steps": 0, "video": False}
    started = time.monotonic()

    def save():
        write(output / "progress.json", report)

    save()
    import numpy as np
    import torch

    torch.set_num_threads(4)
    from omegaconf import OmegaConf
    import omnigibson as og
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    install_wrapper()
    from omnigibson.eval.evaluator import (
        Evaluator,
        resolve_instance_ids,
        DEFAULT_ROBOT_CONFIG_PATH,
    )
    from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED

    seed_everything(DEFAULT_EVAL_SEED)
    # Isolated renderer-only memory profile; no asset/physics/task edits.
    assert 0.05 <= args.texture_memory_budget <= 0.6
    og.launch()
    import carb

    render_settings = carb.settings.get_settings()
    budget_key = "/rtx-transient/resourcemanager/texturestreaming/memoryBudget"
    streaming_key = "/rtx-transient/resourcemanager/texturestreaming/enabled"
    before = {
        budget_key: render_settings.get(budget_key),
        streaming_key: render_settings.get(streaming_key),
    }
    render_settings.set_bool(streaming_key, True)
    render_settings.set_float(budget_key, args.texture_memory_budget)
    viewport_note = {}
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport_note = {"previous_updates_enabled": bool(viewport.updates_enabled)}
            viewport.updates_enabled = False
            viewport_note["updates_enabled"] = bool(viewport.updates_enabled)
    except Exception as exc:
        viewport_note = {"not_applied": str(exc)}
    report["renderer_memory_profile"] = {
        "before": before,
        "requested_texture_budget": args.texture_memory_budget,
        "actual": {
            budget_key: render_settings.get(budget_key),
            streaming_key: render_settings.get(streaming_key),
        },
        "viewport": viewport_note,
        "scope": "renderer texture streaming budget only; physics, scene objects, goal, model and camera resolutions unchanged",
        "comparison_limit": "Texture mip residency may differ; not bit-identical imagery to uncapped baseline",
    }
    save()
    robot_config = OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH)
    robot_config.sensor_config.VisionSensor.sensor_kwargs.image_height = max(
        args.image_size, args.head_image_size or 0, args.wrist_image_size or 0
    )
    robot_config.sensor_config.VisionSensor.sensor_kwargs.image_width = max(
        args.image_size, args.head_image_size or 0, args.wrist_image_size or 0
    )
    report["rendering"] = (
        f"official renderer; no viewer camera; head RGB{args.head_image_size or args.image_size}, wrists RGB{args.wrist_image_size or args.image_size}"
    )
    save()
    cfg = OmegaConf.create(
        {
            "env_wrapper": {
                "_target_": "eval_client.RGBResolutionWrapper",
                "image_size": args.image_size,
                "head_image_size": args.head_image_size,
                "wrist_image_size": args.wrist_image_size,
            },
            "policy_name": "pi05-remote",
            "model": {
                "_target_": "omnigibson.eval.policies.LocalPolicy",
                "action_dim": None,
            },
            "headless": True,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.record_video,
            "video_view": "robot",
            "mode": "public_test",
            "seed": DEFAULT_EVAL_SEED,
            "task": {"name": args.task},
            "robot": robot_config,
        }
    )
    evaluator = Evaluator(cfg)

    import websockets.sync.client, inspect
    from transport import ReliableClient

    options = {"max_size": None, "compression": None, "open_timeout": 20}
    if "ping_interval" in inspect.signature(websockets.sync.client.connect).parameters:
        options["ping_interval"] = None
    weight = args.expected_identity
    robot = evaluator.robot
    indices = {k: array(v) for k, v in robot.controller_action_idx.items()}
    report["control_contract"] = audit_controller_contract(robot)
    assert (
        report["control_contract"] == baseline["results"][0]["control_contract"]
    ), "live control mapping differs from baseline"
    assert abs(og.sim.get_sim_step_dt() - 1 / protocol["action_hz"]) < 1e-8
    assert abs(og.sim.get_physics_dt() - 1 / protocol["physics_hz"]) < 1e-8
    assert abs(og.sim.get_rendering_dt() - 1 / protocol["render_hz"]) < 1e-8
    report.update(
        protocol=protocol,
        source_hashes=local_hashes,
        diagnostic_only=bool(args.preflight_steps),
        oracle_upper_bound=args.oracle_upper_bound,
    )
    extractor = (
        OmniGibsonExtractor(
            evaluator.env, arm_mapping={"left": "left", "right": "right"}, geometry_assets=geometry_assets
        )
        if args.oracle_upper_bound
        else load_provider(args.snapshot_provider, evaluator.env)
    )
    assert (
        not evaluator.should_sync_lights
    ), "Unexpected lighting task: keep official light synchronization"
    assert evaluator._video_path is None
    for trial_index in range(
        args.start_index, args.start_index + 1 if args.preflight_steps else 10
    ):
        directory = output / f"trial_{trial_index+1:02d}"
        directory.mkdir(exist_ok=False)

        seed_everything(DEFAULT_EVAL_SEED)
        evaluator.reset()
        instance = resolve_instance_ids(args.task, [trial_index], mode="public_test")[0]
        evaluator.load_task_instance(instance)
        evaluator.reset()
        assert instance == 301 + trial_index
        report.update(
            status="running",
            trial=trial_index + 1,
            instance_id=instance,
            steps=0,
            inference_queries=0,
            trial_started=time.time(),
            simulator_initialized_seconds=time.monotonic() - started,
        )
        report.pop("metrics", None)
        report.pop("success", None)
        save()
        t0 = time.monotonic()
        step_seconds = 0.0
        client = None

        def event(item):
            item.update(step=report["steps"], time=time.time())
            with (directory / "transport_events.jsonl").open("a") as f:
                f.write(json.dumps(item) + "\n")

        client = ReliableClient(
            lambda: websockets.sync.client.connect(
                f"ws://127.0.0.1:{args.port}", **options
            ),
            weight,
            seed=42,
            on_event=event,
        )
        write(
            directory / "session.json",
            {"session": client.session, "instance": instance, "seed": 42},
        )
        need_obs = True
        terminated = truncated = False
        registry = EntityRegistry()
        extractor.reset()
        actions = np.lib.format.open_memmap(
            directory / "actions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(14000, robot.action_dim),
        )
        try:
            with (directory / "chunk_timing.jsonl").open("w", buffering=1) as timings:
                while not (terminated or truncated):
                    payload = (
                        observation(
                            robot,
                            evaluator.obs,
                            evaluator.robot_camera_names,
                            instruction,
                        )
                        if need_obs
                        else {}
                    )
                    if payload:
                        assert all(
                            hashlib.sha256(
                                (Path(__file__).parent / n).read_bytes()
                            ).hexdigest()
                            == h
                            for n, h in local_hashes.items()
                        ), "execution source changed"
                        sim_time = report["steps"] / 30.0
                        snap = extractor(payload, registry, client.session, sim_time)
                        payload["graph_snapshot"] = (
                            asdict(snap)
                            if hasattr(snap, "__dataclass_fields__")
                            else snap
                        )
                        payload["sim_time"] = sim_time
                        # Actual joint feedback; the base/world transforms are
                        # explicitly Oracle evidence in this upper-bound path.
                        if args.oracle_upper_bound:
                            feedback = robot._get_proprioception_dict()
                            payload["robot_fields"] = live_robot_fields(
                                robot, feedback, sim_time, extractor.geometry_spec
                            )
                        expected = {
                            "head_rgb": 720,
                            "left_wrist_rgb": 480,
                            "right_wrist_rgb": 480,
                        }
                        assert all(
                            tuple(x.shape) == (3, expected[k], expected[k])
                            for k, x in payload["images"].items()
                        )
                    response = client.request(payload)
                    if payload:
                        timings.write(
                            json.dumps(dict(client.last_timing, step=report["steps"]))
                            + "\n"
                        )
                        report["inference_queries"] += 1
                    parts = response["action"]
                    assert set(parts) == {
                        "lower_body",
                        "left_arm",
                        "left_gripper",
                        "right_arm",
                        "right_gripper",
                    }
                    action = action_vector(parts, indices, robot.action_dim)
                    need_obs = response["need_obs"]
                    evaluator.policy.forward = lambda obs, a=action: torch.from_numpy(a)
                    ts = time.monotonic()
                    terminated, truncated = evaluator.step()
                    step_seconds += time.monotonic() - ts
                    actions[report["steps"]] = action
                    report["steps"] += 1
                    if args.preflight_steps and report["steps"] >= args.preflight_steps:
                        break
                    if report["steps"] % 32 == 0 or terminated or truncated:
                        report.update(
                            elapsed_seconds=time.monotonic() - t0,
                            step_seconds=step_seconds,
                        )
                        save()
            metrics = {}
            if args.preflight_steps:
                report.update(
                    status="diagnostic_preflight_completed",
                    not_a_task_result=True,
                    elapsed_seconds=time.monotonic() - t0,
                    step_seconds=step_seconds,
                    source_changed_during_run=False,
                )
                actions.flush()
                write(directory / "result.json", report)
                save()
                continue
            for metric in evaluator.metrics:
                metrics.update(metric.aggregate(evaluator.env))
            report.update(
                source_changed_during_run=False,
                status="completed",
                success=bool(evaluator.env.task.success),
                terminated=bool(terminated),
                truncated=bool(truncated),
                metrics=metrics,
                elapsed_seconds=time.monotonic() - t0,
                step_seconds=step_seconds,
            )
            actions.flush()
            write(directory / "result.json", report)
            save()
            print("TRIAL_RESULT", json.dumps(report, default=str), flush=True)
        except Exception as exc:
            report.update(
                status="infrastructure_error",
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            write(directory / "result.json", report)
            save()
            raise
        finally:
            client.close()
            actions.flush()
            del actions
    report["status"] = "finished"
    save()
    evaluator.env.close()
    og.shutdown()


if __name__ == "__main__":
    main()
