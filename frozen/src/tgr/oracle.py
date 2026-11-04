"""Explicit simulator truth adapter. Does not import or start Isaac at module import."""

import hashlib
import numpy as np
from .state import Entity, Edge, Event, ObservationSnapshot
from .geometry import transform


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class OmniGibsonExtractor:
    """Queries existing environment at chunk boundaries; no stepping/rendering.
    arm_mapping is mandatory if the live arm names are not literally left/right.
    Missing predicates remain unknown; proximity never establishes contact/grasp.
    """

    def __init__(self, env, arm_mapping=None, state_types=None, grasp_states=None):
        self.env = env
        self.arm_mapping = arm_mapping
        self.state_types = state_types
        self.grasp_states = grasp_states
        self.reset()

    def reset(self):
        self.previous = {}
        self.episode = None
        self.registry = None
        self.last_time = -float("inf")

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
            pose = transform(array(pos), array(quat)).tolist()
            eid = registry.resolve(instance)
            info = dict(attrs or {})
            info.update(last_observed_time=timestamp, valid_from=timestamp)
            try:
                lo, hi = obj.aabb
                info["aabb"] = [array(lo).tolist(), array(hi).tolist()]
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
        robot_alias = next(
            (name for name, obj in scope.items() if obj is robot), "robot:base"
        )
        base = add_pose(robot_alias, "robot_base", robot)
        ee = {}
        for side, arm in mapping.items():
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
