"""Decoupled chunk-boundary graph adapter, NOT an RGB detector or robot policy."""

from dataclasses import dataclass
from .state import GraphRuntime
from .tensorize import tensorize, tensorize_history, task_candidate_tensors
from .task import evidence_from_snapshot


@dataclass
class RuntimeInputs:
    episode_id: str
    timestamp: float
    global_graph: object
    local_graph: object
    task_graph: object
    task_labels: dict
    oracle: bool
    candidates: dict | None = None


class ChunkGraphBridge:
    """Call begin_chunk before prefix prefill, then after_prefix with decoder outputs.
    This adapter does not run a decoder, sample actions, score success or step physics.
    Source timestamps must be simulator seconds, not host receive times.
    """

    def __init__(
        self,
        provider,
        task_spec,
        restricted=True,
        k=16,
        device="cpu",
        capacity=64,
        query_width=256,
    ):
        self.runtime = GraphRuntime(provider, restricted, k, task_spec)
        self.device = device
        self.capacity = capacity
        self.pending = {}
        self.query_width = query_width

    def reset(self, env, episode):
        self.runtime.reset(env, episode)
        self.pending.pop(env, None)

    def begin_chunk(self, env, observation, timestamp, feasibility_summary=None):
        if env in self.pending:
            raise ValueError(
                "previous chunk requires after_prefix before new observation"
            )
        s = self.runtime.environments[env]
        # The VLM gets previous task state, never this chunk's decoder output or future labels.
        previous = (
            tensorize_history(s["task"], self.device, self.capacity)
            if s["task"].get_history()
            else tensorize(s["task"].snapshot(), timestamp, self.device, self.capacity)
        )
        manager = s["task_manager"]
        previous_events = s["events"].query()[-64:]
        candidates = task_candidate_tensors(
            manager.task_spec, manager, device=self.device, width=self.query_width
        )
        self.runtime.update(
            env,
            observation,
            timestamp,
            feasibility_summary=feasibility_summary,
            defer_task=True,
        )
        manager = s["task_manager"]
        evidence = evidence_from_snapshot(manager, s["current_snapshot"])
        self.pending[env] = (timestamp, evidence, manager.version)
        from .tensorize import attach_events

        attach_events(previous, previous_events, timestamp)
        return RuntimeInputs(
            s["episode"],
            timestamp,
            tensorize_history(s["global"], self.device, self.capacity),
            tensorize_history(s["local"], self.device, self.capacity, local=True),
            previous,
            dict(manager.states),
            bool(s["global"].oracle or s["local"].oracle or previous.oracle),
            candidates,
        )

    def after_prefix(self, env, predictions):
        if env not in self.pending:
            raise ValueError("begin_chunk must run first")
        timestamp, evidence, version = self.pending[env]
        if self.runtime.environments[env]["task_manager"].version != version:
            raise ValueError("stale VLM task write")
        s = self.runtime.update_task(env, predictions, evidence, timestamp)
        self.pending.pop(env)
        return dict(
            states=dict(s["task_manager"].states),
            legal=s["task_manager"].legal(),
            task_spec_sha256=s["task_manager"].task_spec.sha256,
            candidate_ids=s["task_manager"].task_spec.candidate_ids,
        )


def snapshot_from_decoded_task_info(
    decoded, registry, episode_id, timestamp, ee_world=None
):
    """Consume already schema-decoded fields, NOT a guessed 70-vector layout.
    Keys are canonical BDDL instance names plus _real/_pos/_ori_cos/_ori_sin.
    Privileged task_info is always oracle. No cooked/open/inside labels are inferred.
    ee_world: optional explicitly world-frame 4x4 transforms keyed left/right.
    Robot eef proprio may be base-relative: caller MUST transform using live/calibrated FK.
    """
    import numpy as np
    from .state import Entity, Edge, ObservationSnapshot
    from .geometry import transform, relative

    nodes = []
    edges = []
    hands = {}
    for side, pose in (ee_world or {}).items():
        if side not in ("left", "right"):
            raise ValueError("unknown hand identity")
        relative(pose, pose)
        eid = registry.resolve("robot:ee:" + side)
        hands[side] = eid
        nodes.append(
            Entity(
                eid,
                side + "_ee",
                "robot:ee:" + side,
                np.asarray(pose).tolist(),
                "world",
                timestamp,
                timestamp,
                1.0,
                "oracle",
                True,
                {"provenance": "explicit_world_EE_transform"},
            )
        )
    for key, value in decoded.items():
        if not key.endswith("_real"):
            continue
        name = key[:-5]
        exists = float(np.asarray(value).reshape(-1)[0])
        if exists not in (0.0, 1.0):
            raise ValueError("invalid existence indicator")
        pose = None
        if exists:
            pos = np.asarray(decoded[name + "_pos"], dtype=float)
            c = np.asarray(decoded[name + "_ori_cos"], dtype=float)
            s = np.asarray(decoded[name + "_ori_sin"], dtype=float)
            if (
                pos.shape != (3,)
                or c.shape != (3,)
                or s.shape != (3,)
                or not np.isfinite(np.r_[pos, c, s]).all()
            ):
                raise ValueError("invalid decoded pose")
            if not np.allclose(c * c + s * s, 1.0, atol=2e-3):
                raise ValueError("invalid RPY sine/cosine")
            roll, pitch, yaw = np.arctan2(s, c) / 2
            cr, cp, cy = np.cos([roll, pitch, yaw])
            sr, sp, sy = np.sin([roll, pitch, yaw])
            pose = transform(
                pos,
                [
                    cy * sr * cp - sy * cr * sp,
                    cy * cr * sp + sy * sr * cp,
                    sy * cr * cp - cy * sr * sp,
                    cy * cr * cp + sy * sr * sp,
                ],
            ).tolist()
        eid = registry.resolve(name)
        kind = "robot_base" if name.startswith("agent.") else name.rsplit("_", 1)[0]
        attrs = {"provenance": "recorded_task_info", "existence": bool(exists)}
        nodes.append(
            Entity(
                eid,
                kind,
                name,
                pose,
                "world",
                timestamp,
                timestamp,
                1.0,
                "oracle",
                bool(exists),
                attrs,
            )
        )
        for side, hand in hands.items():
            field = name + "_in_gripper_" + side
            if not exists or field not in decoded:
                continue
            grasp = float(np.asarray(decoded[field]).reshape(-1)[0])
            if grasp not in (-1.0, 0.0, 1.0):
                raise ValueError("invalid IsGraspingState enum")
            value = True if grasp == 1 else False if grasp == -1 else None
            edges.append(
                Edge(
                    eid,
                    hand,
                    "held_by",
                    {"value": value},
                    timestamp,
                    1.0 if value is not None else 0.0,
                    value is True,
                    "oracle",
                )
            )
    return ObservationSnapshot(
        episode_id, timestamp, tuple(nodes), tuple(edges), oracle=True
    )


def phase_supervision(task_spec, phase_candidate_id, *, device="cpu", legal=None):
    # Supervise ONLY the active annotated plan phase. Never previous-phase DONE or goal truth.
    import torch
    from .task import load_task_spec

    spec = load_task_spec(task_spec)
    k = spec.K
    labels = dict(
        state=torch.full((1, k), 4, dtype=torch.long, device=device),
        state_mask=torch.zeros(1, k, dtype=torch.bool, device=device),
        goal=torch.zeros(1, k, device=device),
        goal_mask=torch.zeros(1, k, dtype=torch.bool, device=device),
        next=torch.zeros(1, dtype=torch.long, device=device),
        next_mask=torch.zeros(1, dtype=torch.bool, device=device),
    )
    if phase_candidate_id is None:
        return labels
    i = spec.next_index(phase_candidate_id)
    if legal is not None and (tuple(legal.shape) != (1, k) or not bool(legal[0, i])):
        raise ValueError(
            "phase label violates shared dependency mask; fix artifact/labels, not hidden remapping"
        )
    labels["state"][0, i] = 1
    labels["state_mask"][0, i] = True
    labels["next"][0] = i
    labels["next_mask"][0] = True
    return labels


class _DecodedTaskInfoProvider:
    def snapshot(self, observation, registry, episode_id, timestamp):
        import numpy as np
        from dataclasses import replace
        from .state import Entity
        from .geometry import relative

        fields = observation.get("robot_fields") or {}
        self.last_robot_fields = fields
        eef = fields.get("eef_world", {})
        arms = fields.get("arm_world", {})
        has_world = bool(eef or arms or fields.get("base_world") is not None)
        if has_world and (
            fields.get("reference_frame") != "world"
            or fields.get("world_poses_verified") is not True
            or fields.get("timestamp") != timestamp
        ):
            raise ValueError(
                "verified, synchronized world robot poses required; no implicit proprio/FK assumptions"
            )
        snap = snapshot_from_decoded_task_info(
            observation["decoded_task_info"],
            registry,
            episode_id,
            timestamp,
            ee_world=eef,
        )
        entities = {n.entity_id: n for n in snap.entities}
        if fields.get("base_world") is not None:
            pose = fields["base_world"]
            relative(pose, pose)
            base = next(
                (n for n in entities.values() if n.semantic_type == "robot_base"), None
            )
            name = base.instance_id if base else "robot:base"
            eid = registry.resolve(name)
            entities[eid] = Entity(
                eid,
                "robot_base",
                name,
                np.asarray(pose).tolist(),
                "world",
                timestamp,
                timestamp,
                1.0,
                "oracle",
                True,
                {"provenance": "verified_robot_fields"},
            )
        for side in ("left", "right"):
            if side not in arms:
                name = "robot:arm:" + side
                eid = registry.resolve(name)
                entities[eid] = Entity(
                    eid,
                    side + "_arm",
                    name,
                    None,
                    "world",
                    timestamp,
                    timestamp,
                    0.0,
                    "oracle",
                    False,
                    {
                        "provenance": "arm_world_calibration_unverified",
                        "pose_known": False,
                    },
                )
        for side, pose in arms.items():
            if side not in ("left", "right"):
                raise ValueError("invalid arm side")
            relative(pose, pose)
            name = "robot:arm:" + side
            eid = registry.resolve(name)
            entities[eid] = Entity(
                eid,
                side + "_arm",
                name,
                np.asarray(pose).tolist(),
                "world",
                timestamp,
                timestamp,
                1.0,
                "oracle",
                True,
                {"provenance": "verified_robot_fields"},
            )
        if fields.get("trunk_world") is not None:
            pose = fields["trunk_world"]
            relative(pose, pose)
            name = "robot:trunk"
            eid = registry.resolve(name)
            entities[eid] = Entity(
                eid,
                "robot_trunk",
                name,
                np.asarray(pose).tolist(),
                "world",
                timestamp,
                timestamp,
                1.0,
                "oracle",
                True,
                {"provenance": "measured_URDF_trunk_frame"},
            )
        from .state import Edge

        edges = list(snap.edges)
        base = next(
            (n for n in entities.values() if n.semantic_type == "robot_base"), None
        )
        if base is not None:
            for node in entities.values():
                if node.semantic_type in (
                    "left_arm",
                    "right_arm",
                    "left_ee",
                    "right_ee",
                    "robot_trunk",
                ):
                    edges.append(
                        Edge(
                            node.entity_id,
                            base.entity_id,
                            "connected_to",
                            {},
                            timestamp,
                            node.confidence,
                            node.validity,
                            "oracle",
                        )
                    )
        return _enrich_robot_geometry(
            replace(snap, entities=tuple(entities.values()), edges=tuple(edges)),
            fields,
            registry,
        )


def _history_metadata(graph, capacity, local=False):
    # Same union-slot convention as tensorize_history. IDs are metadata, never RGB-derived GT.
    state = dict(graph.history_base)
    frames = []
    for delta in graph.get_history():
        state.update({n["entity_id"]: n for n in delta["changed_nodes"]})
        if local:
            ids = [
                n["entity_id"]
                for n in state.values()
                if n["semantic_type"] != "feasibility_summary_v1"
            ]
            for e in delta["edges"]:
                if (
                    e["relation_type"] == "relative_to"
                    and e["validity"]
                    and state[e["source_id"]]["validity"]
                    and state[e["target_id"]]["validity"]
                ):
                    ids.append(e["source_id"] + "::" + e["target_id"])
            ids.extend(
                n["entity_id"]
                for n in state.values()
                if n["semantic_type"] == "feasibility_summary_v1" and n["validity"]
            )
        else:
            ids = list(state)
        frames.append((delta["timestamp"], ids))
    if not frames:
        ids = sorted(graph.nodes)
        return dict(timestamps=[0.0], node_ids=[ids + [None] * (capacity - len(ids))])
    ids = sorted({key for _, keys in frames for key in keys})
    if len(ids) > capacity:
        raise ValueError("history capacity exceeded")
    slots = ids + [None] * (capacity - len(ids))
    return dict(
        timestamps=[t for t, _ in frames], node_ids=[list(slots) for _ in frames]
    )


class CPUConditionAdapter:
    """CPU-only sequential condition assembler for both data and evaluation owners.

    robot_fields schema:
      reference_frame='world', world_poses_verified=True, timestamp=current time,
      eef_world={left/right: 4x4}, arm_world={left/right: 4x4}, base_world=4x4.
    All pose fields optional. Missing fields remain UNKNOWN; no FK/IK here.
    TaskSpec artifact is external/data-owned. No image/token/normalization transforms.

    step task_predictions refer to PREVIOUS pending chunk decision. Alternatively call
    commit(predictions) immediately after that chunk's prefix/decoder. Current phase
    annotation only produces labels; it is NEVER fed back as a predicted task state.
    """

    def __init__(
        self,
        task_spec,
        *,
        history=16,
        capacity=64,
        query_width=256,
        expected_spec_sha256=None,
        restricted=False
    ):
        from .task import load_task_spec

        self.spec = load_task_spec(task_spec, expected_sha256=expected_spec_sha256)
        self.bridge = ChunkGraphBridge(
            _DecodedTaskInfoProvider(),
            self.spec,
            restricted=restricted,
            k=history,
            device="cpu",
            capacity=capacity,
            query_width=query_width,
        )
        self.capacity = capacity
        self.steps = {}

    def reset(self, episode_id, env_id=0):
        self.bridge.reset(env_id, episode_id)
        self.steps[env_id] = -1

    def commit(self, task_predictions, env_id=0):
        return self.bridge.after_prefix(env_id, task_predictions)

    def step(
        self,
        decoded_task_info,
        robot_fields,
        *,
        episode_id,
        step_id,
        timestamp,
        env_id=0,
        task_predictions=None,
        phase_candidate_id=None,
        feasibility_summary=None
    ):
        from .model import Conditions

        if env_id not in self.steps:
            raise ValueError("explicit episode reset required")
        state = self.bridge.runtime.environments[env_id]
        if (
            episode_id != state["episode"]
            or not isinstance(step_id, int)
            or step_id <= self.steps[env_id]
        ):
            raise ValueError("episode/step mismatch")
        if env_id in self.bridge.pending:
            self.commit(task_predictions or {}, env_id)
        elif task_predictions is not None:
            raise ValueError("no preceding pending chunk for task_predictions")
        task_meta = _history_metadata(state["task"], self.capacity)
        previous_events = state["events"].query()[-64:]
        c = self.bridge.begin_chunk(
            env_id,
            dict(decoded_task_info=decoded_task_info, robot_fields=robot_fields),
            timestamp,
            feasibility_summary=feasibility_summary,
        )
        self.steps[env_id] = step_id
        v = c.candidates
        from .tensorize import attach_events

        attach_events(c.task_graph, previous_events, timestamp)
        attach_events(c.global_graph, state["events"].query()[-64:], timestamp)
        conditions = Conditions(
            global_graph=c.global_graph,
            local_graph=c.local_graph,
            task_graph=c.task_graph,
            task_condition=v["queries"].mean(dim=1),
            queries=v["queries"],
            legal=v["legal"],
            oracle=c.oracle,
            restricted=self.bridge.runtime.restricted,
        )
        labels = phase_supervision(self.spec, phase_candidate_id, legal=v["legal"])
        # Physical goal labels are absent unless explicitly provided by a separate legal source.
        provenance = dict(
            episode_id=episode_id,
            step_id=step_id,
            timestamp=timestamp,
            graph_source="oracle",
            task_spec_sha256=self.spec.sha256,
            candidate_ids=self.spec.candidate_ids,
            feature_schema="tc_esgr.feature32.v3.approx_center_geometry",
            history_timestamps={},
            history_node_ids={},
            task_context_source="previous_chunk_only",
            phase_labels_are_inputs=False,
            oracle_predicates_inferred=False,
            missing_ee_world=not bool((robot_fields or {}).get("eef_world")),
        )
        for name, local in (("global", False), ("local", True)):
            m = _history_metadata(state[name], self.capacity, local=local)
            provenance["history_timestamps"][name] = m["timestamps"]
            provenance["history_node_ids"][name] = m["node_ids"]
        provenance["history_timestamps"]["task"] = task_meta["timestamps"]
        provenance["history_node_ids"]["task"] = task_meta["node_ids"]
        return dict(conditions=conditions, labels=labels, provenance=provenance)


# Static train-only candidate artifact integration. BDDL terminal goals remain in
# the task-state graph, separately from the EXACT 23 candidate decoder queries.
def _static_plan(plan_spec):
    import json, hashlib
    from pathlib import Path

    default = Path(__file__).resolve().parents[2] / "reports/data_plan_vocabulary.json"
    if isinstance(plan_spec, (str, Path)):
        path = Path(plan_spec)
        raw = path.read_bytes()
        artifact = json.loads(raw)
    else:
        artifact = plan_spec
        # A dict has no file-byte identity. Bind it to the immutable producer
        # artifact rather than inventing another semantic hash.
        raw = default.read_bytes()
        if json.loads(raw) != artifact:
            raise ValueError(
                "plan dict must equal frozen on-disk data artifact; supply a path for another artifact"
            )
    if (
        artifact.get("schema") != "tgr_static_skill_candidates_v1"
        or artifact.get("source_split") != "train"
        or artifact.get("test_annotations_used") is not False
    ):
        raise ValueError("train-only static plan required")
    ids = artifact.get("query_order", [])
    if (
        not ids
        or ids != artifact.get("node_keys")
        or len(set(ids)) != len(ids)
        or [c["id"] for c in artifact["candidates"]] != ids
    ):
        raise ValueError("static plan candidate order mismatch")
    if (
        artifact.get("query_features") != 256
        or artifact.get("query_encoding") != "sha256_counter_bytes_v1"
        or artifact.get("dependency_edges") != []
    ):
        raise ValueError("unsupported query encoding or unverified dependencies")
    return artifact, hashlib.sha256(raw).hexdigest()


def _internal_task_spec(artifact, bddl_spec=None):
    from pathlib import Path
    from .task import parse_bddl, construct_task_spec

    if bddl_spec is None:
        bddl_spec = (
            Path(__file__).resolve().parents[2]
            / "data/task_definitions/cook_hot_dogs.bddl"
        ).read_text()
    b = parse_bddl(bddl_spec) if isinstance(bddl_spec, str) else bddl_spec

    def flatten(x):
        if isinstance(x, list):
            return [z for v in x for z in flatten(v)]
        return [x]

    plans = []
    for c in artifact["candidates"]:
        entities = sorted(
            {
                x
                for x in flatten(c.get("targets", []) + c.get("manipulating", []))
                if x in b["objects"]
            }
        )
        plans.append(
            dict(
                id=c["id"],
                text=c["id"],
                entities=entities,
                dependencies=[],
                terminal=False,
                action_type=(
                    "REPOSITION_BASE" if c["id"] == "plan:REPOSITION_BASE" else "SKILL"
                ),
            )
        )
    return construct_task_spec(
        b,
        plans,
        provenance=dict(
            source="training_demonstration_derived",
            split="train",
            episode_ids=[x["episode_id"] for x in artifact["source_train_episodes"]],
            uses_test_gt=False,
        ),
    )


def _plain_conditions(conditions):
    from dataclasses import asdict

    # asdict preserves torch tensors and ensures each GraphBatch is a dictionary.
    return asdict(conditions)


class StaticConditionAdapter(CPUConditionAdapter):
    def __init__(
        self,
        plan_spec,
        *,
        device="cpu",
        oracle=True,
        task_spec=None,
        history=16,
        capacity=64
    ):
        self.plan, self.plan_sha256 = _static_plan(plan_spec)
        self.node_keys = tuple(self.plan["node_keys"])
        self.device = device
        internal = _internal_task_spec(self.plan, task_spec)
        super().__init__(
            internal, history=history, capacity=capacity, restricted=not oracle
        )
        self.bridge.device = device
        self.memory = None
        self.memory_tokens = None
        self.reachability_capabilities = {
            "ik": "URDF center position-only IK; nominal metric bbox center",
            "collision": "trusted model-internal capsule/oriented-box partial estimate; physical_verified=false",
            "base_pose_feedback": "fresh measured pose required",
            "reposition_trained_validated": False,
            "policy": "REPOSITION_BASE disabled; no distance reachability or actuator commands",
        }

    def reset(self, episode_id, env_id=0):
        super().reset(episode_id, env_id)
        self.memory = None
        self.memory_tokens = None

    def _conditions(self, c):
        import torch
        from .build_batches import plan_queries

        q = plan_queries(self.plan).to(self.device)
        manager = self.bridge.runtime.environments[0]["task_manager"]
        legal_ids = set(manager.legal())
        c.queries = q
        c.task_condition = q.mean(1)
        # Conservative experiment-level provenance: prior task definitions do
        # not wash Oracle context out of an Oracle-conditioned training batch.
        if not self.bridge.runtime.restricted:
            for name in ("global_graph", "local_graph", "task_graph"):
                getattr(c, name).oracle = True
            c.oracle = True
        c.legal = torch.tensor(
            [[x in legal_ids and x != "plan:REPOSITION_BASE" for x in self.node_keys]],
            device=self.device,
            dtype=torch.bool,
        )
        c.memory = self.memory
        c.memory_tokens = None
        state = self.bridge.runtime.environments[0]
        registry = state["registry"]
        task_ids = _history_metadata(state["task"], self.capacity)["node_ids"][-1]
        c.task_candidate_slots = torch.tensor(
            [
                [
                    task_ids.index(registry.resolve("task:" + key))
                    for key in self.node_keys
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )
        global_ids = _history_metadata(state["global"], self.capacity)["node_ids"][-1]
        local_ids = _history_metadata(state["local"], self.capacity, local=True)[
            "node_ids"
        ][-1]
        names = set(self.spec.artifact["objects"])
        # Instruction/previous task references are hard retrieval inputs; no current labels.
        relevant = {registry.resolve(name) for name in names}
        c.global_graph.relevance = torch.tensor(
            [
                [
                    [
                        key in relevant
                        or (
                            node is not None
                            and node.semantic_type
                            in (
                                "robot_base",
                                "robot_trunk",
                                "left_arm",
                                "right_arm",
                                "left_ee",
                                "right_ee",
                            )
                        )
                        for key in global_ids
                        for node in [state["global"].nodes.get(key)]
                    ]
                ]
            ],
            dtype=torch.bool,
            device=self.device,
        )
        masks = []

        def flatten(values):
            if isinstance(values, list):
                return [value for item in values for value in flatten(item)]
            return [values]

        for candidate in self.plan["candidates"]:
            selected = {
                registry.resolve(name)
                for name in flatten(
                    candidate.get("targets", []) + candidate.get("manipulating", [])
                )
                if name in names
            }
            values = []
            for key in local_ids:
                parts = [] if key is None else key.split("::")
                node = state["local"].nodes.get(key)
                target = None if node is None else node.attributes.get("target_id")
                obstacle_reference = any(
                    state["local"].nodes.get(part) is not None
                    and state["local"].nodes[part].semantic_type == "local_obstacle"
                    for part in parts
                )
                obstacle_reference = obstacle_reference or (
                    node is not None and node.semantic_type == "local_obstacle"
                )
                values.append(
                    not selected
                    or obstacle_reference
                    or bool(selected.intersection(parts))
                    or target in selected
                )
            masks.append(values)
        c.candidate_mask = torch.tensor([masks], dtype=torch.bool, device=self.device)
        # Chunk boundaries are legitimate semantic memory update events.
        c.event_changed = torch.ones(1, dtype=torch.bool, device=self.device)
        fields = getattr(self.bridge.runtime.provider, "last_robot_fields", {})
        joints = _robot_joints(fields) if fields else None
        if joints is not None:
            import numpy as np
            from pathlib import Path
            from .kinematics import URDFKinematics

            if not hasattr(self, "joint_backend"):
                self.joint_backend = URDFKinematics(
                    Path(__file__).resolve().parents[2] / "data/robot/r1pro.urdf"
                )
            vectors = []
            roles = []
            for group, prefix, count in [
                (0, "torso_joint", 4),
                (1, "left_arm_joint", 7),
                (2, "right_arm_joint", 7),
            ]:
                for i in range(1, count + 1):
                    key = prefix + str(i)
                    joint = self.joint_backend.joints[key]
                    value = joints[key]
                    vectors.append(
                        [
                            (2 * value - joint.lower - joint.upper)
                            / (joint.upper - joint.lower),
                            joint.lower / np.pi,
                            joint.upper / np.pi,
                            1.0,
                        ]
                    )
                    roles.append(group)
            c.local_graph.joints = torch.tensor(
                [vectors], dtype=torch.float32, device=self.device
            )
            c.local_graph.joint_roles = torch.tensor(
                [roles], dtype=torch.long, device=self.device
            )
        else:
            c.local_graph.joints = torch.zeros(1, 18, 4, device=self.device)
            c.local_graph.joint_roles = torch.tensor(
                [[0] * 4 + [1] * 7 + [2] * 7], device=self.device
            )
        return c

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        result["conditions"] = self._conditions(result["conditions"])
        # No labels supplied here: data owner alone supplies phase annotation masks.
        result.pop("labels", None)
        p = result["provenance"]
        for field in ("history_timestamps", "history_node_ids"):
            p[field] = {k + "_graph": v for k, v in p[field].items()}
        # Empty task history at reset is still a one-frame task definition snapshot.
        p.update(
            plan_sha256=self.plan_sha256,
            condition_schema=self.plan["condition_schema"],
            runtime_factory=self.plan["runtime_factory"],
            candidate_ids=self.node_keys,
            task_spec_sha256=self.plan_sha256,
            query_order=list(self.node_keys),
            reposition_trained_validated=False,
            architecture_version="tc_esgr_v2",
        )
        return result


def build_training_conditions(
    *,
    decoded_task_info,
    ee_world,
    plan_spec,
    episode_id,
    timestamp,
    history,
    device="cpu",
    robot_fields=None
):
    """Replay only same-episode causal graph observations; labels never enter.
    history includes current observation if desired; duplicate current time is
    checked against current input and consumed once. No future history allowed.
    """
    adapter = StaticConditionAdapter(plan_spec, device=device, oracle=True)
    adapter.reset(episode_id)
    rows = []
    last = -float("inf")
    for h in history:
        t = float(h["timestamp"])
        if t <= last or t > timestamp:
            raise ValueError("noncausal graph history")
        if str(h.get("episode_id", episode_id)) != str(episode_id):
            raise ValueError("cross-episode graph history")
        last = t
        if t == timestamp:
            import numpy as np

            if set(h["decoded_task_info"]) != set(decoded_task_info) or any(
                not np.array_equal(h["decoded_task_info"][k], decoded_task_info[k])
                for k in decoded_task_info
            ):
                raise ValueError("current history observation mismatch")
            continue
        rows.append(h)
    rows.append(
        dict(
            timestamp=timestamp,
            decoded_task_info=decoded_task_info,
            ee_world=ee_world,
            robot_fields=robot_fields,
        )
    )
    replay = rows[-16:]
    for i, h in enumerate(replay):
        fields = h.get("robot_fields")
        if fields is not None:
            fields = dict(fields)
            # Previous/current only; never solve IK for every history exposure.
            fields["compute_reachability"] = i >= len(replay) - 2
        if fields is None:
            fields = dict(
                reference_frame="world",
                world_poses_verified=True,
                timestamp=h["timestamp"],
                eef_world=h.get("ee_world") or {},
            )
        if i + 1 < len(replay):
            # Historical observations need graph/state updates only. Materialize
            # all three histories once, immediately before the current prefix.
            # No predictions or annotation targets enter this replay.
            adapter.bridge.runtime.update(
                0,
                dict(decoded_task_info=h["decoded_task_info"], robot_fields=fields),
                float(h["timestamp"]),
            )
        else:
            result = adapter.step(
                h["decoded_task_info"],
                fields,
                episode_id=episode_id,
                step_id=i,
                timestamp=float(h["timestamp"]),
            )
    result["conditions"] = _plain_conditions(result["conditions"])
    return result


class EvalGraphRuntime(StaticConditionAdapter):
    def __init__(self, plan_path, device="cpu", oracle=False, task_spec=None):
        super().__init__(plan_path, device=device, oracle=oracle, task_spec=task_spec)
        from .serve import WireProvider

        self.bridge = ChunkGraphBridge(
            _RobotEvidenceProvider(WireProvider(oracle)),
            self.spec,
            restricted=not oracle,
            k=16,
            device=device,
            capacity=self.capacity,
        )
        self.episode_id = None
        self.chunk = 0

    def reset(self, episode_id, env_id=0):
        self.episode_id = episode_id
        self.chunk = 0
        if episode_id is not None:
            super().reset(episode_id, env_id)
        else:
            self.bridge.runtime.environments.clear()
            self.bridge.pending.clear()
            self.steps.clear()
            self.memory = None
            self.memory_tokens = None

    def begin_chunk(self, raw):
        from .model import Conditions

        if self.episode_id is None or raw["session"] != self.episode_id:
            raise ValueError("explicit matching evaluation reset required")
        c = self.bridge.begin_chunk(
            0,
            raw,
            float(raw["sim_time"]),
            feasibility_summary=raw.get("feasibility_summary"),
        )
        self.chunk += 1
        return self._conditions(
            Conditions(
                global_graph=c.global_graph,
                local_graph=c.local_graph,
                task_graph=c.task_graph,
                oracle=c.oracle,
                restricted=self.bridge.runtime.restricted,
            )
        )

    def after_prefix(self, decision):
        from .task import STATES

        predictions = {}
        if decision is not None:
            from .task_feedback import constrained_plan_states

            values = constrained_plan_states(decision).cpu().tolist()[0]
            if len(values) != len(self.node_keys):
                raise ValueError("decoder query count mismatch")
            predictions = {
                key: STATES[value] for key, value in zip(self.node_keys, values)
            }
            self.memory = decision["memory"].detach()
            self.memory_tokens = None
        return self.bridge.after_prefix(0, predictions)

    def export_state(self):
        import pickle, numpy as np

        def cpu(x):
            return None if x is None else x.detach().cpu().numpy()

        return dict(
            version="tgr.runtime.static.v1",
            plan_sha256=self.plan_sha256,
            episode_id=self.episode_id,
            chunk=self.chunk,
            graph_state=pickle.dumps(
                (self.bridge.runtime.environments, self.bridge.pending, self.steps),
                protocol=4,
            ),
            memory=cpu(self.memory),
            memory_tokens=cpu(self.memory_tokens),
        )

    def restore_state(self, state):
        import pickle, torch

        if (
            state.get("version") != "tgr.runtime.static.v1"
            or state.get("plan_sha256") != self.plan_sha256
        ):
            raise ValueError("runtime state plan/version mismatch")
        self.episode_id = state["episode_id"]
        self.chunk = state["chunk"]
        self.bridge.runtime.environments, self.bridge.pending, self.steps = (
            pickle.loads(state["graph_state"])
        )
        for key in ("memory", "memory_tokens"):
            value = state[key]
            setattr(
                self,
                key,
                None if value is None else torch.as_tensor(value, device=self.device),
            )


def make_eval_runtime(*, plan_path, device="cpu", oracle=False, task_spec=None):
    return EvalGraphRuntime(plan_path, device, oracle, task_spec)


_KINEMATICS = None


def _robot_joints(fields):
    import numpy as np

    groups = fields.get("joints", {})
    joints = {}
    for key, prefix, n in [
        ("trunk_qpos", "torso_joint", 4),
        ("left_arm_qpos", "left_arm_joint", 7),
        ("right_arm_qpos", "right_arm_joint", 7),
    ]:
        if key not in groups:
            return None
        values = np.asarray(groups[key], dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("explicit measured joint layout mismatch")
        joints.update(
            {prefix + str(i + 1): float(value) for i, value in enumerate(values)}
        )
    for side in ("left", "right"):
        key = side + "_gripper_qpos"
        if key in groups:
            values = np.asarray(groups[key], dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("measured gripper joint layout mismatch")
            joints.update(
                {
                    side + "_gripper_finger_joint" + str(i + 1): float(value)
                    for i, value in enumerate(values)
                }
            )
    return joints


def _enrich_robot_geometry(snapshot, fields, registry):
    """Use explicit synchronized robot feedback; object targets remain UNKNOWN.
    No desired EE grasp transform is inferred from an object centre.
    """
    global _KINEMATICS
    _enrich_robot_geometry.last_cache_key = None
    _enrich_robot_geometry.last_assessments = []
    from pathlib import Path
    from .kinematics import URDFKinematics
    from .reachability import (
        RobotState,
        TargetPose,
    )
    from .state import merge_feasibility
    from dataclasses import replace
    from .state import Entity

    entities = {n.entity_id: n for n in snapshot.entities}
    for side in ("left", "right"):
        name = "robot:arm:" + side
        eid = registry.resolve(name)
        if eid not in entities:
            entities[eid] = Entity(
                eid,
                side + "_arm",
                name,
                None,
                "world",
                snapshot.timestamp,
                snapshot.timestamp,
                0.0,
                "oracle" if snapshot.oracle else "observation",
                False,
                {"pose_known": False, "provenance": "arm_world_calibration_unverified"},
            )
    snapshot = replace(snapshot, entities=tuple(entities.values()))
    joints = _robot_joints(fields)
    base = next(
        (
            n
            for n in snapshot.entities
            if n.semantic_type in ("robot_base", "base")
            and n.validity
            and n.pose is not None
        ),
        None,
    )
    if joints is None or base is None:
        return snapshot
    if (
        fields.get("timestamp") != snapshot.timestamp
        or fields.get("reference_frame") != "world"
        or fields.get("world_poses_verified") is not True
    ):
        raise ValueError("synchronized measured robot geometry required")
    if _KINEMATICS is None:
        import json

        root = Path(__file__).resolve().parents[2]
        tool = json.loads((root / "reports/data_tool_frames.json").read_text())
        if tool.get("offset_fitted_to_demonstration") is not False:
            raise ValueError("source-declared tool frame required")
        _KINEMATICS = URDFKinematics(
            root / "data/robot/r1pro.urdf",
            tool_transforms={a: tool["matrix"] for a in ("left", "right")},
        )
    state = RobotState(
        snapshot.timestamp,
        base.pose,
        joints,
        snapshot.episode_id,
        source="oracle" if snapshot.oracle else "robot_feedback",
    )
    from .geometry_data import attach_geometry
    from .approx_reachability import CenterApproxReachabilityEngine, OrientedBoxObstacle
    from collections import OrderedDict

    snapshot = attach_geometry(snapshot, fields, registry)
    if not fields.get("compute_reachability", True):
        return snapshot
    engine = CenterApproxReachabilityEngine(
        _KINEMATICS, max_iterations=32, max_candidates=4
    )
    body_types = {
        "base",
        "robot_base",
        "left_ee",
        "right_ee",
        "left_arm",
        "right_arm",
        "trunk",
        "robot_trunk",
        "feasibility_summary_v1",
        "local_obstacle",
        "floor.n.01",
    }
    targets = [
        node
        for node in snapshot.entities
        if node.semantic_type not in body_types
        and node.pose is not None
        and node.validity
    ]
    obstacles = []
    for node in snapshot.entities:
        attrs = node.attributes
        if (
            not node.validity
            or not attrs.get("extent_known")
            or node.semantic_type == "floor.n.01"
        ):
            continue
        obstacles.append(
            OrientedBoxObstacle(
                node.entity_id,
                attrs["world_T_box"],
                attrs["nominal_extent_m"],
                node.timestamp,
                snapshot.episode_id,
                source=(
                    "initial_static_scene_proxy"
                    if attrs.get("assumed_static")
                    else node.source
                ),
                assumed_static=bool(attrs.get("assumed_static")),
                container=bool(attrs.get("container")),
            )
        )
    if not hasattr(_enrich_robot_geometry, "cache"):
        _enrich_robot_geometry.cache = OrderedDict()
    cache = _enrich_robot_geometry.cache
    import hashlib, json

    from .geometry_cache import make_key, lookup

    payload = {
        "episode": snapshot.episode_id,
        "timestamp": snapshot.timestamp,
        "base": base.pose,
        "joints": joints,
        "targets": [(n.entity_id, n.pose) for n in targets],
        "boxes": [(o.entity_id, o.world_T_box, o.extent_m) for o in obstacles],
    }
    key = make_key(payload)
    persisted = lookup(key, fields)
    if persisted is not None:
        assessments = persisted
        cache[key] = assessments
        cache.move_to_end(key)
    elif key in cache:
        assessments = cache[key]
        cache.move_to_end(key)
    else:
        assessments = []
        for target in targets:
            goal = TargetPose(
                target.entity_id,
                snapshot.timestamp,
                target.pose,
                snapshot.episode_id,
                source=target.source,
                pose_kind="object_center_approx",
            )
            assessment = engine.evaluate(state, goal, snapshot.timestamp, obstacles)
            if assessment.get("valid"):
                if not assessment.get("available_arms"):
                    candidates = engine.propose(
                        state, goal, snapshot.timestamp, obstacles
                    )
                else:
                    candidates = []
                assessment["base_candidates"] = candidates
                assessment["base_reposition_feasibility"] = (
                    "MODEL_PASS"
                    if assessment.get("available_arms")
                    or any(
                        bool(candidate.get("estimated_reposition_feasible"))
                        for candidate in candidates
                    )
                    else "MODEL_REJECT"
                )
            # Robot posture metrics remain defined when no goal IK solution exists.
            for side, info in assessment.get("arms", {}).items():
                if info.get("workspace_margin") is None:
                    names = _KINEMATICS.arm_joints[side]
                    info["workspace_margin"] = float(
                        min(
                            min(
                                joints[name] - _KINEMATICS.joints[name].lower,
                                _KINEMATICS.joints[name].upper - joints[name],
                            )
                            / (
                                _KINEMATICS.joints[name].upper
                                - _KINEMATICS.joints[name].lower
                            )
                            for name in names
                        )
                    )
                    info["manipulability"] = float(
                        __import__("numpy").prod(
                            __import__("numpy").linalg.svd(
                                _KINEMATICS.jacobian(joints, side, base.pose)[:3],
                                compute_uv=False,
                            )
                        )
                    )
                    info["metric_posture_source"] = (
                        "measured_current_when_goal_IK_unsolved"
                    )
                    assessment["workspace_margin"][side] = info["workspace_margin"]
                    assessment["manipulability"][side] = info["manipulability"]
            assessments.append(assessment)
        cache[key] = assessments
        while len(cache) > 64:
            cache.popitem(last=False)
    _enrich_robot_geometry.last_cache_key = key
    _enrich_robot_geometry.last_assessments = assessments
    for assessment in assessments:
        if assessment.get("valid"):
            snapshot = merge_feasibility(snapshot, assessment, registry)
    return snapshot


class _RobotEvidenceProvider:
    """Wire observations share the same center/IK/proxy producer as demonstrations."""

    def __init__(self, provider):
        self.provider = provider

    def snapshot(self, observation, registry, episode_id, timestamp):
        snapshot = self.provider.snapshot(observation, registry, episode_id, timestamp)
        self.last_robot_fields = observation.get("robot_fields") or {}
        return _enrich_robot_geometry(snapshot, self.last_robot_fields, registry)
