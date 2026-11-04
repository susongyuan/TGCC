import re, itertools


def sexpr(text):
    tokens = re.findall(r"[^\s()]+|[()]", re.sub(r";[^\n]*", "", text))
    stack = []
    root = []
    cur = root
    for t in tokens:
        if t == "(":
            new = []
            cur.append(new)
            stack.append(cur)
            cur = new
        elif t == ")":
            if not stack:
                raise ValueError("bad BDDL parentheses")
            cur = stack.pop()
        else:
            cur.append(t)
    if stack or len(root) != 1:
        raise ValueError("bad BDDL")
    return root[0]


def parse_bddl(text):
    root = sexpr(text)
    sections = {
        x[0]: x[1:]
        for x in root
        if isinstance(x, list) and x and isinstance(x[0], str) and x[0].startswith(":")
    }
    obj = {}
    pending = []
    ts = iter(sections[":objects"])
    for t in ts:
        if t == "-":
            kind = next(ts)
            for name in pending:
                obj[name] = kind
            pending = []
        else:
            pending.append(t)
    if pending:
        raise ValueError("untyped objects unsupported")

    def expand(x, bindings):
        if x[0] == "and":
            return [p for child in x[1:] for p in expand(child, bindings)]
        if x[0] == "forall":
            var, sep, kind = x[1]
            if sep != "-":
                raise ValueError("bad quantified variable")
            return [
                p
                for name, typ in obj.items()
                if typ == kind
                for p in expand(x[2], dict(bindings, **{var: name}))
            ]
        if x[0] in ("or", "exists", "imply"):
            raise ValueError("unsupported goal logical operator; do not flatten")
        if x[0] == "not":
            return [("not", *p) for p in expand(x[1], bindings)]
        predicate = {
            "ontop": "on_top",
            "inroom": "in_room",
            "toggledon": "toggled_on",
        }.get(x[0], x[0])
        return [(predicate, *(bindings.get(t, t) for t in x[1:]))]

    goals = [p for x in sections[":goal"] for p in expand(x, {})]
    return {"objects": obj, "initial": sections.get(":init", []), "goals": goals}


STATES = ("PENDING", "ACTIVE", "BELIEVED_DONE", "BLOCKED", "UNKNOWN", "VERIFIED_DONE")


class TaskManager:
    def __init__(self, goals, dependencies):
        self.goals = goals
        self.dependencies = {k: tuple(v) for k, v in dependencies.items()}
        if any(
            d not in self.dependencies
            for deps in self.dependencies.values()
            for d in deps
        ):
            raise ValueError("unknown dependency")

        def visit(k, path):
            if k in path:
                raise ValueError("dependency cycle")
            for d in self.dependencies[k]:
                visit(d, path | {k})

        for k in self.dependencies:
            visit(k, set())
        self.node_goals = {}
        self.entity_ids = {}
        self.reset()

    @classmethod
    def from_bddl(cls, spec, registry):
        from pathlib import Path

        if isinstance(spec, Path) or (
            isinstance(spec, str) and not spec.lstrip().startswith("(")
        ):
            spec = load_task_spec(spec)
        if isinstance(spec, TaskSpec) or (
            isinstance(spec, dict) and "schema_version" in spec
        ):
            return cls.from_task_spec(load_task_spec(spec), registry)
        spec = parse_bddl(spec) if isinstance(spec, str) else spec
        return cls.from_task_spec(construct_task_spec(spec), registry)

    @classmethod
    def from_task_spec(cls, spec, registry):
        spec = load_task_spec(spec)
        a = spec.artifact
        records = spec.nodes
        m = cls(
            [tuple(n["predicate"]) for n in a["goals"]],
            {key: records[key].get("dependencies", []) for key in spec.candidate_ids},
        )
        m.task_spec = spec
        m.records = records
        m.node_goals = {n["id"]: tuple(n["predicate"]) for n in a["goals"]}
        m.verification_predicates = {
            key: tuple(n["predicate"])
            for key, n in records.items()
            if n.get("predicate")
        }
        m.entity_ids = {name: registry.resolve(name) for name in a["objects"]}
        m.initial_predicates = a.get("initial", [])
        return m

    def reset(self):
        self.states = {key: "PENDING" for key in self.dependencies}
        self.evidence = {}
        self.changed_at = {key: 0.0 for key in self.dependencies}
        self.history = []
        self.base_targets = {}
        self.version = 0

    def update(
        self,
        predictions,
        evidence,
        timestamp=None,
        restricted=False,
        expected_version=None,
    ):
        import math

        if expected_version is not None and expected_version != self.version:
            raise ValueError("stale task-state write version")
        # Validate full transaction before mutating any states.
        for key, state in predictions.items():
            if key not in self.states or state not in STATES[:-1]:
                raise ValueError("invalid predicted state")
        for key, ev in evidence.items():
            if key not in self.states:
                raise ValueError("unknown task node")
            if restricted and ev.get("source") == "oracle":
                raise ValueError("Oracle task evidence forbidden")
            if timestamp is not None and (
                not math.isfinite(ev.get("timestamp", float("nan")))
                or ev["timestamp"] > timestamp
            ):
                raise ValueError("untimed/future evidence")
            if "value" in ev and type(ev["value"]) is not bool:
                raise ValueError("predicate value must be boolean")
            if (
                not math.isfinite(ev.get("confidence", 1.0))
                or not 0 <= ev.get("confidence", 1.0) <= 1
            ):
                raise ValueError("invalid evidence confidence")
        proposed = self.states.copy()
        proposed.update(predictions)
        for key, ev in evidence.items():
            trusted = (
                ev.get("verified") is True
                and ev.get("evidence_id")
                and ev.get("source") in ("observation", "task_feedback", "oracle")
                and type(ev.get("value")) is bool
                and 0.5 <= ev.get("confidence", 1.0) <= 1.0
            )
            if (
                trusted
                and ev.get("value")
                and getattr(self, "records", {}).get(key, {}).get("action_type")
                == "REPOSITION_BASE"
            ):
                trusted = (
                    timestamp is None or ev.get("timestamp") == timestamp
                ) and self._base_pose_verified(key, ev)
            if trusted:
                old = self.evidence.get(key)
                if old is not None and ev.get("timestamp", 0) < old.get("timestamp", 0):
                    raise ValueError("stale evidence replay")
                proposed[key] = "VERIFIED_DONE" if ev["value"] else "PENDING"
            else:
                proposed[key] = "PENDING"
        for key, value in proposed.items():
            if self.states[key] != value:
                self.history.append(
                    dict(
                        node=key,
                        previous=self.states[key],
                        current=value,
                        timestamp=timestamp,
                    )
                )
                self.changed_at[key] = timestamp if timestamp is not None else 0.0
        if self.states != proposed:
            self.version += 1
        self.states = proposed
        self.history = self.history[-2048:]
        for key, ev in evidence.items():
            self.evidence[key] = dict(ev)
        return self.states.copy()

    def legal(self):
        records = getattr(self, "records", {})
        has_plan = any(n["kind"] == "plan" for n in records.values())
        result = []
        for key, deps in self.dependencies.items():
            record = records.get(key, {})
            if has_plan and record.get("kind") != "plan":
                continue
            if self.states[key] in ("VERIFIED_DONE", "BLOCKED"):
                continue
            accepted = {"VERIFIED_DONE"}
            if record.get("dependency_policy") == "believed_or_verified":
                accepted.add("BELIEVED_DONE")
            if all(self.states[d] in accepted for d in deps):
                result.append(key)
        return result

    def terminal_success(self):
        # Only formal BDDL predicates determine this graph-level completion claim.
        return bool(self.node_goals) and all(
            self.states[k] == "VERIFIED_DONE" for k in self.node_goals
        )

    def set_base_target(
        self, key, pose, timestamp, translation_tolerance=0.03, rotation_tolerance=0.1
    ):
        import math
        from .geometry import relative

        if (
            getattr(self, "records", {}).get(key, {}).get("action_type")
            != "REPOSITION_BASE"
        ):
            raise ValueError("not a fixed REPOSITION_BASE candidate")
        relative(pose, pose)
        if (
            not math.isfinite(timestamp)
            or not 0 < translation_tolerance <= 0.2
            or not 0 < rotation_tolerance <= 0.5
        ):
            raise ValueError("invalid base target tolerances/time")
        self.base_targets[key] = (
            pose,
            timestamp,
            translation_tolerance,
            rotation_tolerance,
        )

    def _base_pose_verified(self, key, evidence):
        import numpy as np
        from .geometry import relative

        target = self.base_targets.get(key)
        if (
            target is None
            or evidence.get("observed_base_pose") is None
            or evidence.get("reference_frame") != "world"
        ):
            return False
        pose, issued, position_tol, rotation_tol = target
        if evidence.get("timestamp", -float("inf")) <= issued:
            return False
        try:
            x = relative(pose, evidence["observed_base_pose"])
            r = np.column_stack((x[3:6], x[6:9], np.cross(x[3:6], x[6:9])))
            angle = float(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
            return bool(np.linalg.norm(x[:3]) <= position_tol and angle <= rotation_tol)
        except (ValueError, TypeError):
            return False


def evidence_from_snapshot(manager, snapshot):
    import hashlib

    nodes = {n.entity_id: n for n in snapshot.entities}
    out = {}
    # Only a current explicit predicate value is evidence. Missing state means UNKNOWN.
    for key, goal in getattr(
        manager, "verification_predicates", manager.node_goals
    ).items():
        neg = goal[0] == "not"
        pred = goal[1] if neg else goal[0]
        args = goal[2:] if neg else goal[1:]
        ids = [manager.entity_ids[a] for a in args]
        value = None
        source = None
        confidence = 0.0
        t = snapshot.timestamp
        if len(ids) == 1 and ids[0] in nodes:
            n = nodes[ids[0]]
            if n.validity and n.timestamp == t and type(n.attributes.get(pred)) is bool:
                value = n.attributes[pred]
                source = n.source
                confidence = n.confidence
        if value is None and len(ids) == 1:
            for event in snapshot.events:
                if (
                    event.entity_id == ids[0]
                    and event.predicate == pred
                    and event.timestamp == t
                    and type(event.current_value) is bool
                ):
                    value = event.current_value
                    source = event.source
                    confidence = event.confidence
        if len(ids) == 2:
            for edge in snapshot.edges:
                if (
                    edge.source_id == ids[0]
                    and edge.target_id == ids[1]
                    and edge.relation_type == pred
                    and (
                        edge.validity or edge.relative_attributes.get("value") is False
                    )
                    and edge.timestamp == t
                ):
                    value = edge.relative_attributes.get("value", True)
                    source = edge.source
                    confidence = edge.confidence
        if value is not None:
            if neg:
                value = not value
            token = hashlib.sha256(
                f"{snapshot.episode_id}:{t}:{key}:{value}".encode()
            ).hexdigest()[:24]
            out[key] = dict(
                verified=True,
                value=value,
                source=source,
                confidence=confidence,
                evidence_id=token,
                timestamp=t,
            )
        else:
            out[key] = dict(verified=False, source="unknown", timestamp=t)
    return out


def task_snapshot(manager, registry, episode, timestamp):
    from .state import Entity, Edge, ObservationSnapshot

    nodes = []
    edges = []
    oracle = False
    for name, eid in manager.entity_ids.items():
        nodes.append(
            Entity(
                eid,
                "entity_reference",
                name,
                None,
                "world",
                timestamp,
                timestamp,
                1.0,
                "task_definition",
                True,
            )
        )
    for key, state in manager.states.items():
        eid = registry.resolve("task:" + key)
        ev = manager.evidence.get(key, {})
        source = ev.get("source", "prediction")
        oracle = oracle or source == "oracle"
        attrs = dict(
            state=state,
            state_index=STATES.index(state),
            goal=manager.node_goals.get(key),
            evidence=ev,
            changed_at=manager.changed_at[key],
        )
        record = getattr(manager, "records", {}).get(key, {})
        attrs.update(
            kind=record.get("kind", "goal"),
            terminal=record.get("terminal", key in manager.node_goals),
            text=record.get("text", key),
            candidate_index=list(manager.states).index(key),
            action_type=record.get("action_type"),
        )
        nodes.append(
            Entity(
                eid,
                "task_node",
                key,
                None,
                "world",
                timestamp,
                timestamp,
                ev.get("confidence", 0.0),
                source,
                True,
                attrs,
            )
        )
        goal = record.get("entities", manager.node_goals.get(key, ()))
        for name in goal:
            if name in manager.entity_ids:
                edges.append(
                    Edge(
                        eid,
                        manager.entity_ids[name],
                        "relevant_entity",
                        {},
                        timestamp,
                        1.0,
                        True,
                        source,
                    )
                )
        for dep in manager.dependencies[key]:
            edges.append(
                Edge(
                    registry.resolve("task:" + dep),
                    eid,
                    "prerequisite",
                    {},
                    timestamp,
                    1.0,
                    True,
                    "task_definition",
                )
            )
    return ObservationSnapshot(
        episode, timestamp, tuple(nodes), tuple(edges), oracle=oracle
    )


# Shared interface for James (artifact producer) / Meitner (consumer).
# v1: candidates = explicit ordered goal IDs + training-derived plan IDs.
# next is the index in candidate_order (NOT a 13-class phase ordinal).
TASK_SPEC_SCHEMA = "tgr.task_spec.v1"


class TaskSpec:
    """Immutable canonical artifact. No online GT or per-frame labels belong here."""

    def __init__(self, artifact):
        import json, hashlib

        data = json.loads(json.dumps(artifact))
        _validate_task_spec(data)
        self._json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        self.sha256 = hashlib.sha256(self._json.encode()).hexdigest()

    @property
    def artifact(self):
        import json

        return json.loads(self._json)

    @property
    def candidate_ids(self):
        return tuple(self.artifact["candidate_order"])

    @property
    def K(self):
        return len(self.candidate_ids)

    @property
    def nodes(self):
        a = self.artifact
        return {
            n["id"]: dict(n, kind=kind)
            for kind in ("goal", "plan")
            for n in a[kind + "s" if kind == "goal" else "plan"]
        }

    def next_index(self, candidate_id):
        if candidate_id not in self.nodes or self.nodes[candidate_id]["kind"] != "plan":
            raise ValueError(
                "next supervision requires explicit plan candidate ID, not a goal/phase ordinal"
            )
        return self.candidate_ids.index(candidate_id)

    def assert_compatible(self, sha256, candidate_ids=None):
        if sha256 != self.sha256 or (
            candidate_ids is not None and tuple(candidate_ids) != self.candidate_ids
        ):
            raise ValueError(
                "TaskSpec fingerprint/order mismatch: training and inference must share artifact"
            )


def _validate_task_spec(a):
    if a.get("schema_version") != TASK_SPEC_SCHEMA:
        raise ValueError("unsupported TaskSpec schema")
    allowed = {
        "schema_version",
        "task_name",
        "objects",
        "initial",
        "goals",
        "plan",
        "candidate_order",
        "provenance",
    }
    if set(a) - allowed:
        raise ValueError(
            "TaskSpec cannot contain labels, timestamps, rollout observations or unknown fields"
        )
    if not isinstance(a.get("task_name"), str) or not a["task_name"]:
        raise ValueError("task_name required")
    if not isinstance(a.get("objects"), dict) or not a["objects"]:
        raise ValueError("canonical BDDL objects required")
    goals = a.get("goals")
    plan = a.get("plan")
    if not isinstance(goals, list) or not goals or not isinstance(plan, list):
        raise ValueError("goal/plan lists required")
    nodes = {}
    for kind, records in (("goal", goals), ("plan", plan)):
        for n in records:
            fields = {
                "id",
                "text",
                "entities",
                "dependencies",
                "dependency_policy",
                "predicate",
                "terminal",
                "action_type",
            }
            if not isinstance(n, dict) or set(n) - fields:
                raise ValueError("invalid candidate schema / per-frame data forbidden")
            if not isinstance(n.get("id"), str) or not n["id"] or n["id"] in nodes:
                raise ValueError("duplicate/empty candidate ID")
            if not isinstance(n.get("text"), str) or not n["text"]:
                raise ValueError("candidate text required")
            if type(n.get("terminal")) is not bool or n["terminal"] != (kind == "goal"):
                raise ValueError("only BDDL goal nodes are terminal")
            if n.get("action_type") == "REPOSITION_BASE" and kind != "plan":
                raise ValueError("base reposition is nonterminal plan only")
            if kind == "goal" and not n.get("predicate"):
                raise ValueError("goal predicate required")
            if n.get("dependency_policy", "verified_only") not in (
                "verified_only",
                "believed_or_verified",
            ):
                raise ValueError("bad dependency policy")
            if (
                kind == "goal"
                and n.get("dependency_policy", "verified_only") != "verified_only"
            ):
                raise ValueError("goal dependencies require verification")
            if not isinstance(n.get("entities", []), list) or any(
                e not in a["objects"] for e in n.get("entities", [])
            ):
                raise ValueError("noncanonical target entity")
            pred = n.get("predicate")
            if pred is not None:
                if not isinstance(pred, list) or not all(
                    isinstance(x, str) for x in pred
                ):
                    raise ValueError("invalid predicate")
                args = pred[2:] if pred and pred[0] == "not" else pred[1:]
                if not args or any(x not in a["objects"] for x in args):
                    raise ValueError("predicate entity not in BDDL objects")
            if not isinstance(n.get("dependencies", []), list) or len(
                set(n.get("dependencies", []))
            ) != len(n.get("dependencies", [])):
                raise ValueError("invalid dependencies")
            nodes[n["id"]] = n
    order = a.get("candidate_order")
    if (
        not isinstance(order, list)
        or len(order) != len(nodes)
        or len(set(order)) != len(order)
        or set(order) != set(nodes)
    ):
        raise ValueError("candidate_order must list each goal/plan node exactly once")

    def visit(k, path):
        if k in path:
            raise ValueError("dependency cycle")
        for dep in nodes[k].get("dependencies", []):
            if dep not in nodes:
                raise ValueError("unknown dependency")
            visit(dep, path | {k})

    for key in nodes:
        visit(key, set())
    if plan:
        p = a.get("provenance", {})
        if (
            p.get("source") != "training_demonstration_derived"
            or p.get("split") != "train"
            or not p.get("episode_ids")
        ):
            raise ValueError(
                "plan vocabulary must be derived from identified training demonstrations only"
            )
        # This is a provenance contract, not a forensic guarantee that source annotations are honest.
        if p.get("uses_test_gt", False):
            raise ValueError("test GT prohibited")


def construct_task_spec(
    bddl_spec,
    plan_candidates=(),
    *,
    task_name="task",
    provenance=None,
    candidate_order=None,
):
    """Construct a task-agnostic template from TRAIN-derived candidate records.
    No hardcoded 13 phases, no automatic sequence dependencies/completion inference.
    Data owner writes the returned artifact; runtime never mines test observations.
    """
    spec = parse_bddl(bddl_spec) if isinstance(bddl_spec, str) else bddl_spec
    goals = []
    for pred in spec["goals"]:
        pred = list(pred)
        args = pred[2:] if pred[0] == "not" else pred[1:]
        goals.append(
            dict(
                id="goal:" + ":".join(pred),
                text=" ".join(pred),
                predicate=pred,
                entities=list(args),
                dependencies=[],
                terminal=True,
            )
        )
    plan = []
    for node in plan_candidates:
        n = dict(node)
        n.setdefault("entities", [])
        n.setdefault("dependencies", [])
        n.setdefault("terminal", False)
        n.setdefault("dependency_policy", "verified_only")
        plan.append(n)
    artifact = dict(
        schema_version=TASK_SPEC_SCHEMA,
        task_name=task_name,
        objects=spec["objects"],
        initial=spec.get("initial", []),
        goals=goals,
        plan=plan,
        provenance=provenance or {"source": "bddl_definition"},
        candidate_order=candidate_order or [n["id"] for n in goals + plan],
    )
    return TaskSpec(artifact)


def load_task_spec(artifact, *, bddl_spec=None, expected_sha256=None):
    """Same JSON artifact loader for training and inference; fingerprint is semantic JSON SHA."""
    import json
    from pathlib import Path

    if isinstance(artifact, TaskSpec):
        spec = artifact
    else:
        a = (
            json.loads(Path(artifact).read_text())
            if isinstance(artifact, (str, Path))
            else artifact
        )
        spec = TaskSpec(a)
    if expected_sha256 is not None:
        spec.assert_compatible(expected_sha256)
    if bddl_spec is not None:
        b = parse_bddl(bddl_spec) if isinstance(bddl_spec, str) else bddl_spec
        if spec.artifact["objects"] != b["objects"] or sorted(
            tuple(n["predicate"]) for n in spec.artifact["goals"]
        ) != sorted(tuple(g) for g in b["goals"]):
            raise ValueError("artifact terminal predicates differ from official BDDL")
    return spec
