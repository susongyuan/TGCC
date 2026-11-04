import hashlib, numpy as np, torch
from .encoders import GraphBatch


def bucket(text, n):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % n


def tensorize(graph, timestamp, device="cpu", capacity=64, features=32):
    if features < 32:
        raise ValueError("features>=32 required for geometry/state schema")
    nodes = sorted(graph["nodes"], key=lambda n: n["entity_id"])
    if len(nodes) > capacity:
        raise ValueError(
            "explicit task-relevance selection required; refusing silent truncation"
        )
    roles = np.zeros((1, 1, capacity), np.int64)
    x = np.zeros((1, 1, capacity, features), np.float32)
    sem = np.zeros((1, 1, capacity), np.int64)
    rel = np.zeros((1, 1, capacity, capacity), np.int64)
    mask = np.zeros((1, 1, capacity), bool)
    idx = {n["entity_id"]: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        roles[0, 0, i] = role_index(n)
        if n["timestamp"] > timestamp or n["last_seen"] > timestamp:
            raise ValueError("future node")
        sem[0, 0, i] = bucket(
            n["semantic_type"]
            + ":"
            + str((n.get("attributes", {}).get("goal") or [""])[0]),
            256,
        )
        mask[0, 0, i] = n["validity"]
        x[0, 0, i, :4] = [
            n["confidence"],
            timestamp - n["last_seen"],
            n["semantic_type"] == "left_ee",
            n["semantic_type"] == "right_ee",
        ]
        if n["pose"] is not None:
            t = np.asarray(n["pose"])
            x[0, 0, i, 4:7] = t[:3, 3]
            x[0, 0, i, 7:13] = np.r_[t[:3, 0], t[:3, 1]]
        if features >= 32:
            attrs = n.get("attributes", {})
            if attrs.get("extent_known") and "nominal_extent_m" in attrs:
                extent = np.asarray(attrs["nominal_extent_m"], dtype=float)
                if (
                    extent.shape != (3,)
                    or not np.isfinite(extent).all()
                    or (extent <= 0).any()
                ):
                    raise ValueError("valid metric nominal extent required")
                x[0, 0, i, 17:20] = extent * 0.2
                x[0, 0, i, 20] = 1.0
                x[0, 0, i, 21] = float(bool(attrs.get("assumed_static")))
            state = attrs.get("state_index")
            if state is not None and 0 <= state < 6:
                x[0, 0, i, 17 + state] = 1
            for j, key in enumerate(("cooked", "open", "toggled_on")):
                if type(attrs.get(key)) is bool:
                    x[0, 0, i, 23 + j] = float(attrs[key])
                    x[0, 0, i, 26 + j] = 1
            x[0, 0, i, 29] = float(attrs.get("state") == "VERIFIED_DONE")
            identity = bucket(n["entity_id"], 65536) / 65536 * 2 * np.pi
            x[0, 0, i, 30:32] = [np.sin(identity), np.cos(identity)]
        if n["semantic_type"] == "feasibility_summary_v1":
            attrs = n.get("attributes", {})
            geometry = attrs.get("geometry9")
            if geometry is not None:
                x[0, 0, i, 4:13] = geometry
            # Node-typed schema; other semantic types keep every existing slot unchanged.
            x[0, 0, i, 17:29] = 0
            for j, key in enumerate(FEASIBILITY_FIELDS):
                value = attrs.get("feasibility", {}).get(key)
                if value is not None:
                    x[0, 0, i, 17 + j] = float(value)
                    x[0, 0, i, 23 + j] = 1
        age = timestamp - n["timestamp"]
        x[0, 0, i, 13:17] = [
            np.sin(age),
            np.cos(age),
            np.sin(age * 0.1),
            np.cos(age * 0.1),
        ]
    for e in graph["edges"]:
        if e["timestamp"] > timestamp:
            raise ValueError("future edge")
        if (
            e["validity"]
            and e.get("relative_attributes", {}).get("value", True) is not False
            and e["source_id"] in idx
            and e["target_id"] in idx
        ):
            a, b = idx[e["source_id"]], idx[e["target_id"]]
            rel[0, 0, a, b] = 1 + bucket(e["relation_type"], 31)
    return GraphBatch(
        *(torch.from_numpy(a).to(device) for a in (x, sem, rel, mask)),
        oracle=bool(graph.get("oracle", False))
        or any(n.get("source") == "oracle" for n in nodes)
        or any(e.get("source") == "oracle" for e in graph["edges"]),
        roles=torch.from_numpy(roles).to(device),
    )


# Local geometry must remain separately indexed by hand and target, never averaged.
def local_pairs(graph):
    return {
        (e["source_id"], e["target_id"]): e["relative_attributes"]["geometry9"]
        for e in graph["edges"]
        if e["relation_type"] == "relative_to" and e["validity"]
    }


def tensorize_local(graph, timestamp, device="cpu", capacity=64, features=32):
    from .state import Entity, Edge

    by_id = {n["entity_id"]: n for n in graph["nodes"]}
    # Keep explicit body/target/obstacle identities and masks, no world xyz in local tokens.
    nodes = [
        dict(n, pose=None)
        for n in graph["nodes"]
        if n["semantic_type"] != "feasibility_summary_v1"
    ]
    edges = list(graph["edges"])
    for e in graph["edges"]:
        if e["relation_type"] != "relative_to" or not e["validity"]:
            continue
        hand = by_id[e["source_id"]]
        target = by_id[e["target_id"]]
        if not hand["validity"] or not target["validity"]:
            continue
        key = e["source_id"] + "::" + e["target_id"]
        nodes.append(
            {
                "entity_id": key,
                "semantic_type": hand["semantic_type"],
                "timestamp": e["timestamp"],
                "last_seen": min(hand["last_seen"], target["last_seen"]),
                "confidence": e["confidence"],
                "validity": True,
                "pose": None,
            }
        )
        edges.extend(
            [
                dict(
                    e,
                    source_id=hand["entity_id"],
                    target_id=key,
                    relation_type="local_reference",
                ),
                dict(
                    e,
                    source_id=key,
                    target_id=target["entity_id"],
                    relation_type="local_target",
                ),
            ]
        )
    nodes.extend(
        n
        for n in graph["nodes"]
        if n["semantic_type"] == "feasibility_summary_v1" and n["validity"]
    )
    batch = tensorize(
        {"nodes": nodes, "edges": edges}, timestamp, device, capacity, features
    )
    batch.oracle = (
        bool(graph.get("oracle", False))
        or any(n.get("source") == "oracle" for n in graph["nodes"])
        or any(e.get("source") == "oracle" for e in graph["edges"])
    )
    lookup = {
        e["source_id"] + "::" + e["target_id"]: e
        for e in graph["edges"]
        if e["relation_type"] == "relative_to" and e["validity"]
    }
    # Only synthetic relative-pair nodes carry this flag. Actual body anchors
    # retain their role without being filtered by the current subgoal target.
    batch.pair_nodes = torch.zeros_like(batch.valid)
    for i, node in enumerate(sorted(nodes, key=lambda n: n["entity_id"])):
        if node["entity_id"] not in lookup:
            continue
        batch.pair_nodes[0, 0, i] = True
        attrs = lookup[node["entity_id"]]["relative_attributes"]
        batch.features[0, 0, i, 4:13] = torch.tensor(attrs["geometry9"], device=device)
        velocity = attrs.get("relative_velocity")
        if velocity is not None:
            batch.features[0, 0, i, 17:20] = torch.tensor(velocity, device=device)
            batch.features[0, 0, i, 20] = 1
        for offset, key in enumerate(("contact_state", "grasp_state", "reachability")):
            if type(attrs.get(key)) is bool:
                batch.features[0, 0, i, 21 + offset] = float(attrs[key])
                batch.features[0, 0, i, 24 + offset] = 1
    return batch


def tensorize_history(graph, device="cpu", capacity=64, local=False):
    # Reconstruct bounded deltas with the state immediately preceding the oldest delta.
    frames = []
    state = dict(graph.history_base)
    for delta in graph.get_history():
        for node in delta["changed_nodes"]:
            state[node["entity_id"]] = node
        frame = dict(
            timestamp=delta["timestamp"],
            nodes=list(state.values()),
            edges=delta["edges"],
            oracle=delta.get("oracle", False),
        )
        if local:
            # Local pair identities are ordered (EE,target), never shared between hands.
            pairs = [
                n["entity_id"]
                for n in frame["nodes"]
                if n["semantic_type"] != "feasibility_summary_v1"
            ]
            byid = {n["entity_id"]: n for n in frame["nodes"]}
            for e in frame["edges"]:
                if e["relation_type"] != "relative_to" or not e["validity"]:
                    continue
                a, b = byid[e["source_id"]], byid[e["target_id"]]
                if a["validity"] and b["validity"]:
                    pairs.append(e["source_id"] + "::" + e["target_id"])
            pairs.extend(
                n["entity_id"]
                for n in frame["nodes"]
                if n["semantic_type"] == "feasibility_summary_v1" and n["validity"]
            )
            frame["slots"] = pairs
        else:
            frame["slots"] = [n["entity_id"] for n in frame["nodes"]]
        frames.append(frame)
    if not frames:
        raise ValueError("history is empty; capture observation first")
    ids = sorted({key for f in frames for key in f["slots"]})
    if len(ids) > capacity:
        raise ValueError(
            "history union exceeds capacity; explicit relevance selection required"
        )
    # Pack pure observed features on CPU and transfer each full history once.
    # No gradients exist in this numeric state adapter; learned encoders remain
    # fully differentiable after the returned GraphBatch is consumed.
    features = np.zeros((1, len(frames), capacity, 32), dtype=np.float32)
    semantics = np.zeros((1, len(frames), capacity), dtype=np.int64)
    relations = np.zeros((1, len(frames), capacity, capacity), dtype=np.int64)
    valid = np.zeros((1, len(frames), capacity), dtype=bool)
    roles = np.zeros((1, len(frames), capacity), dtype=np.int64)
    pair_nodes = np.zeros((1, len(frames), capacity), dtype=bool)
    oracle = False
    index = {key: i for i, key in enumerate(ids)}
    for t, frame in enumerate(frames):
        fn = tensorize_local if local else tensorize
        batch = fn(frame, frame["timestamp"], "cpu", capacity)
        oracle = oracle or batch.oracle
        present = sorted(frame["slots"])
        count = len(present)
        if present:
            new = np.array([index[key] for key in present], dtype=np.int64)
            features[0, t, new] = batch.features.numpy()[0, 0, :count]
            semantics[0, t, new] = batch.semantics.numpy()[0, 0, :count]
            roles[0, t, new] = batch.roles.numpy()[0, 0, :count]
            if batch.pair_nodes is not None:
                pair_nodes[0, t, new] = batch.pair_nodes.numpy()[0, 0, :count]
            valid[0, t, new] = batch.valid.numpy()[0, 0, :count]
            relations[0, t][np.ix_(new, new)] = batch.relations.numpy()[
                0, 0, :count, :count
            ]
    return GraphBatch(
        *(
            torch.from_numpy(value).to(device)
            for value in (features, semantics, relations, valid)
        ),
        oracle=bool(oracle),
        roles=torch.from_numpy(roles).to(device),
        pair_nodes=torch.from_numpy(pair_nodes).to(device) if local else None,
    )


def task_candidate_tensors(task_spec, manager=None, *, device="cpu", width=256):
    """Fixed K/order for BOTH train and rollout; continuous query features, not text token IDs.
    Training next labels must use TaskSpec.next_index(id). Goals share state head slots,
    but are never phase labels and are excluded from next when a plan is present.
    """
    import hashlib, json
    from .task import load_task_spec

    spec = load_task_spec(task_spec)
    records = spec.nodes
    ids = spec.candidate_ids
    if width < 1:
        raise ValueError("positive query width required")
    values = []
    for key in ids:
        text = json.dumps(
            dict(id=key, record=records[key]), sort_keys=True, separators=(",", ":")
        )
        u = np.frombuffer(
            hashlib.shake_256(text.encode()).digest(width * 4), dtype="<u4"
        ).astype(np.float64)
        values.append((u / (2**32 - 1) * 2 - 1).astype(np.float32))
    queries = torch.from_numpy(np.stack(values)[None]).to(device)
    plan = any(n["kind"] == "plan" for n in records.values())
    legal_ids = (
        set(manager.legal())
        if manager is not None
        else {key for key in ids if not records[key].get("dependencies")}
    )
    legal = torch.tensor(
        [
            [
                key in legal_ids and (not plan or records[key]["kind"] == "plan")
                for key in ids
            ]
        ],
        dtype=torch.bool,
        device=device,
    )
    goal = torch.tensor(
        [[records[key]["kind"] == "goal" for key in ids]],
        dtype=torch.bool,
        device=device,
    )
    return dict(
        queries=queries,
        legal=legal,
        goal_query_mask=goal,
        candidate_ids=ids,
        task_spec_sha256=spec.sha256,
        feature_schema="tgr.feature32.v2.typed_feasibility",
    )


FEASIBILITY_FIELDS = (
    "left_reachability",
    "right_reachability",
    "workspace_margin",
    "manipulability",
    "collision_feasibility",
    "base_reposition_feasibility",
)


def role_index(node):
    kind = node["semantic_type"]
    roles = {
        "robot_base": 2,
        "base": 2,
        "robot_trunk": 3,
        "trunk": 3,
        "left_arm": 4,
        "right_arm": 5,
        "left_ee": 6,
        "right_ee": 7,
        "obstacle": 8,
        "local_obstacle": 8,
        "feasibility_summary_v1": 11,
    }
    if kind == "task_node":
        return 10 if node.get("attributes", {}).get("kind") == "plan" else 9
    return roles.get(kind, 1)


def attach_events(batch, events, timestamp, capacity=64):
    """Bounded source/time-labelled event tensor; no future or completed-skill inference."""
    from .task import STATES

    x = np.zeros((1, capacity, 32), np.float32)
    valid = np.zeros((1, capacity), bool)
    for i, event in enumerate(events[-capacity:]):
        if event["timestamp"] > timestamp:
            raise ValueError("future event")
        valid[0, i] = True
        x[0, i, :4] = [
            event.get("confidence", 0.0),
            np.log1p(max(0.0, timestamp - event["timestamp"])) / 4,
            bucket(event["predicate"], 256) / 255,
            event.get("source") == "oracle",
        ]
        old, new = event.get("previous_value"), event.get("current_value")
        if isinstance(old, str) and old in STATES:
            x[0, i, 4] = STATES.index(old) / 5
        if isinstance(new, str) and new in STATES:
            x[0, i, 5] = STATES.index(new) / 5
        if type(new) is bool:
            x[0, i, 6] = float(new)
            x[0, i, 7] = 1
        if event["predicate"] == "pose" and old is not None and new is not None:
            a = np.asarray(old, dtype=float)
            b = np.asarray(new, dtype=float)
            if a.shape == (4, 4) and b.shape == (4, 4):
                x[0, i, 8:11] = (b[:3, 3] - a[:3, 3]) * 0.2
                x[0, i, 11:17] = np.r_[b[:3, 0] - a[:3, 0], b[:3, 1] - a[:3, 1]]
    device = batch.features.device
    batch.events = torch.from_numpy(x).to(device)
    batch.event_valid = torch.from_numpy(valid).to(device)
    return batch
