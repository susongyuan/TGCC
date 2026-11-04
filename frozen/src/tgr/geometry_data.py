"""Nominal metric asset boxes and fixed scene proxies; no dynamic state guesses."""

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import numpy as np
from .geometry import transform
from .state import Entity, Edge

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "reports/asset_bounds_readonly_audit.json"
STATIC_NAMES = ("bottom_cabinet_no_top_gjeoer_0", "bottom_cabinet_no_top_rkgjer_0")


@lru_cache(maxsize=1)
def asset_records():
    data = json.loads(CATALOG.read_text())
    if data["units"] != "metres":
        raise ValueError("metric asset bounds required")
    return {(row["category"], row["model"]): row for row in data["records"]}


def _spec(args):
    row = asset_records().get((args.get("category"), args.get("model")))
    if row is None:
        return None
    scale = np.asarray(args.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    if scale.shape == ():
        scale = np.repeat(scale, 3)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("positive dimensionless asset scale required")
    return {
        "category": args["category"],
        "model": args["model"],
        "extent_m": (scale * np.asarray(row["native_bbox_extent_m"])).tolist(),
        "center_offset_m": (
            scale * np.asarray(row["native_bbox_center_offset_from_root_m"])
        ).tolist(),
        "source_sha256": row["metadata_source_sha256"],
        "container": args["category"] in ("fridge", "microwave"),
        "geometry_source": "nominal_original_asset_metadata",
        "physical_verified": False,
    }


def geometry_from_metadata(meta):
    scene = meta["scene_file"]
    if isinstance(scene, str):
        scene = json.loads(scene)
    loaded = scene["objects_info"]["init_info"]
    mapping = scene["metadata"]["task"]["inst_to_name"]
    result = {"task_objects": {}, "static_objects": []}
    for canonical, name in mapping.items():
        item = loaded.get(name)
        if item is None:
            continue
        spec = _spec(item["args"])
        if spec is not None:
            result["task_objects"][canonical] = spec
    states = scene["state"]["registry"]["object_registry"]
    for name in STATIC_NAMES:
        item = loaded.get(name)
        if item is None:
            continue
        args = item["args"]
        if args.get("fixed_base") is not True or args.get("visual_only", False):
            raise ValueError("fixed loaded static proxy required")
        spec = _spec(args)
        if spec is None:
            raise ValueError("static proxy asset bounds missing")
        root = states[name]["root_link"]
        spec.update(
            instance_id="scene_static:" + name,
            root_world=transform(root["pos"], root["ori"]).tolist(),
            source_timestamp=0.0,
            assumed_static=True,
        )
        result["static_objects"].append(spec)
    return result


def _centered(node, spec):
    if node.pose is None:
        return node
    attrs = dict(node.attributes)
    if attrs.get("canonical_pose_kind") == "nominal_bbox_center":
        return node
    root = np.asarray(node.pose, dtype=float)
    centered = root.copy()
    centered[:3, 3] += root[:3, :3] @ np.asarray(spec["center_offset_m"])
    attrs.update(
        spec,
        original_root_world_pose=root.tolist(),
        canonical_pose_kind="nominal_bbox_center",
        nominal_extent_m=spec["extent_m"],
        extent_known=True,
        world_T_box=centered.tolist(),
    )
    return replace(node, pose=centered.tolist(), attributes=attrs)


def attach_geometry(snapshot, fields, registry):
    """One canonical center frame shared by all three views; static source time stays0."""
    spec = fields.get("geometry_spec") or {}
    task_specs = spec.get("task_objects", {})
    nodes = []
    for node in snapshot.entities:
        if node.instance_id in task_specs:
            node = _centered(node, task_specs[node.instance_id])
        elif (
            node.pose is not None
            and "aabb" in node.attributes
            and not node.semantic_type.startswith(("robot", "left_", "right_"))
        ):
            # Only diagnostic/wire inputs lacking the nominal model specification.
            attrs = dict(node.attributes)
            lo, hi = np.asarray(attrs["aabb"])
            if node.semantic_type not in (
                "robot_base",
                "base",
            ) and not node.instance_id.startswith("agent"):
                pose = np.asarray(node.pose).copy()
                pose[:3, 3] = (lo + hi) / 2
                box = np.eye(4)
                box[:3, 3] = pose[:3, 3]
                attrs.update(
                    canonical_pose_kind="measured_world_aabb_center",
                    nominal_extent_m=(hi - lo).tolist(),
                    extent_known=True,
                    world_T_box=box.tolist(),
                    geometry_source="actual_world_aabb",
                    container=node.semantic_type.startswith(
                        ("electric_refrigerator", "microwave")
                    ),
                    physical_verified=False,
                )
                node = replace(node, pose=pose.tolist(), attributes=attrs)
        nodes.append(node)
    existing = {node.entity_id for node in nodes}
    for item in spec.get("static_objects", []):
        name = item["instance_id"]
        eid = registry.resolve(name)
        if eid in existing:
            continue
        root = np.asarray(item["root_world"])
        center = root.copy()
        center[:3, 3] += root[:3, :3] @ np.asarray(item["center_offset_m"])
        attrs = dict(
            item,
            nominal_extent_m=item["extent_m"],
            extent_known=True,
            canonical_pose_kind="nominal_bbox_center",
            world_T_box=center.tolist(),
            provenance="initial_static_scene_proxy_not_fresh_observation",
        )
        nodes.append(
            Entity(
                eid,
                "local_obstacle",
                name,
                center.tolist(),
                "world",
                0.0,
                0.0,
                1.0,
                "oracle",
                True,
                attrs,
            )
        )
    edges = list(snapshot.edges)
    base = next(
        (node for node in nodes if node.semantic_type in ("base", "robot_base")), None
    )
    if base is not None:
        for node in nodes:
            if node.semantic_type == "local_obstacle":
                edges.append(
                    Edge(
                        base.entity_id,
                        node.entity_id,
                        "collision_context",
                        {},
                        snapshot.timestamp,
                        1.0,
                        True,
                        "oracle",
                    )
                )
    return replace(snapshot, entities=tuple(nodes), edges=tuple(edges))
