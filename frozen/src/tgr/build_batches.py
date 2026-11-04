"""Audited prepared-tensor packer and official metadata inspection (CPU only).

No RGB->oracle invention, no 256D->61D truncation, no locally fitted normalizer.
`build` consumes synchronized tensors produced by a verified upstream adapter;
`inspect` inventories the actual parquet/meta/annotation without fabricating labels.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from .data import validate_manifest

GRAPHS = ("global_graph", "local_graph", "task_graph")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, path):
    p = (Path(root) / path).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes manifest directory")
    return p


def task_fields(keys, values):
    """Decode only explicit official task_obs_keys widths; grasp enum stays numeric.

    These are privileged poses/grasp values, not cooked/open/inside/on_top labels.
    Metadata scene_file is INITIAL state, never per-frame truth.
    """
    import numpy as np

    numeric = np.asarray(values, dtype=np.float64)
    if (
        numeric.ndim != 1
        or not np.isfinite(numeric).all()
        or (np.abs(numeric) > np.finfo(np.float32).max).any()
    ):
        raise ValueError("nonfinite task info")
    fields = {}
    offset = 0
    for key in keys:
        suffix = key.rsplit("_", 1)[-1]
        width = 3 if key.endswith(("_pos", "_ori_cos", "_ori_sin")) else 1
        if not key.endswith(
            (
                "_real",
                "_pos",
                "_ori_cos",
                "_ori_sin",
                "_in_gripper_left",
                "_in_gripper_right",
            )
        ):
            raise ValueError("unrecognized task observation key: " + key)
        if offset + width > len(values):
            raise ValueError("task observation width mismatch")
        fields[key] = [float(v) for v in values[offset : offset + width]]
        offset += width
    if offset != len(values):
        raise ValueError("task observation trailing values")
    return fields


def skill_target(annotation, frame):
    """Active annotated skill is supervision ONLY, not physical completion truth.

    Gaps remain masked; interval ends are exclusive. Never derive cooked/open
    from skill names or completed-previous-skill from future annotation boundaries.
    """
    hits = [
        s
        for s in annotation["skill_annotation"]
        if s["frame_duration"][0] <= frame < s["frame_duration"][1]
    ]
    if len(hits) > 1:
        raise ValueError("overlapping skill annotation")
    return {
        "mask": bool(hits),
        "skill_idx": hits[0]["skill_idx"] if hits else -1,
        "source": "human_skill_annotation",
        "physical_state_labels_available": False,
        "supervision_only": True,
    }


def validate_batch(batch, row, norm_sha):
    if batch.get("normalization_sha256") != norm_sha:
        raise ValueError("checkpoint normalization mismatch")
    if "sequence" in batch:
        packed = pack_sequence(batch["sequence"])
        for item in packed["sequence"]:
            if item["episode_id"] != row["episode_id"]:
                raise ValueError("sequence manifest episode mismatch")
            r = dict(row, timestamp=item["timestamp"], step_id=item["step_id"])
            validate_batch(item, r, norm_sha)
        return packed

    proof = batch.get("provenance", {})
    for key in (
        "adapter_revision",
        "proprio_layout_sha256",
        "action_layout_sha256",
        "label_evidence",
    ):
        if not proof.get(key):
            raise ValueError("missing audited provenance: " + key)
    if proof.get("normalization") != "checkpoint_owned_quantile":
        raise ValueError("unverified normalization")
    obs = batch.get("observation", {})
    if (
        not {
            "image",
            "image_mask",
            "state",
            "tokenized_prompt",
            "tokenized_prompt_mask",
        }
        <= obs.keys()
    ):
        raise ValueError("missing prepared official observation fields")
    # Official Observation uses image/image_mask (not RGB inferred labels).
    actions = batch.get("actions")
    if (
        not isinstance(actions, torch.Tensor)
        or actions.ndim != 3
        or actions.shape[:2] != (1, 32)
    ):
        raise ValueError("expected one 32-step normalized action chunk")
    if actions.shape[-1] == 23:
        batch["actions"] = torch.nn.functional.pad(actions, (0, 9))
    elif actions.shape[-1] != 32:
        raise ValueError("native23 or padded32 actions required")
    if not torch.isfinite(batch["actions"]).all() or torch.count_nonzero(
        batch["actions"][..., 23:]
    ):
        raise ValueError("nonfinite actions or nonzero padding")
    if obs["state"].shape != (1, 32) or not torch.isfinite(obs["state"]).all():
        raise ValueError(
            "official normalized and padded state required; no raw256 truncation"
        )
    conditions = batch.get("conditions", {})
    for key in GRAPHS:
        if key not in conditions:
            raise ValueError("missing real synchronized graph: " + key)
        g = conditions[key]
        x = g["features"]
        valid = g["valid"]
        sem = g["semantics"]
        rel = g["relations"]
        if (
            x.ndim != 4
            or x.shape[0] != 1
            or not 1 <= x.shape[1] <= 16
            or x.shape[-1] != 32
        ):
            raise ValueError("graph shape/history")
        b, t, n, _ = x.shape
        if (
            valid.shape != (b, t, n)
            or sem.shape != (b, t, n)
            or rel.shape != (b, t, n, n)
        ):
            raise ValueError("graph mask/topology shape")
        if (
            valid.dtype != torch.bool
            or sem.dtype != torch.long
            or rel.dtype != torch.long
        ):
            raise ValueError("graph index/mask dtype")
        if (
            not torch.isfinite(x).all()
            or (sem < 0).any()
            or (sem >= 256).any()
            or (rel < 0).any()
            or (rel >= 32).any()
        ):
            raise ValueError("invalid graph features/indices")
        times = proof.get("history_timestamps", {}).get(key)
        ids = proof.get("history_node_ids", {}).get(key)
        if (
            times is None
            or len(times) != t
            or any(
                not isinstance(v, (float, int))
                or not float("-inf") < v <= row["timestamp"]
                for v in times
            )
            or any(a >= b for a, b in zip(times, times[1:]))
        ):
            raise ValueError("missing/noncausal graph timestamps")
        if ids is None or len(ids) != t or any(len(frame) != n for frame in ids):
            raise ValueError("stable entity indexing required")
        for slot in zip(*ids):
            if len({v for v in slot if v is not None}) > 1:
                raise ValueError("entity identity changed in recurrent slot")
        for i, frame in enumerate(ids):
            occupied = [v for j, v in enumerate(frame) if bool(valid[0, i, j])]
            if None in occupied or len(set(occupied)) != len(occupied):
                raise ValueError("missing/duplicate valid entity ID")
        if bool(g.get("oracle", False)) != (row["graph_source"] == "oracle"):
            raise ValueError("oracle provenance mismatch")
    labels = batch.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(
            "missing trusted graph labels; RGB/proprio/actions alone insufficient"
        )
    required = ("state", "state_mask", "goal", "goal_mask", "next", "next_mask")
    if any(k not in labels for k in required):
        raise ValueError("missing trusted labels/masks")
    q = conditions.get("queries")
    legal = conditions.get("legal")
    if (
        q is None
        or q.ndim != 3
        or q.shape[0] != 1
        or q.shape[-1] != 256
        or legal is None
        or legal.shape != q.shape[:2]
        or legal.dtype != torch.bool
    ):
        raise ValueError("task queries/legal shape")
    size = q.shape[1]
    for k in ("state", "goal", "state_mask", "goal_mask"):
        if labels[k].shape != (1, size):
            raise ValueError("task label shape")
    for k in ("state_mask", "goal_mask", "next_mask"):
        if labels[k].dtype != torch.bool:
            raise ValueError("label mask must be bool")
    if labels["next"].shape != (1,) or labels["next_mask"].shape != (1,):
        raise ValueError("next label shape")
    for k, limit in (("state", 5), ("next", size)):
        y = labels[k][labels[k + "_mask"]]
        if y.dtype != torch.long or ((y < 0) | (y >= limit)).any():
            raise ValueError("label range/dtype")
    goals = labels["goal"][labels["goal_mask"]]
    if not torch.isfinite(goals).all() or ((goals < 0) | (goals > 1)).any():
        raise ValueError("goal label range")
    if labels["next_mask"].any() and not legal[0, int(labels["next"][0])]:
        raise ValueError("next target is illegal")
    batch["has_graph_supervision"] = any(
        bool(labels[k].any()) for k in ("state_mask", "goal_mask", "next_mask")
    )
    return batch


def build(
    manifest,
    checkpoint,
    output,
    split_by="episode_id",
    max_bytes=1024**3,
    min_free_bytes=50 * 1024**3,
):
    manifest = Path(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError("refuse to overwrite batch directory")
    norms = list(Path(checkpoint).glob("assets/**/norm_stats.json"))
    if len(norms) != 1:
        raise ValueError("ambiguous checkpoint normalizer")
    norm_sha = sha256(norms[0])
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    if not rows:
        raise ValueError("empty manifest")
    validate_manifest(rows, split_by=split_by)
    seen = set()
    packed = []
    total_bytes = 0
    # Validate all sources BEFORE creating output; no incomplete training manifest.
    for row in rows:
        key = (row["episode_id"], row["step_id"])
        if key in seen:
            raise ValueError("duplicate episode/frame")
        seen.add(key)
        if row["action_horizon"] != 32 or row["split"] not in (
            "train",
            "validation",
            "test",
        ):
            raise ValueError("horizon/split mismatch")
        src = inside(manifest.parent, row["tensor_path"])
        if row.get("tensor_sha256") != sha256(src):
            raise ValueError("missing or mismatched tensor source hash")
        batch = validate_batch(
            torch.load(src, weights_only=True, map_location="cpu"), row, norm_sha
        )
        total_bytes += src.stat().st_size
        if total_bytes > max_bytes:
            raise ValueError(
                "prepared-tensor pilot exceeds disk cap; use lazy video decoding, not dataset-wide float batches"
            )
        packed.append((dict(row), src))
        del batch
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < min_free_bytes + 2 * total_bytes:
        raise ValueError("refuse to violate 50GiB free-space reserve")
    output.mkdir(parents=True, exist_ok=False)
    out = []
    for i, (row, src) in enumerate(packed):
        if sha256(src) != row["tensor_sha256"]:
            raise ValueError("source changed after validation")
        batch = validate_batch(
            torch.load(src, weights_only=True, map_location="cpu"), row, norm_sha
        )
        if shutil.disk_usage(output).free < min_free_bytes + 2 * src.stat().st_size:
            raise ValueError("free-space reserve reached")
        name = f"batch_{i:08d}.pt"
        torch.save(batch, output / name)
        row.update(tensor_path=name, tensor_sha256=sha256(output / name))
        out.append(row)
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    receipt = {
        "status": "prepared_tensor_batches_validated",
        "batches": len(out),
        "split_by": split_by,
        "normalization_sha256": norm_sha,
        "oracle_upper_bound": any(r["graph_source"] == "oracle" for r in out),
        "raw_dataset_conversion_performed": False,
        "pilot_only": True,
        "source_bytes": total_bytes,
        "disk_cap_bytes": max_bytes,
        "min_free_bytes": min_free_bytes,
        "source_manifest_sha256": sha256(manifest),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def inspect(parquet, meta, annotation):
    import pyarrow.parquet as pq

    m = json.loads(Path(meta).read_text())
    a = json.loads(Path(annotation).read_text())
    t = pq.read_table(parquet)
    keys = m["task_obs_keys"]
    rows = t.to_pylist()
    dimensions = {
        k: sorted({len(r[k]) for r in rows})
        for k in (
            "observation.state",
            "observation.task_info",
            "action",
            "observation.cam_rel_poses",
        )
    }
    decoded = [task_fields(keys, r["observation.task_info"]) for r in rows]
    cfg = json.loads(m["config"]) if isinstance(m["config"], str) else m["config"]
    result = {
        "rows": len(rows),
        "dimensions": dimensions,
        "task_obs_keys": keys,
        "all_task_info_decoded": True,
        "frame_synchronized_privileged_fields": [
            "object existence",
            "world position",
            "RPY cosine/sine",
            "numeric grasp enum",
        ],
        "physical_predicates_not_recorded": [
            "cooked",
            "open",
            "toggled_on",
            "inside",
            "on_top",
        ],
        "initial_scene_file_is_not_per_frame_truth": True,
        "skill_annotation_count": len(a["skill_annotation"]),
        "annotation_frame_extent": a["meta_data"],
        "source_hashes": {str(p): sha256(p) for p in (parquet, meta, annotation)},
        "scene_id": cfg.get("scene", {}).get("scene_instance"),
        "proprio_obs_order": cfg["robots"][0]["proprio_obs"],
        "training_ready": False,
        "blockers": [
            "verified 256D proprio layout -> official baseline adapter",
            "decoded synchronized RGB/depth/segmentation",
            "EE frame calibration and local graph",
            "trusted predicate/state labels or explicit partial-label masks",
            "non-oracle inference state provider for restricted baseline comparison",
        ],
        "notes": [
            "Skill names are active-skill labels only, not proof of physical predicate success.",
            "Privileged task_info supports oracle pose supervision only, never pass it to restricted evaluation.",
            "No joint efforts used; no forward-looking completion labels synthesized.",
        ],
    }
    result["annotation_coverage_frames"] = sum(
        skill_target(a, i)["mask"] for i in range(len(rows))
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("build")
    b.add_argument("--manifest", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--max-bytes", type=int, default=1024**3)
    b.add_argument(
        "--split-by", choices=["episode_id", "scene_id"], default="episode_id"
    )
    i = s.add_parser("inspect")
    i.add_argument("--parquet", required=True)
    i.add_argument("--meta", required=True)
    i.add_argument("--annotation", required=True)
    i.add_argument("--report", required=True)
    a = p.parse_args()
    if a.cmd == "build":
        r = build(a.manifest, a.checkpoint, a.output, a.split_by, a.max_bytes)
    else:
        r = inspect(a.parquet, a.meta, a.annotation)
        Path(a.report).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


class NumericEpisodeRows:
    """One episode's native numeric columns in RAM, no Python-float/RGB cache."""

    def __init__(self, table):
        import numpy as np
        import pyarrow as pa

        self.columns = {}
        self.length = len(table)
        for name in table.column_names:
            column = table[name].combine_chunks()
            if column.null_count:
                raise ValueError("null numeric episode column")
            if (
                pa.types.is_list(column.type)
                or pa.types.is_large_list(column.type)
                or pa.types.is_fixed_size_list(column.type)
            ):
                if column.values.null_count:
                    raise ValueError("null numeric vector item")
                if pa.types.is_fixed_size_list(column.type):
                    width = column.type.list_size
                else:
                    widths = np.diff(column.offsets.to_numpy(zero_copy_only=False))
                    if not len(widths) or (widths != widths[0]).any():
                        raise ValueError("variable numeric vector layout")
                    width = int(widths[0])
                value = column.values.to_numpy(zero_copy_only=False).reshape(
                    self.length, width
                )
            else:
                value = column.to_numpy(zero_copy_only=False)
            self.columns[name] = value

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.length:
            raise IndexError(index)
        return {
            key: (
                values[index].item()
                if getattr(values[index], "ndim", 0) == 0
                else values[index]
            )
            for key, values in self.columns.items()
        }


class LazyDemonstration:
    """Decoupled real-parquet -> train.py minibatches without dense disk caches.

    Adapter contracts:
      transform(raw): observation/actions normalized by official checkpoint and
        provenance with adapter_revision, proprio_layout_sha256, action_layout_sha256.
        raw.state is FULL256; adapter MUST use audited named field slices.
      graphs(context): conditions/labels/provenance for current+past sensor times;
        task_fields are explicitly privileged and must yield oracle=True.
      read_images(frame): current RGB dictionary only; no future exposures.
    This class itself never guesses proprio offsets, makes up predicates or fits norms.
    Last31 anchors are excluded rather than inventing future target actions. It
    preserves raw action ordering exactly; the transform owns official mapping.
    """

    def __init__(
        self,
        parquet,
        meta,
        annotation,
        checkpoint,
        transform,
        graphs,
        read_images,
        split="train",
        stride=32,
        history=16,
        label_builder=None,
    ):
        import pyarrow.parquet as pq

        if (
            split not in ("train", "validation", "test")
            or not 1 <= history <= 16
            or stride < 1
        ):
            raise ValueError("split/stride/history")
        self.records = NumericEpisodeRows(
            pq.read_table(
                parquet,
                columns=[
                    "index",
                    "episode_index",
                    "timestamp",
                    "observation.state",
                    "observation.task_info",
                    "observation.cam_rel_poses",
                    "action",
                ],
            )
        )
        self.meta = json.loads(Path(meta).read_text())
        self.annotation = json.loads(Path(annotation).read_text())
        self.transform = transform
        self.graphs = graphs
        self.read_images = read_images
        self.split = split
        self.history = history
        self.label_builder = label_builder
        self.config = (
            json.loads(self.meta["config"])
            if isinstance(self.meta["config"], str)
            else self.meta["config"]
        )
        norms = list(Path(checkpoint).glob("assets/**/norm_stats.json"))
        if len(norms) != 1:
            raise ValueError("ambiguous checkpoint normalizer")
        self.norm_sha = sha256(norms[0])
        self.anchors = list(range(0, max(0, len(self.records) - 31), stride))
        if not self.records:
            raise ValueError("empty demonstration")
        self.episode = self.records[0]["episode_index"]
        import numpy as np

        columns = self.records.columns
        timestamps = columns["timestamp"]
        if (
            (columns["episode_index"] != self.episode).any()
            or not np.isfinite(timestamps).all()
            or (np.diff(timestamps) <= 0).any()
        ):
            raise ValueError("noncausal/cross-episode source parquet")
        if columns["observation.state"].shape != (len(self.records), 256) or columns[
            "action"
        ].shape != (len(self.records), 23):
            raise ValueError("unexpected raw state/action layout")
        task_fields(
            self.meta["task_obs_keys"], self.records[0]["observation.task_info"]
        )
        numeric = columns["observation.task_info"]
        if (
            not np.isfinite(numeric).all()
            or (np.abs(numeric) > np.finfo(np.float32).max).any()
        ):
            raise ValueError("nonfinite task info")
        self.parquet_hash = sha256(parquet)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        frame = self.anchors[index]
        r = self.records[frame]
        t = r["timestamp"]
        fields = task_fields(self.meta["task_obs_keys"], r["observation.task_info"])
        # Context contains no action targets or future labels/observations.
        history_indices = list(
            reversed(range(frame, max(-1, frame - self.history * 32), -32))
        )
        context = {
            "episode_id": str(self.episode),
            "step_id": frame,
            "timestamp": t,
            "raw_state": r["observation.state"],
            "camera_relative_poses": r["observation.cam_rel_poses"],
            "task_fields": fields,
            "task_obs_keys": self.meta["task_obs_keys"],
            "history": [
                {
                    "timestamp": self.records[i]["timestamp"],
                    "raw_state": self.records[i]["observation.state"],
                    "task_fields": task_fields(
                        self.meta["task_obs_keys"],
                        self.records[i]["observation.task_info"],
                    ),
                }
                for i in history_indices
            ],
            "source": "oracle",
            "initial_config": self.config,
        }
        # Future skill annotation is NEVER supplied to graph input builder.
        # Current phase is a target-only field; builder may use its static task
        # vocabulary, but current annotated state must not enter graph features.
        graph = self.graphs(context)
        if self.label_builder is not None:
            graph["labels"] = self.label_builder(
                skill_target(self.annotation, frame), context
            )
        raw = {
            "state": r["observation.state"],
            "images": self.read_images(frame),
            "task": "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked.",
            "actions": [self.records[i]["action"] for i in range(frame, frame + 32)],
        }
        if isinstance(self.read_images, RGBVideoReader):
            raw["images"] = self.read_images(frame, timestamp=t)
        model = self.transform(raw)
        if model.get("normalization_sha256") != self.norm_sha:
            raise ValueError("official transform normalization proof missing")
        batch = {
            k: model[k] for k in ("observation", "actions", "normalization_sha256")
        }
        batch.update(conditions=graph["conditions"], labels=graph["labels"])
        batch["provenance"] = {
            **model.get("provenance", {}),
            **graph.get("provenance", {}),
            "source_parquet_sha256": self.parquet_hash,
        }
        if isinstance(self.read_images, RGBVideoReader):
            batch["provenance"]["rgb_exposures"] = self.read_images.last_receipt
        # No direct current-skill labels are injected into model conditions.
        batch["supervision"] = skill_target(self.annotation, frame)
        batch["episode_id"] = str(self.episode)
        batch["step_id"] = frame
        batch["timestamp"] = t
        row = {
            "episode_id": str(self.episode),
            "scene_id": self.config["scene"]["scene_instance"],
            "step_id": frame,
            "timestamp": t,
            "object_ids": list(fields),
            "action_horizon": 32,
            "split": self.split,
            "graph_source": "oracle",
            "graph_timestamp": t,
        }
        return validate_batch(batch, row, self.norm_sha)


def privileged_pose_snapshot(fields, meta, episode, timestamp, registry, ee_world=None):
    # One canonical implementation shared with live runtime; never model-name IDs.
    from .runtime_bridge import snapshot_from_decoded_task_info

    return snapshot_from_decoded_task_info(
        fields, registry, str(episode), timestamp, ee_world
    )


def episode_partition(episode_id, validation_fraction=0.1):
    """Deterministic episode-level holdout, independent of any rollout scores."""
    x = int(hashlib.sha256(str(episode_id).encode()).hexdigest()[:8], 16) / 2**32
    return "validation" if x < validation_fraction else "train"


def canonical_skill(skill, meta):
    cfg = (
        json.loads(meta["config"])
        if isinstance(meta["config"], str)
        else meta["config"]
    )
    scene = cfg["scene"]["scene_file"]
    scene = json.loads(scene) if isinstance(scene, str) else scene
    mapping = scene.get("metadata", {}).get("task", {}).get("inst_to_name", {})
    if not mapping:
        scene = meta["scene_file"]
        scene = json.loads(scene) if isinstance(scene, str) else scene
        mapping = scene["metadata"]["inst_to_name"]
    reverse = {v: k for k, v in mapping.items()}

    def convert(x):
        if isinstance(x, list):
            return [convert(v) for v in x]
        # Never keep scene-specific model names as candidate identities.
        return reverse.get(x, "UNKNOWN")

    return {
        "skill_ids": skill["skill_id"],
        "targets": convert(skill.get("object_id", [])),
        "manipulating": convert(skill.get("manipulating_object_id", [])),
    }


def candidate_key(canonical):
    return (
        "plan:"
        + hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )


def compile_plan(sources, output):
    """Static candidate vocabulary from TRAIN episodes only, shared at deployment.

    Not a ground-truth dependency DAG or test-time phase oracle. Order is canonical
    hash order, not episode timing. Unknown scene targets stay UNKNOWN. Runtime
    must load this exact artifact and use its query_order; two BDDL goal queries
    cannot be substituted for these phase candidates.
    """
    candidates = {}
    provenance = []
    for row in sources:
        if row["split"] != "train":
            raise ValueError("plan vocabulary must use training episodes only")
        a = json.loads(Path(row["annotation"]).read_text())
        m = json.loads(Path(row["meta"]).read_text())
        for skill in a["skill_annotation"]:
            canonical = canonical_skill(skill, m)
            key = candidate_key(canonical)
            candidates.setdefault(key, {"id": key, **canonical})
        provenance.append(
            {
                "episode_id": str(row["episode_id"]),
                "annotation_sha256": sha256(row["annotation"]),
                "meta_sha256": sha256(row["meta"]),
            }
        )
    if not candidates:
        raise ValueError("empty training candidate vocabulary")
    candidates["plan:REPOSITION_BASE"] = {
        "id": "plan:REPOSITION_BASE",
        "skill_ids": [],
        "targets": [],
        "manipulating": [],
        "supervision": "UNKNOWN; no IK/collision-verified labels",
    }
    artifact = {
        "schema": "tgr_static_skill_candidates_v1",
        "condition_schema": "tgr_static_skill_candidates_v1",
        "node_keys": sorted(candidates),
        "source_split": "train",
        "test_annotations_used": False,
        "runtime_factory": "tgr.runtime_bridge:make_eval_runtime",
        "query_order": sorted(candidates),
        "candidates": [candidates[k] for k in sorted(candidates)],
        "query_features": 256,
        "query_encoding": "sha256_counter_bytes_v1",
        "source_train_episodes": provenance,
        "unknown_entity_token": "UNKNOWN",
        "dependency_edges": [],
        "semantic_scope": "active annotated skill candidate selection; NOT verified physical completion or a learned DAG",
        "required_runtime": "Load artifact and use identical query order/count; retain BDDL goal predicates separately. Never read evaluation annotations.",
        "physical_goal_labels": "masked unless per-frame verified predicate evidence exists",
    }
    Path(output).write_text(json.dumps(artifact, indent=2))
    return artifact


def plan_queries(artifact):
    if artifact["schema"] != "tgr_static_skill_candidates_v1":
        raise ValueError("unsupported plan artifact")
    data = []
    for key in artifact["query_order"]:
        values = b"".join(
            hashlib.sha256((key + ":" + str(i)).encode()).digest() for i in range(8)
        )
        data.append([(v / 127.5 - 1) / 16 for v in values])
    return torch.tensor(data, dtype=torch.float32)[None]


def phase_labels(artifact, annotation, meta, frame):
    """Partial labels: only current annotated phase is ACTIVE; no inferred DONE."""
    n = len(artifact["query_order"])
    active = skill_target(annotation, frame)
    labels = {
        "state": torch.zeros(1, n, dtype=torch.long),
        "state_mask": torch.zeros(1, n, dtype=torch.bool),
        "goal": torch.zeros(1, n),
        "goal_mask": torch.zeros(1, n, dtype=torch.bool),
        "next": torch.zeros(1, dtype=torch.long),
        "next_mask": torch.zeros(1, dtype=torch.bool),
    }
    if not active["mask"]:
        return labels
    match = [
        s
        for s in annotation["skill_annotation"]
        if s["skill_idx"] == active["skill_idx"]
    ]
    if len(match) != 1:
        raise ValueError("ambiguous annotated skill index")
    key = candidate_key(canonical_skill(match[0], meta))
    if key not in artifact["query_order"]:
        return labels  # heldout unseen skills are unknown, never added to vocabulary
    i = artifact["query_order"].index(key)
    labels["state"][0, i] = 1
    labels["state_mask"][0, i] = True
    labels["next"][0] = i
    labels["next_mask"][0] = True
    return labels


def pack_sequence(items):
    """Two-item minimum for memory gradient audit; no future/cross-episode."""
    if not 2 <= len(items) <= 16:
        raise ValueError("temporal sequence length must be 2..16")
    episode = items[0]["episode_id"]
    norm = items[0]["normalization_sha256"]
    step = -1
    time = -float("inf")
    for item in items:
        if (
            "sequence" in item
            or item["episode_id"] != episode
            or item["normalization_sha256"] != norm
        ):
            raise ValueError("nested/cross-episode/normalizer-mixed sequence")
        if item["step_id"] <= step or item["timestamp"] <= time:
            raise ValueError("noncausal temporal sequence")
        for key in (
            "plan_sha256",
            "condition_schema",
            "feature_schema",
            "query_order",
            "proprio_layout_sha256",
            "action_layout_sha256",
        ):
            if item.get("provenance", {}).get(key) != items[0].get(
                "provenance", {}
            ).get(key):
                raise ValueError("mixed sequence provenance: " + key)
        step = item["step_id"]
        time = item["timestamp"]
    return {
        "normalization_sha256": norm,
        "sequence": items,
        "episode_id": episode,
        "step_id": items[0]["step_id"],
        "timestamp": items[0]["timestamp"],
        "provenance": dict(items[0].get("provenance", {})),
        "supervision_scope": "partial labels; memory resets per sequence; graph inputs causal",
    }


# Official metadata commit 38b4f389.. layouts, verified against all6805 first-episode
# frames: actual parquet excludes the two scalar grasp_* entries in old constants.
# Do not read erroneous joint effort channels112:140.
PROPRIO_256 = {
    "robot_pos": (140, 143),
    "robot_ori_cos": (143, 146),
    "robot_ori_sin": (146, 149),
    "arm_left_qpos": (158, 165),
    "arm_left_qvel": (179, 186),
    "eef_left_pos": (186, 189),
    "eef_left_quat": (189, 193),
    "gripper_left_qpos": (193, 195),
    "gripper_left_qvel": (195, 197),
    "arm_right_qpos": (197, 204),
    "arm_right_qvel": (218, 225),
    "eef_right_pos": (225, 228),
    "eef_right_quat": (228, 232),
    "gripper_right_qpos": (232, 234),
    "gripper_right_qvel": (234, 236),
    "trunk_qpos": (236, 240),
    "trunk_qvel": (240, 244),
    "base_qpos": (244, 247),
    "base_qvel": (253, 256),
}


def named_proprio(state):
    """Audited actual256 -> named proprio, with explicit current-baseline body vel."""
    import numpy as np

    s = np.asarray(state, dtype=np.float32)
    if s.shape != (256,):
        raise ValueError("expected audited actual256 state")
    fields = {k: s[a:b].copy() for k, (a, b) in PROPRIO_256.items()}
    if any(not np.isfinite(v).all() for v in fields.values()):
        raise ValueError("invalid named proprio")
    for name, actual, expected in (
        ("arm_left", s[158:165], s[10:24:2]),
        ("arm_right", s[197:204], s[11:24:2]),
        ("trunk", s[236:240], s[6:10]),
        ("gripper_left", s[193:195], s[24:26]),
        ("gripper_right", s[232:234], s[26:28]),
        ("base_position", s[244:247], s[[0, 1, 5]]),
        ("base_velocity", s[253:256], s[[84, 85, 89]]),
    ):
        if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise ValueError("actual256 redundant layout check failed: " + name)
    fields["base_qvel_legacy_world"] = fields["base_qvel"].copy()
    yaw = fields["base_qpos"][2]
    c = np.cos(yaw)
    sn = np.sin(yaw)
    vx, vy, w = fields["base_qvel"]
    fields["base_qvel"] = np.array([c * vx + sn * vy, -sn * vx + c * vy, w], np.float32)
    return fields


def base_world_from_named(fields):
    """Recorded articulation-root WORLD pose, not virtual base joint pose."""
    import numpy as np
    from .geometry import transform

    c = fields["robot_ori_cos"]
    s = fields["robot_ori_sin"]
    if not np.allclose(c * c + s * s, 1.0, atol=2e-3):
        raise ValueError("invalid base RPY cos/sin")
    roll, pitch, yaw = np.arctan2(s, c) / 2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    q = [
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp,
    ]
    return transform(fields["robot_pos"], q)


def ee_world_from_named(fields):
    from .geometry import transform

    base = base_world_from_named(fields)
    return {
        side: (
            base
            @ transform(fields["eef_" + side + "_pos"], fields["eef_" + side + "_quat"])
        ).tolist()
        for side in ("left", "right")
    }


def robot_fields_from_named(fields, timestamp):
    """Recorded root/EE plus URDF arm roots; numeric demo agreement, not live."""
    from .kinematics import URDFKinematics

    owner = Path(__file__).resolve().parents[2]
    if not hasattr(robot_fields_from_named, "_kinematics"):
        robot_fields_from_named._kinematics = URDFKinematics(
            owner / "data/robot/r1pro.urdf"
        )
    kin = robot_fields_from_named._kinematics
    base = base_world_from_named(fields)
    joints = {f"torso_joint{i+1}": float(v) for i, v in enumerate(fields["trunk_qpos"])}
    for arm in ("left", "right"):
        joints.update(
            {
                f"{arm}_arm_joint{i+1}": float(v)
                for i, v in enumerate(fields[f"arm_{arm}_qpos"])
            }
        )
    arm_world = {
        arm: kin.frames(joints, arm, base)[0][f"{arm}_arm_base_link"].tolist()
        for arm in ("left", "right")
    }
    return {
        "reference_frame": "world",
        "world_poses_verified": True,
        "timestamp": float(timestamp),
        "base_world": base.tolist(),
        "eef_world": ee_world_from_named(fields),
        "arm_world": arm_world,
        "trunk_world": kin.frames(joints, "left", base)[0]["torso_link4"].tolist(),
        "joints": {
            "trunk_qpos": fields["trunk_qpos"].tolist(),
            "left_arm_qpos": fields["arm_left_qpos"].tolist(),
            "right_arm_qpos": fields["arm_right_qpos"].tolist(),
            "left_gripper_qpos": fields["gripper_left_qpos"].tolist(),
            "right_gripper_qpos": fields["gripper_right_qpos"].tolist(),
        },
        "joint_positions": joints,
        "joint_source": "audited_demonstration256",
        "source": "oracle",
        "arm_world_source": "actual_R1Pro_URDF_FK",
        "kinematics_sha256": kin.sha256,
        "tool_frame_artifact_sha256": sha256(owner / "reports/data_tool_frames.json"),
        "frame_audit_sha256": sha256(
            owner / "reports/data_real_fk_audit_20261023.json"
        ),
        "validation_scope": "demonstration_numeric_not_live",
        "live_calibration_verified": False,
    }


class RGBVideoReader:
    """Bounded uint8 decoder with exact PTS/parquet timestamp checks.

    Forward gaps/backward requests seek to the preceding keyframe, then use PTS
    to identify the requested frame. No approximate nearest exposure is accepted.
    """

    def __init__(self, root, episode_id, fps=30.0):
        import av

        self.av = av
        self.fps = fps
        self.handles = {}
        self.last_receipt = {}
        for camera in ("head", "left_wrist", "right_wrist"):
            p = (
                Path(root)
                / "videos/task-0045"
                / ("observation.images.rgb." + camera)
                / f"episode_{int(episode_id):08d}.mp4"
            )
            if not p.exists():
                raise FileNotFoundError(p)
            container = av.open(str(p))
            stream = container.streams.video[0]
            if abs(float(stream.average_rate) - fps) > 1e-6:
                raise ValueError("unexpected source RGB fps")
            stream.thread_type = "AUTO"
            stream.thread_count = 2
            self.handles[camera] = {
                "container": container,
                "stream": stream,
                "iterator": iter(container.decode(stream)),
                "index": -1,
                "image": None,
                "receipt": None,
                "source_path": str(p),
            }

    def __call__(self, index, timestamp=None):
        if not isinstance(index, int) or index < 0:
            raise ValueError("invalid video index")
        expected = index / self.fps if timestamp is None else float(timestamp)
        if abs(expected - index / self.fps) > 1e-6:
            raise ValueError("parquet frame/time is not 30Hz source exposure")
        out = {}
        receipt = {}
        for camera, h in self.handles.items():
            if index < h["index"] or index - h["index"] > 96:
                h["container"].seek(
                    int(expected / h["stream"].time_base),
                    stream=h["stream"],
                    backward=True,
                    any_frame=False,
                )
                h["iterator"] = iter(h["container"].decode(h["stream"]))
                h["index"] = -1
            if h["index"] != index:
                found = False
                last_decoded = None
                # MP4 keyframe seeking can use decode time and return a display
                # frame just AFTER the requested PTS. Retry bounded earlier seeks;
                # accept only the original exact requested frame, never a neighbour.
                for preroll in (None, 0.1, 1.0):
                    if preroll is not None:
                        seek_time = max(0.0, expected - preroll)
                        h["container"].seek(
                            int(seek_time / h["stream"].time_base),
                            stream=h["stream"],
                            backward=True,
                            any_frame=False,
                        )
                        h["iterator"] = iter(h["container"].decode(h["stream"]))
                        h["index"] = -1
                    for frame in h["iterator"]:
                        if frame.pts is None:
                            raise ValueError("missing RGB PTS")
                        exposed = float(frame.pts * frame.time_base)
                        decoded_index = round(exposed * self.fps)
                        last_decoded = decoded_index
                        if abs(exposed - decoded_index / self.fps) > 1e-6:
                            raise ValueError("RGB PTS is off source frame grid")
                        if decoded_index < index:
                            continue
                        if decoded_index > index:
                            break
                        if abs(exposed - expected) > 1e-6:
                            raise ValueError("RGB/parquet timestamp mismatch")
                        h["index"] = decoded_index
                        h["image"] = frame.to_ndarray(format="rgb24")
                        h["receipt"] = {
                            "frame_index": index,
                            "pts": frame.pts,
                            "time_base": str(frame.time_base),
                            "timestamp": exposed,
                            "parquet_timestamp": expected,
                            "error_seconds": exposed - expected,
                            "native_size": [frame.height, frame.width],
                            "dtype": "uint8",
                            "source_path": h["source_path"],
                            "seek_recovery_used": preroll is not None,
                            "seek_preroll_seconds": preroll or 0.0,
                        }
                        found = True
                        break
                    if found:
                        break
                if not found:
                    raise ValueError(
                        f'RGB/parquet exact exposure missing after bounded seek recovery: camera={camera}, expected_frame={index}, last_decoded={last_decoded}, file={h["source_path"]}'
                    )
            out[camera] = h["image"]
            receipt[camera] = dict(h["receipt"])
        self.last_receipt = receipt
        return out

    def close(self):
        for h in self.handles.values():
            h["container"].close()
        self.handles.clear()


def make_demonstration(root, episode_id, split, checkpoint, plan_path, adapter=None):
    """One episode in RAM, native current RGB only; all proof hashes explicit."""
    import numpy as np
    from .input_adapter import OfficialInputAdapter
    from .runtime_bridge import build_training_conditions

    root = Path(root)
    plan_path = Path(plan_path)
    owner = Path(__file__).resolve().parents[2]
    artifact = json.loads(plan_path.read_text())
    plan_sha = sha256(plan_path)
    if artifact["node_keys"] != artifact["query_order"] or len(
        set(artifact["query_order"])
    ) != len(artifact["query_order"]):
        raise ValueError("plan order mismatch")
    if artifact["source_split"] != "train" or artifact.get("test_annotations_used"):
        raise ValueError("plan source leakage")
    prop_sha = sha256(owner / "reports/data_proprio_layout.json")
    act_sha = sha256(owner / "reports/data_action_layout.json")
    if adapter is None:
        adapter = OfficialInputAdapter(
            checkpoint, proprio_layout_sha256=prop_sha, action_layout_sha256=act_sha
        )
    if (
        adapter.proof["proprio_layout_sha256"] != prop_sha
        or adapter.proof["action_layout_sha256"] != act_sha
    ):
        raise ValueError("input adapter layout proof mismatch")
    filename = f"episode_{int(episode_id):08d}"
    annotation = root / "annotations/task-0045" / (filename + ".json")
    meta = root / "meta/episodes/task-0045" / (filename + ".json")
    annotations = json.loads(annotation.read_text())
    metainfo = json.loads(meta.read_text())
    from .geometry_data import geometry_from_metadata

    geometry_spec = geometry_from_metadata(metainfo)
    reader = RGBVideoReader(root, int(episode_id))

    def transform(raw):
        return adapter(
            named_proprio=named_proprio(raw["state"]),
            rgb={k + "_rgb": v for k, v in raw["images"].items()},
            actions=np.asarray(raw["actions"], np.float32),
            task=raw["task"],
        )

    def graphs(ctx):
        history = []
        for h in ctx["history"]:
            fields = named_proprio(h["raw_state"])
            history_robot_fields = robot_fields_from_named(fields, h["timestamp"])
            history_robot_fields["geometry_spec"] = geometry_spec
            history.append(
                {
                    "episode_id": ctx["episode_id"],
                    "timestamp": h["timestamp"],
                    "decoded_task_info": h["task_fields"],
                    "ee_world": ee_world_from_named(fields),
                    "robot_fields": history_robot_fields,
                }
            )
        fields = named_proprio(ctx["raw_state"])
        current_robot_fields = robot_fields_from_named(fields, ctx["timestamp"])
        current_robot_fields["geometry_spec"] = geometry_spec
        result = build_training_conditions(
            decoded_task_info=ctx["task_fields"],
            ee_world=ee_world_from_named(fields),
            robot_fields=current_robot_fields,
            plan_spec=artifact,
            episode_id=ctx["episode_id"],
            timestamp=ctx["timestamp"],
            history=history,
            device="cpu",
        )
        if not torch.equal(result["conditions"]["queries"], plan_queries(artifact)):
            raise ValueError("runtime/data query encoding mismatch")
        result["labels"] = phase_labels(artifact, annotations, metainfo, ctx["step_id"])
        result.setdefault("provenance", {}).update(
            label_evidence=sha256(annotation),
            plan_sha256=plan_sha,
            proprio_layout_sha256=prop_sha,
            action_layout_sha256=act_sha,
            query_order=artifact["query_order"],
            condition_schema=artifact["condition_schema"],
        )
        return result

    try:
        data = LazyDemonstration(
            root / "data/task-0045" / (filename + ".parquet"),
            meta,
            annotation,
            checkpoint,
            transform,
            graphs,
            reader,
            split=split,
        )
    except BaseException:
        reader.close()
        raise
    return data, reader


def prepare_pilot(
    root, checkpoint, plan_path, output, input_adapter=None, condition_adapter=None
):
    """Actual disjoint train/validation frame32/64 seq2; CPU only, tiny cache."""
    root = Path(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("refuse overwrite pilot")
    if shutil.disk_usage(root).free < 51 * 1024**3:
        raise ValueError("50GiB free-space reserve")
    owner = Path(__file__).resolve().parents[2]
    split_rows = json.loads((owner / "reports/data_episode_splits.json").read_text())[
        "rows"
    ]
    chosen = [
        next(r for r in split_rows if r["split"] == split)
        for split in ("train", "validation")
    ]
    if chosen[0]["episode_id"] == chosen[1]["episode_id"]:
        raise ValueError("episode holdout overlap")
    artifact = json.loads(Path(plan_path).read_text())
    plan_sha = sha256(plan_path)
    rows = []
    batches = []
    for source in chosen:
        split = source["split"]
        ep = source["episode_id"]
        data, reader = make_demonstration(root, ep, split, checkpoint, plan_path)
        try:
            items = [data[1], data[2]]
            batch = pack_sequence(items)
            row = {
                "episode_id": str(ep),
                "scene_id": data.config["scene"]["scene_instance"],
                "step_id": 32,
                "timestamp": items[0]["timestamp"],
                "object_ids": artifact["query_order"],
                "action_horizon": 32,
                "split": split,
                "graph_source": "oracle",
                "graph_timestamp": items[0]["timestamp"],
                "tensor_path": f"{split}.pt",
            }
            validate_batch(batch, row, data.norm_sha)
            rows.append(row)
            batches.append(batch)
        finally:
            reader.close()
    output.mkdir(parents=True, exist_ok=False)
    for row, batch in zip(rows, batches):
        torch.save(batch, output / row["tensor_path"])
        row["tensor_sha256"] = sha256(output / row["tensor_path"])
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {
        "status": "real_rgb_parquet_causal_pilot_validate_batch_passed",
        "episodes": [r["episode_id"] for r in rows],
        "split_disjoint": True,
        "sequence_length": 2,
        "steps": [32, 64],
        "plan_sha256": plan_sha,
        "condition_schema": artifact["condition_schema"],
        "queries": len(artifact["query_order"]),
        "oracle_upper_bound": True,
        "disk_bytes": sum(p.stat().st_size for p in output.iterdir()),
        "normalization_sha256": batches[0]["normalization_sha256"],
        "formal_training_started": False,
        "rgb_exposures": {
            r["split"]: [b["provenance"]["rgb_exposures"] for b in batch["sequence"]]
            for r, batch in zip(rows, batches)
        },
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# Lazy training API exported here for existing train.BatchSource compatibility.
from .data import LazySequenceDataset, collate_sequences

if __name__ == "__main__":
    main()
