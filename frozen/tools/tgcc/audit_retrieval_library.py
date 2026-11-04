#!/usr/bin/env python3
"""CPU-only independent acceptance of the frozen success-support library.

Checks every source anchor and cached tensor; never decodes RGB, loads a policy,
starts training, or reads evaluation trajectories. Reports data acceptance only.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"),
               str(ROOT / "code/packages/openpi-client/src")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch

from tgcc.retrieval_data import RetrievalDataset, FrozenFeatureCodec, digest, past_action_summary
from tgcc.robot import ActionNormalizer
from tgr.build_batches import NumericEpisodeRows, phase_labels


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message, tolerance=3e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, message + ": shape")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), message + ": finite")
    require(np.allclose(actual, expected, atol=tolerance, rtol=2e-6), message)
    return float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))) if actual.size else 0.


def audit(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    library = RetrievalDataset(args.library)
    identity = library.identity
    codec = FrozenFeatureCodec.load(args.library)
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    require(digest(args.manifest) == identity["demonstrations_manifest_sha256"], "Manifest identity")
    require(digest(args.plan) == identity["plan_sha256"], "Plan identity")
    plan = json.loads(args.plan.read_text())
    require(plan["source_split"] == "train" and plan["test_annotations_used"] is False, "Plan holdout leak")
    for name in ("vision_encoder.safetensors", "vision_config.json"):
        key = "vision_encoder_sha256" if name.endswith("safetensors") else "vision_config_sha256"
        require(digest(args.library/name) == identity[key], "Independent frozen encoder artifact: " + name)
    for name, expected in identity["extraction_source_sha256"].items():
        require(digest(args.extraction_root/name) == expected, "Extraction implementation changed: " + name)
    require(identity["encoder_frozen"] is True and identity["future_targets_as_context_inputs"] is False,
            "Frozen encoder/current-context boundary")
    require(identity["failed_or_synthetic_targets_used"] is False, "Failed/synthetic support")
    require(identity["graph_source"] == "oracle_upper_bound" and identity["privileged_graphs_authorized"] is True,
            "Authorized graph provenance")

    groups = defaultdict(list)
    for index, row in enumerate(library.rows):
        groups[row["episode_id"]].append(index)
    train = {row["episode_id"] for row in library.rows if row["split"] == "train"}
    validation = {row["episode_id"] for row in library.rows if row["split"] == "validation"}
    require(len(train) == 184 and len(validation) == 16 and not train & validation, "184/16 episode isolation")
    require(train == set(identity["index_episodes"]) and validation == set(identity["validation_episodes"]),
            "Declared/reference episode ownership")
    require(train == {str(row["episode_id"]) for row in plan["source_train_episodes"]}, "Plan train source ownership")
    sampling = json.loads((args.library/"sampling_plan.json").read_text())
    require([row["manifest_index"] for row in library.rows] == sampling["manifest_indices"], "Frozen sampling order")
    require(all(len(indices) == 32 for indices in groups.values()), "32 anchors per source episode")
    require(all(not row["phases_without_aligned_manifest_anchor"] for row in sampling["per_episode"].values()),
            "Annotated phase lacks aligned anchor")

    a = library.arrays
    expected_shapes = dict(context=(6400,3097), after_features=(6400,1518), after_feature_mask=(6400,1518),
        actions=(6400,32,23), actual_native=(6400,32,23), action_mask=(6400,32),
        next_index=(6400,), next_mask=(6400,), reference_state=(6400,32),
        reference_after_state=(6400,32), reference_visual_context=(6400,256))
    require(set(a) == set(expected_shapes), "Frozen support tensor keys")
    for name, shape in expected_shapes.items():
        require(a[name].shape == shape and np.isfinite(a[name]).all(), "Tensor finite/shape: " + name)
    require(a["action_mask"].all(), "Incomplete actual native prefix")
    require(not a["reference_state"][:,23:].any() and not a["reference_after_state"][:,23:].any(), "State32 padding")
    require(np.count_nonzero(a["after_features"][~a["after_feature_mask"]]) == 0, "Unknown after targets must be zero and masked")
    require(np.all((a["next_index"] >= 0) & (a["next_index"] < len(plan["candidates"]))), "Skill vocabulary index")
    columns = [column for group in identity["retrieval_distance_groups"]
               for lo,hi in group["slices"] for column in range(lo,hi)]
    require(sorted(columns) == list(range(identity["context_width"])), "Retrieval groups must cover every context column once")
    require(abs(sum(g["weight"] for g in identity["retrieval_distance_groups"])-1) < 1e-6, "Retrieval group weights")

    normalizer = ActionNormalizer(args.base)
    raw = defaultdict(list)
    checked_hashes = {}
    checks = Counter()
    max_diffs = defaultdict(float)
    for episode, indices in sorted(groups.items()):
        with np.load(args.library/f"raw_episode_{episode}.npz", allow_pickle=False) as archive:
            cache = {key:archive[key] for key in archive.files}
        records = json.loads((args.library/f"raw_episode_{episode}.rows.json").read_text())
        require(records == [library.rows[index] for index in indices], "Cached row provenance: " + episode)
        require(all(len(value) == len(indices) and np.isfinite(value).all() for value in cache.values()), "Cached tensor integrity")
        for key in ("raw_context", "raw_after_features", "raw_after_feature_mask", "raw_visual", "raw_after_visual"):
            raw[key].append(cache[key])
        for key in ("actions", "action_mask", "actual_native", "next_index", "next_mask", "reference_state", "reference_after_state"):
            max_diffs[key] = max(max_diffs[key],same(a[key][indices],cache[key],"Cached final tensor: " + key, tolerance=0))
        source = manifest[records[0]["manifest_index"]]
        dataset_root = Path(source["dataset_root"])
        for relative, expected in source["source_sha256"].items():
            path = dataset_root/relative
            if str(path) not in checked_hashes:
                require(digest(path) == expected, "Original source identity: " + str(path))
                checked_hashes[str(path)] = expected
                checks["source_files_hashed"] += 1
        import pyarrow.parquet as parquet
        numeric = NumericEpisodeRows(parquet.read_table(
            dataset_root/f"data/task-0045/episode_{int(episode):08d}.parquet",
            columns=["action", "timestamp"])).columns
        annotation = json.loads((dataset_root/f"annotations/task-0045/episode_{int(episode):08d}.json").read_text())
        meta = json.loads((dataset_root/f"meta/episodes/task-0045/episode_{int(episode):08d}.json").read_text())
        for position, row in enumerate(records):
            original = manifest[row["manifest_index"]]
            for key in ("episode_id", "split", "scene_id", "sequence_steps", "source_sha256", "plan_sha256"):
                require(row[key] == original[key], "Original anchor ownership: " + key)
            frame = row["sequence_steps"][0]
            require(abs(row["before_timestamp"]-float(numeric["timestamp"][frame])) < 2e-4 and
                    abs(row["after_timestamp"]-float(numeric["timestamp"][frame+32])) < 2e-4, "Actual parquet prefix timing")
            require(np.array_equal(cache["actual_native"][position], np.asarray(numeric["action"][frame:frame+32],np.float32)),
                    "Native command source slice")
            summary, valid = past_action_summary(numeric["action"],frame,normalizer)
            same(cache["raw_context"][position,3456+23:3456+69],summary,"Executed past actions/current-future exclusion", tolerance=0)
            require(cache["raw_context"][position,3456+69] == float(valid), "Past action validity")
            labels = phase_labels(plan,annotation,meta,frame)
            require(int(cache["next_index"][position]) == int(labels["next"][0]) and
                    bool(cache["next_mask"][position]) == bool(labels["next_mask"][0]), "Current skill target provenance")
            for step, exposure in zip(row["sequence_steps"],row["rgb_exposures"]):
                require(set(exposure) == {"head", "left_wrist", "right_wrist"}, "Exactly three current RGB exposures")
                for camera, receipt in exposure.items():
                    require(receipt["frame_index"] == step and receipt["dtype"] == "uint8" and
                            abs(receipt["timestamp"]-float(numeric["timestamp"][step])) < 2e-4 and
                            abs(receipt["error_seconds"]) < 2e-4, "Actual RGB/parquet alignment")
                    path = receipt["source_path"]
                    expected = row["video_sha256"][camera]
                    require(checked_hashes.get(path,expected) == expected, "Conflicting source video identity")
                    if path not in checked_hashes:
                        require(digest(path) == expected, "Original RGB video identity")
                        checked_hashes[path] = expected
                        checks["video_files_hashed"] += 1
            checks["anchors_native_history_labels_rgb_aligned"] += 1
        max_diffs["normalized_native"] = max(max_diffs["normalized_native"],
            same(cache["actions"],normalizer.transform(cache["actual_native"]),"Official native normalization", tolerance=3e-6))
        checks["episodes"] += 1
        if checks["episodes"] % 25 == 0:
            print(json.dumps(dict(stage="independent_data_audit",**checks)),flush=True)

    raw = {key:np.concatenate(value) for key,value in raw.items()}
    c = codec.artifact
    projection = lambda values:(values-c["visual_mean"])@c["visual_components"]
    context = np.concatenate([projection(raw["raw_context"][:,:3456]),raw["raw_context"][:,3456:]],1)
    after = np.concatenate([projection(raw["raw_after_features"][:,:3456]),raw["raw_after_features"][:,3456:]],1)
    mask = np.concatenate([np.ones((6400,256),bool),raw["raw_after_feature_mask"][:,3456:]],1)
    require(np.array_equal(mask,a["after_feature_mask"]), "Exact after target mask propagation")
    max_diffs["encoded_context"] = same(a["context"],(context-c["context_mean"])/c["context_std"],"Frozen context encoding")
    expected_after = ((after-c["after_mean"])/c["after_std"]).astype(np.float32)
    expected_after[~mask] = 0
    max_diffs["encoded_after"] = same(a["after_features"],expected_after,"Frozen after encoding")
    max_diffs["reference_visual"] = same(a["reference_visual_context"],projection(raw["raw_visual"]),"Reference visual codec")
    train_mask = np.asarray([row["split"] == "train" for row in library.rows])
    visual_train = np.concatenate([raw["raw_visual"][train_mask],raw["raw_after_visual"][train_mask]])
    max_diffs["pca_train_mean"] = same(c["visual_mean"],visual_train.mean(0),"Train-only visual basis mean")
    max_diffs["context_train_mean"] = same(c["context_mean"],context[train_mask].mean(0),"Train-only context mean")
    max_diffs["context_train_std"] = same(c["context_std"],np.maximum(context[train_mask].std(0),.01),"Train-only context scale")
    count = mask[train_mask].sum(0)
    mean = np.where(count,(after[train_mask]*mask[train_mask]).sum(0)/np.maximum(count,1),0)
    variance = ((after[train_mask]-mean)**2*mask[train_mask]).sum(0)/np.maximum(count,1)
    max_diffs["after_train_mean"] = same(c["after_mean"],mean,"Train-only masked after mean")
    max_diffs["after_train_std"] = same(c["after_std"],np.maximum(np.sqrt(variance),.01),"Train-only masked after scale")
    query_checks = []
    for episode, indices in sorted(groups.items()):
        for index in (indices[0],indices[-1]):
            reference, distance = library.query(a["context"][index],exclude_episode_id=episode)[0]
            require(reference["split"] == "train" and reference["episode_id"] != episode,"Whole-episode query exclusion")
            query_checks.append(dict(query_episode=episode,query_step=library.rows[index]["sequence_steps"][0],
                                     reference_episode=reference["episode_id"],distance=distance))
    distance_groups = {}
    for group in identity["retrieval_distance_groups"]:
        columns = [column for lo,hi in group["slices"] for column in range(lo,hi)]
        distance_groups[group["name"]] = dict(weight=group["weight"],columns=len(columns),
            varying_training_columns=int(np.count_nonzero(np.var(a["context"][train_mask][:,columns],axis=0)>1e-8)))
        require(distance_groups[group["name"]]["varying_training_columns"] > 0,"Disconnected retrieval feature group")
    return dict(schema="tgcc.independent_retrieval_data_audit.v1",status="passed",data_acceptance=True,
        training_or_runtime_readiness=False,library=str(args.library),identity_sha256=digest(args.library/"identity.json"),
        auditor_sha256=digest(Path(__file__)),extraction_root=str(args.extraction_root),elapsed_seconds=time.monotonic()-started,
        train_episodes=len(train),validation_episodes=len(validation),train_anchors=int(train_mask.sum()),
        validation_anchors=int((~train_mask).sum()),valid_training_targets=identity["valid_training_targets"],
        checks=dict(checks),tensor_shapes={key:list(value.shape) for key,value in a.items()},
        max_absolute_differences=dict(max_diffs),distance_groups=distance_groups,
        whole_episode_exclusion_queries=len(query_checks),query_evidence=query_checks,
        source_semantics=identity["source_semantics"],terminal_task_success_independently_verified=False,
        limitations=["Official curated human demonstrations provide normal support; no independent per-episode terminal success receipt exists in these source files.",
                     "One source scene and 200 independent episodes; 6400 sampled anchors are correlated contexts.",
                     "Absent depth and unauthenticated held/open/cooked/toggled states stay UNKNOWN.",
                     "This CPU audit does not establish gradient, checkpoint, inference-interface or BEHAVIOR task performance."])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=Path("/media/data5/ssy/TGR/reports/hotdog_lazy_seq2_manifest.jsonl"))
    parser.add_argument("--plan",type=Path,default=ROOT/"reports/data_plan_vocabulary.json")
    parser.add_argument("--base",default="/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32")
    parser.add_argument("--extraction-root",type=Path,default=ROOT,
                        help="Preserved frozen extraction checkout whose source hashes are recorded by the library")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(),"Audit output already exists; preserve original evidence")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report = audit(args)
    except Exception as error:
        report = dict(schema="tgcc.independent_retrieval_data_audit.v1",status="failed",data_acceptance=False,
                      training_or_runtime_readiness=False,library=str(args.library),error=repr(error))
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        raise
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key != "query_evidence"}),flush=True)


if __name__ == "__main__":
    main()
