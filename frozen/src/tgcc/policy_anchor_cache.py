"""Immutable original prepared first exposures, never learned prefix features."""

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import digest


SCHEMA = "tgcc.prepared_first_policy_anchors.v1"


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_signature(path):
    value = Path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, size=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def anchor_sources(library, demonstrations, index):
    support = library.rows[int(index)]
    original = demonstrations.rows[int(support["manifest_index"])]
    result = {str((Path(original["dataset_root"]) / relative).resolve()): sha
              for relative, sha in original["source_sha256"].items()}
    cameras = {"head", "left_wrist", "right_wrist"}
    if len(support.get("rgb_exposures", [])) != 2 or set(support.get("video_sha256", {})) != cameras:
        raise ValueError("Prepared cache requires hash-bound original three-camera RGB")
    for exposure in support["rgb_exposures"]:
        if set(exposure) != cameras:
            raise ValueError("Missing prepared-cache original camera exposure")
        for camera, receipt in exposure.items():
            path = str(Path(receipt["source_path"]).resolve())
            sha = support["video_sha256"][camera]
            if path in result and result[path] != sha:
                raise ValueError("Conflicting prepared-cache video identity")
            result[path] = sha
    return result


def cache_header(library, demonstrations):
    root = Path(__file__).resolve().parents[2]
    implementations = [*sorted((root / "src/tgr").glob("*.py")),
        root / "src/tgcc/policy_anchor_cache.py", root / "src/tgcc/distribution_data.py",
        root / "src/tgcc/robot.py", root / "code/src/openpi/transforms.py",
        root / "code/src/openpi/training/config.py",
        root / "code/src/behavior/learning/datas/dataset.py",
        root / "code/src/openpi/policies/b1k_policy.py",
        root / "code/src/openpi/models/tokenizer.py",
        root / "code/src/openpi/training/checkpoints.py",
        root / "code/packages/openpi-client/src/openpi_client/image_tools.py",
        *sorted((root / "reports").glob("data_*.json")),
        root / "reports/robot_gripper_proxy_source.json",
        root / "data/robot/r1pro.urdf"]
    owned_sources = {}
    for index in library.train_indices:
        support = library.rows[int(index)]
        row = demonstrations.rows[int(support["manifest_index"])]
        if row["split"] != "train" or str(row["episode_id"]) != str(support["episode_id"]):
            raise ValueError("Prepared cache requires original successful training sources")
        for path, sha in anchor_sources(library, demonstrations, index).items():
            if path in owned_sources and owned_sources[path] != sha:
                raise ValueError("Conflicting prepared-cache source identities")
            owned_sources[path] = sha
    return dict(schema=SCHEMA, feature_library_sha256=digest(library.directory / "identity.json"),
        demonstrations_manifest_sha256=digest(demonstrations.manifest_path),
        plan_sha256=digest(demonstrations.plan_path),
        normalization_sha256=library.identity["normalization_sha256"],
        loader_source_sha256={str(p.relative_to(root)): digest(p) for p in implementations},
        train_anchor_indices=[int(i) for i in library.train_indices],
        train_anchor_count=len(library.train_indices), source_files_sha256=owned_sources,
        exposures_per_policy_anchor=1, future_policy_exposures_used=False,
        learned_features_cached=False, action_expert_or_vlm_prefix_cached=False,
        failed_or_synthetic_targets_used=False)


def anchor_owner(library, demonstrations, index):
    support = library.rows[int(index)]
    original = demonstrations.rows[int(support["manifest_index"])]
    return json_digest(dict(support_index=int(index), original_manifest_row=original, support_row=support,
        before_timestamp=support["before_timestamp"], sequence_steps=support["sequence_steps"],
        normalization_sha256=library.identity["normalization_sha256"]))


def prepare_first_policy(library, demonstrations, normalizer, index):
    """Run the existing complete seq2 validator before retaining its first item."""
    support = library[int(index)]
    policy = demonstrations[int(support["manifest_index"])]
    actions = normalizer.transform(np.asarray(support["actual_native"], np.float32))
    if (len(policy["sequence"]) != 2 or
            not np.allclose(policy["sequence"][0]["actions"][0, :, :23].numpy(), actions,
                            atol=3e-6, rtol=2e-6) or
            abs(float(policy["sequence"][0]["timestamp"]) - support["before_timestamp"]) > 2e-4 or
            policy["normalization_sha256"] != library.identity["normalization_sha256"]):
        raise ValueError("Original prepared policy/time/target identity mismatch")
    return dict(policy, sequence=policy["sequence"][:1])


def entry_paths(root, index):
    root = Path(root)
    return root / "anchors" / f"anchor{int(index):05d}.pt", root / "anchors" / f"anchor{int(index):05d}.json"


def write_policy_anchor(root, library, demonstrations, normalizer, index):
    """Atomic per-anchor writes allow a CPU build to resume without overwrites."""
    target, receipt = entry_paths(root, index)
    owner = anchor_owner(library, demonstrations, index)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["owner_sha256"] != owner or digest(target) != saved["sha256"]:
            raise ValueError("Existing prepared anchor is foreign or corrupt")
        return saved
    if target.exists():
        raise ValueError("Uncommitted prepared anchor requires explicit investigation")
    policy = prepare_first_policy(library, demonstrations, normalizer, index)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".building.{os.getpid()}.{time.time_ns()}")
    torch.save(dict(schema=SCHEMA, support_index=int(index), owner_sha256=owner, policy=policy), temporary)
    # The durable representation must exactly preserve all original tensors,
    # causal metadata and masks. weights_only rejects executable pickle types.
    restored = torch.load(temporary, map_location="cpu", weights_only=True)
    if not equal_prepared(policy, restored["policy"]):
        raise ValueError("Prepared policy serialization changed original tensors/metadata")
    record = dict(path=str(target.relative_to(root)), support_index=int(index),
        manifest_index=int(library.rows[int(index)]["manifest_index"]),
        episode_id=str(library.rows[int(index)]["episode_id"]),
        sha256=digest(temporary), owner_sha256=owner, original_serialization_exact=True)
    os.replace(temporary, target)
    receipt_tmp = receipt.with_name(receipt.name + f".building.{os.getpid()}")
    receipt_tmp.write_text(json.dumps(record, indent=2))
    os.replace(receipt_tmp, receipt)
    return record


def equal_prepared(first, second):
    if isinstance(first, torch.Tensor):
        return (isinstance(second, torch.Tensor) and first.dtype == second.dtype and
                first.shape == second.shape and torch.equal(first, second))
    if isinstance(first, dict):
        return (isinstance(second, dict) and first.keys() == second.keys() and
                all(equal_prepared(first[k], second[k]) for k in first))
    if isinstance(first, (tuple, list)):
        return (type(first) is type(second) and len(first) == len(second) and
                all(equal_prepared(a, b) for a, b in zip(first, second)))
    return type(first) is type(second) and first == second


class PolicyAnchorCache:
    def __init__(self, path, library, demonstrations, normalizer):
        self.path = Path(path)
        self.library, self.demonstrations = library, demonstrations
        self.identity = json.loads((self.path / "identity.json").read_text())
        header = cache_header(library, demonstrations)
        if (self.identity.get("status") != "complete" or
                any(self.identity.get(k) != v for k, v in header.items()) or
                demonstrations.adapter.normalization_sha256 != header["normalization_sha256"]):
            raise ValueError("Prepared first-policy cache identity changed or incomplete")
        expected = {str(i) for i in header["train_anchor_indices"]}
        if set(self.identity.get("entries", {})) != expected:
            raise ValueError("Prepared cache does not cover exactly all successful training anchors")
        signatures = self.identity.get("source_file_signatures", {})
        if set(signatures) != set(header["source_files_sha256"]):
            raise ValueError("Prepared cache missing original source-file signatures")
        # Verify original file bytes once in the parent before workers inherit
        # the immutable source signatures. Every cache hit checks those again.
        for source, sha in header["source_files_sha256"].items():
            if digest(source) != sha or source_signature(source) != signatures[source]:
                raise ValueError("Prepared cache original source changed: " + source)
        self.identity_sha256 = digest(self.path / "identity.json")

    def get(self, index):
        index = int(index)
        record = self.identity["entries"].get(str(index))
        if record is None:
            raise ValueError("Non-training or missing anchor requested from prepared cache")
        support = self.library.rows[index]
        original = self.demonstrations.rows[int(support["manifest_index"])]
        for source in anchor_sources(self.library, self.demonstrations, index):
            if source_signature(source) != self.identity["source_file_signatures"][source]:
                raise ValueError("Prepared cache original source changed: " + source)
        expected, _ = entry_paths(self.path, index)
        if (record["path"] != str(expected.relative_to(self.path)) or
                record["owner_sha256"] != anchor_owner(self.library, self.demonstrations, index) or
                digest(expected) != record["sha256"]):
            raise ValueError("Prepared policy anchor identity changed/corrupt")
        value = torch.load(expected, map_location="cpu", weights_only=True)
        if (value.get("schema") != SCHEMA or value.get("support_index") != index or
                value.get("owner_sha256") != record["owner_sha256"] or
                len(value["policy"]["sequence"]) != 1):
            raise ValueError("Prepared policy anchor payload ownership changed")
        return value["policy"]
