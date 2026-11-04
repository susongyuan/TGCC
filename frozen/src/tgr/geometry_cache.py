"""Compact, signed, read-only geometry reuse for audited demonstrations only."""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "tgr.compact_geometry_cache.v1"
SOURCES = (
    "src/tgr/geometry.py", "src/tgr/kinematics.py", "src/tgr/geometry_data.py",
    "src/tgr/approx_reachability.py", "src/tgr/runtime_bridge.py", "src/tgr/build_batches.py",
    "src/tgr/geometry_cache.py", "src/tgr/state.py", "src/tgr/tensorize.py",
    "reports/asset_bounds_readonly_audit.json", "reports/robot_gripper_proxy_source.json",
    "reports/data_tool_frames.json", "reports/data_proprio_layout.json",
    "reports/data_action_layout.json", "reports/data_real_fk_audit_20261023.json",
    "reports/data_plan_vocabulary.json", "data/robot/r1pro.urdf",
)
TOP_FIELDS = (
    "schema", "episode_id", "timestamp", "target_id", "oracle", "frame_id", "valid",
    "estimate_trusted", "geometry_identity", "approximate", "physical_verified", "source",
    "status", "reason", "known_meaning", "available_arms", "required_arms",
    "left_reachability", "right_reachability", "workspace_margin", "workspace_margin_kind",
    "manipulability", "manipulability_kind", "collision_feasibility", "base_reposition_feasibility",
    "simultaneous_bimanual_assessment", "T_base_target", "center_source",
)
ARM_FIELDS = (
    "estimated_reachability", "estimated_collision_feasible", "workspace_margin", "manipulability",
    "status", "ik_status", "metric_posture_source", "quality_insufficient", "reposition_advice",
)
CANDIDATE_FIELDS = (
    "candidate_id", "timestamp", "episode_id", "world_T_base", "score", "source",
    "base_reposition_feasibility", "estimated_reposition_feasible", "geometry_identity",
)


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("non-numeric geometry cache value: " + type(value).__name__)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_value, allow_nan=False)


def make_key(payload):
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


cache_key = make_key


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def implementation_metadata(root=ROOT, manifest=None):
    root = Path(root).resolve()
    manifest = Path(manifest or root / "reports/hotdog_lazy_seq2_manifest.jsonl").resolve()
    if not manifest.is_relative_to(root):
        raise ValueError("owned TGR manifest required")
    hashes = {relative: sha256_file(root / relative) for relative in SOURCES}
    hashes[str(manifest.relative_to(root))] = sha256_file(manifest)
    identity = dict(schema=SCHEMA, source_sha256=hashes, manifest=str(manifest.relative_to(root)),
                    input_scope="audited_demonstration256_only", geometry_only=True,
                    no_rgb=True, no_history_graph=True, no_labels=True, no_neural_predictions=True)
    return dict(identity, implementation_digest=make_key(identity))


def compact_assessments(assessments):
    result = []
    for assessment in assessments:
        if not isinstance(assessment, dict) or assessment.get("schema") != "tgr.reachability.v1":
            raise ValueError("geometry reachability assessment schema required")
        from .state import assessment_fingerprint
        digest = assessment.get("full_assessment_sha256") or assessment_fingerprint(assessment)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("original full geometry assessment hash required")
        item = {key: assessment[key] for key in TOP_FIELDS if key in assessment}
        item["arms"] = {}
        for side, info in assessment.get("arms", {}).items():
            if side not in ("left", "right"):
                raise ValueError("only explicit left/right arm assessments may be cached")
            item["arms"][side] = {key: info[key] for key in ARM_FIELDS if key in info}
            if "collision" in info:
                item["arms"][side]["collision"] = {
                    key: info["collision"][key] for key in ("verified", "scope", "status")
                    if key in info["collision"]
                }
        item["base_candidates"] = [
            {key: candidate[key] for key in CANDIDATE_FIELDS if key in candidate}
            for candidate in assessment.get("base_candidates", [])
        ]
        item.update(full_assessment_sha256=digest, compact_geometry_schema=SCHEMA)
        # Normalize NumPy scalar/list values and reject NaN before persistence.
        result.append(json.loads(canonical_json(item)))
    return result


class GeometryCache:
    def __init__(self, path, expected_metadata, *, allow_pilot=False):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        try:
            self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
            if self.metadata.get("implementation") != expected_metadata:
                raise ValueError("geometry cache implementation/source/manifest digest mismatch")
            if not self.metadata.get("build_complete") or (not self.metadata.get("production_complete") and not allow_pilot):
                raise ValueError("geometry cache is incomplete or an unapproved pilot")
        except BaseException:
            self.connection.close()
            raise

    def lookup(self, key):
        with self.lock:
            row = self.connection.execute("SELECT payload,payload_sha256 FROM entries WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError("compressed geometry row integrity mismatch")
        return json.loads(raw)

    def close(self):
        self.connection.close()


class GeometryCacheWriter:
    """One parent process owns all SQLite writes; publish only complete coverage."""

    def __init__(self, final_path, implementation, planned_counts, *, production=True, max_bytes=1024**3):
        self.final_path = Path(final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError("published geometry cache is immutable: " + str(self.final_path))
        self.lock_path = self.final_path.with_name(self.final_path.name + ".build.lock")
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as lock_file:
            lock_file.write(str(os.getpid()))
        self.temp_path = self.final_path.with_name(self.final_path.name + f".building.{os.getpid()}")
        self.connection = sqlite3.connect(self.temp_path)
        self.connection.executescript("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE entries(key TEXT PRIMARY KEY,episode TEXT NOT NULL,anchor INTEGER NOT NULL,payload BLOB NOT NULL,payload_sha256 TEXT NOT NULL,raw_bytes INTEGER NOT NULL,UNIQUE(episode,anchor));")
        self.planned = {str(key): int(value) for key, value in planned_counts.items()}
        self.counts = Counter()
        self.max_bytes = max_bytes
        self.metadata = dict(implementation=implementation, build_complete=False, production_complete=False,
                             production_requested=production, planned_counts=self.planned, created_at=time.time(),
                             entries=0, raw_bytes=0, compressed_bytes=0)
        self._metadata()
        self.connection.commit()

    def _metadata(self):
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES('build',?)", (canonical_json(self.metadata),))

    def add(self, key, assessments, episode, anchor):
        if len(key) != 64 or str(episode) not in self.planned:
            raise ValueError("known episode and SHA geometry input key required")
        raw = canonical_json(compact_assessments(assessments)).encode()
        compressed = zlib.compress(raw, 6)
        self.connection.execute("INSERT INTO entries VALUES(?,?,?,?,?,?)", (key, str(episode), int(anchor), compressed, hashlib.sha256(raw).hexdigest(), len(raw)))
        self.counts[str(episode)] += 1
        self.metadata["entries"] += 1
        self.metadata["raw_bytes"] += len(raw)
        self.metadata["compressed_bytes"] += len(compressed)
        if self.metadata["entries"] % 128 == 0:
            self.flush()

    def flush(self):
        self.metadata["actual_counts"] = dict(self.counts)
        self._metadata()
        self.connection.commit()
        if self.temp_path.stat().st_size > self.max_bytes:
            raise OSError("compact geometry cache exceeded its declared size budget")

    def prepare_validation(self):
        if dict(self.counts) != self.planned:
            raise ValueError("cannot validate geometry cache with missing episode anchors")
        self.metadata.update(build_complete=True, production_complete=False, validation_pending=True)
        self.flush()
        return self.temp_path

    def finalize(self):
        if dict(self.counts) != self.planned:
            raise ValueError("cannot publish geometry cache with missing episode anchors")
        if self.final_path.exists():
            raise FileExistsError("immutable cache already published by another owner")
        self.metadata.update(build_complete=True, production_complete=self.metadata["production_requested"], validation_pending=False, finished_at=time.time())
        self.flush()
        if self.connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite final integrity check failed")
        self.connection.close()
        with self.temp_path.open("rb") as file:
            os.fsync(file.fileno())
        self.temp_path.chmod(0o444)
        self.temp_path.rename(self.final_path)
        self.lock_path.unlink()
        directory_fd = os.open(self.final_path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(self.metadata, path=str(self.final_path), file_bytes=self.final_path.stat().st_size,
                    database_sha256=sha256_file(self.final_path))

    def abort(self, error):
        self.metadata.update(build_complete=False, production_complete=False, error=str(error), stopped_at=time.time())
        self._metadata()
        self.connection.commit()
        self.connection.close()
        self.lock_path.unlink(missing_ok=True)


_reader = None
_path = None
_lock = threading.RLock()


def configure(path, expected_metadata=None, *, allow_pilot=False):
    global _reader, _path
    with _lock:
        if _reader is not None:
            _reader.close()
        _reader, _path = None, str(Path(path).resolve())
        try:
            expected = expected_metadata or implementation_metadata()
            _reader = GeometryCache(path, expected, allow_pilot=allow_pilot)
            logging.getLogger(__name__).info("geometry cache configured: %s", _path)
            return True
        except (OSError, ValueError, sqlite3.Error, TypeError) as error:
            logging.getLogger(__name__).warning("geometry cache rejected; explicit fresh fallback: %s: %s", _path, error)
            return False


def lookup(key, fields):
    if os.environ.get("TGR_GEOMETRY_CACHE_BYPASS") == "1" or fields.get("joint_source") != "audited_demonstration256":
        return None
    path = os.environ.get("TGR_GEOMETRY_CACHE_PATH")
    if not path:
        return None
    with _lock:
        if str(Path(path).resolve()) != _path:
            configure(path, allow_pilot=os.environ.get("TGR_GEOMETRY_CACHE_ALLOW_PILOT") == "1")
        if _reader is None:
            return None
        try:
            return _reader.lookup(key)
        except (ValueError, zlib.error, sqlite3.Error) as error:
            logging.getLogger(__name__).warning("geometry cache row rejected; fresh fallback: %s", error)
            return None
