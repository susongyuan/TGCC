"""Compile exactly the training support's original first policy inputs on CPU."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"),
               str(ROOT / "code/packages/openpi-client/src"), str(ROOT / "tools/tgcc")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TGR_GEOMETRY_CACHE_PATH",
    "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3")

from tgcc.checkpoint import digest
from tgcc.policy_anchor_cache import (PolicyAnchorCache, anchor_owner, cache_header,
    entry_paths, source_signature, write_policy_anchor)

_state = None


def construct(library_path, manifest, plan):
    import torch
    torch.set_num_threads(1)
    from tgr.factory import ensure_legacy_imports
    ensure_legacy_imports()
    from tgr.data import LazySequenceDataset
    from tgcc.retrieval_data import RetrievalDataset
    from tgcc.robot import ActionNormalizer
    from train import BASE
    library = RetrievalDataset(library_path)
    raw = LazySequenceDataset(manifest, BASE, plan)
    return library, raw, ActionNormalizer(BASE)


def initialize(library_path, manifest, plan, output):
    global _state
    from multiprocessing.util import Finalize
    library, raw, normalizer = construct(library_path, manifest, plan)
    _state = (Path(output), library, raw, normalizer)
    Finalize(None, raw.close, exitpriority=20)


def episode(indices):
    output, library, raw, normalizer = _state
    started = time.monotonic()
    try:
        records = [write_policy_anchor(output, library, raw, normalizer, i) for i in indices]
        return dict(episode_id=records[0]["episode_id"], anchors=len(records),
                    seconds=time.monotonic() - started)
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl")
    parser.add_argument("--plan", type=Path, default=ROOT / "reports/data_plan_vocabulary.json")
    parser.add_argument("--sample-anchors", type=int, default=0,
        help="Prepare this many dispersed anchors as an incomplete disposable build, then resume without this option")
    args = parser.parse_args()
    if args.workers < 1 or args.sample_anchors < 0:
        raise ValueError("Positive CPU workers and nonnegative sample count required")
    args.output.mkdir(parents=True, exist_ok=True)
    library, raw, normalizer = construct(args.feature_library, args.manifest, args.plan)
    started = time.monotonic()
    header = cache_header(library, raw)
    snapshot_path = args.output / "build_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if any(snapshot.get(k) != v for k, v in header.items()):
            raise ValueError("Prepared-cache build source identity changed")
        if any(source_signature(p) != signature
               for p, signature in snapshot["source_file_signatures"].items()):
            raise ValueError("Prepared-cache build original file changed")
    else:
        signatures = {}
        for path, sha in header["source_files_sha256"].items():
            before = source_signature(path)
            if digest(path) != sha or before != source_signature(path):
                raise ValueError("Original policy/RGB source changed during cache build: " + path)
            signatures[path] = before
        snapshot = dict(header, source_file_signatures=signatures)
        temporary = snapshot_path.with_suffix(".building.json")
        temporary.write_text(json.dumps(snapshot, indent=2))
        os.replace(temporary, snapshot_path)
    all_indices = header["train_anchor_indices"]
    selected = all_indices
    if args.sample_anchors and args.sample_anchors < len(all_indices):
        import numpy as np
        selected = [all_indices[int(i)] for i in np.linspace(0, len(all_indices)-1,
                    args.sample_anchors, dtype=int)]
    groups = {}
    for index in selected:
        groups.setdefault(str(library.rows[index]["episode_id"]), []).append(index)
    raw.close()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize, initargs=(str(args.feature_library), str(args.manifest),
                str(args.plan), str(args.output))) as pool:
        futures = [pool.submit(episode, indices) for indices in groups.values()]
        for future in as_completed(futures):
            record = future.result()
            completed += record["anchors"]
            print(json.dumps(dict(stage="prepared_policy_cache_build", completed=completed,
                requested=len(selected), **record)), flush=True)
    if any(source_signature(p) != signature
           for p, signature in snapshot["source_file_signatures"].items()):
        raise ValueError("Original policy/RGB source changed during parallel build")
    entries = {}
    for index in all_indices:
        target, receipt = entry_paths(args.output, index)
        if not receipt.exists():
            continue
        record = json.loads(receipt.read_text())
        if (record["owner_sha256"] != anchor_owner(library, raw, index) or
                record.get("original_serialization_exact") is not True or
                digest(target) != record["sha256"]):
            raise ValueError("Prepared-cache commit refuses corrupt/foreign/inexact anchor")
        entries[str(index)] = record
    complete = len(entries) == len(all_indices)
    identity = dict(snapshot, entries=entries, status="complete" if complete else "incomplete",
        original_serialization_exact_count=len(entries), workers=args.workers,
        build_elapsed_seconds=time.monotonic() - started,
        compiled_inputs="original successful demonstration seq2 validated, then sequence[:1] retained")
    temporary = args.output / ".identity.building.json"
    temporary.write_text(json.dumps(identity, indent=2))
    os.replace(temporary, args.output / "identity.json")
    if complete:
        cache = PolicyAnchorCache(args.output, library, raw, normalizer)
        identity_sha = cache.identity_sha256
    else:
        identity_sha = digest(args.output / "identity.json")
    print(json.dumps(dict(stage="prepared_policy_cache_finished", status=identity["status"],
        completed=len(entries), expected=len(all_indices), identity_sha256=identity_sha,
        elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
