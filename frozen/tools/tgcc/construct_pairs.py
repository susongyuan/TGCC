"""Benchmark/export certified pairs from real numeric demonstrations, no simulator.

Does not collect physical failure/recovery trajectories. Output records explicitly
carry this distinction; test episodes are excluded by the existing manifest split.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tgcc.construction import construct_pairs, preserved_grasp_constraints
from tgcc.data import decoded_evidence
from tgcc.robot import action_spec
from tgr.build_batches import NumericEpisodeRows, task_fields
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("/media/data5/ssy/TGR/reports/hotdog_lazy_seq2_manifest.jsonl"),
    )
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--export-pairs",
        action="store_true",
        help="Write native action pairs; default measures construction only",
    )
    args = p.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "benchmark.json"
    records_path = args.output / "pairs.jsonl"
    if report_path.exists() or (args.export_pairs and records_path.exists()):
        raise FileExistsError(
            "Use a new output directory; existing evidence is never overwritten"
        )
    start = time.perf_counter()
    raw = args.manifest.read_bytes()
    rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
    selected = [r for r in rows if r["split"] == args.split][: args.limit]
    if not selected:
        raise ValueError("No rows for requested split")
    if any(301 <= int(r["episode_id"]) <= 320 for r in selected):
        raise ValueError("Public evaluation instances must not become training data")
    spec = action_spec(ROOT)
    loaded = {}
    timings = []
    loads = []
    decode_times = []
    counts = Counter()
    rejection = Counter()
    subtypes = Counter()
    dimensions = Counter()
    bound_sides = Counter()
    output = records_path.open("x") if args.export_pairs else None
    export_seconds = 0.0
    try:
        for i, row in enumerate(selected):
            episode = str(row["episode_id"])
            dataset = Path(row["dataset_root"])
            stem = f"episode_{int(episode):08d}"
            if episode not in loaded:
                t = time.perf_counter()
                meta = json.loads(
                    (dataset / "meta/episodes/task-0045" / f"{stem}.json").read_text()
                )
                data = NumericEpisodeRows(
                    pq.read_table(
                        dataset / "data/task-0045" / f"{stem}.parquet",
                        columns=["timestamp", "observation.task_info", "action"],
                    )
                )
                loaded[episode] = (meta, data)
                loads.append(time.perf_counter() - t)
            meta, data = loaded[episode]
            frame = int(row["step_id"])
            if row["sequence_steps"] != [frame, frame + 32] or frame + 32 >= len(data):
                raise ValueError("Missing real aligned after-state")
            x = np.asarray(data.columns["action"][frame : frame + 32], np.float32)
            now = float(data.columns["timestamp"][frame])
            after = float(data.columns["timestamp"][frame + 32])
            if abs(after - now - 32 / 30) > 2e-4:
                raise ValueError("Unaligned real 32-step prefix")
            t = time.perf_counter()
            decoded = task_fields(
                meta["task_obs_keys"], data.columns["observation.task_info"][frame]
            )
            ev = decoded_evidence(decoded, now)
            context = f"{episode}:{frame}"
            constraints = preserved_grasp_constraints(context, ev, spec, now)
            decode_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            pairs, rejected = construct_pairs(
                x,
                np.ones(len(x), bool),
                spec,
                context,
                i,
                constraints=constraints,
                evidence=ev,
                now=now,
            )
            timings.append(time.perf_counter() - t)
            for kind, why in rejected.items():
                rejection[kind + ":" + why] += 1
            for pair in pairs:
                counts[pair.kind] += 1
                if pair.kind == "physical":
                    subtypes[pair.certificate["subtype"]] += 1
                    dimensions[pair.certificate["dimension_name"]] += 1
                    bound_sides[pair.certificate["bound_side"]] += 1
                if output:
                    t = time.perf_counter()
                    record = pair.record()
                    record.update(
                        episode_id=episode,
                        step_id=frame,
                        split=args.split,
                        before_timestamp=now,
                        actual_demonstration_after_timestamp=after,
                        after_state_applies_to="positive_demonstration_only",
                    )
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    export_seconds += time.perf_counter() - t
    finally:
        if output:
            output.close()
    total = time.perf_counter() - start
    latency = lambda values: dict(
        mean_s=float(np.mean(values)),
        median_s=float(np.median(values)),
        p95_s=float(np.quantile(values, 0.95)),
        total_s=float(sum(values)),
    )
    result = dict(
        schema="tgcc.certified_pair_benchmark.v1",
        samples=len(selected),
        episodes=len(loaded),
        split=args.split,
        construction_latency=latency(timings),
        evidence_decode_latency=latency(decode_times),
        episode_load_latency=latency(loads),
        end_to_end_wall_s=total,
        end_to_end_s_per_context=total / len(selected),
        export_seconds=export_seconds,
        valid_pairs=dict(counts),
        masked=dict(rejection),
        physical_subtypes=dict(subtypes),
        dimensions=dict(dimensions),
        bound_sides=dict(bound_sides),
        maximum_physical_pairs_per_positive=1,
        maximum_task_pairs_per_positive=1,
        real_failure_recovery_trajectories_collected=0,
        simulator_executed=False,
        rgb_decoded=False,
        repair_supervision="original action under same pre-action context only; not post-failure physical recovery",
        after_state_supervision="real demonstration only; synthetic negatives never receive outcomes",
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        action_spec=asdict(spec),
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                ROOT / "src/tgcc/construction.py",
                ROOT / "src/tgcc/data.py",
                ROOT / "src/tgcc/contracts.py",
            ]
        },
    )
    report_path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("action_spec", "source_sha256", "dimensions")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
