"""Audit trusted self-generated three-graph records and actual executed prefixes."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tgcc.contracts import Evidence, Truth, build_contract
from tgcc.runtime import snapshot_evidence


def load_verified(directory, receipt):
    path = directory / receipt["file"]
    if path.parent.resolve() != directory.resolve():
        raise ValueError("exposure outside owned directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("record hash changed")
    with gzip.open(path, "rb") as stream:
        record = pickle.load(stream)  # Own producer only, not arbitrary uploads.
    if record["timestamp"] != receipt["timestamp"]:
        raise ValueError("receipt timestamp mismatch")
    return record


def truths(contract, evidence, now):
    return [
        evidence.get(p.key, Evidence()).truth_for(p, now) for p in contract.predicates
    ]


def audit(directory):
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    counts = Counter()
    episodes = set()
    rgb_transitions = 0
    for path in sorted(directory.glob("graphs_*/transitions.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            before = load_verified(path.parent, row["before"])
            after = load_verified(path.parent, row["after"])
            n = row["executed_steps"]
            actions = np.asarray(row["executed_native"])
            if (
                actions.shape != (n, 23)
                or not np.isfinite(actions).all()
                or not 1 <= n <= 32
                or row["action_mask"] != [True] * n + [False] * (32 - n)
                or abs(after["timestamp"] - before["timestamp"] - n / 30) > 2e-4
                or before["session"] != after["session"]
            ):
                raise ValueError("execution alignment or session mismatch")
            if (
                row["recovery_success_label"] is not None
                or row["task_success_label"] is not None
            ):
                raise ValueError("generic recorder must not fabricate outcome labels")
            episodes.add(before["session"])
            rgb_transitions += bool(before["raw"]["images"]) and bool(
                after["raw"]["images"]
            )
            for record in (before, after):
                raw = record["raw"]
                now = record["timestamp"]
                for node in raw["graph_snapshot"]["entities"]:
                    if node["timestamp"] > now or node["last_seen"] > now:
                        raise ValueError("future physical evidence")
                for name in ("global_graph", "task_graph", "local_graph"):
                    if record["conditions"][name] is None:
                        raise ValueError("missing three-graph condition")
            be = snapshot_evidence(before["raw"]["graph_snapshot"], before["timestamp"])
            ae = snapshot_evidence(after["raw"]["graph_snapshot"], after["timestamp"])
            for ev in (be, ae):
                for key, value in ev.items():
                    if key[0] == "held":
                        counts[
                            "held_"
                            + (
                                "positive"
                                if value.value is True
                                else "negative"
                                if value.value is False
                                else "unknown"
                            )
                        ] += 1
            for candidate in plan["candidates"]:
                contract = build_contract(candidate)
                bs = truths(contract, be, before["timestamp"])
                at = truths(contract, ae, after["timestamp"])
                post = [
                    i for i, p in enumerate(contract.predicates) if p.category == "post"
                ]
                hand = [
                    i
                    for i, p in enumerate(contract.predicates)
                    if p.category == "handoff"
                ]

                def known(indices):
                    return indices and all(
                        bs[i] != Truth.UNKNOWN and at[i] != Truth.UNKNOWN
                        for i in indices
                    )

                if known(post):
                    delta = sum(
                        (at[i] == Truth.TRUE) == contract.predicates[i].expected
                        for i in post
                    ) - sum(
                        (bs[i] == Truth.TRUE) == contract.predicates[i].expected
                        for i in post
                    )
                    counts[
                        "progress_positive"
                        if delta > 0
                        else "progress_negative"
                        if delta < 0
                        else "progress_zero"
                    ] += 1
                if known(hand):
                    good = all(
                        (at[i] == Truth.TRUE) == contract.predicates[i].expected
                        for i in hand
                    )
                    counts["handoff_positive" if good else "handoff_negative"] += 1
            counts["actual_transitions"] += 1
            counts["full_32_step_transitions"] += n == 32
    report = dict(
        schema="tgcc.collection_quality.v1",
        directory=str(directory),
        candidate_specific_label_counts=dict(counts),
        sessions=len(episodes),
        rgb_transitions=rgb_transitions,
        fabricated_outcomes=0,
        training_adapter_verified=False,
        formal_training_ready=False,
        physical_recovery_scope="base_xy_return_only",
        evaluation_instances_used=False,
    )
    (directory / "quality_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
