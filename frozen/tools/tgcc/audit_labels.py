"""Complete numeric-only TGCC evidence/time/certificate audit; no image decoding.

Coverage enumerates the existing fixed candidates, rather than pretending a
numeric-only audit knows which candidate the neural policy will select. Negative
eligibility is a certificate opportunity, not an executed failure trajectory.
"""

from pathlib import Path
import argparse
import collections
from dataclasses import asdict
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"), str(ROOT / "code/packages/openpi-client/src")]
from tgcc.robot import action_spec
from tgcc.data import decoded_evidence
from tgcc.contracts import build_contract, Truth
from tgcc.checkpoint import sources
from tgcc.readiness import supervision_gate
from tgr.build_batches import NumericEpisodeRows, task_fields
import pyarrow.parquet as pq


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def scalar(decoded, key):
    value = decoded.get(key)
    return None if value is None else float(np.asarray(value).reshape(-1)[0])


def valid_states(contract, evidence, now):
    """Exactly the contract_tensors validity/expected-value rule, without tensors."""
    states = []
    for p in contract.predicates:
        ev = evidence.get(p.key)
        t = Truth.UNKNOWN if ev is None else ev.truth_for(p, now)
        states.append(None if not p.required_at_current_stage or t == Truth.UNKNOWN else (t == Truth.TRUE) == p.expected)
    return states


def transition_counts(contract, before, after):
    required = [i for i, p in enumerate(contract.predicates) if p.required_at_current_stage and p.category in ("post", "invariant")]
    post = [i for i, p in enumerate(contract.predicates) if p.required_at_current_stage and p.category == "post"]
    hand = [i for i, p in enumerate(contract.predicates) if p.required_at_current_stage and p.category == "handoff"]
    progress_valid = bool(post) and all(before[i] is not None and after[i] is not None for i in required)
    handoff_valid = bool(hand) and all(before[i] is not None and after[i] is not None for i in hand)
    progress = sum(int(after[i]) - int(before[i]) for i in post) / len(post) if progress_valid else None
    handoff = all(after[i] for i in hand) if handoff_valid else None
    return progress_valid, progress, handoff_valid, handoff


def main():
    parser = argparse.ArgumentParser()
    default_manifest = ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl"
    if not default_manifest.exists():
        default_manifest = Path("/media/data5/ssy/TGR/reports/hotdog_lazy_seq2_manifest.jsonl")
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/interface_audit")
    parser.add_argument("--examples-per-kind", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path, default=Path("/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"))
    args = parser.parse_args()
    if args.examples_per_kind < 1:
        raise ValueError("At least one preflight example per kind")
    norm_paths = list(args.checkpoint.glob("assets/**/norm_stats.json"))
    if len(norm_paths) != 1:
        raise ValueError("Exactly one checkpoint normalization file required")
    normalization_sha = file_sha(norm_paths[0])
    stats = json.loads(norm_paths[0].read_text())["norm_stats"]["actions"]
    quantile_lo, quantile_hi = np.asarray(stats["q01"][:23], dtype=np.float32), np.asarray(stats["q99"][:23], dtype=np.float32)
    if quantile_lo.shape != (23,) or quantile_hi.shape != (23,) or not np.isfinite(quantile_lo).all() or not np.isfinite(quantile_hi).all() or np.any(quantile_hi < quantile_lo):
        raise ValueError("Invalid original checkpoint native quantiles")
    spec = action_spec(ROOT)
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    initial_sources = sources(ROOT)
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    counts, episodes, bounds, totals = {}, {}, {}, {}
    supervision, negatives, timing, recorded_evidence = {}, {}, {}, {}
    raw_grasp_fields = {}
    grouped = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    lo, hi, known = np.asarray(spec.lower, dtype=np.float32), np.asarray(spec.upper, dtype=np.float32), np.asarray(spec.bounds_valid, bool)
    known_dimensions = np.flatnonzero(known)
    if not len(known_dimensions):
        raise ValueError("No certified native control limits")
    for global_index, row in enumerate(rows):
        grouped[str(row["episode_id"])].append((global_index, row))
    consumed = 0
    for epi, indexed_rows in grouped.items():
        row = indexed_rows[0][1]
        root = Path(row["dataset_root"])
        stem = f"episode_{int(epi):08d}"
        meta = json.loads((root / "meta/episodes/task-0045" / f"{stem}.json").read_text())
        data = NumericEpisodeRows(pq.read_table(root / "data/task-0045" / f"{stem}.parquet", columns=["timestamp", "observation.task_info", "action"]))
        split = row["split"]
        if split not in ("train", "validation") or any(r["split"] != split for _, r in indexed_rows):
            raise ValueError("Episode split changed or unknown")
        episodes.setdefault(split, []).append(epi)
        counter = counts.setdefault(split, collections.Counter())
        raw_counter = recorded_evidence.setdefault(split, collections.Counter())
        sup = supervision.setdefault(split, collections.Counter())
        neg = negatives.setdefault(split, collections.Counter())
        time_counter = timing.setdefault(split, dict(checked_sequences=0, full_executed_prefixes=0, partial_prefixes=0, failures=0, maximum_chunk_dt_error_s=0.0, maximum_native_normalization_roundtrip_error=0.0))
        totals[split] = totals.get(split, 0) + len(indexed_rows)
        native = np.asarray(data.columns["action"], dtype=np.float32)
        timestamps = np.asarray(data.columns["timestamp"], dtype=float).reshape(-1)
        if native.ndim != 2 or native.shape[1] != 23 or not np.isfinite(native).all() or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Nonfinite/misaligned native numeric episode {epi}")
        violations = ((native < lo) | (native > hi)) & known
        tolerant = ((native < lo - 1e-5) | (native > hi + 1e-5)) & known
        bounds[epi] = dict(samples=len(native), any_native_limit_violation=int(tolerant.any(-1).sum()), strict_native_limit_violation=int(violations.any(-1).sum()), dimensions=tolerant.sum(0).tolist())
        cache = {}
        # Count recorded field values independently of expected contract truth.
        # `held:false` satisfying `expected=False` is never a positive grasp.
        for frame in sorted({int(t) for _, r in indexed_rows for t in r["sequence_steps"]}):
            rr = data[frame]
            ev = decoded_evidence(task_fields(meta["task_obs_keys"], rr["observation.task_info"]), float(rr["timestamp"]))
            for key, fact in ev.items():
                if key[0] == "held":
                    truth = fact.truth(float(rr["timestamp"]))
                    raw_counter["held_positive" if truth == Truth.TRUE else "held_negative" if truth == Truth.FALSE else "held_unknown"] += 1
        offset = 0
        task_info = np.asarray(data.columns["observation.task_info"])
        for key in meta["task_obs_keys"]:
            width = 3 if key.endswith(("_pos", "_ori_cos", "_ori_sin")) else 1
            if "_in_gripper_" in key:
                values, frequencies = np.unique(task_info[:, offset], return_counts=True)
                field = raw_grasp_fields.setdefault(split, {}).setdefault(key, collections.Counter())
                field.update({str(float(value)): int(n) for value, n in zip(values, frequencies)})
            offset += width
        if offset != task_info.shape[1]:
            raise ValueError("Raw grasp-field offsets differ from exported task schema")
        def exposure(frame):
            if frame not in cache:
                rr = data[int(frame)]
                now = float(rr["timestamp"])
                decoded = task_fields(meta["task_obs_keys"], rr["observation.task_info"])
                cache[frame] = (now, decoded, decoded_evidence(decoded, now))
            return cache[frame]
        for global_index, r in indexed_rows:
            if r.get("normalization_sha256") != normalization_sha:
                raise ValueError("Manifest/checkpoint normalization identity mismatch")
            start = int(r["step_id"])
            sequence = r.get("sequence_steps")
            if sequence != [start, start + 32] or int(r.get("action_horizon", 32)) != 32 or start < 0 or start + 64 > len(data):
                raise ValueError(f"Invalid exact two-chunk horizon: {epi}/{start}")
            end = start + 32
            before_time, before_decoded, before_evidence = exposure(start)
            after_time, _, after_evidence = exposure(end)
            chunk = native[start:end]
            normalized = (chunk - quantile_lo) / (quantile_hi - quantile_lo + 1e-6) * 2 - 1
            roundtrip = (normalized + 1) / 2 * (quantile_hi - quantile_lo + 1e-6) + quantile_lo
            roundtrip_error = float(np.abs(roundtrip - chunk).max())
            if not np.isfinite(normalized).all() or roundtrip_error > 1e-6:
                raise ValueError("Checkpoint normalization native-unit roundtrip failed")
            time_counter["maximum_native_normalization_roundtrip_error"] = max(time_counter["maximum_native_normalization_roundtrip_error"], roundtrip_error)
            dt_error = abs(after_time - before_time - 32 / 30.0)
            time_counter["maximum_chunk_dt_error_s"] = max(time_counter["maximum_chunk_dt_error_s"], dt_error)
            time_counter["checked_sequences"] += 1
            time_counter["full_executed_prefixes"] += int(len(chunk) == 32)
            if len(chunk) != 32 or dt_error > 2e-4 or abs(float(r["timestamp"]) - before_time) > 2e-4 or np.max(np.abs(np.diff(timestamps[start:end + 1]) - 1 / 30)) > 2e-4:
                time_counter["failures"] += 1
                raise ValueError(f"Actual 32-step transition timing mismatch: {epi}/{start}")
            held_names, held_sides, certificates = [], {}, []
            for key in before_decoded:
                if not key.endswith("_real"):
                    continue
                name = key[:-5]
                if scalar(before_decoded, key) != 1.0:
                    continue
                recorded = before_evidence.get(("held", name, ""))
                if recorded is None or recorded.truth(before_time) != Truth.TRUE:
                    continue
                sides = tuple(side for side in ("left", "right") if scalar(before_decoded, name + "_in_gripper_" + side) == 1.0)
                if not sides or before_evidence.get(("held", name, "")) is None:
                    continue
                held_names.append(name)
                held_sides[name] = sides
                for side, dimension in (("left", 14), ("right", 22)):
                    if side in sides and np.all(chunk[:, dimension] < -0.5):
                        certificates.append(dict(object_id=name, side=side, gripper_dimension=dimension,
                            evidence_source="recorded_existence_pose_and_recorded_grasp_state",
                            source_fields={key: 1.0, name + "_in_gripper_" + side: 1.0},
                            closed_positive_native_max=float(chunk[:, dimension].max()),
                            candidate_native_command=float(hi[dimension]),
                            native_unit=spec.units[dimension], proof="explicit_preserve_grasp_invariant",
                            not_a_failure_trajectory=True))
            positive_valid = not violations[start:end].any()
            # The same construction as max_count=1; one candidate, not 23 negatives.
            dimension = int(known_dimensions[(global_index * 3) % len(known_dimensions)])
            physical_value = float(np.float32(hi[dimension] + max(.1, .1 * abs(hi[dimension] - lo[dimension]))))
            physical_valid = bool(positive_valid and physical_value > hi[dimension])
            neg["positive_sequences_valid_native_limits"] += int(positive_valid)
            neg["physical_valid_sequences"] += int(physical_valid)
            neg["physical_masked_sequences"] += int(not physical_valid)
            # Release is certified only for a contract that explicitly preserves
            # the actually recorded grasp; placement candidates intentionally do not.
            task_eligible_ids = []
            progress_ids, handoff_ids = [], []
            for candidate in plan["candidates"]:
                contract = build_contract(candidate, ("hotdog.n.02_1", "hotdog.n.02_2") if candidate["skill_ids"] == [69] and "microwave" in str(candidate["targets"]) else (), preserved_grasps=held_names)
                before = valid_states(contract, before_evidence, before_time)
                after = valid_states(contract, after_evidence, after_time)
                for i, p in enumerate(contract.predicates):
                    counter[p.kind + ":total"] += 1
                    counter[p.kind + (":unknown" if after[i] is None else ":true" if after[i] else ":false")] += 1
                    sup["clauses"] += int(p.required_at_current_stage)
                    sup["before_known"] += int(before[i] is not None)
                    sup["after_known"] += int(after[i] is not None)
                    sup["both_exposures_known"] += int(before[i] is not None and after[i] is not None)
                    sup["before_unknown_masked"] += int(p.required_at_current_stage and before[i] is None)
                    sup["after_unknown_masked"] += int(p.required_at_current_stage and after[i] is None)
                    if p.category == "pre":
                        sup["precondition_valid"] += int(before[i] is not None)
                        sup["precondition_unknown_masked"] += int(p.required_at_current_stage and before[i] is None)
                    if p.category == "invariant" and p.kind == "held" and p.expected and any(c["object_id"] == p.subject for c in certificates):
                        task_eligible_ids.append(candidate["id"])
                pv, progress, hv, handoff = transition_counts(contract, before, after)
                sup["progress_valid"] += int(pv)
                sup["progress_unknown_masked"] += int(not pv)
                sup["transition_valid"] += int(pv)
                sup["handoff_valid"] += int(hv)
                sup["handoff_unknown_masked"] += int(not hv)
                sup["progress_positive"] += int(pv and progress > 0)
                sup["handoff_positive"] += int(hv and handoff)
                if pv and progress > 0:
                    progress_ids.append(candidate["id"])
                if hv and handoff and any(p.kind == "held" for p in contract.predicates if p.category == "handoff"):
                    handoff_ids.append(candidate["id"])
            task_valid = bool(positive_valid and certificates and task_eligible_ids)
            neg["task_valid_sequences_potential"] += int(task_valid)
            neg["task_masked_sequences"] += int(not task_valid)
            neg["maximum_negatives_per_positive"] = max(neg["maximum_negatives_per_positive"], int(physical_valid) + int(task_valid))
            neg["task_candidate_contracts_with_proof"] += len(set(task_eligible_ids))
            if split == "train":
                reference = dict(manifest_index=global_index, episode_id=epi, step_id=start, sequence_steps=[start, end], before_timestamp=before_time, after_timestamp=after_time, executed_steps=32)
                for kind, eligible, extra in (
                    ("preserved_closed_grasp", task_valid, dict(candidate_ids=sorted(set(task_eligible_ids)), certificates=certificates[:1])),
                    ("positive_progress", bool(progress_ids), dict(candidate_ids=progress_ids)),
                    ("held_handoff", bool(handoff_ids), dict(candidate_ids=handoff_ids)),
                    ("actual_held_state_change", any(before_evidence.get(k) is not None and after_evidence.get(k) is not None and before_evidence[k].truth(before_time) != Truth.UNKNOWN and after_evidence[k].truth(after_time) != Truth.UNKNOWN and before_evidence[k].value != after_evidence[k].value for k in set(before_evidence) | set(after_evidence) if k[0] == "held"), {}),
                ):
                    if eligible and len(examples[kind]) < args.examples_per_kind:
                        examples[kind].append(dict(reference, **extra))
            consumed += 1
        print(json.dumps(dict(episode=epi, split=split, audited_sequences=consumed)), flush=True)
    if set(episodes.get("train", ())) & set(episodes.get("validation", ())):
        raise ValueError("Training/validation overlap")
    if {k: len(v) for k, v in episodes.items()} != {"train": 184, "validation": 16}:
        raise ValueError("Expected complete 184 train / 16 validation episodes")
    final_sources = sources(ROOT)
    if final_sources != initial_sources:
        raise ValueError("Source tree changed during complete numeric audit; rerun after freeze")
    report = dict(schema="tgcc.revision_numeric_label_audit.v2", train_sequences=totals["train"], validation_sequences=totals["validation"],
        episodes={k: len(v) for k, v in episodes.items()}, contract_satisfaction_coverage=counts,
        coverage_semantics="Contract expected-value satisfaction, NOT raw predicate truth or positive grasp coverage",
        recorded_evidence=recorded_evidence, raw_grasp_fields=raw_grasp_fields,
        supervision=supervision, negative_certificates=negatives,
        timing=timing, native_bounds_by_episode=bounds, preflight_training_examples=examples,
        candidate_coverage="Enumerates existing fixed candidates; selected neural candidate eligibility is checked again in real gradient preflight.",
        proof_rules=dict(maximum_physical_negatives_per_positive=1, maximum_task_negatives_per_positive=1,
            physical="float32 native bounds exactly matching Torch checker in audited controller units, then checkpoint normalization",
            task="recorded real=1, recorded held side=1, explicit preserve-grasp invariant, all 32 positive native gripper commands closed (<-0.5), candidate is native fully open",
            negatives_are_executed_failures=False, negatives_supervise_state_transition=False, same_current_context=True),
        masks="Unknown physical predicates are excluded; labels use corresponding actual before and after states, never an unexecuted next chunk.",
        time_alignment=dict(control_hz=30, executed_prefix_steps=32, before_action_interval="[step_id, step_id+32)", after_exposure="step_id+32", next_chunk_not_in_transition=True),
        no_future_facts_as_forward_inputs=True, annotation_completion_labels_used=False,
        demonstration_inputs="Existing oracle/existence/grasp metadata only; no new privileged facts added to an observable-only interface.",
        native_comparison_dtype="float32, identical to Torch float native_actions/lower/upper",
        normalization=dict(path=str(norm_paths[0]), sha256=normalization_sha, clipped=False, native_order_rearranged=False, delta_action_used=False, audit="checkpoint quantile native-unit roundtrip; exact official pipeline equality is independently checked by real-batch preflight"),
        rgb_decoded=False, manifest=dict(path=str(args.manifest), sha256=file_sha(args.manifest)), source_sha256=final_sources,
        adaptation="Open/cooked/inside/on_top/toggled/operable unavailable in demonstrations remain UNKNOWN. Real held/existence fields supply supported supervision only.")
    report["legacy_grasp_export"] = "No authenticated AG ownership provenance; all legacy grasp fields are UNKNOWN, including -1/-1. Raw enum frequencies are reported separately."
    report["supervision_gate"] = supervision_gate(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "robot_action_spec.json").write_text(json.dumps(asdict(spec), indent=2))
    for name in ("revision_label_coverage.json", "complete_label_coverage.json"):
        (args.output / name).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("schema", "train_sequences", "validation_sequences", "episodes", "supervision", "negative_certificates", "timing")}), flush=True)


if __name__ == "__main__":
    main()
