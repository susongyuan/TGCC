"""Validation-only finite-sample calibration; no model/optimizer updates."""

from pathlib import Path
from dataclasses import replace
import sys, os, argparse, json, math

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "code/src"),
    str(ROOT / "code/packages/openpi-client/src"),
]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "TGR_GEOMETRY_CACHE_PATH",
    "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3",
)
import numpy as np
import torch
from safetensors.torch import load_file
from tgr.factory import create
from tgr.data import LazySequenceDataset
from tgcc.robot import action_spec, ActionNormalizer
from tgcc.data import DemonstrationMetadata, transition_labels
from tgcc.features import extract_pair
from tgcc.feature_cache import slice_observation
from tgcc.policy import PolicyFeatures
from tgcc.train_system import TrainSystem
from tgcc.specs import ExecutedTransition
from tgcc.checkpoint import digest, sources
from tgcc.joint_checkpoint import load_student
from tgr.serve import checkpoint_gate


def binary_calibration(calibration, heldout, alpha=0.05, minimum_class_count=16):
    """Fit on one episode partition, certify on another; UNKNOWNs never enter.

    Scores concern the named predicate only, never generic task success. A
    learned result can request verification but cannot replace physical evidence.
    """
    report = {"enabled": False, "temperature": None}
    for name, rows in (("calibration", calibration), ("gate", heldout)):
        report[name + "_positive_count"] = sum(y > 0.5 for _, y in rows)
        report[name + "_negative_count"] = sum(y <= 0.5 for _, y in rows)
    if any(
        report[n + "_" + c + "_count"] < minimum_class_count
        for n in ("calibration", "gate")
        for c in ("positive", "negative")
    ):
        report["disabled_reason"] = (
            "insufficient_independent_positive_negative_coverage"
        )
        return report
    x = torch.tensor([x for x, _ in calibration], dtype=torch.float64)
    y = torch.tensor([y for _, y in calibration], dtype=torch.float64)
    temperatures = torch.logspace(-1, 1, 81, dtype=torch.float64)
    losses = [
        float(torch.nn.functional.binary_cross_entropy_with_logits(x / t, y))
        for t in temperatures
    ]
    temperature = float(temperatures[int(np.argmin(losses))])
    cp = (x / temperature).sigmoid().numpy()
    # Learn a conservative alert threshold on negative calibration outcomes.
    cn = sorted(cp[y.numpy() <= 0.5])
    threshold = float(cn[min(len(cn) - 1, math.ceil((len(cn) + 1) * (1 - alpha)) - 1)])
    z = torch.tensor([x for x, _ in heldout], dtype=torch.float64)
    truth = torch.tensor([y for _, y in heldout], dtype=torch.float64)
    prob = (z / temperature).sigmoid()
    positive = prob[truth > 0.5].numpy()
    negative = prob[truth <= 0.5].numpy()
    auc = float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )
    fpr = float((negative > threshold).mean())
    recall = float((positive > threshold).mean())
    brier = float((prob - truth).square().mean())
    report.update(
        temperature=temperature,
        threshold=threshold,
        gate_brier=brier,
        gate_bce=float(
            torch.nn.functional.binary_cross_entropy_with_logits(z / temperature, truth)
        ),
        gate_auroc=auc,
        gate_false_alarm_rate=fpr,
        gate_recall=recall,
        enabled=bool(auc >= 0.8 and brier <= 0.2 and fpr <= alpha and recall >= 0.5),
    )
    return report


def progress_calibration(calibration, heldout, minimum_class_count=16):
    """Calibrate signed contract progress, requiring real gains AND regressions.

    Flat demonstrations are not negative recovery evidence. This scalar remains
    an advisory executed-transition score, never a task completion predicate.
    """
    report = {"enabled": False, "scale": None}
    for name, rows in (("calibration", calibration), ("gate", heldout)):
        report[name + "_positive_count"] = sum(y > 0 for _, y in rows)
        report[name + "_negative_count"] = sum(y < 0 for _, y in rows)
        report[name + "_zero_count"] = sum(y == 0 for _, y in rows)
    if any(
        report[n + "_" + c + "_count"] < minimum_class_count
        for n in ("calibration", "gate")
        for c in ("positive", "negative")
    ):
        report["disabled_reason"] = "insufficient_real_progress_and_regression_coverage"
        return report
    x = np.asarray([x for x, _ in calibration])
    y = np.asarray([y for _, y in calibration])
    scale = float(np.clip(np.dot(x, y) / max(float(np.dot(x, x)), 1e-12), 0.1, 10.0))
    gx = np.asarray([x for x, _ in heldout]) * scale
    gy = np.asarray([y for _, y in heldout])
    moving = gy != 0
    sign_accuracy = float((np.sign(gx[moving]) == np.sign(gy[moving])).mean())
    mae = float(np.abs(gx - gy).mean())
    report.update(
        scale=scale,
        gate_mae=mae,
        gate_sign_accuracy=sign_accuracy,
        enabled=bool(sign_accuracy >= 0.8 and mae <= 0.2),
    )
    return report


def select(rows, episodes, per_episode=16):
    out = []
    for ep in episodes:
        ids = [
            i
            for i, r in enumerate(rows)
            if r["episode_id"] == ep and r["split"] == "validation"
        ]
        if not ids:
            raise ValueError("Calibration episode has no validation sequences")
        out.extend(ids[int(x)] for x in np.linspace(0, len(ids) - 1, per_episode))
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgcc", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    md = json.loads((a.tgcc / "metadata.json").read_text())
    cfg = md["config"]
    torch.set_num_threads(4)
    if (
        md.get("schema") != "tgcc.joint.v2"
        or md["new_updates"] != 6000
        or md["optimizer_steps"] != 11000
        or not md["training_complete"]
        or cfg.get("critic_warmup_updates") != 1000
        or cfg.get("joint_updates") != 5000
        or cfg.get("new_updates") != 6000
        or md["source_sha256"] != sources(ROOT)
        or digest(a.tgcc / "tgcc.safetensors") != md["checkpoint_sha256"]
        or digest(a.tgcc / "student.safetensors") != md["student_sha256"]
        or digest(Path(md["parent_checkpoint"]) / "model.safetensors")
        != md["parent_model_sha256"]
    ):
        raise ValueError("Only final frozen source/checkpoint may be calibrated")
    base = "/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"
    protocol = json.loads((ROOT / "configs/tgcc/evaluation_pairing.json").read_text())
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    norms = list(Path(base).glob("assets/**/norm_stats.json"))
    if len(norms) != 1 or digest(norms[0]) != md["normalization_sha256"]:
        raise ValueError("Normalizer identity changed")
    checkpoint_gate(
        Path(md["parent_checkpoint"]),
        "tri_graph",
        digest(Path(base) / "model.safetensors"),
        digest(norms[0]),
        protocol,
        plan,
        digest(ROOT / "reports/data_plan_vocabulary.json"),
    )
    _, parent = create(base, "tri_graph", a.device, md["parent_checkpoint"])
    load_student(parent, a.tgcc, md)
    parent.requires_grad_(False)
    parent.eval()
    normalizer = ActionNormalizer(base)
    system = TrainSystem(parent, action_spec(ROOT), normalizer, cfg).to(a.device)
    system.load_state_dict(load_file(str(a.tgcc / "tgcc.safetensors")), strict=True)
    system.eval()
    system.requires_grad_(False)
    if (
        digest(ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl")
        != md["manifest_sha256"]
        or digest(ROOT / "reports/data_plan_vocabulary.json") != md["plan_sha256"]
    ):
        raise ValueError("Held-out manifest or training vocabulary identity changed")
    raw = LazySequenceDataset(
        ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl",
        base,
        ROOT / "reports/data_plan_vocabulary.json",
    )
    dataset = DemonstrationMetadata(raw, normalizer)
    tap = PolicyFeatures(parent)
    episodes = sorted({r["episode_id"] for r in raw.rows if r["split"] == "validation"})
    if len(episodes) != 16:
        raise ValueError("Exactly sixteen held-out demonstration episodes required")
    training_episodes = {r["episode_id"] for r in raw.rows if r["split"] == "train"}
    if set(episodes) & training_episodes:
        raise ValueError("Train/validation episode leakage")
    groups = {"calibration": episodes[::2], "gate": episodes[1::2]}
    results = {}
    for name, eps in groups.items():
        positive = []
        negative = []
        negative_by_kind = {"physical": [], "task": []}
        paired_margins_by_kind = {"physical": [], "task": []}
        records = []
        outcome_records = {"transition": [], "progress": [], "handoff": []}
        for index in select(raw.rows, eps):
            sample = dataset[index]
            before, after, native, _ = extract_pair(
                parent, [sample], tap, a.device, plan
            )
            before = slice_observation(before, 0, a.device, True)
            after = slice_observation(after, 0, a.device, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = system.critic.forward_action(before)
                state = system.critic.retrospective(
                    ExecutedTransition(
                        before,
                        after,
                        before.candidate_actions,
                        before.candidate_action_mask,
                        before.timestamp,
                        after.timestamp,
                    )
                )
            positive_valid = ~system.checker.violations(
                native, before.candidate_action_mask
            )
            if positive_valid[0]:
                positive.append(float(out["action_energy_total"][0]))
            negatives = system.checker.generate(
                native,
                before.candidate_action_mask,
                normalizer.transform,
                max_count=1,
                offset=index * 3,
            )
            negatives.append(
                system.checker.task_negative(
                    native,
                    before.candidate_action_mask,
                    before.preserve_gripper_mask,
                    normalizer.transform,
                )
            )
            for actions, valid, proof in negatives:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = system.critic.forward_action(
                        replace(before, candidate_actions=actions)
                    )
                if valid[0] and positive_valid[0]:
                    energy = float(o["action_energy_total"][0])
                    negative.append(energy)
                    kind = (
                        "task"
                        if proof["kind"] == "explicit_preserve_grasp_invariant"
                        else "physical"
                    )
                    negative_by_kind[kind].append(energy)
                    paired_margins_by_kind[kind].append(
                        energy - float(out["action_energy_total"][0])
                    )
            labels = transition_labels(before, after)
            for label, output in (
                ("transition", "transition_logit"),
                ("progress", "progress"),
                ("handoff", "handoff_logit"),
            ):
                if bool(labels[label + "_valid"][0]):
                    outcome_records[label].append(
                        (float(state[output][0]), float(labels[label][0]))
                    )
            for j in range(before.contract_ids.shape[1]):
                if labels["predicate_valid"][0, j]:
                    records.append(
                        (
                            int(before.contract_ids[0, j]),
                            float(state["predicate_logits"][0, j]),
                            float(labels["predicate"][0, j]),
                        )
                    )
        results[name] = dict(
            episodes=eps,
            positive=positive,
            negative=negative,
            negative_by_kind=negative_by_kind,
            paired_margins_by_kind=paired_margins_by_kind,
            predicate_records=records,
            outcome_records=outcome_records,
        )
        print(
            json.dumps(
                {
                    "completed_group": name,
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "negative_counts_by_kind": {
                        k: len(v) for k, v in negative_by_kind.items()
                    },
                }
            ),
            flush=True,
        )
    cal = results["calibration"]
    gate = results["gate"]
    alpha = cfg["calibration_alpha"]
    n = len(cal["positive"])
    if n == 0 or not gate["positive"] or not gate["negative"]:
        raise ValueError("No valid held-out action-ranking calibration samples")
    threshold = sorted(cal["positive"])[
        min(n - 1, math.ceil((n + 1) * (1 - alpha)) - 1)
    ]
    gpos = np.asarray(gate["positive"])
    gneg = np.asarray(gate["negative"])
    fpr = float((gpos > threshold).mean())
    auc = float(
        (gneg[:, None] > gpos[None, :]).mean()
        + 0.5 * (gneg[:, None] == gpos[None, :]).mean()
    )
    per_kind = {}
    for kind, energies in gate["negative_by_kind"].items():
        values = np.asarray(energies)
        per_kind[kind] = dict(
            calibration_negative_count=len(cal["negative_by_kind"][kind]),
            gate_negative_count=len(values),
            same_context_ranking_accuracy=(
                float((np.asarray(gate["paired_margins_by_kind"][kind]) > 0).mean())
                if len(values)
                else None
            ),
            same_context_mean_energy_margin=(
                float(np.mean(gate["paired_margins_by_kind"][kind]))
                if len(values)
                else None
            ),
            heldout_auroc=(
                float(
                    (values[:, None] > gpos[None, :]).mean()
                    + 0.5 * (values[:, None] == gpos[None, :]).mean()
                )
                if len(values)
                else None
            ),
        )
    physical_auc = per_kind["physical"]["heldout_auroc"]
    ranking_gates = {}
    for kind in ("physical", "task"):
        metrics = per_kind[kind]
        ranking_gates[kind] = bool(
            n >= 64
            and len(gpos) >= 64
            and metrics["calibration_negative_count"] >= 64
            and metrics["gate_negative_count"] >= 64
            and metrics["same_context_ranking_accuracy"] is not None
            and metrics["same_context_ranking_accuracy"] >= 0.8
            and metrics["heldout_auroc"] is not None
            and metrics["heldout_auroc"] >= 0.8
            and fpr <= alpha
        )
    temperatures = {}
    for kind in sorted({r[0] for r in cal["predicate_records"]}):
        rows = [(r[1], r[2]) for r in cal["predicate_records"] if r[0] == kind]
        held = [(r[1], r[2]) for r in gate["predicate_records"] if r[0] == kind]
        temperatures[str(kind)] = binary_calibration(rows, held, alpha)
    outcome_calibrations = {
        name: binary_calibration(
            cal["outcome_records"][name], gate["outcome_records"][name], alpha
        )
        for name in ("transition", "handoff")
    }
    outcome_calibrations["progress"] = progress_calibration(
        cal["outcome_records"]["progress"], gate["outcome_records"]["progress"]
    )
    result = dict(
        schema="tgcc.validation_calibration.v2",
        source_sha256=md["source_sha256"],
        student_sha256=md["student_sha256"],
        new_updates=6000,
        cumulative_steps=11000,
        checkpoint_sha256=md["checkpoint_sha256"],
        alpha=alpha,
        calibration_episodes=groups["calibration"],
        gate_episodes=groups["gate"],
        evaluation_instances_used=False,
        action_energy_threshold=threshold,
        heldout_action_ranking_auroc=auc,
        heldout_native_admissibility_auroc=physical_auc,
        negative_metrics_by_kind=per_kind,
        maximum_negative_counts_per_positive={"physical": 1, "task": 1},
        heldout_positive_false_alarm_rate=fpr,
        action_energy_gate_enabled=ranking_gates["physical"],
        action_ranking_gate_enabled=ranking_gates["physical"],
        task_ranking_gate_enabled=ranking_gates["task"],
        ranking_scope=(
            "task_contract"
            if ranking_gates["task"]
            else "physical_limits"
            if ranking_gates["physical"]
            else "disabled"
        ),
        ranking_candidate_count=2,
        ranking_candidate_count_bounds=[2, 4],
        ranking_minimum_paired_count_per_split=64,
        ranking_minimum_same_context_accuracy=0.8,
        retrospective_gate_enabled=bool(
            outcome_calibrations["transition"]["enabled"]
            and outcome_calibrations["progress"]["enabled"]
        ),
        handoff_gate_enabled=outcome_calibrations["handoff"]["enabled"],
        outcome_calibrations=outcome_calibrations,
        learned_outputs_are_advisory=True,
        task_completion_authority="verified_physical_contract_evidence",
        analytic_contract_gate_enabled=True,
        shadow_chunks_per_episode=16,
        predicate_temperatures=temperatures,
        action_energy_scope="native control-limit gate; separately reported certified preserve-grasp ranking; not a task-success probability",
        validation_samples_per_episode=16,
        manifest_sha256=md["manifest_sha256"],
        plan_sha256=md["plan_sha256"],
        effective_negative_counts={
            name: {
                kind: len(energies)
                for kind, energies in group["negative_by_kind"].items()
            }
            for name, group in results.items()
        },
        independent_ood_head_enabled=False,
        unsupported_learned_predicates=[
            "open",
            "inside",
            "on_top",
            "toggled_on",
            "cooked",
            "operable",
        ],
        critic_role="evaluation_only",
        correction_dispatch_enabled=False,
    )
    (a.tgcc / "calibration.json").write_text(json.dumps(result, indent=2))
    (a.tgcc / "calibration_samples.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
    raw.close()
    tap.close()


if __name__ == "__main__":
    main()
