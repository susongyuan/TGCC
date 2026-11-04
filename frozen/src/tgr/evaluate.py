"""Fail-closed paired receipt comparator. Does NOT launch an untrained policy."""

import argparse, json, math
from pathlib import Path

PROTOCOL_KEYS = (
    "task",
    "instances",
    "action_seed",
    "environment_seed",
    "action_horizon",
    "executed_chunk",
    "max_steps",
    "norm_stats_sha256",
    "base_weight_sha256",
    "oracle_inputs",
    "camera_capture",
    "model_image_size",
    "physics_hz",
    "action_hz",
    "render_hz",
    "texture_budget",
    "action_mapping",
    "evaluation_domain",
)


def validate_results(results, config):
    if not isinstance(results, list):
        raise ValueError("missing result list")
    expected = config["instances"]
    seen = {}
    out = {}
    for row in results:
        instance = row["instance_id"]
        if instance in seen:
            raise ValueError("duplicate attempt/instance; cannot cherry-pick successes")
        seen[instance] = True
        if instance not in expected:
            raise ValueError("unpaired instance")
        if row.get("status") != "completed":
            raise ValueError("infrastructure interruption is not a valid task result")
        success = row.get("success")
        if type(success) is not bool:
            raise ValueError("success must be boolean")
        if not (row.get("terminated") or row.get("truncated")):
            raise ValueError("no natural task terminal state")
        q = row.get("metrics", {}).get("q_score", {}).get("final")
        if not isinstance(q, (float, int)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("missing/invalid official final Q")
        if success and q != 1.0:
            raise ValueError("success/Q disagreement")
        if (
            not isinstance(row.get("steps"), int)
            or not 0 < row["steps"] <= config["max_steps"] + 1
        ):
            raise ValueError("invalid task step count")
        out[instance] = {
            "success": int(success),
            "q": float(q),
            "steps": row["steps"],
            "elapsed_seconds": row.get("elapsed_seconds"),
        }
    if set(out) != set(expected):
        raise ValueError("all ten paired instances required; no partial final result")
    return out


def validate_protocol(protocol, config):
    for k in PROTOCOL_KEYS:
        if k not in protocol or protocol[k] != config[k]:
            raise ValueError("unmatched protocol: " + k)
    if protocol["oracle_inputs"]:
        raise ValueError("oracle upper bound is not comparable to RGB/proprio baseline")


def compare(baseline, candidate, config):
    validate_protocol(baseline["protocol"], config)
    validate_protocol(candidate["protocol"], config)
    if candidate.get("graph_checkpoint_trained") is not True or not candidate.get(
        "graph_checkpoint_sha256"
    ):
        raise ValueError("untrained or unidentified candidate forbidden")
    if not candidate.get("source_hashes") or candidate.get(
        "source_changed_during_run", True
    ):
        raise ValueError("candidate execution source not proven frozen")
    b = validate_results(baseline["results"], config)
    c = validate_results(candidate["results"], config)
    pairs = [
        {
            "instance_id": i,
            "baseline": b[i],
            "candidate": c[i],
            "delta_q": c[i]["q"] - b[i]["q"],
        }
        for i in config["instances"]
    ]
    n = len(pairs)
    wins = sum(c[i]["success"] > b[i]["success"] for i in b)
    losses = sum(c[i]["success"] < b[i]["success"] for i in b)
    discordant = wins + losses
    p = (
        1.0
        if not discordant
        else min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
            / 2**discordant,
        )
    )
    return {
        "paired_n": n,
        "baseline_success_rate": sum(r["success"] for r in b.values()) / n,
        "candidate_success_rate": sum(r["success"] for r in c.values()) / n,
        "baseline_mean_q": sum(r["q"] for r in b.values()) / n,
        "candidate_mean_q": sum(r["q"] for r in c.values()) / n,
        "improved_success_pairs": wins,
        "regressed_success_pairs": losses,
        "mcnemar_exact_two_sided_p": p,
        "pairs": pairs,
        "limitation": "Ten fixed paired instances, one action seed: exploratory evidence, not generalization proof.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate")
    p.add_argument("--output")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    b = json.loads(Path(a.baseline).read_text())
    if a.candidate:
        if not a.output:
            p.error("--output required with --candidate")
        r = compare(b, json.loads(Path(a.candidate).read_text()), config)
        Path(a.output).write_text(json.dumps(r, indent=2))
    else:
        validate_protocol(b["protocol"], config)
        rs = validate_results(b["results"], config)
        r = {
            "status": "baseline_pairing_validated",
            "n": len(rs),
            "success_rate": sum(x["success"] for x in rs.values()) / len(rs),
            "mean_q": sum(x["q"] for x in rs.values()) / len(rs),
            "candidate_evaluation_started": False,
        }
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
