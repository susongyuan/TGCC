"""Semantic training gates: valid counts must never stand for positive coverage."""


def supervision_gate(report):
    """Return explicit failures without reinterpreting UNKNOWN as a negative.

    Numeric coverage enumerates candidate contracts, so contract satisfaction
    (e.g. `not-held`) is not a recorded positive grasp. Only the separate raw
    evidence counters may certify grasp-positive coverage.
    """
    failures = []
    for split in ("train", "validation"):
        raw = report.get("recorded_evidence", {}).get(split, {})
        supervision = report.get("supervision", {}).get(split, {})
        negatives = report.get("negative_certificates", {}).get(split, {})
        for key in ("held_positive", "held_negative"):
            if raw.get(key, 0) <= 0:
                failures.append(f"{split}: no recorded {key} evidence")
        for key in ("progress_positive", "handoff_positive"):
            if supervision.get(key, 0) <= 0:
                failures.append(f"{split}: no {key} supervision")
        if negatives.get("task_valid_sequences_potential", 0) <= 0:
            failures.append(f"{split}: no certified task negatives")
        timing = report.get("timing", {}).get(split, {})
        if timing.get("failures", 0) or timing.get("checked_sequences", 0) <= 0:
            failures.append(f"{split}: missing or failed executed-prefix alignment")
    return dict(status="failed" if failures else "passed", failures=failures,
                valid_counts_are_not_positive_coverage=True)


def require_supervision(report):
    result = supervision_gate(report)
    if result["status"] != "passed":
        raise ValueError("Unsupported formal supervision: " + "; ".join(result["failures"]))
    return result
