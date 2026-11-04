"""Bounded evaluation feedback. No failure-reason or recovery-class ontology.

This supervisor reuses the trained fixed skill vocabulary. It does not generate
free-text plans, certify geometric returns, or infer terminal task outcomes.
"""

from dataclasses import replace
import hashlib
import math

import torch

from .contracts import Evidence, Truth


class ControlSafetyStop(RuntimeError):
    """No executable candidate exists; no action is dispatched."""


def known_precondition_status(contract, evidence, now):
    values = []
    for predicate in contract.predicates:
        if predicate.category != "pre" or not predicate.required_at_current_stage:
            continue
        truth = evidence.get(predicate.key, Evidence()).truth_for(predicate, now)
        values.append(Truth.UNKNOWN if truth == Truth.UNKNOWN else
                      Truth.TRUE if (truth == Truth.TRUE) == predicate.expected else Truth.FALSE)
    if Truth.FALSE in values:
        return "FALSE"
    if values and all(v == Truth.TRUE for v in values):
        return "TRUE"
    return "UNKNOWN"


def observed_invariant_loss(contract, before, after, before_time, after_time):
    """An actual TRUE -> FALSE required invariant is local execution failure.

    A missing/stale/future measurement and an unmet postcondition are excluded.
    The returned value is an assessment; it carries no diagnosis or remedy.
    """
    for predicate in contract.predicates:
        if predicate.category != "invariant" or not predicate.required_at_current_stage:
            continue
        earlier = before.get(predicate.key, Evidence()).truth_for(predicate, before_time)
        later = after.get(predicate.key, Evidence()).truth_for(predicate, after_time)
        if (earlier != Truth.UNKNOWN and later != Truth.UNKNOWN
                and (earlier == Truth.TRUE) == predicate.expected
                and (later == Truth.TRUE) != predicate.expected):
            return True
    return False


def calibrated_binary(logit, calibration):
    """Return probability and decision only for an enabled finite calibration."""
    if not calibration.get("enabled", False):
        return None, None
    temperature = float(calibration.get("temperature", float("nan")))
    threshold = float(calibration.get("threshold", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < threshold < 1:
        raise ValueError("Invalid enabled binary calibration")
    probability = float(torch.as_tensor(logit).float().div(temperature).sigmoid())
    return probability, probability >= threshold


def action_gate(calibration, energy):
    """An energy threshold is scoped to certified action ranking only."""
    enabled = bool(calibration.get("action_energy_gate_enabled", False))
    if not enabled:
        return False
    threshold = float(calibration.get("action_energy_threshold", float("nan")))
    if not math.isfinite(threshold) or not math.isfinite(float(energy)):
        raise ValueError("Invalid enabled action calibration")
    return float(energy) > threshold


def choose_five(energies, legal, *, ranking_enabled):
    """Choose from exactly five *new* draws; unsafe controls never participate."""
    if len(energies) != 5 or len(legal) != 5:
        raise ValueError("Action regeneration requires exactly five fresh candidates")
    allowed = [i for i, valid in enumerate(legal) if valid]
    if not allowed:
        raise ControlSafetyStop("All five regenerated action candidates violate control constraints")
    if not all(math.isfinite(float(v)) for v in energies):
        raise FloatingPointError("Nonfinite candidate energy")
    return min(allowed, key=lambda i: float(energies[i])) if ranking_enabled else allowed[0]


def feedback_conditions(conditions, contracts, evidence, now, feedback, *, retry_index=None):
    """Send structured failure status through existing graph and legal queries.

    The original instruction and vocabulary stay fixed. A new prefix consumes
    this graph event and reselects an executable skill; no old rule is dispatched.
    """
    if feedback not in ("execution_failed", "planning_failed"):
        raise ValueError("Unsupported evaluation feedback")
    legal = conditions.legal.clone()
    for index, contract in enumerate(contracts):
        if known_precondition_status(contract, evidence, now) == "FALSE":
            legal[:, index] = False
        # This vocabulary includes an explicitly unsupported reposition query.
        if contract.candidate_id == "plan:REPOSITION_BASE" or contract.skill < 0:
            legal[:, index] = False
    if retry_index is not None and bool(legal[:, retry_index].all()):
        legal.zero_()
        legal[:, retry_index] = True
    if not bool(legal.any(-1).all()):
        raise ControlSafetyStop("No legal fixed skill remains after evaluation feedback")
    graph = conditions.task_graph
    events = (graph.features.new_zeros(len(legal), 64, 32)
              if graph.events is None else graph.events.clone())
    valid = (torch.zeros(len(legal), 64, device=legal.device, dtype=torch.bool)
             if graph.event_valid is None else graph.event_valid.clone())
    for batch in range(len(legal)):
        count = int(valid[batch].sum())
        position = min(count, events.shape[1] - 1)
        if count >= events.shape[1]:
            events[batch, :-1] = events[batch, 1:].clone()
            valid[batch, :-1] = valid[batch, 1:].clone()
        event = events.new_zeros(32)
        event[0] = 1.0
        event[2] = int.from_bytes(hashlib.sha256(feedback.encode()).digest()[:4], "big") % 256 / 255
        event[4] = 1.0  # assessed failed status, not a physical terminal-goal bit
        events[batch, position] = event
        valid[batch, position] = True
    return replace(conditions, legal=legal,
                   task_graph=replace(graph, events=events, event_valid=valid),
                   event_changed=torch.ones(len(legal), device=legal.device, dtype=torch.bool))
