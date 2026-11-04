"""Proof-carrying action pairs. No synthetic execution outcomes are produced.

A repaired action is an action-space target under the SAME pre-action context;
it is not a recovery trajectory from an executed, perturbed physical state.
"""

from dataclasses import dataclass
import numpy as np
from .contracts import Truth


@dataclass(frozen=True)
class CommandConstraint:
    """Explicit stage contract in native units, not an estimated reachability test.

    Examples: a verified controller's no-base-motion phase or a preserved grasp.
    The producer must provide the actual clause and supporting current evidence.
    """

    constraint_id: str
    context_id: str
    dimension: int
    lower: float
    upper: float
    evidence_key: tuple
    expected: bool = True


@dataclass
class CertifiedPair:
    context_id: str
    kind: str
    positive_native: np.ndarray
    negative_native: np.ndarray
    action_mask: np.ndarray
    certificate: dict

    def record(self):
        return dict(
            context_id=self.context_id,
            kind=self.kind,
            positive_native=self.positive_native.tolist(),
            negative_native=self.negative_native.tolist(),
            action_mask=self.action_mask.tolist(),
            certificate=self.certificate,
            state_transition_supervision=False,
            repair_target_semantics="same_context_action_space_only",
            executed_failure=False,
        )


def construct_pairs(
    native, valid, spec, context_id, offset=0, *, constraints=(), evidence=None, now=0.0
):
    """Return <=1 physical and <=1 task pair, or masks as explicit reasons.

    Bounds comparisons use float32 exactly like online native controls. Only
    executed-prefix positions are modified. Unknown/stale evidence never proves
    a task negative. Per-dimension command contracts must be physically feasible.
    """
    x = np.asarray(native, dtype=np.float32)
    mask = np.asarray(valid)
    lo, hi = np.asarray(spec.lower, np.float32), np.asarray(spec.upper, np.float32)
    known = np.asarray(spec.bounds_valid, bool)
    if (
        x.ndim != 2
        or x.shape[-1] != len(lo)
        or mask.shape != x.shape[:1]
        or mask.dtype != bool
    ):
        raise ValueError(
            "Expected native time x action and boolean executed-prefix mask"
        )
    if (
        not np.isfinite(x).all()
        or not np.isfinite(lo[known]).all()
        or not np.isfinite(hi[known]).all()
    ):
        raise ValueError("Nonfinite actions/bounds")
    if not mask.any():
        return [], {
            "physical": "empty_executed_prefix",
            "task": "empty_executed_prefix",
        }
    if ((((x < lo) | (x > hi)) & known)[mask]).any():
        return [], {
            "physical": "positive_outside_native_limits",
            "task": "positive_outside_native_limits",
        }
    pairs, rejected = [], {}
    dimensions = np.flatnonzero(known)
    if len(dimensions):
        d = int(dimensions[offset % len(dimensions)])
        upper = (offset // len(dimensions)) % 2 == 0
        y = x.copy()
        # Alternate a single command impulse and a sustained invalid command;
        # both have the same controller-limit proof, no invented failure labels.
        indices = np.flatnonzero(mask)
        changed = (
            indices
            if (offset // (len(dimensions) * 2)) % 2
            else indices[offset % len(indices) : offset % len(indices) + 1]
        )
        margin = max(0.1, 0.1 * float(hi[d] - lo[d]))
        value = np.float32(hi[d] + margin if upper else lo[d] - margin)
        y[changed, d] = value
        if not (value > hi[d] if upper else value < lo[d]):
            raise ValueError("Negative construction failed to cross exact native bound")
        pairs.append(
            CertifiedPair(
                context_id,
                "physical",
                x.copy(),
                y,
                mask.copy(),
                dict(
                    rule="native_controller_limit",
                    dimension=d,
                    dimension_name=spec.names[d],
                    native_unit=spec.units[d],
                    bound=float(hi[d] if upper else lo[d]),
                    bound_side="upper" if upper else "lower",
                    candidate_value=float(value),
                    modified_steps=changed.tolist(),
                    subtype="sustained" if len(changed) > 1 else "impulse",
                ),
            )
        )
    else:
        rejected["physical"] = "no_verified_native_bounds"
    evidence = evidence or {}
    usable = []
    for c in constraints:
        if c.context_id != context_id:
            raise ValueError("Task constraint belongs to another pre-action context")
        d = c.dimension
        if (
            not 0 <= d < len(lo)
            or not known[d]
            or not np.isfinite([c.lower, c.upper]).all()
            or not lo[d] <= c.lower <= c.upper <= hi[d]
        ):
            raise ValueError("Task constraint is not feasible in audited native units")
        ev = evidence.get(c.evidence_key)
        state = Truth.UNKNOWN if ev is None else ev.truth(now)
        if (
            state == Truth.UNKNOWN
            or ev.information_source not in ("OBSERVED", "MEMORY_INFERRED")
            or bool(state == Truth.TRUE) != c.expected
        ):
            continue
        if not ((x[mask, d] >= c.lower) & (x[mask, d] <= c.upper)).all():
            continue
        choices = ([float(lo[d])] if lo[d] < c.lower else []) + (
            [float(hi[d])] if hi[d] > c.upper else []
        )
        if choices:
            usable.append((c, ev, choices))
    if usable:
        c, ev, choices = usable[offset % len(usable)]
        y = x.copy()
        value = choices[offset % len(choices)]
        y[mask, c.dimension] = value
        pairs.append(
            CertifiedPair(
                context_id,
                "task",
                x.copy(),
                y,
                mask.copy(),
                dict(
                    rule="explicit_stage_native_command_constraint",
                    constraint_id=c.constraint_id,
                    dimension=c.dimension,
                    dimension_name=spec.names[c.dimension],
                    native_unit=spec.units[c.dimension],
                    permitted_interval=[c.lower, c.upper],
                    candidate_value=value,
                    evidence_key=list(c.evidence_key),
                    evidence_source=ev.source,
                    information_source=ev.information_source,
                    evidence_timestamp=ev.timestamp,
                    confidence=ev.confidence,
                ),
            )
        )
    else:
        rejected["task"] = "no_verified_satisfied_current_stage_command_constraint"
    return pairs, rejected


def preserved_grasp_constraints(
    context_id, evidence, spec, now, *, preserved_objects=()
):
    """Require an explicit preserve-grasp clause AND reliable side ownership evidence."""
    out = []
    for key, ev in evidence.items():
        # Side ownership is explicit. Generic ('held',object,'') is insufficient.
        if (
            len(key) != 3
            or key[0] != "held"
            or key[1] not in preserved_objects
            or key[2] not in ("left", "right")
            or ev.truth(now) != Truth.TRUE
        ):
            continue
        d = 14 if key[2] == "left" else 22
        out.append(
            CommandConstraint(
                "preserve_grasp:" + ":".join(key[1:]),
                context_id,
                d,
                float(spec.lower[d]),
                -0.5,
                key,
            )
        )
    return tuple(out)
