"""Observed, timestamped three-valued task contracts; no annotation-time progress."""

from dataclasses import dataclass
from enum import IntEnum
import math


class Truth(IntEnum):
    UNKNOWN = -1
    FALSE = 0
    TRUE = 1


@dataclass(frozen=True)
class Evidence:
    value: object = None
    timestamp: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    frame: str = "world"
    source: str = "unknown"
    information_source: str = "OBSERVED"

    def truth(self, now, max_age=2.2):
        if (
            not self.valid
            or type(self.value) is not bool
            or not math.isfinite(self.timestamp)
            or self.timestamp > now
            or now - self.timestamp > max_age
            or self.confidence < 0.5
            or not math.isfinite(self.confidence)
        ):
            return Truth.UNKNOWN
        return Truth.TRUE if self.value else Truth.FALSE

    def truth_for(self, predicate, now):
        if self.confidence < predicate.minimum_confidence:
            return Truth.UNKNOWN
        if self.information_source != "OBSERVED" and (
            predicate.requires_fresh_observation or not predicate.memory_acceptable
        ):
            return Truth.UNKNOWN
        return self.truth(now, predicate.maximum_observation_age)


@dataclass(frozen=True)
class Predicate:
    kind: str
    subject: str
    target: str = ""
    expected: bool = True
    tolerance: float = 0.0
    frame: str = "world"
    category: str = "post"
    required_at_current_stage: bool = True
    requires_fresh_observation: bool = False
    memory_acceptable: bool = True
    minimum_confidence: float = 0.5
    maximum_observation_age: float = 2.2

    def __post_init__(self):
        if not 0 <= self.minimum_confidence <= 1 or self.maximum_observation_age <= 0:
            raise ValueError("Invalid per-predicate evidence policy")

    @property
    def key(self):
        return (self.kind, self.subject, self.target)


@dataclass(frozen=True)
class TaskContract:
    skill: int
    candidate_id: str
    predicates: tuple

    def partitions(self, capacity=16):
        # Explicit partitions preserve ALL clauses, including mandatory invariants.
        return [
            self.predicates[i : i + capacity]
            for i in range(0, len(self.predicates), capacity)
        ]


class ContractEvaluator:
    def __init__(self, max_age=2.2, consecutive=3):
        self.max_age = max_age
        self.consecutive = consecutive
        self.streak = {}
        self.last = {}

    def reset(self):
        self.streak.clear()
        self.last.clear()

    def evaluate(self, episode, contract, evidence, now):
        states = []
        for p in contract.predicates:
            e = evidence.get(p.key, Evidence())
            t = e.truth_for(p, now)
            if t != Truth.UNKNOWN:
                t = Truth.TRUE if bool(t) == p.expected else Truth.FALSE
            states.append(t)
        rates = {}
        for cat in ("pre", "post", "invariant", "handoff"):
            v = [
                s
                for p, s in zip(contract.predicates, states)
                if p.category == cat and p.required_at_current_stage
            ]
            known = [x for x in v if x != Truth.UNKNOWN]
            rates[cat] = {
                "satisfied": sum(x == Truth.TRUE for x in known),
                "known": len(known),
                "total": len(v),
                "all_known_true": bool(v)
                and len(known) == len(v)
                and all(x == Truth.TRUE for x in v),
            }
        key = (episode, contract.candidate_id)
        if now < self.last.get(key, -math.inf):
            raise ValueError("Stale contract observation")
        # Repeated evaluation of the same exposure must NOT advance debounce.
        # An unchanged historical estimate is NOT three new confirmations.
        post_sources = [
            evidence.get(p.key, Evidence())
            for p in contract.predicates
            if p.category == "post" and p.required_at_current_stage
        ]
        fresh_completion = bool(post_sources) and all(
            e.information_source == "OBSERVED" and e.timestamp == now
            for e in post_sources
        )
        if now != self.last.get(key):
            self.streak[key] = (
                self.streak.get(key, 0) + 1
                if rates["post"]["all_known_true"]
                and fresh_completion
                and not any(
                    p.required_at_current_stage
                    and p.category == "invariant"
                    and s != Truth.TRUE
                    for p, s in zip(contract.predicates, states)
                )
                else 0
            )
            self.last[key] = now
        return {
            "states": states,
            "rates": rates,
            "unknown_count": states.count(Truth.UNKNOWN),
            "completed": self.streak.get(key, 0) >= self.consecutive,
            "handoff_ready": rates["handoff"]["all_known_true"],
            "violated": [
                p.key
                for p, s in zip(contract.predicates, states)
                if p.required_at_current_stage and s == Truth.FALSE
            ],
            "required_unknown": [
                p.key
                for p, s in zip(contract.predicates, states)
                if p.required_at_current_stage
                and p.category in ("pre", "invariant")
                and s == Truth.UNKNOWN
            ],
            "information_sources": [
                evidence.get(p.key, Evidence()).information_source
                if s != Truth.UNKNOWN
                else "UNKNOWN"
                for p, s in zip(contract.predicates, states)
            ],
            "contract_residual": sum(
                p.required_at_current_stage
                and p.category in ("pre", "invariant")
                and s == Truth.FALSE
                for p, s in zip(contract.predicates, states)
            ),
        }


def build_contract(candidate, uncompleted_food=(), preserved_grasps=()):
    def flat(v):
        return [a for x in v for a in (flat(x) if isinstance(x, list) else [x])]

    ids = candidate["skill_ids"]
    skill = ids[0] if len(ids) == 1 else -1
    targets = flat(candidate["targets"])
    clauses = []
    subject = targets[0] if targets else "UNKNOWN"
    target = targets[1] if len(targets) > 1 else ""

    def p(kind, s=subject, t="", expected=True, category="post"):
        # Existence/last reliable localization is not current camera visibility.
        # Contact, containment and articulation change during manipulation and
        # require fresh evidence; stationary appliance localization can persist.
        static = any(
            w in s for w in ("refrigerator", "microwave", "cabinet", "floor", "room")
        )
        fresh = kind in ("held", "inside", "on_top", "open", "toggled_on", "operable")
        clauses.append(
            Predicate(
                kind,
                s,
                t,
                expected=expected,
                category=category,
                requires_fresh_observation=fresh,
                memory_acceptable=not fresh,
                minimum_confidence=0.6 if fresh else 0.5,
                maximum_observation_age=(600.0 if static else 2.2)
                if not fresh
                else (1.2 if kind in ("held", "operable") else 2.2),
            )
        )

    p("observed", category="pre")
    if skill == 1:
        p(
            "operable", category="post"
        )  # must be backed by real reachability/operating-zone evidence
    elif skill == 10:
        p("open", expected=False, category="pre")
        p("open")
    elif skill == 12:
        p("open", category="pre")
        p("open", expected=False)
    elif skill == 2:
        p("held", expected=False, category="pre")
        p("held")
        p("held", category="handoff")
    elif skill in (3, 4):
        p("held", category="pre")
        p("inside" if skill == 4 else "on_top", t=target)
        p("held", expected=False)
        if skill == 4 and target:
            p("open", s=target, category="pre")
        p("inside" if skill == 4 else "on_top", t=target, category="handoff")
    elif skill == 69:
        p("open", expected=False, category="pre")
        p("toggled_on")
        p("open", expected=False, category="invariant")
        for food in uncompleted_food:
            p("inside", s=food, t=subject, category="pre")
            p("inside", s=food, t=subject, category="invariant")
    else:
        p("unsupported_contract", category="post")
    if skill not in (3, 4):
        for held_name in preserved_grasps:
            p("held", s=held_name, category="invariant")
    return TaskContract(skill, candidate["id"], tuple(clauses))


class NegativeValidityChecker:
    """Hard negatives require actual native-space bounds, not generic normalized limits."""

    def __init__(self, spec):
        self.spec = spec

    def violations(self, native_actions, valid):
        import torch

        lo = native_actions.new_tensor(self.spec.lower)
        hi = native_actions.new_tensor(self.spec.upper)
        known = torch.tensor(self.spec.bounds_valid, device=native_actions.device)
        return (
            ((native_actions < lo) | (native_actions > hi)) & known & valid[..., None]
        ).any((-2, -1))

    def generate(self, positive_native, valid, normalize, max_count=3, offset=0):
        import torch

        if valid.shape != positive_native.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Action validity must be boolean batch x time")
        known = [i for i, v in enumerate(self.spec.bounds_valid) if v]
        out = []
        # Rotate the actual training negatives across native subsystems, lower
        # and upper limits, and impulse versus sustained violations. One
        # candidate per positive remains the formal training default.
        for k in range(min(max_count, len(known))):
            schedule = offset + k
            i = known[schedule % len(known)]
            lower = (schedule // len(known)) % 2 == 1
            impulse = (schedule // (2 * len(known))) % 2 == 1
            margin = max(0.1, 0.1 * abs(self.spec.upper[i] - self.spec.lower[i]))
            value = (
                self.spec.lower[i] - margin if lower else self.spec.upper[i] + margin
            )
            x = positive_native.clone()
            changed = valid.clone()
            if impulse:
                # Per-row first real command; never corrupt padding to claim a
                # violation or invent an unexecuted failure transition.
                first = valid.to(torch.int64).argmax(-1)
                changed.zero_()
                changed.scatter_(1, first[:, None], valid.any(-1)[:, None])
            x[..., i] = torch.where(changed, x.new_tensor(value), x[..., i])
            proof = self.violations(x, valid)
            if proof.any():
                out.append(
                    (
                        normalize(x),
                        proof,
                        {
                            "kind": "native_joint_or_velocity_limit",
                            "dimension": i,
                            "limit_side": "lower" if lower else "upper",
                            "duration": "impulse" if impulse else "sustained",
                            "not_a_failure_trajectory": True,
                        },
                    )
                )
        return out

    def task_negative(self, native, valid, preserve_gripper_mask, normalize):
        """Certified release of an explicitly preserved grasp. +1 commands
        fully open in the audited smooth-gripper mapping, not a normalized
        joint-angle threshold. Only positive chunks that keep it closed qualify.
        No kinematic reachability estimate is treated as a collision certificate.
        """
        import torch

        mask = (
            torch.zeros(len(native), 2, dtype=torch.bool, device=native.device)
            if preserve_gripper_mask is None
            else preserve_gripper_mask
        )
        if mask.shape != (len(native), 2) or mask.dtype != torch.bool:
            raise ValueError("Preserved grasp mask must be boolean batch x 2")
        if valid.shape != native.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Action validity must be boolean batch x time")
        x = native.clone()
        proof = torch.zeros(len(native), dtype=torch.bool, device=native.device)
        for side, dimension in enumerate((14, 22)):
            eligible = (
                mask[:, side]
                & valid.any(-1)
                & ((native[:, :, dimension] < -0.5) | ~valid).all(-1)
                & ~proof
            )
            x[eligible, :, dimension] = self.spec.upper[dimension]
            proof |= eligible
        proof &= ~self.violations(x, valid)
        return (
            normalize(x),
            proof,
            {
                "kind": "explicit_preserve_grasp_invariant",
                "not_a_failure_trajectory": True,
            },
        )
