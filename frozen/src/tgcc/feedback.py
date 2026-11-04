"""Shared analytic legality and causal task-token feedback, without phase labels."""

from dataclasses import replace
from .contracts import Evidence, Truth, ContractEvaluator, build_contract


def filter_legal(conditions, contracts, evidence, now):
    legal = conditions.legal.clone()
    blocked = []
    for i, contract in enumerate(contracts):
        prefalse = False
        for p in contract.predicates:
            t = evidence.get(p.key, Evidence()).truth_for(p, now)
            if (
                p.category == "pre"
                and p.required_at_current_stage
                and t != Truth.UNKNOWN
                and (t == Truth.TRUE) != p.expected
            ):
                prefalse = True
        subject = contract.predicates[0].subject
        unsupported = contract.skill in (10, 12) and not any(
            w in subject for w in ("refrigerator", "microwave")
        )
        unsupported |= contract.skill == 4 and any(
            p.target.startswith("floor.") for p in contract.predicates
        )
        unsupported |= contract.skill == 1 and subject == "UNKNOWN"
        if prefalse or unsupported:
            legal[:, i] = False
            blocked.append(contract.candidate_id)
    # An empty legal set requests recovery; do not fabricate a legal skill.
    if legal.any():
        conditions.legal = legal
    return blocked, bool(legal.any())


def training_feedback(conditions, aux, plan):
    """The task prefix reads only PREVIOUS physical evidence. Current facts
    may constrain current selection but never stand in for a preceding write.
    """
    features = conditions.task_graph.features.clone()
    for b, a in enumerate(aux):
        if a["frame"] < 512:
            continue
        contracts = [
            build_contract(
                c,
                ("hotdog.n.02_1", "hotdog.n.02_2")
                if c["skill_ids"] == [69] and "microwave" in str(c["targets"])
                else (),
            )
            for c in plan["candidates"]
        ]
        one = replace(conditions, legal=conditions.legal[b : b + 1].clone())
        filter_legal(one, contracts, a["evidence"], float(a["timestamp"][0]))
        conditions.legal[b : b + 1] = one.legal
        for k, contract in enumerate(contracts):
            evaluator = ContractEvaluator()
            result = None
            for when, evidence in a["previous_facts"]:
                if when >= float(a["timestamp"][0]):
                    raise ValueError("Future task-prefix contract feedback")
                result = evaluator.evaluate(a["episode_id"], contract, evidence, when)
            if result is None:
                continue
            slot = int(conditions.task_candidate_slots[b, k])
            post = result["rates"]["post"]
            if result["completed"]:
                features[b, -1, slot, 17:23] = 0
                features[b, -1, slot, 22] = 1
                features[b, -1, slot, 29] = 1
                features[b, -1, slot, 0] = 1
            elif post["known"] == post["total"] and not post["all_known_true"]:
                features[b, -1, slot, 17:23] = 0
                features[b, -1, slot, 17] = 1
                features[b, -1, slot, 29] = 0
                features[b, -1, slot, 0] = 1
    conditions.task_graph = replace(conditions.task_graph, features=features)
    return conditions
