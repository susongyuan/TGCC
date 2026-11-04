"""Three scoped critic assessments and bounded fixed-vocabulary feedback."""
from dataclasses import asdict, replace
from collections import deque
import contextlib
import pickle
import numpy as np
import torch
from .contracts import Evidence, Truth, ContractEvaluator, NegativeValidityChecker, build_contract
from .data import HISTORY_WIDTH, pose9, critic_observation, transition_labels
from .feature_cache import slice_observation
from .policy import PolicyFeatures
from .specs import ExecutedTransition
from .evaluation_supervisor import (ControlSafetyStop, action_gate, calibrated_binary,
                                    choose_five, feedback_conditions,
                                    known_precondition_status, observed_invariant_loss)

def snapshot_evidence(snapshot, now):
    out = {}
    nodes = {n["entity_id"]: n for n in snapshot["entities"]}
    for n in nodes.values():
        name = n["instance_id"]
        t = float(n.get("last_seen", n["timestamp"]))
        conf = float(n["confidence"])
        out[("observed", name, "")] = Evidence(
            True if n["validity"] and n["pose"] is not None else None,
            t,
            conf,
            bool(n["validity"] and n["pose"] is not None),
            source=n["source"],
            information_source=n.get("attributes", {}).get(
                "information_source", "OBSERVED"
            ),
        )
        for kind in ("open", "cooked", "toggled_on"):
            value = n.get("attributes", {}).get(kind)
            if type(value) is bool:
                out[(kind, name, "")] = Evidence(
                    value,
                    t,
                    conf,
                    bool(n["validity"]),
                    source=n["source"],
                    information_source=n.get("attributes", {}).get(
                        "information_source", "OBSERVED"
                    ),
                )
    holds = {}
    for e in snapshot["edges"]:
        kind = e["relation_type"]
        value = e.get("relative_attributes", {}).get("value")
        if e["source_id"] not in nodes or e["target_id"] not in nodes:
            continue
        source = nodes[e["source_id"]]["instance_id"]
        target = nodes[e["target_id"]]["instance_id"]
        # Edge validity controls positive graph adjacency. The audited oracle
        # exporter intentionally sets it False for measured negative relations;
        # that must not erase reliable FALSE predicate evidence.
        attributes = e.get("relative_attributes", {})
        evidence_valid = attributes.get("evidence_valid")
        if evidence_valid is None:
            evidence_valid = e["validity"] or (
                value is False and e["source"] == "oracle"
            )
        evidence_valid = (
            evidence_valid is True
            and type(value) is bool
            and bool(nodes[e["source_id"]]["validity"])
            and bool(nodes[e["target_id"]]["validity"])
        )
        evidence = Evidence(
            value,
            float(e["timestamp"]),
            float(e["confidence"]),
            evidence_valid,
            source=e["source"],
            information_source=e.get("relative_attributes", {}).get(
                "information_source", "OBSERVED"
            ),
        )
        if kind == "held_by":
            if target in ("robot:ee:left", "robot:ee:right"):
                hands = holds.setdefault(source, {})
                previous = hands.get(target)
                hands[target] = (
                    Evidence()
                    if previous is not None and previous.value != evidence.value
                    else evidence
                )
        elif kind in ("inside", "on_top"):
            out[(kind, source, target)] = evidence
    for name, hands in holds.items():
        values = list(hands.values())
        known = [(v, v.truth(now)) for v in values]
        positives = [v for v, t in known if t == Truth.TRUE]
        negatives = [v for v, t in known if t == Truth.FALSE]
        # A fresh positive on either hand is enough. A stale opposite-hand edge
        # must not backdate that evidence. Absence requires both actual sensors.
        supports = positives if positives else negatives
        value = (
            True
            if positives
            else False
            if len(negatives) == 2 and len(negatives) == len(values)
            else None
        )
        out[("held", name, "")] = Evidence(
            value,
            min((v.timestamp for v in supports), default=now),
            min((v.confidence for v in supports), default=0.0),
            value is not None,
            source="current_grasp_feedback",
            information_source="OBSERVED"
            if supports and all(v.information_source == "OBSERVED" for v in supports)
            else "MEMORY_INFERRED",
        )
    return out

class TGCCController:
    def __init__(self, parent, system, normalizer, plan, calibration, config):
        self.parent, self.system, self.normalizer = parent, system, normalizer
        self.plan, self.calibration, self.config = plan, calibration, config
        self.device = next(parent.parameters()).device
        self.tap = PolicyFeatures(parent)
        self.checker = NegativeValidityChecker(system.critic.action_spec)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=16)
        self.previous = None
        self.evaluator = ContractEvaluator()
        self.chunk = 0
        self.raw = self.conditions = None
        self.current_evidence = {}
        self.trace = {}
        self.critic_previous_decision = None
        self.feedback = None

    def export(self):
        return pickle.dumps(dict(schema="tgcc.evaluation_state.v2", history=self.history,
                                 previous=self.previous, evaluator=self.evaluator,
                                 chunk=self.chunk,
                                 critic_previous_decision=self.critic_previous_decision), protocol=4)

    def restore(self, blob):
        state = pickle.loads(blob)
        if state.pop("schema", None) != "tgcc.evaluation_state.v2":
            raise ValueError("Old correction-controller state is not evaluation state")
        for name, value in state.items():
            setattr(self, name, value)
        self.raw = self.conditions = None
        self.current_evidence, self.trace = {}, {}
        self.feedback = None

    def enrich(self, raw, task_states=None):
        # Evaluation does not modify the policy's task, graph or legal candidates.
        return raw

    def prepare(self, raw, conditions):
        self.raw, self.conditions = raw, conditions
        now = float(raw["sim_time"])
        self.chunk += 1
        self.current_evidence = snapshot_evidence(raw["graph_snapshot"], now)
        self.contracts = self._contracts(self.current_evidence, now)
        self.feedback = None
        self.trace = dict(chunk=self.chunk, mode="three_scope_evaluation_with_bounded_feedback", events=[])
        if self.previous is None:
            return
        receipt = raw.get("execution_receipt")
        if receipt is None:
            raise ValueError("Evaluation requires actual executed-prefix receipt")
        previous = self.previous
        count = int(receipt["step_count"])
        native = np.asarray(receipt["actions"], np.float32)
        if (receipt.get("source") != "native_controller_commands_after_successful_env_step"
                or not 1 <= count <= 32 or native.shape != (count, 23)
                or not np.isfinite(native).all()
                or int(receipt["previous_request_id"]) != previous["request_id"]
                or abs(float(receipt["start_time"]) - previous["timestamp"]) > 1e-7
                or abs(float(receipt["end_time"]) - now) > 1e-7
                or abs(now - previous["timestamp"] - count / 30) > 1e-6
                or not np.array_equal(native, previous["dispatched_native"][:count])):
            raise ValueError("Execution receipt identity/time/action mismatch")
        normalized = self.normalizer.transform(torch.from_numpy(native)).float()
        padded = torch.zeros(1, 32, 23)
        padded[0, :count] = normalized
        mask = torch.zeros(1, 32, dtype=torch.bool)
        mask[:, :count] = True
        previous["executed_actions"], previous["executed_mask"] = padded, mask
        self.history.append(self._history_record(previous, normalized, now))
        lost = observed_invariant_loss(previous["contract"], previous["evidence"],
                                       self.current_evidence, previous["timestamp"], now)
        if lost:
            self.feedback = dict(scope="local_executed_contract", status="failure",
                                 source="observed_required_invariant_true_to_false",
                                 previous_request_id=previous["request_id"],
                                 terminal_subtask_failure=None, reason=None)
            self.trace["events"].append(self.feedback.copy())

    def commit_task(self, runtime):
        # The original task manager consumes only the final selected decision.
        # Neither a retry nor a candidate selection is physical completion.
        return None

    def _contracts(self, evidence, now):
            pending = [
                name
                for name in ("hotdog.n.02_1", "hotdog.n.02_2")
                if evidence.get(("cooked", name, ""), Evidence()).truth(now) != Truth.TRUE
            ]
            return [
                build_contract(
                    c,
                    pending
                    if c["skill_ids"] == [69] and "microwave" in str(c["targets"])
                    else (),
                    preserved_grasps=tuple(
                        k[1]
                        for k, e in evidence.items()
                        if k[0] == "held" and e.truth(now) == Truth.TRUE
                    ),
                )
                for c in self.plan["candidates"]
            ]

    def _history_record(self, previous, actions, end):
            v = np.zeros(HISTORY_WIDTH, np.float32)
            r = previous["robot"][0].numpy()
            v[:96] = previous["graph_summary"]
            v[96:119] = actions.mean(0).numpy()
            v[119:142] = actions[-1].numpy()
            v[142:174] = r[:32]
            v[174:183] = r[32:41]
            v[183:186] = r[41:44]
            v[186:204] = r[44:62]
            # Prior skill slot remains UNKNOWN to match demonstration availability.
            v[227] = end - previous["timestamp"]
            v[228:237] = 1.0
            v[238] = 1.0
            return previous["timestamp"], v

    def _aux(self, observation):
            now = float(self.raw["sim_time"])
            robot = np.zeros(64, np.float32)
            robot[:32] = observation.state[0].detach().float().cpu().numpy()
            f = self.raw["robot_fields"]
            robot[32:41] = pose9(f["base_world"])
            robot[41:44] = np.asarray(self.raw["state"]["base_qvel"])
            nodes = {n["instance_id"]: n for n in self.raw["graph_snapshot"]["entities"]}
            for side, sl in (("left", slice(44, 53)), ("right", slice(53, 62))):
                n = nodes.get("robot:ee:" + side)
                if n and n["pose"] is not None:
                    robot[sl] = pose9(n["pose"])
                    robot[62 if side == "left" else 63] = 1.0
            h = np.zeros((16, HISTORY_WIDTH), np.float32)
            hm = np.zeros(16, bool)
            ht = np.zeros(16, np.float64)
            for i, (t, v) in enumerate(self.history, 16 - len(self.history)):
                h[i] = v
                hm[i] = True
                ht[i] = t
            return dict(
                timestamp=torch.tensor([now], dtype=torch.float64),
                history=torch.from_numpy(h)[None],
                history_mask=torch.from_numpy(hm)[None],
                history_timestamps=torch.from_numpy(ht)[None],
                evidence=self.current_evidence,
                held_sides={
                    n["instance_id"]: tuple(
                        next(
                            x["instance_id"]
                            for x in self.raw["graph_snapshot"]["entities"]
                            if x["entity_id"] == e["target_id"]
                        ).rsplit(":", 1)[-1]
                        for e in self.raw["graph_snapshot"]["edges"]
                        if e["source_id"] == n["entity_id"]
                        and e["relation_type"] == "held_by"
                        and e.get("relative_attributes", {}).get("value") is True
                        and any(
                            x["entity_id"] == e["target_id"]
                            for x in self.raw["graph_snapshot"]["entities"]
                        )
                    )
                    for n in self.raw["graph_snapshot"]["entities"]
                },
                robot_current=torch.from_numpy(robot)[None],
            )

    def _observation(self, decision, aux, actions, contract):
            o = critic_observation(
                self.tap.values, decision, aux, actions, contract, self.conditions
            )
            # Same compact precision as the deterministic frozen-feature training cache.
            o.graph_tokens = tuple(x.half() for x in o.graph_tokens)
            o.semantic_context = o.semantic_context.half()
            return o

    def _clean_prefix(self, observation, conditions):
        # Keep an independent learned memory: the original scheduler's memory
        # may contain privileged task/predicate feedback from earlier chunks.
        from .evaluation_data import clean_conditions
        from .data import conditions_from_dict
        from tgr.task_feedback import roll_forward_task_graph
        cleaned = conditions_from_dict(clean_conditions(asdict(conditions)))
        previous = self.critic_previous_decision
        cleaned.memory = None if previous is None else previous["decision"]["memory"]
        cleaned.memory_tokens = None
        if previous is not None:
            elapsed = torch.tensor([float(self.raw["sim_time"]) - previous["timestamp"]],
                                   device=observation.state.device)
            cleaned.task_graph = roll_forward_task_graph(
                cleaned.task_graph, previous["decision"], cleaned.task_candidate_slots, elapsed)
        self.tap.clear()
        _, _, _, _, decision = self.parent.prefix(observation, cleaned, False)
        return cleaned, decision

    def _score(self, decision, aux, actions, contract, conditions):
        original = self.conditions
        self.conditions = conditions
        try:
            current = self._observation(decision, aux, actions, contract)
        finally:
            self.conditions = original
        with (torch.autocast("cuda", dtype=torch.bfloat16)
              if actions.device.type == "cuda" else contextlib.nullcontext()):
            return current, self.system.critic.forward_action(current)

    def _generate(self, state, valid, cache, tokens, device, noise, num_steps):
        x = self.parent.base.sample_noise((len(state), 32, 32), device) if noise is None else noise.clone()
        dt = -1.0 / num_steps
        for i in range(num_steps):
            time = torch.full((len(x),), 1 + i * dt, device=device, dtype=torch.float32)
            x = x + dt * self.parent.velocity(state, valid, cache, tokens, x, time)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite generated action")
        return x

    def _native_safe(self, actions, observation):
        native = self.normalizer.inverse(actions[..., :23])
        safe = not bool(self.checker.violations(native, observation.candidate_action_mask)[0])
        # The explicit maintained-grasp condition forbids opening a hand whose
        # current contract requires that grasp. Legitimate placements omit it.
        preserve = observation.preserve_gripper_mask
        if preserve is not None:
            for side, dimension in enumerate((14, 22)):
                if bool(preserve[0, side]) and bool((native[0, :, dimension] > 0.5).any()):
                    safe = False
        return native, safe

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10,
                       conditions=None, return_decision=False):
        self.parent.validate(conditions)
        if num_steps < 1:
            raise ValueError("Positive Euler steps required")
        if len(observation.state) != 1:
            raise ValueError("Durable evaluator handles one physical environment")
        now = float(self.raw["sim_time"])
        action_conditions = conditions
        if self.feedback is not None:
            previous_index = next(i for i, c in enumerate(self.contracts)
                                  if c.candidate_id == self.previous["contract"].candidate_id)
            action_conditions = feedback_conditions(
                conditions, self.contracts, self.current_evidence, now,
                "execution_failed", retry_index=previous_index)
            self.trace["fixed_skill_feedback"] = dict(
                status="execution_failed", interface="graph_event_and_legal_skill_queries",
                free_text_planning=False, source_request_id=self.previous["request_id"])
        self.tap.clear()
        state, valid, cache, tokens, decision = self.parent.prefix(observation, action_conditions, False)
        selected = int(decision["selected_index"][0])
        contract = self.contracts[selected]
        pre_status = known_precondition_status(contract, self.current_evidence, now)
        if pre_status == "FALSE":
            # Rebuild the original VLM prefix once with proven incompatible
            # queries removed. The instruction and executable vocabulary stay fixed.
            action_conditions = feedback_conditions(
                action_conditions, self.contracts, self.current_evidence, now, "planning_failed")
            self.tap.clear()
            state, valid, cache, tokens, decision = self.parent.prefix(observation, action_conditions, False)
            selected = int(decision["selected_index"][0])
            contract = self.contracts[selected]
            self.trace["events"].append(dict(scope="subtask_precondition_error", status="failure",
                                              source="known_false_required_precondition", reason=None))
            self.trace["fixed_skill_feedback"] = dict(
                status="planning_failed", interface="graph_event_and_legal_skill_queries",
                free_text_planning=False)
        if not bool(action_conditions.legal[0, selected]) or not bool(decision["subgoal_valid"].all()):
            raise ControlSafetyStop("Fixed skill decoder selected no executable candidate")
        adapted = [(self.system.adapters[k](t), m)
                   for k, (t, m) in zip(("local", "subgoal"), tokens)]
        x = self._generate(state, valid, cache, adapted, device, noise, num_steps)
        cleaned, clean_decision = self._clean_prefix(observation, conditions)
        aux = self._aux(observation)
        current, score = self._score(clean_decision, aux, x, contract, cleaned)
        native, safe = self._native_safe(x, current)
        energy = float(score["action_energy_total"][0])
        rejected = not safe or action_gate(self.calibration, energy)
        candidate_trace = []
        if rejected:
            # The rejected original draw is not one of the five new candidates.
            candidates, contexts, outputs, energies, admissible = [], [], [], [], []
            for candidate_id in range(5):
                candidate = self._generate(state, valid, cache, adapted, device, None, num_steps)
                context, output = self._score(clean_decision, aux, candidate, contract, cleaned)
                _, candidate_safe = self._native_safe(candidate, context)
                e = float(output["action_energy_total"][0])
                candidates.append(candidate)
                contexts.append(context)
                outputs.append(output)
                energies.append(e)
                # A learned rejection is applied only by its held-out gate.
                admissible.append(candidate_safe and not action_gate(self.calibration, e))
                candidate_trace.append(dict(candidate_id=candidate_id, action_energy=e,
                                            native_constraints_satisfied=candidate_safe,
                                            learned_action_accepted=not action_gate(self.calibration, e)))
            ranking = bool(self.calibration.get("action_ranking_gate_enabled", False)
                           or self.calibration.get("task_ranking_gate_enabled", False))
            choice = choose_five(energies, admissible, ranking_enabled=ranking)
            x, current, score = candidates[choice], contexts[choice], outputs[choice]
            native, safe = self._native_safe(x, current)
            self.trace["action_regeneration"] = dict(
                original_rejected=True, fresh_candidates=5, selected_candidate=choice,
                independent_noise=True, ranked_by_calibrated_critic=ranking,
                candidates=candidate_trace)
        planning_calibration = self.calibration.get("planning_precondition_calibration", {})
        pre_mask = current.contract_mask[0] & current.contract_valid[0] & (current.contract_category[0] == 0)
        planning_predictions = []
        if planning_calibration.get("enabled", False):
            for logit in score["precondition_logits"][0, pre_mask]:
                probability, status = calibrated_binary(logit, planning_calibration)
                planning_predictions.append(dict(probability=probability, satisfied=status))
        self.trace.update(selected_candidate=contract.candidate_id,
                          generated_action_energy=float(score["action_energy_total"][0]),
                          native_limits_satisfied=safe,
                          selected_subtask_preconditions_satisfied=known_precondition_status(
                              contract, self.current_evidence, now),
                          calibrated_planning_predictions=planning_predictions,
                          learned_gate_scopes=dict(
                              action=bool(self.calibration.get("action_energy_gate_enabled", False)),
                              planning=bool(planning_calibration.get("enabled", False)),
                              execution=bool(self.calibration.get("outcome_calibrations", {}).get("transition", {}).get("enabled", False))),
                          subtask_success=None, subtask_failure=None)
        previous = self.previous
        if previous is not None:
            before = slice_observation(previous["observation"], 0, device)
            executed = previous["executed_actions"].to(device)
            mask = previous["executed_mask"].to(device)
            after, _ = self._score(clean_decision, aux, x, previous["contract"], cleaned)
            before = replace(before, candidate_actions=executed, candidate_action_mask=mask)
            after = replace(after, candidate_actions=executed, candidate_action_mask=mask)
            transition = ExecutedTransition(before, after, executed, mask, before.timestamp, after.timestamp)
            with (torch.autocast("cuda", dtype=torch.bfloat16)
                  if torch.device(device).type == "cuda" else contextlib.nullcontext()):
                result = self.system.critic.retrospective(transition)
            labels = transition_labels(before, after)
            calibration = self.calibration.get("outcome_calibrations", {}).get("transition", {})
            probability, model_satisfied = calibrated_binary(result["transition_logit"][0], calibration)
            known = bool(labels["transition_valid"][0])
            self.trace["executed_prefix_evaluation"] = dict(
                previous_request_id=previous["request_id"], executed_steps=int(mask.sum()),
                raw_contract_consistency_probability=float(result["transition_logit"].sigmoid()[0]),
                calibrated_probability=probability, calibrated_contract_satisfied=model_satisfied if known else None,
                contract_evidence_known=known,
                observed_contract_satisfied=bool(labels["transition"][0]) if known else None,
                observed_invariant_loss=self.feedback is not None,
                subtask_failure=None, attempt_complete=False,
                scope="local_executed_contract; unmet postcondition alone is not terminal subtask failure")
        summary = []
        for name in ("global_graph", "task_graph", "local_graph"):
            graph = getattr(cleaned, name)
            m, f = graph.valid[0, -1], graph.features[0, -1]
            summary.extend((f[m].mean(0) if m.any() else torch.zeros(32, device=f.device)).cpu().tolist())
        self.previous = dict(timestamp=now, request_id=int(self.raw["id"]),
                             observation=slice_observation(current, 0), contract=contract,
                             robot=aux["robot_current"], graph_summary=np.asarray(summary, np.float32),
                             evidence=self.current_evidence.copy(),
                             dispatched_native=native[0].detach().cpu().numpy().astype(np.float32))
        self.critic_previous_decision = dict(timestamp=now,
                                             decision={k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                                                       for k, v in clean_decision.items()})
        self.parent.last_decision = decision
        return dict(actions=x, decision=decision) if return_decision else x
