"""Success-distribution monitoring and retrieval-conditioned ActionExpert.

Only actually acknowledged control prefixes have an after-state assessment.
High likelihood surprise requests assistance; it is never a fabricated task
failure. High-level feedback uses the original fixed skill decoder vocabulary.
"""

from collections import deque
from dataclasses import replace
import math
import pickle

import numpy as np
import torch

from .contracts import NegativeValidityChecker
from .retrieval_adapter import RetrievalContext


TRUSTED_FEEDBACK = {
    "execution_failure": "The executed action segment failed its required grasp-preservation condition, confirmed by fresh actual before-and-after grasp feedback. The overall task outcome is unknown. Reconsider the next executable subtask and use the current observation and retrieved reference to generate a new action segment.",
    "execution_deviation": "The executed action segment deviated from successful demonstration progress. Use the current observation and the retrieved reference to choose a new executable action segment.",
    "planning_deviation": "The selected subtask deviated from demonstrated subtask order. Reconsider the next executable subtask using the original task and the current observation.",
}


class DistributionCapabilityStop(RuntimeError):
    """The bounded model path cannot supply a supported safe action; no Q label."""


def assessment(calibration, name, nll, *, executed_steps=None):
    head = calibration.get("heads", {}).get(name, {})
    enabled = head.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("Density calibration enabled flag must be boolean")
    if not math.isfinite(float(nll)):
        raise FloatingPointError("Nonfinite demonstration-consistency score")
    threshold = head.get("threshold") if enabled else None
    if enabled and (type(threshold) not in (float, int) or not math.isfinite(threshold)):
        raise ValueError("Enabled density head needs a finite validation threshold")
    calibrated = enabled
    disabled_reason = head.get("disabled_reason") if not enabled else None
    if name == "execution" and enabled:
        counts = head.get("calibrated_prefix_counts", [])
        if (not isinstance(counts, list) or not counts
                or any(type(n) is not int or not 1 <= n <= 32 for n in counts)):
            raise ValueError("Enabled execution density needs its held-out actual prefix counts")
        if executed_steps not in counts:
            calibrated = False
            disabled_reason = "actual_prefix_length_absent_from_validation_calibration"
    return dict(nll=float(nll), threshold=float(threshold) if calibrated else None,
                calibrated=calibrated, distribution_deviation=(float(nll) > threshold) if calibrated else None,
                status=("outside_demonstration_support" if float(nll) > threshold else "within_demonstration_support")
                if calibrated else "uncalibrated_partial_prefix" if enabled else "uncalibrated",
                disabled_reason=disabled_reason, task_failure=None)


def current_grasp_evidence(raw):
    """Fresh explicit Oracle feedback, with both hands needed to establish FALSE.

    This is a physical evidence verifier, independent of success-only density
    training. Missing, contradictory, stale or inferred entries remain unknown.
    """
    snapshot = raw.get("graph_snapshot")
    if snapshot is None:
        return {}
    now = float(raw["sim_time"])
    if (snapshot.get("episode_id") != raw["session"]
            or snapshot.get("timestamp") != now or snapshot.get("oracle") is not True):
        raise ValueError("Actual grasp evidence snapshot episode/time/Oracle provenance mismatch")
    nodes = {node["entity_id"]: node for node in snapshot.get("entities", [])}
    if len(nodes) != len(snapshot.get("entities", [])):
        raise ValueError("Duplicate actual evidence entity identity")
    hands = {}
    for edge in snapshot.get("edges", []):
        if edge.get("relation_type") != "held_by":
            continue
        source, target = nodes.get(edge.get("source_id")), nodes.get(edge.get("target_id"))
        if source is None or target is None or target.get("instance_id") not in ("robot:ee:left", "robot:ee:right"):
            continue
        value = edge.get("relative_attributes", {}).get("value")
        def fresh(node):
            confidence = node.get("confidence")
            return (node.get("source") == "oracle" and node.get("validity") is True
                    and node.get("timestamp") == now and node.get("last_seen") == now
                    and type(confidence) in (float, int) and math.isfinite(confidence) and confidence >= .6)
        confidence = edge.get("confidence")
        valid = (fresh(source) and fresh(target) and edge.get("source") == "oracle"
                 and edge.get("timestamp") == now and type(value) is bool
                 and type(confidence) in (float, int) and math.isfinite(confidence) and confidence >= .6
                 and (value is False or edge.get("validity") is True))
        entries = hands.setdefault(source["instance_id"], {})
        hand = target["instance_id"]
        # Duplicate sensor entries are ambiguous even when their values agree.
        entries[hand] = None if hand in entries else value if valid else None
    result = {}
    for name, entries in hands.items():
        values = list(entries.values())
        known = True if any(value is True for value in values) else False if (
            set(entries) == {"robot:ee:left", "robot:ee:right"}
            and all(value is False for value in values)) else None
        result[name] = dict(value=known, timestamp=now, source="current_oracle_robot_is_grasping",
                            hand_values=entries)
    return result


def preserved_grasp_subjects(plan, selected, evidence, semantics=None):
    """Physical keep-grasp contract; only an explicit placement may release its object."""
    candidates = plan.get("candidates", [])
    if semantics is None or not candidates:
        return ()
    candidate = candidates[selected]
    if candidate.get("id") != plan["node_keys"][selected]:
        raise ValueError("Physical contract candidate order changed")
    skills = candidate.get("skill_ids")
    if not isinstance(skills, list) or len(skills) != 1 or str(skills[0]) not in semantics["skills"]:
        return ()
    def flat(items):
        return [value for item in items for value in (flat(item) if isinstance(item, list) else [item])]
    targets = flat(candidate.get("targets", []))
    description = semantics["skills"][str(skills[0])]
    # These names are verified from the human demonstration annotation table,
    # not from the differently numbered simulator action primitive enum.
    placement = description in ("place on", "place in")
    if placement and (not targets or targets[0] == "UNKNOWN"
            or targets[0] not in candidate.get("manipulating", [])):
        return ()
    released = targets[0] if placement else None
    return tuple(sorted(name for name, entry in evidence.items()
                        if entry["value"] is True and name != released))


class DistributionController:
    def __init__(self, parent, critic, retrieval, normalizer, library,
                 feature_provider, calibration, plan, feedback_observation=None,
                 max_consecutive_feedback=8, num_steps=10, native_action_spec=None, skill_semantics=None):
        self.parent, self.critic, self.retrieval = parent, critic, retrieval
        self.normalizer, self.library = normalizer, library
        self.feature_provider, self.calibration, self.plan = feature_provider, calibration, plan
        self.feedback_observation = feedback_observation
        self.skill_semantics = skill_semantics
        self.device = next(parent.parameters()).device
        if native_action_spec is None:
            raise ValueError("Independent native control-limit specification is required")
        self.checker = NegativeValidityChecker(native_action_spec)
        if (type(max_consecutive_feedback) is not int or not 1 <= max_consecutive_feedback <= 32
                or type(num_steps) is not int or not 1 <= num_steps <= 64):
            raise ValueError("Distribution assistance and flow budgets must be bounded positive integers")
        self.max_consecutive_feedback, self.num_steps = max_consecutive_feedback, num_steps
        self.reset()

    def reset(self):
        self.previous = None
        self.history = deque(maxlen=16)
        self.consecutive_feedback = 0
        self.raw, self.conditions, self.context = None, None, None
        self.after_features, self.trace, self.pending_feedback = None, {}, None
        self.grasp_evidence = {}

    def export(self):
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: cpu(v) for key, v in value.items()}
            return value
        return pickle.dumps(dict(schema="tgcc.distribution_runtime.v1", previous=cpu(self.previous),
                                 history=self.history, consecutive_feedback=self.consecutive_feedback), protocol=4)

    def restore(self, blob):
        state = pickle.loads(blob)
        if state.pop("schema", None) != "tgcc.distribution_runtime.v1":
            raise ValueError("Evaluation-rule controller state is incompatible with distribution inference")
        expected = {"previous", "history", "consecutive_feedback"}
        if set(state) != expected:
            raise ValueError("Distribution durable state fields changed")
        self.previous, self.history = state["previous"], state["history"]
        self.consecutive_feedback = state["consecutive_feedback"]
        self.raw, self.conditions, self.context = None, None, None
        self.after_features, self.trace, self.pending_feedback = None, {}, None
        self.grasp_evidence = {}

    def _receipt(self, raw):
        previous = self.previous
        receipt = raw.get("execution_receipt")
        if previous is None:
            if receipt is not None:
                raise ValueError("First observation cannot refer to an unknown execution")
            return None, None
        if receipt is None:
            raise ValueError("Fresh after observation needs a successful-step execution receipt")
        n = receipt.get("step_count")
        native = np.asarray(receipt.get("actions"), np.float32)
        now = float(raw["sim_time"])
        start, end = float(receipt.get("start_time", math.nan)), float(receipt.get("end_time", math.nan))
        if (type(n) is not int or not 1 <= n <= 32 or native.shape != (n, 23)
                or not np.isfinite(native).all()
                or receipt.get("source") != "native_controller_commands_after_successful_env_step"
                or receipt.get("previous_request_id") != previous["request_id"]
                or not math.isfinite(start) or not math.isfinite(end)
                or abs(start - previous["timestamp"]) > 1e-7
                or abs(end - now) > 1e-7
                or abs(now - previous["timestamp"] - n / 30) > 1e-6
                or not np.array_equal(native, previous["native_actions"][:n])):
            raise ValueError("Execution receipt identity, time or dispatched-prefix mismatch")
        return native, n

    @torch.no_grad()
    def prepare(self, raw, conditions, *, enforce_assistance_budget=True):
        self.raw, self.conditions = raw, conditions
        self.pending_feedback = None
        now = float(raw["sim_time"])
        if not math.isfinite(now):
            raise ValueError("Nonfinite current observation time")
        native, count = self._receipt(raw)
        if native is not None:
            self.history.append(dict(start_time=self.previous["timestamp"], end_time=now,
                                     actions=native.copy(), request_id=self.previous["request_id"],
                                     state=self.previous["official_state"].copy()))
        features = self.feature_provider(raw, conditions, list(self.history))
        context, after = features[:2]
        after_mask = features[2] if len(features) > 2 else np.ones_like(after, dtype=bool)
        self.context = torch.as_tensor(context, device=self.device, dtype=torch.float32).reshape(1, -1)
        self.after_features = torch.as_tensor(after, device=self.device, dtype=torch.float32).reshape(1, -1)
        self.after_mask = torch.as_tensor(after_mask, device=self.device, dtype=torch.bool).reshape(1, -1)
        self.grasp_evidence = current_grasp_evidence(raw)
        self.trace = dict(schema="tgcc.distribution_trace.v1", timestamp=now,
                          feedback=[], task_success=None, task_failure=None,
                          scope="success_demonstration_consistency", retry_is_task_success=False)
        if native is not None:
            executed = torch.zeros(1, 32, 23, device=self.device)
            executed[0, :count] = self.normalizer.transform(torch.from_numpy(native).to(self.device)).float()
            mask = torch.arange(32, device=self.device)[None] < count
            target, target_mask = self.after_features, self.after_mask
            if hasattr(self.feature_provider,'transition'):
                if 'outcome_before' not in self.previous:
                    raise ValueError('Old absolute-outcome durable state cannot resume delta critic')
                change, known = self.feature_provider.transition(
                    self.previous['outcome_before'],self.after_features.cpu().numpy(),
                    self.previous['outcome_before_mask'],self.after_mask.cpu().numpy())
                target = torch.as_tensor(change,device=self.device,dtype=torch.float32)
                target_mask = torch.as_tensor(known,device=self.device,dtype=torch.bool)
            output = self.critic.score_execution(self.previous["context"].to(self.device),
                                                 executed, target, mask, target_mask)
            result = assessment(self.calibration, "execution", float(output["execution_nll"][0]), executed_steps=count)
            result.update(previous_request_id=self.previous["request_id"], executed_steps=count,
                          start_time=self.previous["timestamp"], end_time=now,
                          after_source="actual_observation_after_acknowledged_control_prefix")
            self.trace["execution"] = result
            losses = []
            for subject in self.previous.get("required_preserved_grasps", ()):
                before = self.previous["grasp_evidence"].get(subject, {})
                after = self.grasp_evidence.get(subject, {})
                if (before.get("value") is True and before.get("timestamp") == self.previous["timestamp"]
                        and after.get("value") is False and after.get("timestamp") == now):
                    losses.append(dict(subject=subject, predicate="preserve_grasp", before=before, after=after))
            self.trace["execution_evidence"] = dict(
                scope="local_executed_grasp_preservation_contract", reliable_local_failure=bool(losses),
                verified_local_execution_failure=bool(losses),
                status="verified_local_execution_failure" if losses else "no_verified_local_execution_failure",
                required_preserved_grasps=list(self.previous.get("required_preserved_grasps", ())),
                selected_candidate=self.plan["node_keys"][self.previous["selected_index"]],
                contract_source="actual_selected_frozen_train_candidate_plus_actual_train_human_annotation_descriptions",
                contract_semantics_verified=self.skill_semantics is not None,
                contract_semantics_sha256=self.skill_semantics["sha256"] if self.skill_semantics else None,
                previous_request_id=self.previous["request_id"], executed_steps=count,
                start_time=self.previous["timestamp"], end_time=now, losses=losses,
                source="fresh_actual_grasp_truth_before_and_after_exact_acknowledged_prefix",
                task_failure=None, Q=None, independent_of_density_calibration=True)
            if losses:
                self.pending_feedback = "execution_failure"
            elif result["distribution_deviation"] is True:
                self.pending_feedback = "execution_deviation"
        if self.pending_feedback is not None:
            self.consecutive_feedback += 1
        else:
            self.consecutive_feedback = 0
        if enforce_assistance_budget and self.consecutive_feedback > self.max_consecutive_feedback:
            raise DistributionCapabilityStop("Repeated execution assistance requests exceed the bounded assistance budget")

    def _prefix(self, observation, conditions, feedback=None):
        if feedback is not None:
            if self.feedback_observation is None:
                raise DistributionCapabilityStop("Trusted semantic feedback adapter is unavailable")
            observation = self.feedback_observation(observation, TRUSTED_FEEDBACK[feedback])
            conditions = replace(conditions, event_changed=torch.ones(1, device=self.device, dtype=torch.bool))
            self.trace["feedback"].append(dict(kind=feedback, text=TRUSTED_FEEDBACK[feedback],
                                               interface="original_task_plus_trusted_feedback_and_fixed_skill_decoder",
                                               free_text_planning=False))
        state, valid, cache, tokens, decision = self.parent.prefix(observation, conditions, False)
        selected = int(decision["selected_index"][0])
        if not bool(conditions.legal[0, selected]) or not bool(decision["subgoal_valid"].all()):
            raise DistributionCapabilityStop("The fixed skill vocabulary has no executable selection")
        return state, valid, cache, tokens, decision

    def _reference(self, conditions):
        sample, distance = self.library.query(self.context[0].detach().cpu().numpy(), k=1,
                                              eligible_next=conditions.legal[0].detach().cpu().numpy())[0]
        actions = torch.as_tensor(sample["actual_native"], device=self.device, dtype=torch.float32)[None]
        reference = RetrievalContext(self.normalizer.transform(actions).float(),
                                     torch.as_tensor(sample["action_mask"], device=self.device, dtype=torch.bool)[None],
                                     torch.as_tensor(sample["context"], device=self.device, dtype=torch.float32)[None],
                                     torch.as_tensor(sample["after_features"], device=self.device, dtype=torch.float32)[None],
                                     torch.ones(1, device=self.device, dtype=torch.bool),
                                     torch.as_tensor(sample["after_feature_mask"], device=self.device, dtype=torch.bool)[None])
        row = dict(episode_id=sample["episode_id"], source_steps=sample["sequence_steps"],
                   distance=float(distance), split=sample["split"],
                   source="closest_train_demonstration_segment", used_as_actuator_commands=False)
        if row["split"] != "train":
            raise ValueError("Validation/test trajectory cannot become inference retrieval support")
        self.trace["reference"] = row
        return reference

    def _generate(self, prefix, noise, steps, reference=None):
        state, valid, cache, tokens, _ = prefix
        tokens = self.retrieval(tokens, reference)
        if getattr(self.retrieval,'last_metrics',None):
            self.trace['reference_conditioning']={name:{key:float(value.float().mean().cpu())
                for key,value in metrics.items()} for name,metrics in self.retrieval.last_metrics.items()}
        actions = self.parent.base.sample_noise((len(state), 32, 32), self.device) if noise is None else noise.clone()
        dt = -1.0 / steps
        for index in range(steps):
            when = torch.full((len(actions),), 1 + index * dt, device=self.device)
            actions = actions + dt * self.parent.velocity(state, valid, cache, tokens, actions, when)
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite retrieval-conditioned ActionExpert output")
        return actions

    def _safe(self, actions):
        native = self.normalizer.inverse(actions[..., :23])
        if bool(self.checker.violations(native, torch.ones(1, 32, device=self.device, dtype=torch.bool))[0]):
            return native, False
        return native, bool(torch.isfinite(native).all())

    def commit_dispatched(self, native_actions):
        """Bind receipts to the official output-transform bytes actually sent.

        Its NumPy quantile arithmetic may round differently from the internal
        torch safety check; the controller must compare to the real wire reply.
        """
        actions = np.asarray(native_actions, np.float32)
        if self.previous is None or actions.shape != (32, 23) or not np.isfinite(actions).all():
            raise ValueError("Official dispatched action chunk mismatch")
        values = torch.from_numpy(actions).to(self.device)[None]
        mask = torch.ones(1, 32, device=self.device, dtype=torch.bool)
        if bool(self.checker.violations(values, mask)[0]):
            raise DistributionCapabilityStop("Official output transform produced a native control-limit violation")
        self.previous["native_actions"] = actions.copy()

    @torch.no_grad()
    def finalize(self, raw, conditions):
        """Assess the last acknowledged prefix without requesting another action.

        The caller's terminal marker does not supply any success/Q label. A
        short final prefix remains outside the full-32 calibration guarantee.
        """
        self.prepare(raw, conditions, enforce_assistance_budget=False)
        self.trace.update(final_execution_assessment=True, new_action_generated=False,
                          undispatched_assistance=self.pending_feedback)
        return self.trace

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10,
                       conditions=None, return_decision=False):
        self.parent.validate(conditions)
        if type(num_steps) is not int or not 1 <= num_steps <= 64 or len(observation.state) != 1 or self.context is None:
            raise ValueError("One prepared physical observation and positive flow steps are required")
        action_conditions = conditions
        prefix = self._prefix(observation, action_conditions, self.pending_feedback)
        decision = prefix[-1]
        original_selected_index=int(decision['selected_index'][0])
        planning = self.critic.score_planning(self.context, decision["selected_index"].long())
        planning_result = assessment(self.calibration, "planning", float(planning["planning_nll"][0]))
        self.trace["planning_original"] = planning_result
        if planning_result["distribution_deviation"] is True:
            # One fresh VLM prefix. Its fixed vocabulary is restricted by both
            # existing legality and validation-calibrated demonstrated order.
            threshold = self.calibration["heads"]["planning"]["threshold"]
            nlls = -planning["planning_logits"].float().log_softmax(-1)
            legal = conditions.legal & (nlls <= threshold)
            if not bool(legal.any(-1).all()):
                raise DistributionCapabilityStop("No legal demonstrated-order subtask remains")
            action_conditions = replace(conditions, legal=legal)
            prefix = self._prefix(observation, action_conditions, "planning_deviation")
            decision = prefix[-1]
            output = self.critic.score_planning(self.context, decision["selected_index"].long())
            self.trace["planning_reselected"] = assessment(self.calibration, "planning", float(output["planning_nll"][0]))
        reference = self._reference(action_conditions) if self.pending_feedback is not None else None
        actions = self._generate(prefix, noise, num_steps, reference)
        mask = torch.ones(1, 32, device=self.device, dtype=torch.bool)
        output = self.critic.forward_action(self.context, actions[..., :23], mask)
        original_assessment = assessment(self.calibration, "action", float(output["action_nll"][0]))
        native, safe = self._safe(actions)
        self.trace["action_original"] = original_assessment
        if original_assessment["distribution_deviation"] is True or not safe:
            if reference is None:
                reference = self._reference(action_conditions)
            regenerated = self._generate(prefix, None, num_steps, reference)
            output = self.critic.forward_action(self.context, regenerated[..., :23], mask)
            evaluated = assessment(self.calibration, "action", float(output["action_nll"][0]))
            regen_native, regen_safe = self._safe(regenerated)
            self.trace["action_regenerated"] = dict(evaluated, fresh_draws=1,
                                                      independent_noise=True, native_constraints_satisfied=regen_safe)
            if not regen_safe and not safe:
                raise DistributionCapabilityStop("Neither the original nor retrieved-context candidate supplies a safe control segment")
            select_regenerated = regen_safe and (not safe or evaluated["nll"] < original_assessment["nll"])
            if select_regenerated:
                actions, native, safe = regenerated, regen_native, regen_safe
            self.trace["action_regenerated"]["selected_for_execution"] = select_regenerated
            self.trace["action_regenerated"]["selection_scope"] = "lowest_conditional_nll_among_safe_original_and_one_reference_draw"
            # Distribution shift is advisory rather than a physical failure
            # claim. Bounded assistance never loops until a desired score.
        self.trace.update(selected_index=int(decision["selected_index"][0]),
                          original_selected_index=original_selected_index,
                          planning_feedback_triggered=planning_result['distribution_deviation'] is True,
                          actual_subtask_selection_changed=int(decision['selected_index'][0])!=original_selected_index,
                          selected_candidate=self.plan["node_keys"][int(decision["selected_index"][0])],
                          native_constraints_satisfied=safe,
                          original_subtask_selection_changed=int(decision['selected_index'][0])!=original_selected_index)
        self.previous = dict(context=self.context.detach().cpu(), timestamp=float(self.raw["sim_time"]),
                             outcome_before=self.after_features.detach().cpu().numpy().copy(),
                             outcome_before_mask=self.after_mask.detach().cpu().numpy().copy(),
                             request_id=int(self.raw["id"]), native_actions=native[0].detach().cpu().numpy().astype(np.float32),
                             selected_index=int(decision["selected_index"][0]),
                             official_state=observation.state[0].detach().cpu().numpy().astype(np.float32),
                             grasp_evidence=self.grasp_evidence,
                             required_preserved_grasps=preserved_grasp_subjects(
                                 self.plan, int(decision["selected_index"][0]), self.grasp_evidence, self.skill_semantics))
        self.parent.last_decision = decision
        return dict(actions=actions, decision=decision) if return_decision else actions
