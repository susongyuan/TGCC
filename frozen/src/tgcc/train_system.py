"""Shared critic and adapters; JointSystem owns the trainable policy separately."""

from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import NegativeValidityChecker
from .data import transition_labels
from .losses import critic_loss, masked_mean
from .network import TGCC
from .policy import ResidualAdapter
from .specs import ExecutedTransition


class TrainSystem(nn.Module):
    def __init__(self, parent, action_spec, normalizer, config):
        super().__init__()
        from .data import HISTORY_WIDTH

        self.critic = TGCC(action_spec, HISTORY_WIDTH)
        width = parent.base.action_in_proj.out_features
        self.adapters = nn.ModuleDict(
            {
                k: ResidualAdapter(width, config["adapter_bottleneck"])
                for k in ("local", "subgoal")
            }
        )
        # JointSystem registers the policy exactly once for full fine-tuning.
        # Keep it outside this submodule so compact critic/adapter exports do
        # not duplicate the complete student or change optimizer traversal.
        object.__setattr__(self, "parent", parent)
        self.normalizer = normalizer
        self.config = config
        self.checker = NegativeValidityChecker(action_spec)

    def forward(self, before, after, native_actions, new_step, flow_context=None,
                evaluation_supervision=None):
        reference = before
        if evaluation_supervision is not None:
            if len(evaluation_supervision) != 1 or len(native_actions) != 1:
                raise ValueError("Supplemental evaluations require isolated sample forwards")
            supervision = evaluation_supervision[0]
            if supervision is None:
                raise ValueError("Missing supplemental supervision")
            if flow_context and not supervision.get("policy_imitation_allowed", False):
                raise ValueError("Failed/collected actions cannot enter imitation loss")
            if supervision.get("reference_native") is not None:
                reference_native = native_actions.new_tensor(supervision["reference_native"])[None]
                if reference_native.shape != native_actions.shape:
                    raise ValueError("Ranking reference shape differs from actual prefix")
                reference = replace(before, candidate_actions=self.normalizer.transform(reference_native))
            else:
                reference_native = native_actions
        else:
            reference_native = native_actions
        positive = self.critic.forward_action(reference)
        positive_valid = ~self.checker.violations(
            reference_native, before.candidate_action_mask
        )
        if evaluation_supervision is None:
            negatives = self.checker.generate(
                native_actions,
                before.candidate_action_mask,
                self.normalizer.transform,
                max_count=1,
                offset=new_step * 3,
            )
            negatives.append(self.checker.task_negative(
                native_actions, before.candidate_action_mask,
                before.preserve_gripper_mask, self.normalizer.transform))
        else:
            negatives = []
            if supervision.get("rank_valid", False):
                # Certificate is independently rechecked by EvaluationDataset;
                # loss never creates an outcome for this counterfactual.
                bad = native_actions.new_tensor(supervision["negative_native"])[None]
                negatives = [(self.normalizer.transform(bad), positive_valid,
                              {"kind": supervision["negative_kind"]})]
        outputs = []
        masks = []
        negative_preconditions = []
        for actions, proof, _ in negatives:
            negative = replace(reference, candidate_actions=actions)
            if evaluation_supervision is not None and supervision.get("negative_contract"):
                from .evaluation_data import read_contract
                from .data import contract_tensors
                # Evidence is supervision-only, not added to graph features.
                contract = read_contract(supervision["negative_contract"])
                evidence = supervision["before_evidence"]
                tensor = contract_tensors(contract, evidence, float(before.timestamp[0]), actions.device)
                negative = replace(negative, **tensor,
                                   skill_ids=torch.full_like(negative.skill_ids, contract.skill + 1))
            outputs.append(self.critic.forward_action(negative))
            negative_preconditions.append(negative)
            masks.append(proof & positive_valid)
        negative_valid = (
            torch.stack(masks, -1) if masks else positive_valid[:, None][:, :0]
        )
        tr = ExecutedTransition(
            before,
            after,
            before.candidate_actions,
            before.candidate_action_mask,
            before.timestamp,
            after.timestamp,
        )
        state = self.critic.retrospective(tr)
        labels = transition_labels(before, after)
        if new_step < self.config.get("critic_warmup_updates", 1000):
            labels["handoff_valid"] = torch.zeros_like(labels["handoff_valid"])
        total, parts, counts = critic_loss(
            positive,
            outputs,
            negative_valid,
            state,
            labels,
            self.config["margin"],
            self.config["loss_weights"],
        )
        if evaluation_supervision is None:
            counts["rank_physical"] = int(masks[0].sum()) if masks else 0
            counts["rank_task"] = int(masks[-1].sum()) if masks else 0
        else:
            kind = supervision.get("negative_kind", "none")
            for key in ("physical", "task", "planning"):
                counts["rank_" + key] = int(negative_valid.sum()) if kind == key else 0
        pre_valid = (
            before.contract_valid
            & before.contract_mask
            & (before.contract_category == 0)
        )
        pre_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                positive["precondition_logits"],
                before.contract_values.float(),
                reduction="none",
            ),
            pre_valid,
        )
        total = total + self.config["loss_weights"].get("precondition", 0.5) * pre_loss
        parts["precondition"] = pre_loss
        counts["precondition"] = int(pre_valid.sum())
        if evaluation_supervision is not None and supervision.get("negative_kind") == "planning":
            # Same learned precondition head sees a contract whose prerequisite
            # is *actually known false*. The evidence values are BCE targets
            # only; TGCC.encode never consumes contract_values/contract_valid.
            terms = []
            known_count = 0
            for output, candidate in zip(outputs, negative_preconditions):
                known = (candidate.contract_mask & candidate.contract_valid &
                         (candidate.contract_category == 0))
                terms.append(masked_mean(F.binary_cross_entropy_with_logits(
                    output["precondition_logits"], candidate.contract_values.float(),
                    reduction="none"), known))
                known_count += int(known.sum())
            if terms:
                planning_pre = torch.stack(terms).mean()
                total = total + self.config["loss_weights"].get("precondition", 0.5) * planning_pre
                parts["planning_precondition"] = planning_pre
                counts["planning_precondition"] = known_count
        if flow_context:
            flows = []
            consistencies = []
            skills = []
            for flow_item in flow_context:
                state_, valid, cache, tokens, actions, noise, time = flow_item[:7]
                changed = [
                    (self.adapters[k](t), m)
                    for k, (t, m) in zip(("local", "subgoal"), tokens)
                ]
                x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
                if len(flow_item) == 9:
                    teacher, skill = flow_item[7:]
                    skills.append(skill)
                else:
                    # Legacy seven-item contexts have no teacher supervision.
                    # Never manufacture a frozen reference from this student.
                    teacher = None
                student = self.parent.velocity(state_, valid, cache, changed, x, time)
                flows.append(
                    (student[..., :23] - (noise - actions)[..., :23]).square().mean()
                )
                if teacher is not None:
                    consistencies.append(
                        (student[..., :23] - teacher[..., :23]).square().mean()
                    )
            flow = torch.stack(flows).mean()
            total = total + self.config["loss_weights"]["flow"] * flow
            parts["flow"] = flow
            if consistencies:
                consistency = torch.stack(consistencies).mean()
                total = total + self.config["loss_weights"].get("teacher_consistency", 0.0) * consistency
                parts["teacher_consistency"] = consistency
            if skills:
                skill_loss = torch.stack(skills).mean()
                total = (
                    total + self.config["loss_weights"].get("skill", 0.01) * skill_loss
                )
                parts["skill"] = skill_loss
        else:
            # All ranks retain identical DDP parameter traversal in every phase.
            total = total + sum(p.sum() * 0.0 for p in self.adapters.parameters())
        return total, parts, counts
