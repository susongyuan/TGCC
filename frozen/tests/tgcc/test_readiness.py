"""Reject the actual zero-positive audit pattern before optimizer creation."""

import copy

import pytest

from tgcc.readiness import require_supervision, supervision_gate


def report():
    return dict(
        recorded_evidence={s: dict(held_positive=12, held_negative=30) for s in ("train", "validation")},
        supervision={s: dict(progress_valid=100, progress_positive=5, handoff_valid=100, handoff_positive=8) for s in ("train", "validation")},
        negative_certificates={s: dict(task_valid_sequences_potential=6) for s in ("train", "validation")},
        timing={s: dict(checked_sequences=100, failures=0) for s in ("train", "validation")},
    )


def test_valid_counts_and_not_held_satisfaction_do_not_prove_positive_grasps():
    data = report()
    data["coverage"] = {"train": {"held:true": 568854}}
    data["recorded_evidence"]["train"]["held_positive"] = 0
    data["supervision"]["train"]["progress_positive"] = 0
    with pytest.raises(ValueError, match="held_positive.*progress_positive"):
        require_supervision(data)


def test_old_reports_without_raw_truth_counts_are_rejected():
    data = report()
    del data["recorded_evidence"]
    assert supervision_gate(data)["status"] == "failed"


@pytest.mark.parametrize("split", ["train", "validation"])
def test_missing_task_certificate_or_handoff_cannot_pass(split):
    data = report()
    data["negative_certificates"][split]["task_valid_sequences_potential"] = 0
    data["supervision"][split]["handoff_positive"] = 0
    result = supervision_gate(data)
    assert result["status"] == "failed" and len(result["failures"]) == 2


def test_report_with_actual_positive_coverage_and_alignment_passes_unchanged():
    data = report()
    before = copy.deepcopy(data)
    assert require_supervision(data)["status"] == "passed"
    assert data == before


def test_handoff_is_disabled_through_warmup_and_enabled_at_joint_boundary():
    import torch
    from test_core import observation, spec
    from tgcc.train_system import TrainSystem

    parent = torch.nn.Module()
    parent.base = torch.nn.Module()
    parent.base.action_in_proj = torch.nn.Linear(1, 1024)

    class Normalizer:
        def transform(self, value):
            return value

    config = dict(adapter_bottleneck=16, margin=0.2, critic_warmup_updates=1000,
                  loss_weights=dict(rank=1., transition=1., predicate=1., progress=.5, handoff=.5))
    model = TrainSystem(parent, spec(), Normalizer(), config).eval()
    before, after = observation(1), observation(1)
    before.contract_category[:] = after.contract_category[:] = torch.tensor([[0, 1, 3]])
    after.timestamp += 32 / 30
    with torch.no_grad():
        _, parts, counts = model(before, after, before.candidate_actions, 999)
        assert counts["handoff"] == 0 and parts["handoff"] == 0
        _, _, counts = model(before, after, before.candidate_actions, 1000)
        assert counts["handoff"] == 1


@pytest.mark.parametrize("left,right", [(-1., -1.), (1., -1.), (0., -1.)])
def test_legacy_grasp_exports_without_provenance_are_unknown(left, right):
    from tgcc.data import decoded_evidence
    from tgcc.contracts import Truth

    decoded = {"hotdog_real": [1.], "hotdog_in_gripper_left": [left],
               "hotdog_in_gripper_right": [right]}
    facts = decoded_evidence(decoded, 1.)
    assert facts[("held", "hotdog", "")].truth(1.) == Truth.UNKNOWN
    assert facts[("observed", "hotdog", "")].truth(1.) == Truth.TRUE


def test_authenticated_grasp_enum_can_supply_true_and_false():
    from tgcc.data import decoded_evidence
    from tgcc.contracts import Truth

    decoded = {"hotdog_real": [1.], "hotdog_in_gripper_left": [1.],
               "hotdog_in_gripper_right": [-1.]}
    assert decoded_evidence(decoded, 1., grasp_fields_reliable=True)[("held", "hotdog", "")].truth(1.) == Truth.TRUE
    decoded["hotdog_in_gripper_left"] = [-1.]
    assert decoded_evidence(decoded, 1., grasp_fields_reliable=True)[("held", "hotdog", "")].truth(1.) == Truth.FALSE
