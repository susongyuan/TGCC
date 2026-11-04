"""Check independent teacher history on the actual two-chunk extraction path."""

import copy
import sys
from types import SimpleNamespace

import torch

import tgcc.features as features
import tgcc.feedback as feedback


def raw_conditions():
    graph = dict(
        features=torch.zeros(1, 1, 2, 32),
        semantics=torch.zeros(1, 1, 2, 256),
        relations=torch.zeros(1, 1, 2, 2, dtype=torch.long),
        valid=torch.ones(1, 1, 2, dtype=torch.bool),
    )
    return dict(
        global_graph=copy.deepcopy(graph),
        task_graph=copy.deepcopy(graph),
        local_graph=copy.deepcopy(graph),
        legal=torch.ones(1, 2, dtype=torch.bool),
        task_candidate_slots=torch.tensor([[0, 1]]),
    )


def test_causal_conditions_isolate_legality_and_keep_only_student_bptt(monkeypatch):
    def mutate_legal(c, aux, plan):
        c.legal[0, 0] = False
        return c

    monkeypatch.setattr(feedback, "training_feedback", mutate_legal)
    raw = raw_conditions()
    memory = torch.tensor([[2.0]], requires_grad=True)
    previous = (
        dict(memory=memory, selected_index=torch.tensor([1]), next=torch.zeros(1, 2)),
        torch.tensor([1.0]),
    )
    student = features.causal_conditions(raw, [], previous, torch.tensor([2.0]), {}, True)
    teacher = features.causal_conditions(raw, [], previous, torch.tensor([2.0]), {}, False)
    assert raw["legal"].all()
    assert student.legal.data_ptr() != teacher.legal.data_ptr()
    assert student.memory is memory and student.memory.requires_grad
    assert not teacher.memory.requires_grad
    student.memory.square().sum().backward()
    assert torch.equal(memory.grad, torch.tensor([[4.0]]))


class TinyHistoryPolicy(torch.nn.Module):
    def __init__(self, weight, tap=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(weight)))
        self.base = SimpleNamespace(
            sample_noise=lambda shape, device: torch.zeros(shape, device=device),
            sample_time=lambda size, device: torch.full((size,), 0.5, device=device),
        )
        self.tap = tap
        self.received = []

    def prefix(self, observation, c, train):
        self.received.append(c)
        memory = self.weight.reshape(1, 1)
        if c.memory is not None:
            memory = memory + c.memory
        selected = torch.tensor([int(float(self.weight.detach())) % 2])
        decision = dict(
            memory=memory,
            context=memory,
            selected_index=selected,
            next=torch.stack((memory[:, 0], -memory[:, 0]), -1),
        )
        if self.tap is not None:
            self.tap.values = {
                key: (memory[:, None], torch.ones(1, 1, dtype=torch.bool))
                for key in ("global_encoder", "task_encoder", "local_encoder")
            }
        return memory, torch.ones(1, 1, dtype=torch.bool), None, [], decision

    def velocity(self, state, valid, cache, tokens, x, time):
        return state[:, None].expand_as(x)


def pair_inputs(monkeypatch):
    actions = torch.zeros(1, 2, 23)
    items = [
        dict(conditions=raw_conditions(), observation={}, actions=actions,
            labels=dict(next=torch.tensor([0]), next_mask=torch.tensor([True])))
        for _ in range(2)
    ]
    monkeypatch.setitem(sys.modules, "tgr.data", SimpleNamespace(
        collate_sequences=lambda samples: dict(sequence=copy.deepcopy(items))
    ))
    monkeypatch.setitem(sys.modules, "openpi.models.model", SimpleNamespace(
        Observation=SimpleNamespace(from_dict=lambda observation: observation)
    ))
    monkeypatch.setattr(features, "build_contract", lambda *args, **kwargs: {})
    monkeypatch.setattr(features, "critic_observation", lambda *args: args[1]["context"])
    monkeypatch.setattr(features, "concatenate_observations", lambda items: items)
    samples = [dict(
        sequence=items,
        tgcc_aux=[dict(timestamp=torch.tensor([float(k + 1)]), frame=0,
            native_actions=actions) for k in range(2)],
    )]
    plan = dict(candidates=[dict(skill_ids=[1], targets=[])] * 2)
    return samples, plan


def test_teacher_chunk2_is_invariant_to_student_memory_and_task_writes(monkeypatch):
    samples, plan = pair_inputs(monkeypatch)

    def run(student_weight):
        tap = SimpleNamespace(values={}, clear=lambda: None)
        student = TinyHistoryPolicy(student_weight, tap)
        teacher = TinyHistoryPolicy(2)
        teacher.requires_grad_(False)
        _, _, _, flow = features.extract_pair(
            student, samples, tap, "cpu", plan, with_flow=True,
            trainable=True, teacher=teacher,
        )
        return student, teacher, flow

    student_a, teacher_a, first = run(1)
    student_b, teacher_b, second = run(3)
    assert teacher_a.received[1].memory.item() == 2
    assert teacher_b.received[1].memory.item() == 2
    assert not teacher_a.received[1].memory.requires_grad
    for a, b in zip(first, second):
        assert torch.equal(a[7], b[7])
    assert torch.equal(teacher_a.received[1].task_graph.features,
        teacher_b.received[1].task_graph.features)
    assert not torch.equal(student_a.received[1].task_graph.features,
        teacher_a.received[1].task_graph.features)
    assert student_a.received[1].memory.requires_grad
    first[1][0].sum().backward()
    assert student_a.weight.grad.item() == 2


def test_teacher_free_pair_retains_real_skill_ce_and_two_chunk_bptt(monkeypatch):
    samples, plan = pair_inputs(monkeypatch)
    tap = SimpleNamespace(values={}, clear=lambda: None)
    student = TinyHistoryPolicy(1, tap)
    _, _, _, flow = features.extract_pair(
        student, samples, tap, "cpu", plan, with_flow=True,
        trainable=True, teacher=None,
    )
    assert len(flow) == 2 and all(len(item) == 9 and item[7] is None for item in flow)
    assert all(item[8].requires_grad and item[8].item() > 0 for item in flow)
    assert student.received[1].memory.requires_grad
    sum(item[8] for item in flow).backward()
    assert student.weight.grad.abs().item() > 0
