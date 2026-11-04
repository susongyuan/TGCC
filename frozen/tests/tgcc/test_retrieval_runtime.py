import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tgcc.distribution_runtime import (DistributionController, DistributionCapabilityStop,
                                       assessment, current_grasp_evidence, preserved_grasp_subjects)
from tgcc.retrieval_adapter import RetrievalAdapter, RetrievalContext
from test_core import spec


def reference(batch=2):
    return RetrievalContext(torch.randn(batch, 32, 23), torch.ones(batch, 32, dtype=torch.bool),
                            torch.randn(batch, 4), torch.randn(batch, 3), torch.ones(batch, dtype=torch.bool),
                            torch.ones(batch, 3, dtype=torch.bool))


def tokens(batch=2):
    return [(torch.randn(batch, size, 16), torch.ones(batch, size, dtype=torch.bool)) for size in (8, 2)]


def test_zero_gate_exact_bypass_has_no_rng_or_suffix_shape_change():
    model = RetrievalAdapter(16, 4, 3, hidden=8, heads=4).eval()
    values, ref = tokens(), reference()
    rng = torch.get_rng_state().clone()
    out = model(values, ref)
    assert out is values
    assert torch.equal(rng, torch.get_rng_state())


def test_train_zero_gate_has_gradient_and_open_gate_reference_changes_tokens():
    model = RetrievalAdapter(16, 4, 3, hidden=8, heads=4)
    values, ref = tokens(), reference()
    out = model(values, ref)
    assert all(torch.equal(a, b) for (a, _), (b, _) in zip(values, out))
    sum(a.sum() for a, _ in out).backward()
    assert all(g.grad is not None and g.grad.isfinite().all() and g.grad.abs().sum() > 0
               for g in model.gates.values())
    with torch.no_grad():
        for gate in model.gates.values():
            gate.fill_(0.1)
    model.eval()
    first = model(values, ref)
    ref.normalized_actions += 0.5
    second = model(values, ref)
    assert any(not torch.equal(a, b) for (a, _), (b, _) in zip(first, second))


def test_reference_padding_and_unknown_never_pollute_tokens():
    model = RetrievalAdapter(16, 4, 3, hidden=8, heads=4).eval()
    with torch.no_grad():
        for gate in model.gates.values():
            gate.fill_(0.1)
    values, ref = tokens(), reference()
    ref.valid[1] = False
    ref.action_mask[:, 16:] = False
    clean = model(values, ref)
    ref.normalized_actions[:, 16:] = float("nan")
    ref.normalized_actions[1] = float("nan")
    ref.before_features[1] = ref.after_features[1] = float("nan")
    changed = model(values, ref)
    assert all(torch.equal(a, b) for (a, _), (b, _) in zip(clean, changed))
    assert all(torch.equal(a[1], b[1]) for (a, _), (b, _) in zip(values, changed))


@dataclass
class Conditions:
    legal: torch.Tensor
    event_changed: object = None


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.last_decision = None
        self.noise_calls = 0
        self.prefixes = 0
        self.base = SimpleNamespace(sample_noise=self.noise)

    def noise(self, shape, device):
        self.noise_calls += 1
        return torch.zeros(shape, device=device)

    def validate(self, conditions):
        assert conditions is not None

    def prefix(self, observation, conditions, train):
        self.prefixes += 1
        chosen = conditions.legal.long().argmax(-1)
        decision = dict(selected_index=chosen, subgoal_valid=conditions.legal.any(-1)[:, None].expand(-1, 2),
                        next=torch.zeros(1, 2), memory=torch.ones(1, 4) * self.prefixes)
        return observation.state, torch.ones(1, 1, dtype=torch.bool), None, tokens(1), decision

    def velocity(self, state, valid, cache, values, x, time):
        return torch.zeros_like(x)


class Critic:
    def __init__(self, action=0, execution=0, planning=0):
        self.action, self.execution, self.planning = action, execution, planning

    def forward_action(self, context, actions, mask):
        return {"action_nll": torch.tensor([self.action], dtype=torch.float32)}

    def score_execution(self, context, actual_actions, after_features, mask, after_mask=None):
        return {"execution_nll": torch.tensor([self.execution], dtype=torch.float32)}

    def score_planning(self, context, next_index):
        return {"planning_nll": torch.tensor([self.planning], dtype=torch.float32),
                "planning_logits": torch.tensor([[-4.0, 4.0]])}


class Library:
    def __init__(self):
        self.queries = 0

    def query(self, context, k, eligible_next):
        self.queries += 1
        return [(dict(actual_native=np.zeros((32, 23), np.float32), action_mask=np.ones(32, bool),
                      context=np.zeros(4, np.float32), after_features=np.zeros(3, np.float32), after_feature_mask=np.ones(3, bool),
                      episode_id="different_training_episode", sequence_steps=[64, 96], split="train"), 0.2)]


class Normalizer:
    transform = staticmethod(lambda x: x)
    inverse = staticmethod(lambda x: x)


def controller(**scores):
    parent, library = Parent(), Library()
    calibration = dict(heads={name: dict(enabled=True, threshold=1.0) for name in ("action", "execution", "planning")})
    calibration["heads"]["execution"]["calibrated_prefix_counts"] = [32]
    runtime = DistributionController(parent, Critic(**scores), RetrievalAdapter(16, 4, 3, hidden=8).eval(),
                                     Normalizer(), library, lambda raw, c, h: (np.zeros(4), np.zeros(3)),
                                     calibration, {"node_keys": ["pick", "place"]},
                                     feedback_observation=lambda observation, text: observation,
                                     native_action_spec=spec())
    return runtime, parent, library


def raw(step=0):
    return dict(id=step // 32 + 1, session="physical_episode", sim_time=step / 30)


def advance(runtime, count=32):
    previous = runtime.previous
    current = raw(count)
    current["execution_receipt"] = dict(start_time=previous["timestamp"], end_time=current["sim_time"],
                                         actions=previous["native_actions"][:count].copy(), step_count=count,
                                         previous_request_id=previous["request_id"],
                                         source="native_controller_commands_after_successful_env_step")
    return current


def test_action_deviation_regenerates_one_fresh_retrieval_conditioned_draw():
    runtime, parent, library = controller(action=2)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    assert parent.noise_calls == 2 and library.queries == 1
    assert runtime.trace["action_regenerated"]["fresh_draws"] == 1
    assert runtime.trace["action_regenerated"]["task_failure"] is None
    assert parent.last_decision["selected_index"].item() == runtime.previous["selected_index"]


def test_execution_uses_acknowledged_prefix_and_only_then_notifies_vlm():
    runtime, parent, library = controller(execution=2)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    observation = SimpleNamespace(state=torch.zeros(1, 32))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", observation, conditions=conditions)
    runtime.prepare(advance(runtime, 32), conditions)
    runtime.sample_actions("cpu", observation, conditions=conditions)
    assert runtime.trace["execution"]["executed_steps"] == 32
    assert runtime.trace["execution"]["task_failure"] is None
    assert runtime.trace["feedback"][0]["kind"] == "execution_deviation"
    assert runtime.trace["feedback"][0]["free_text_planning"] is False
    assert library.queries == 1


@pytest.mark.parametrize("change", ["action", "id", "future", "nan"])
def test_receipt_rejects_unexecuted_or_unaligned_evidence(change):
    runtime, _, _ = controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    current = advance(runtime)
    if change == "action":
        current["execution_receipt"]["actions"][0, 0] += 0.1
    elif change == "id":
        current["execution_receipt"]["previous_request_id"] += 1
    elif change == "future":
        current["execution_receipt"]["end_time"] += 1 / 30
    else:
        current["execution_receipt"]["start_time"] = float("nan")
    with pytest.raises(ValueError, match="receipt"):
        runtime.prepare(current, conditions)


def test_planning_reselects_only_from_existing_legal_candidates_once():
    runtime, parent, _ = controller(planning=2)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    assert parent.prefixes == 2 and parent.last_decision["selected_index"].item() == 1
    assert runtime.trace["feedback"][0]["kind"] == "planning_deviation"
    runtime, _, _ = controller(planning=2)
    conditions = Conditions(torch.tensor([[True, False]]))
    runtime.prepare(raw(), conditions)
    with pytest.raises(DistributionCapabilityStop, match="No legal"):
        runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)


def test_durable_state_roundtrip_preserves_previous_prefix_and_history():
    runtime, _, _ = controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    restored, _, _ = controller()
    restored.restore(runtime.export())
    restored.prepare(advance(runtime), conditions)
    assert restored.history[0]["request_id"] == runtime.previous["request_id"]
    assert restored.trace["execution"]["executed_steps"] == 32


def test_uncalibrated_density_does_not_turn_into_failure_or_deviation():
    result = assessment({"heads": {"action": {"enabled": False, "threshold": None}}}, "action", 1000)
    assert result["distribution_deviation"] is None and result["task_failure"] is None


def test_receipt_binds_exact_official_posttransform_commands():
    runtime, _, _ = controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    sent = runtime.previous["native_actions"].copy()
    sent[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    runtime.commit_dispatched(sent)
    receipt = advance(runtime)
    runtime.prepare(receipt, conditions)
    assert runtime.history[0]["actions"][0, 0] == sent[0, 0]


def test_structured_controller_stop_survives_legacy_client_without_text_guessing():
    from tgcc.transport import _StatusSocket, ControllerCapabilityStop
    from tgr.transport import packb
    reply = dict(error="No safe control segment", status="capability_stop", transaction_rolled_back=True,
                 task_failure=None, Q=None)
    socket = _StatusSocket(SimpleNamespace(recv=lambda **kwargs: packb(reply)))
    with pytest.raises(ControllerCapabilityStop) as error:
        socket.recv(timeout=1)
    assert error.value.reply == reply
    text_only = dict(error="TGCC_CAPABILITY_STOP: text alone does not establish controller status")
    socket = _StatusSocket(SimpleNamespace(recv=lambda **kwargs: packb(text_only)))
    assert socket.recv(timeout=1) == packb(text_only)


def grasp_snapshot(current, left=True, right=False):
    timestamp = current["sim_time"]
    def node(eid, name):
        return dict(entity_id=eid, instance_id=name, timestamp=timestamp, last_seen=timestamp,
                    source="oracle", validity=True, confidence=1.0)
    def edge(target, value):
        return dict(source_id="food", target_id=target, relation_type="held_by",
                    relative_attributes=dict(value=value), timestamp=timestamp,
                    confidence=1.0 if value is not None else 0.0, validity=value is True, source="oracle")
    current["graph_snapshot"] = dict(episode_id=current["session"], timestamp=timestamp, oracle=True,
        entities=[node("food", "food"), node("left", "robot:ee:left"), node("right", "robot:ee:right")],
        edges=[edge("left", left), edge("right", right)])
    return current


def grasp_controller(skill=1, **scores):
    runtime, parent, library = controller(**scores)
    runtime.skill_semantics = dict(sha256="a" * 64, skills={"1":"move to", "2":"pick up from", "3":"place on", "4":"place in", "10":"open door", "12":"close door", "69":"turn on switch"})
    runtime.plan["candidates"] = [dict(id="pick", skill_ids=[skill], targets=[["food", "container"]], manipulating=["food"]),
                                   dict(id="place", skill_ids=[4], targets=[["food", "container"]], manipulating=["food"])]
    return runtime, parent, library


def test_actual_required_grasp_loss_not_density_deviation_feedbacks_vlm_and_reference():
    runtime, parent, library = grasp_controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    observation = SimpleNamespace(state=torch.zeros(1, 32))
    runtime.prepare(grasp_snapshot(raw()), conditions)
    runtime.sample_actions("cpu", observation, conditions=conditions)
    runtime.prepare(grasp_snapshot(advance(runtime, 16), left=False, right=False), conditions)
    runtime.sample_actions("cpu", observation, conditions=conditions)
    evidence = runtime.trace["execution_evidence"]
    assert evidence["reliable_local_failure"] is True and evidence["task_failure"] is None
    assert evidence["previous_request_id"] == 1 and evidence["executed_steps"] == 16
    assert runtime.trace["execution"]["calibrated"] is False
    assert runtime.trace["feedback"][0]["kind"] == "execution_failure"
    assert library.queries == 1 and parent.prefixes == 2


@pytest.mark.parametrize("skill,left,right,expected", [(4,False,False,False), (3,False,False,False),
    (1,False,True,False), (1,False,None,False), (1,False,False,True)])
def test_release_transfer_and_unknown_do_not_fabricate_execution_failure(skill,left,right,expected):
    runtime, _, _ = grasp_controller(skill=skill)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(grasp_snapshot(raw()), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    runtime.prepare(grasp_snapshot(advance(runtime), left=left, right=right), conditions)
    assert runtime.trace["execution_evidence"]["reliable_local_failure"] is expected


@pytest.mark.parametrize("change", ["stale", "inferred", "duplicate", "missing"])
def test_unreliable_grasp_negatives_remain_unknown(change):
    current = grasp_snapshot(raw(), left=False, right=False)
    edges = current["graph_snapshot"]["edges"]
    if change == "stale":
        edges[0]["timestamp"] = -1.0
    elif change == "inferred":
        edges[0]["source"] = "memory"
    elif change == "duplicate":
        edges.append(copy.deepcopy(edges[0]))
    else:
        edges.pop()
    assert current_grasp_evidence(current)["food"]["value"] is None


def test_grasp_verifier_binds_physical_session_time_and_candidate_identity():
    current = grasp_snapshot(raw())
    current["graph_snapshot"]["episode_id"] = "another_episode"
    with pytest.raises(ValueError, match="episode/time"):
        current_grasp_evidence(current)
    runtime, _, _ = grasp_controller()
    runtime.plan["candidates"][0]["id"] = "another_selection"
    with pytest.raises(ValueError, match="candidate order"):
        preserved_grasp_subjects(runtime.plan, 0, {"food": dict(value=True)}, runtime.skill_semantics)


def test_partial_actual_after_prefix_is_recorded_without_full32_threshold_claim():
    runtime, _, _ = controller(execution=100)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    runtime.prepare(advance(runtime, 16), conditions)
    result = runtime.trace["execution"]
    assert result["executed_steps"] == 16 and result["nll"] == 100
    assert result["status"] == "uncalibrated_partial_prefix"
    assert result["calibrated"] is False and result["distribution_deviation"] is None
    assert runtime.pending_feedback is None


def test_final_prefix_assessment_does_not_generate_action_or_apply_full32_calibration():
    runtime, parent, library = controller(execution=100)
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(raw(), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    before = (parent.noise_calls, parent.prefixes, library.queries)
    final = runtime.finalize(advance(runtime, 7), conditions)
    assert before == (parent.noise_calls, parent.prefixes, library.queries)
    assert final["final_execution_assessment"] is True and final["new_action_generated"] is False
    assert final["execution"]["executed_steps"] == 7
    assert final["execution"]["distribution_deviation"] is None
    assert final["task_success"] is None and final["task_failure"] is None


def test_final_assessment_transaction_reconnect_and_duplicate_never_regenerate_action(tmp_path):
    from tgcc.transport import ExecutedSessionStore
    runtime, parent, _ = controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    store = ExecutedSessionStore(tmp_path / "final.msgpack", "identity", torch.manual_seed,
        lambda: torch.get_rng_state().numpy(), lambda x: torch.set_rng_state(torch.from_numpy(np.array(x, copy=True))),
        runtime.export, runtime.restore)
    store.start("physical_episode", 42)
    def generate():
        runtime.prepare(raw(), conditions)
        runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
        return runtime.previous["native_actions"], .1
    store.infer("physical_episode", 0, "original", generate)
    def finish():
        return runtime.finalize(advance(runtime, 5), conditions), .1
    first = store.finalize("physical_episode", 1, "terminal", finish)
    assert "actions" not in first and first["new_action_generated"] is False
    rng = torch.get_rng_state().clone()
    restored, _, _ = controller()
    resumed = ExecutedSessionStore(tmp_path / "final.msgpack", "identity", torch.manual_seed,
        lambda: torch.get_rng_state().numpy(), lambda x: torch.set_rng_state(torch.from_numpy(np.array(x, copy=True))),
        restored.export, restored.restore)
    second = resumed.finalize("physical_episode", 1, "terminal", lambda: 1 / 0)
    assert second["cached"] is True and torch.equal(rng, torch.get_rng_state())
    assert second["final_execution_assessment"] == first["final_execution_assessment"]
    with pytest.raises(ValueError, match="Finalized"):
        resumed.infer("physical_episode", 2, "forbidden", generate)
    with pytest.raises(ValueError, match="changed payload"):
        resumed.finalize("physical_episode", 1, "different", finish)


def test_final_client_lost_reply_retries_exact_bytes_without_executing_suffix(monkeypatch):
    from tgcc.transport import ExecutedClient
    from tgr.transport import packb, decode
    sent = []
    state = {"connections": 0, "messages": []}
    class Socket:
        def __init__(self):
            self.i = state["connections"]
            state["connections"] += 1
            self.receives = 0
        def send(self, message):
            request = decode(message)
            if request["op"] == "finalize":
                sent.append(message)
                self.request = request
        def recv(self, **kwargs):
            self.receives += 1
            if self.receives == 1:
                return packb(dict(weight_sha256="identity", transport="tgr_durable_v1"))
            if self.receives == 2:
                return packb(dict(session="physical_episode"))
            if self.i == 0:
                raise OSError("final reply lost")
            return packb(dict(session="physical_episode", id=self.request["id"], finalized=True,
                              new_action_generated=False, cached=True))
        def close(self):
            pass
    monkeypatch.setattr("tgcc.transport.time.sleep", lambda seconds: None)
    client = ExecutedClient(Socket, "identity", session="physical_episode")
    client.seq, client.exposure_time = 1, 0.
    client.executed = [np.zeros(23, np.float32)] * 5
    client.pending.extend(np.zeros((27, 23), np.float32))
    reply = client.finalize(dict(sim_time=5 / 30))
    assert reply["cached"] is True and len(sent) == 2 and sent[0] == sent[1]
    assert len(client.pending) == 27 and client.finalized is True
    assert decode(sent[0])["execution_receipt"]["step_count"] == 5
    with pytest.raises(ValueError, match="Finalized"):
        client.request({})


@pytest.mark.parametrize("head", [dict(enabled=True,threshold=None), dict(enabled="false",threshold=1.),
                                      dict(enabled=True,threshold=float("inf"))])
def test_invalid_enabled_or_nonboolean_calibration_is_rejected(head):
    with pytest.raises(ValueError, match="calibration|threshold"):
        assessment({"heads":{"action":head}}, "action", 0.)


def test_controller_transaction_failure_and_cached_reply_preserve_rng_prefix_and_grasp(tmp_path):
    from tgr.transport import SessionStore
    runtime, _, _ = grasp_controller()
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    observation = SimpleNamespace(state=torch.zeros(1, 32))
    store = SessionStore(tmp_path / "state.msgpack", "owned-identity", torch.manual_seed,
        lambda: torch.get_rng_state().numpy(), lambda x: torch.set_rng_state(torch.from_numpy(np.array(x, copy=True))),
        runtime.export, runtime.restore)
    store.start("physical_episode", 42)
    def generate():
        runtime.prepare(grasp_snapshot(raw()), conditions)
        runtime.sample_actions("cpu", observation, conditions=conditions)
        torch.rand(3)
        return runtime.previous["native_actions"], 0.1
    reply = store.infer("physical_episode", 0, "payload0", generate)
    state = runtime.export()
    rng = torch.get_rng_state().clone()
    assert store.infer("physical_episode", 0, "payload0", lambda: 1 / 0)["cached"] is True
    assert torch.equal(rng, torch.get_rng_state()) and runtime.export() == state
    def fail():
        runtime.prepare(grasp_snapshot(advance(runtime), left=False, right=False), conditions)
        torch.rand(3)
        raise DistributionCapabilityStop("bounded execution assistance exhausted")
    with pytest.raises(DistributionCapabilityStop):
        store.infer("physical_episode", 1, "payload1", fail)
    assert store.state["next"] == 1
    import pickle
    before, after = pickle.loads(state), pickle.loads(runtime.export())
    assert before["schema"] == after["schema"] and not after["history"]
    assert before["consecutive_feedback"] == after["consecutive_feedback"]
    assert before["previous"]["grasp_evidence"] == after["previous"]["grasp_evidence"]
    assert before["previous"]["required_preserved_grasps"] == after["previous"]["required_preserved_grasps"]
    assert torch.equal(before["previous"]["context"], after["previous"]["context"])
    assert np.array_equal(before["previous"]["native_actions"], after["previous"]["native_actions"])
    assert torch.equal(rng, torch.get_rng_state())
    restored, _, _ = grasp_controller()
    resumed = SessionStore(tmp_path / "state.msgpack", "owned-identity", torch.manual_seed,
        lambda: torch.get_rng_state().numpy(), lambda x: torch.set_rng_state(torch.from_numpy(np.array(x, copy=True))),
        restored.export, restored.restore)
    assert resumed.resume("physical_episode")["next"] == 1
    restored.prepare(grasp_snapshot(advance(restored), left=False, right=False), conditions)
    assert restored.trace["execution_evidence"]["reliable_local_failure"] is True


def test_placement_release_exception_requires_frozen_manipulated_target_identity():
    runtime, _, _ = grasp_controller(skill=4)
    evidence = {"food": dict(value=True), "other_food": dict(value=True)}
    assert preserved_grasp_subjects(runtime.plan, 0, evidence, runtime.skill_semantics) == ("other_food",)
    runtime.plan["candidates"][0]["manipulating"] = ["other_food"]
    assert preserved_grasp_subjects(runtime.plan, 0, evidence, runtime.skill_semantics) == ()


def test_unknown_reference_after_nan_is_masked_and_missingness_is_encoded():
    model = RetrievalAdapter(16, 4, 3, hidden=8, heads=4).eval()
    with torch.no_grad():
        for gate in model.gates.values():
            gate.fill_(0.1)
    values, ref = tokens(), reference()
    ref.after_features.zero_()
    ref.after_feature_mask[:, 0] = False
    unknown = model(values, ref)
    ref.after_features[:, 0] = float("nan")
    changed_unknown = model(values, ref)
    assert all(torch.equal(a, b) for (a, _), (b, _) in zip(unknown, changed_unknown))
    ref.after_features[:, 0] = 0
    ref.after_feature_mask[:, 0] = True
    measured_zero = model(values, ref)
    assert any(not torch.equal(a, b) for (a, _), (b, _) in zip(unknown, measured_zero))


def test_unverified_human_skill_description_mapping_disables_local_failure_contract():
    runtime, _, _ = grasp_controller()
    runtime.skill_semantics = None
    conditions = Conditions(torch.ones(1, 2, dtype=torch.bool))
    runtime.prepare(grasp_snapshot(raw()), conditions)
    runtime.sample_actions("cpu", SimpleNamespace(state=torch.zeros(1, 32)), conditions=conditions)
    runtime.prepare(grasp_snapshot(advance(runtime), left=False, right=False), conditions)
    assert runtime.trace["execution_evidence"]["contract_semantics_verified"] is False
    assert runtime.trace["execution_evidence"]["verified_local_execution_failure"] is False
