"""Frozen success-distribution and retrieval inference; no simulator or training."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from tgr.factory import create, ensure_legacy_imports
from tgr.serve import GraphAdapter, PROMPT, WireProvider, checkpoint_gate, model_input, sources as all_sources, sha
from tgr.transport import decode, packb

from .checkpoint import digest, sources
from .distribution_calibration import distribution_readiness
from .distribution_checkpoint import load_distribution_student
from .distribution_critic import DistributionCritic
from .distribution_runtime import DistributionController, DistributionCapabilityStop
from .retrieval_adapter import RetrievalAdapter
from .retrieval_data import FrozenFeatureCodec, FrozenVisualEncoder, RetrievalDataset, after_feature_mask
from .robot import ActionNormalizer, action_spec
from .transport import ExecutedSessionStore

ROOT = Path(__file__).resolve().parents[2]


def batched_input(policy, raw):
    """Use the exact original policy-owned transforms, including image resize."""
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return torch.from_numpy(np.array(value, copy=True))[None]
    return convert(policy._input_transform(model_input(raw)))


class CurrentFeatureProvider:
    def __init__(self, policy, encoder, codec, normalizer):
        self.policy, self.encoder, self.codec, self.normalizer = policy, encoder, codec, normalizer

    def transition(self, before, after, before_mask, after_mask):
        from .transition_features import transition_target,normalize_transition
        change, valid = transition_target(before,after,before_mask,after_mask)
        a = self.codec.artifact
        if 'after_supported' not in a:
            raise ValueError('Corrected execution critic requires rebuilt delta codec')
        return normalize_transition(change,valid,a)

    def __call__(self, raw, conditions, history):
        transformed = batched_input(self.policy, raw)
        state = transformed["state"][0].numpy().astype(np.float32)
        visual = self.encoder(transformed)[0]
        history = list(history)[-16:]
        for record in history:
            if record["start_time"] >= float(raw["sim_time"]) or record["end_time"] > float(raw["sim_time"]):
                raise ValueError("Future execution/robot feedback in current retrieval key")
        previous = np.concatenate([h["actions"] for h in history], 0)[-32:] if history else np.zeros((0, 23), np.float32)
        if len(previous):
            normalized = np.asarray(self.normalizer.transform(previous), np.float32)
            summary = np.r_[normalized.mean(0), normalized[-1]].astype(np.float32)
        else:
            summary = np.zeros(46, np.float32)
        named = raw.get("named_proprio", {})
        if raw.get("named_proprio_source") != "actual_robot_proprioception_at_current_exposure":
            raise ValueError("Current named feedback provenance missing")
        camera = raw.get("cam_rel_poses")
        kwargs = dict(named_proprio=named, cam_rel_poses=camera, conditions=conditions,
                      depth_available=raw.get("depth_available", [False] * 3))
        context = self.codec.encode_context(visual, state, summary, bool(len(previous)),
                                            history_states=[h["state"] for h in history],
                                            timestamp=float(raw["sim_time"]),
                                            history_timestamps=[h["start_time"] for h in history], **kwargs)
        after = self.codec.encode_after(visual, state, **kwargs)
        visual_width = self.codec.artifact["visual_components"].shape[1]
        mask = after_feature_mask(named, conditions, visual_width=visual_width, cam_rel_poses=camera)
        if context.ndim != 1 or after.ndim != 1 or mask.shape != after.shape:
            raise ValueError("Frozen query/after codec dimensions changed")
        return context, after, mask


class DistributionGraphAdapter(GraphAdapter):
    def __init__(self, policy, model, task_spec, oracle, device, plan_path, plan,
                 critic, retrieval, normalizer, library, provider, calibration, runtime_config, skill_semantics):
        super().__init__(policy, model, task_spec, oracle, device, plan_path, plan)
        self.device = torch.device(device)
        self.controller = DistributionController(model, critic, retrieval, normalizer, library,
                                                 provider, calibration, plan,
                                                 feedback_observation=self.feedback_observation,
                                                 native_action_spec=action_spec(ROOT), skill_semantics=skill_semantics,
                                                 **runtime_config)
        policy._sample_actions = self.controller.sample_actions

    def reset(self):
        super().reset()
        if hasattr(self, "controller"):
            self.controller.reset()

    def export(self):
        value = super().export()
        value["distribution"] = self.controller.export()
        return value

    def restore(self, state):
        super().restore(state)
        if state is not None:
            self.controller.restore(state["distribution"])

    def feedback_observation(self, original, text):
        from openpi.models.model import Observation
        # Only trusted constants selected by the server may extend the prompt.
        # The external request continues to contain the immutable original task.
        raw = dict(self.current_raw, task=PROMPT + "\nSupervisor feedback: " + text)
        values = batched_input(self.policy, raw)
        values = {key: ({name: value.to(self.device) for name, value in item.items()}
                        if isinstance(item, dict) else item.to(self.device))
                  for key, item in values.items()}
        return Observation.from_dict(values)

    def infer(self, raw):
        if raw["task"] != PROMPT:
            raise ValueError("Frozen task instruction changed")
        if self.episode != raw["session"]:
            if self.episode is not None:
                raise ValueError("A new episode needs an explicit session start")
            self.runtime.reset(raw["session"])
            self.episode = raw["session"]
        from tgr.state import EntityRegistry
        WireProvider(self.oracle).snapshot(raw, EntityRegistry(), self.episode, float(raw["sim_time"]))
        self.current_raw = raw
        conditions = self.runtime.begin_chunk(raw)
        self.controller.prepare(raw, conditions)
        self.policy._sample_kwargs["conditions"] = conditions
        try:
            result = self.policy.infer(model_input(raw))
        finally:
            self.policy._sample_kwargs.pop("conditions", None)
        actions = np.asarray(result["actions"], np.float32)
        self.controller.commit_dispatched(actions)
        decision = self.model.take_decision()
        self.runtime.after_prefix(decision)
        self.last_task_trace = dict(session=self.episode, sim_time=float(raw["sim_time"]),
                                    selected_index=int(decision["selected_index"][0]),
                                    distribution=self.controller.trace,
                                    oracle_upper_bound=self.oracle,
                                    graph_completion_authority="existing_physical_evidence; independent_official_metrics")
        return actions

    def finalize(self, raw):
        if raw["task"] != PROMPT or self.episode != raw["session"]:
            raise ValueError("Final after-state task/session changed")
        from tgr.state import EntityRegistry
        WireProvider(self.oracle).snapshot(raw, EntityRegistry(), self.episode, float(raw["sim_time"]))
        self.current_raw = raw
        conditions = self.runtime.begin_chunk(raw)
        trace = self.controller.finalize(raw, conditions)
        self.last_task_trace = dict(session=self.episode, sim_time=float(raw["sim_time"]),
                                    distribution=trace, oracle_upper_bound=self.oracle,
                                    graph_completion_authority="independent_official_metrics")
        return trace


def artifact_recipe_consistent(md, initial):
    """A served checkpoint must be a completed, non-disposable corrected run whose counters match its own
    recipe (any warmup/joint budget, including zero warmup with a reused critic), not one hard-coded budget."""
    cfg = md.get('config') or {}
    warm, joint = cfg.get('critic_warmup_updates'), cfg.get('joint_updates')
    if not (isinstance(warm, int) and isinstance(joint, int) and warm >= 0 and joint > 0):
        return False
    return (md.get('schema') == 'tgcc.distribution_joint.v1' and md.get('training_complete') is True
            and md.get('audit_disposable') is False
            and cfg.get('schema') == 'tgcc.corrected_full_success_distribution.recipe.v1'
            and cfg.get('new_updates') == warm + joint and md.get('new_updates') == warm + joint
            and md.get('full_model_updates') == joint and md.get('fresh_critic_only_updates') == warm
            and md.get('critic_actual_total_updates') == cfg.get('critic_initialization_actual_updates', 0) + warm + joint
            and md.get('policy_new_updates') == joint and md.get('policy_cumulative_updates') == initial + joint)


def artifact_gate(directory, parent, library, root=ROOT):
    directory, parent = Path(directory), Path(parent)
    md = json.loads((directory / "metadata.json").read_text())
    initial=md.get('policy_initialization_updates',5000)
    if not artifact_recipe_consistent(md, initial):
        raise ValueError('Completed corrected run whose warmup/joint counters match its own recipe required')
    for file, key in (("distribution.safetensors", "distribution_sha256"), ("student.safetensors", "student_sha256")):
        if digest(directory / file) != md[key]:
            raise ValueError("Frozen distribution checkpoint changed")
    if digest(parent / "model.safetensors") != md["parent_model_sha256"] or md["source_sha256"] != sources(root):
        raise ValueError("Parent or trained-source identity changed")
    calibration = json.loads((directory / "calibration.json").read_text())
    if (set(calibration.get("heads", {})) != {"action", "execution", "planning"}
            or calibration.get("evaluation_instances_used") is not False
            or calibration.get("failure_labels_used") is not False):
        raise ValueError("Three independently calibrated success-only heads are required")
    if calibration.get("distribution_sha256") != md["distribution_sha256"]:
        raise ValueError("Calibration model identity mismatch")
    if (calibration.get("student_sha256") != md["student_sha256"]
            or calibration.get("feature_library_sha256") != digest(library.directory / "identity.json")
            or md.get("feature_library_sha256") != digest(library.directory / "identity.json")):
        raise ValueError("Calibration/student/retrieval identity mismatch")
    for key in ("source_sha256", "feature_codec_sha256", "retrieval_index_sha256", "demonstrations_manifest_sha256",
                "normalization_sha256", "encoder_model_sha256"):
        if calibration.get(key) != md.get(key):
            raise ValueError("Calibration artifact identity mismatch: " + key)
    ready = distribution_readiness(md, library.identity, calibration)
    if ready["status"] != "passed":
        raise ValueError("Distribution inference readiness failed: " + str(ready["failures"]))
    return md, calibration


def verified_skill_semantics(plan, metadata, root=ROOT):
    """Bind keep-grasp rules to actual TRAIN human descriptions and frozen plan."""
    relative = "reports/annotation_skill_semantics.json"
    path = Path(root) / relative
    expected = metadata.get("source_sha256", {}).get(relative)
    if not expected or digest(path) != expected:
        raise ValueError("Human annotation skill semantics are absent or changed")
    report = json.loads(path.read_text())
    annotated = {entry["annotation_sha256"] for entry in plan.get("source_train_episodes", [])}
    actual = set(report.get("annotation_sha256", {}).values())
    count = report.get("train_episodes")
    expected_names = {"1":"move to", "2":"pick up from", "3":"place on", "4":"place in",
                      "10":"open door", "12":"close door", "69":"turn on switch"}
    if (report.get("source") != "actual_train_owned_human_skill_annotation; distinct_from_StarterSemanticActionPrimitiveSet"
            or report.get("test_annotations_used") is not False
            or report.get("manifest_sha256") != metadata["demonstrations_manifest_sha256"]
            or type(count) is not int or count <= 0 or len(actual) != count or actual != annotated
            or len(plan.get("source_train_episodes", [])) != count
            or plan.get("source_split") != "train" or plan.get("test_annotations_used") is not False
            or set(report.get("skills", {})) != set(expected_names)):
        raise ValueError("Human skill semantics must match the exact train-owned plan/manifest")
    for key, name in expected_names.items():
        entry = report["skills"][key]
        if (set(entry.get("descriptions", {})) != {name}
                or entry.get("source_episode_count") != count or entry["descriptions"][name] < count):
            raise ValueError("Ambiguous or unsupported human annotation skill semantics: " + key)
    return dict(skills=expected_names, source=report["source"], sha256=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--distribution", "--tgcc", dest="distribution", type=Path, required=True)
    parser.add_argument("--retrieval-data", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "reports/data_plan_vocabulary.json")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18827)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--oracle-upper-bound", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ensure_legacy_imports()
    args.output.mkdir(parents=True, exist_ok=True)
    from .native_controller_policy import install
    install(DistributionController)
    library = RetrievalDataset(args.retrieval_data)
    md, calibration = artifact_gate(args.distribution, args.parent, library)
    protocol = json.loads(args.protocol.read_text())
    if not args.oracle_upper_bound or protocol.get("oracle_inputs") is not True:
        raise ValueError("This current privileged three-graph experiment must be labelled Oracle Upper Bound")
    plan = json.loads(args.plan.read_text())
    if digest(args.plan) != md["plan_sha256"]:
        raise ValueError("Frozen executable skill vocabulary changed")
    skill_semantics = verified_skill_semantics(plan, md)
    norms = list(args.checkpoint.glob("assets/**/norm_stats.json"))
    if len(norms) != 1 or sha(norms[0]) != md["normalization_sha256"] or sha(norms[0]) != protocol["norm_stats_sha256"]:
        raise ValueError("Frozen original normalizer changed")
    base_sha = sha(args.checkpoint / "model.safetensors")
    checkpoint_gate(args.parent, "tri_graph", base_sha, sha(norms[0]), protocol, plan, sha(args.plan))
    frozen = all_sources(ROOT)
    identity = hashlib.sha256(json.dumps(dict(distribution=md["distribution_sha256"], student=md["student_sha256"],
                                              source=frozen, calibration=sha(args.distribution / "calibration.json"),
                                              retrieval=library.identity, protocol=protocol), sort_keys=True).encode()).hexdigest()
    policy, parent = create(str(args.checkpoint), "tri_graph", args.device, str(args.parent))
    load_distribution_student(parent, args.distribution, md)
    parent.requires_grad_(False).eval()
    critic = DistributionCritic(md["distribution_config"]).to(args.device).eval()
    retrieval = RetrievalAdapter(**md["retrieval_adapter_config"]).to(args.device).eval()
    weights = load_file(str(args.distribution / "distribution.safetensors"))
    critic.load_state_dict({k.removeprefix("critic."): v for k, v in weights.items() if k.startswith("critic.")}, strict=True)
    retrieval.load_state_dict({k.removeprefix("retrieval."): v for k, v in weights.items() if k.startswith("retrieval.")}, strict=True)
    if any(not (key.startswith("critic.") or key.startswith("retrieval.")) for key in weights):
        raise ValueError("Unexpected old evaluator/rule state in distribution checkpoint")
    encoder = FrozenVisualEncoder.from_library(args.retrieval_data, args.device)
    codec, normalizer = FrozenFeatureCodec.load(args.retrieval_data), ActionNormalizer(args.checkpoint)
    provider = CurrentFeatureProvider(policy, encoder, codec, normalizer)
    adapter = DistributionGraphAdapter(policy, parent, (ROOT / "data/task_definitions/cook_hot_dogs.bddl").read_text(),
                                        True, args.device, args.plan, plan, critic, retrieval, normalizer, library,
                                        provider, calibration, md.get("runtime_config", {}), skill_semantics)
    def seed(value):
        torch.manual_seed(value)
        adapter.reset()
    def rng():
        return dict(cpu=torch.get_rng_state().numpy(), cuda=torch.cuda.get_rng_state(args.device).cpu().numpy())
    def restore_rng(state):
        torch.set_rng_state(torch.from_numpy(np.array(state["cpu"], copy=True)))
        torch.cuda.set_rng_state(torch.from_numpy(np.array(state["cuda"], copy=True)), args.device)
    store = ExecutedSessionStore(args.output / "durable_state.msgpack", identity, seed, rng, restore_rng, adapter.export, adapter.restore)
    ready = dict(transport="tgr_durable_v1", weight_sha256=identity, pid=os.getpid(), port=args.port,
                 graph_checkpoint_sha256=md["distribution_sha256"], student_checkpoint_sha256=md["student_sha256"],
                 parent_checkpoint_sha256=md["parent_model_sha256"], base_model_sha256=base_sha,
                 norm_stats_sha256=sha(norms[0]), protocol=protocol, source_hashes=frozen,
                 oracle_upper_bound=True, record_video=False, calibration=calibration,
                 new_updates=md['new_updates'],cumulative_schedule_steps=md['cumulative_schedule_steps'],
                 policy_cumulative_updates=md['policy_cumulative_updates'],full_model_updates=md['full_model_updates'],
                 critic_actual_total_updates=md['critic_actual_total_updates'],
                 feature_schema=md['feature_schema'],execution_target=md['execution_target'],
                 memory_unroll_blocks=md['memory_unroll_blocks'],native_limits_retained=True,
                 statistical_feedback_is_task_terminal=False,
                 critic_role="success_distribution_monitor", retrieval_conditioned_action_expert=True,
                 recovery_class_rules_present=False, planning_interface="fixed_executable_skill_decoder")
    (args.output / "server_ready.json").write_text(json.dumps(ready, indent=2))
    watched = [args.distribution / name for name in ("metadata.json", "distribution.safetensors", "student.safetensors", "calibration.json")]
    watched.extend([args.parent / "model.safetensors", args.parent / "metadata.json", args.plan, norms[0]])
    watched.extend(p for p in args.retrieval_data.iterdir() if p.is_file())
    stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
    lock = asyncio.Lock()
    import websockets
    async def handler(socket):
        await socket.send(packb(ready))
        async for message in socket:
            try:
                raw = decode(message)
                async with lock:
                    if all_sources(ROOT) != frozen or any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != stamp for p, stamp in stamps.items()):
                        raise RuntimeError("Frozen inference source or artifact changed")
                    if raw["op"] == "start":
                        if raw["seed"] != protocol["action_seed"]:
                            raise ValueError("Frozen action seed changed")
                        reply = store.start(raw["session"], raw["seed"])
                    elif raw["op"] == "resume":
                        reply = store.resume(raw["session"])
                    elif raw["op"] in ("infer", "finalize"):
                        final = raw["op"] == "finalize"
                        def run():
                            start = time.monotonic()
                            value = adapter.finalize(raw) if final else adapter.infer(raw)
                            with (args.output / "task_progress.jsonl").open("a") as stream:
                                stream.write(json.dumps(dict(adapter.last_task_trace, request_id=int(raw["id"])), default=int) + "\n")
                            return value, time.monotonic() - start
                        operation = store.finalize if final else store.infer
                        reply = await asyncio.to_thread(operation, raw["session"], int(raw["id"]), hashlib.sha256(message).hexdigest(), run)
                    else:
                        raise ValueError("Unknown durable operation")
                await socket.send(packb(reply))
            except websockets.exceptions.ConnectionClosed:
                break
            except DistributionCapabilityStop as error:
                await socket.send(packb(dict(error=str(error), status="capability_stop", transaction_rolled_back=True,
                                             task_failure=None, Q=None)))
            except Exception as error:
                await socket.send(packb(dict(error=str(error), status="infrastructure_error")))
    async def serve():
        async with websockets.serve(handler, "0.0.0.0", args.port, max_size=None, compression=None, ping_interval=None):
            print("SERVER_READY", json.dumps(ready), flush=True)
            await asyncio.Future()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
