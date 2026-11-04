"""Frozen TGR remote inference. Never launches Isaac or trains a checkpoint."""

import argparse, asyncio, hashlib, json, os, pickle, sys, time, types, math
from pathlib import Path
import numpy as np
from .transport import SessionStore, decode, packb

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def sources(root=ROOT):
    files = [
        p
        for folder in (
            "src",
            "deployment",
            "scripts",
            "configs",
            "code/src",
            "code/packages",
        )
        for p in (root / folder).rglob("*")
        if p.is_file() and p.suffix in (".py", ".json", ".bddl")
    ]
    files += list((root / "data/task_definitions").glob("*.bddl"))
    files += [root / "reports/asset_bounds_readonly_audit.json", root / "reports/robot_gripper_proxy_source.json"]
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def assert_frozen(expected, root=ROOT):
    if sources(root) != expected:
        raise RuntimeError("TGR source/config changed during frozen evaluation")


def checkpoint_gate(
    delta, mode, base_sha, norm_sha, protocol, plan=None, plan_sha=None, *, root=ROOT
):
    directory = Path(delta)
    if (directory / "RETIRED_CHECKPOINT.json").exists() or (directory.parent / "RETIRED_CHECKPOINT.json").exists():
        raise ValueError("retired source-bug checkpoint is forbidden")
    if not directory.is_dir():
        raise ValueError("only TC-ESGR V2 full checkpoint directories are active")
    md = json.loads((directory / "metadata.json").read_text())
    if (
        md.get("format") != "tc_esgr_full_v2"
        or md.get("architecture_version") != "tc_esgr_v2"
    ):
        raise ValueError("V2 full checkpoint required")
    user_stop = md.get("user_stop_evaluation", {})
    authorized_stop = (user_stop.get("schema") == "tgr.user_stop_fixed_checkpoint_evaluation.v1"
        and user_stop.get("selected_optimizer_step") == md.get("optimizer_steps") == 5000
        and user_stop.get("five_epoch_budget_waived") is True
        and user_stop.get("run") == str(directory.parent)
        and md.get("training_stopped_by_user") is True
        and md.get("training_complete") is False)
    if (
        not (md.get("training_complete") or authorized_stop)
        or md.get("evaluation_authorized") is not True
        or md.get("completed_phase") != "formal"
    ):
        raise ValueError("precheck checkpoints cannot run formal task evaluation")
    if (
        md.get("training_scope") != "full_backbone_and_graph"
        or md.get("expert_lora") is not False
    ):
        raise ValueError("only requested full fine-tuning scope allowed")
    expected_identity = {
        "feature_schema": "tc_esgr.feature32.v3.approx_center_geometry",
        "reachability_revision": "center_position_ik_capsule_box_v1",
        "task_feedback_revision": "procedural_pending_feedback_v2",
    }
    if any(md.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("approximate center/box procedural identity required")
    if md.get("task_manager_trainable_parameters") != 0 or md.get("task_status_classifier_present") is not False:
        raise ValueError("procedural TaskManager with no state/goal NN classifier required")
    if md.get("global_batch") != 48 or md.get("micro_batch") not in (2, 4):
        raise ValueError("formal batch48 and supported micro-batch provenance required")
    if md.get("micro_batch") == 2 and (md.get("world_size") != 3 or md.get("grad_accum_steps") != 8):
        raise ValueError("micro2 must retain world3/accum8/global48")
    completion = md.get("completed_training_budget", md)
    budget = completion.get("formal_budget", {})
    sequences = budget.get("train_sequences", 0)
    required = budget.get("required_samples", 0)
    if not authorized_stop and (sequences <= 0 or budget.get("required_epochs", 0) < 5
            or required < 5 * sequences or completion.get("coverage_budget_met") is not True
            or completion.get("completed_epochs", 0) < 5
            or completion.get("formal_consumed_samples", 0) < required
            or completion.get("formal_optimizer_steps", 0) < math.ceil(required / 48)
            or completion.get("formal_consumed_samples") != completion.get("formal_optimizer_steps", 0) * 48):
        raise ValueError("complete five-epoch formal sampling budget required")
    if authorized_stop and (completion.get("formal_optimizer_steps") != 4970
            or completion.get("formal_consumed_samples") != 4970 * 48
            or completion.get("coverage_budget_met") is not False):
        raise ValueError("user-stop checkpoint must retain actual incomplete step5000 budget")
    source_hashes = md.get("implementation_source_sha256", {})
    required_sources = {
        "src/tgr/geometry_data.py", "src/tgr/approx_reachability.py",
        "reports/asset_bounds_readonly_audit.json", "reports/robot_gripper_proxy_source.json",
    }
    if not required_sources.issubset(source_hashes):
        raise ValueError("complete geometry modules and catalog SHA provenance required")
    resolved_root = Path(root).resolve()
    for relative, expected in source_hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("implementation source path must remain relative to TGR")
        target = (resolved_root / path).resolve()
        if not target.is_relative_to(resolved_root) or not target.is_file() or sha(target) != expected:
            raise ValueError("implementation source changed: " + relative)
    geometry_assets = json.loads((resolved_root / "deployment/geometry_assets.json").read_text())
    if (geometry_assets.get("schema") != "tgr.deployment_nominal_geometry_assets.v1"
            or geometry_assets.get("units") != "metres"
            or geometry_assets.get("derived_from", {}).get("sha256") != source_hashes["reports/asset_bounds_readonly_audit.json"]):
        raise ValueError("deployment geometry assets differ from the trained nominal catalog")
    full_catalog = json.loads((resolved_root / "reports/asset_bounds_readonly_audit.json").read_text())
    record_keys = ("category", "model", "native_bbox_extent_m", "native_bbox_center_offset_from_root_m", "metadata_source_sha256")
    expected_records = sorted([{key: row[key] for key in record_keys} for row in full_catalog["records"]], key=lambda row: (row["category"], row["model"]))
    if (geometry_assets.get("records") != expected_records
            or geometry_assets.get("static_object_names") != ["bottom_cabinet_no_top_gjeoer_0", "bottom_cabinet_no_top_rkgjer_0"]):
        raise ValueError("deployment nominal dimensions/offsets differ from the trained asset catalog")
    if md.get("optimizer_steps", 0) < 1 or md.get("base_origin_sha256") != base_sha:
        raise ValueError("trained origin proof invalid")
    if md.get("normalization_sha256") != norm_sha:
        raise ValueError("original normalization changed")
    if sha(directory / "model.safetensors") != md.get("model_sha256"):
        raise ValueError("full weights changed")
    selection = md.get("validated_selection", {})
    if (
        selection.get("split") != "validation"
        or selection.get("test_instances_used") is not False
        or selection.get("samples") != 128
        or selection.get("validation_episodes") != 16
        or selection.get("implementation_source_sha256") != source_hashes
        or not math.isfinite(selection.get("flow", float("nan")))
        or not math.isfinite(selection.get("next_macro_f1", float("nan")))
        or selection.get("next_macro_f1", 0) < 0.55
    ):
        raise ValueError("independent heldout selection required")
    if (
        plan is None
        or plan.get("source_split") != "train"
        or plan.get("test_annotations_used") is not False
    ):
        raise ValueError("train-only Plan required")
    if (
        md.get("plan_sha256") != plan_sha
        or md.get("condition_schema") != plan.get("condition_schema")
        or md.get("runtime_factory") != plan.get("runtime_factory")
    ):
        raise ValueError("Plan/runtime/encoding mismatch")
    if md.get("graph_source") == "oracle" and not protocol["oracle_inputs"]:
        raise ValueError("Oracle result must be labelled Upper Bound")
    return {"trained": True, "metadata": md}


class WireProvider:
    def __init__(self, oracle):
        self.oracle = oracle

    def snapshot(self, observation, registry, episode_id, timestamp):
        from .state import (
            Entity,
            Edge,
            Event,
            ObservationSnapshot,
            contains_oracle,
            validate_snapshot,
        )

        x = observation["graph_snapshot"]
        s = ObservationSnapshot(
            x["episode_id"],
            x["timestamp"],
            tuple(Entity(**v) for v in x["entities"]),
            tuple(Edge(**v) for v in x.get("edges", [])),
            tuple(Event(**v) for v in x.get("events", [])),
            bool(x.get("oracle", False)),
        )
        if s.episode_id != episode_id or s.timestamp != timestamp:
            raise ValueError("snapshot episode/time mismatch")
        if contains_oracle(s) and not self.oracle:
            raise ValueError("Oracle forbidden in restricted evaluation")
        if self.oracle and not s.oracle:
            raise ValueError("Oracle upper bound input must be labeled")
        for n in s.entities:
            if registry.resolve(n.instance_id) != n.entity_id:
                raise ValueError("entity identity encoding mismatch")
        validate_snapshot(s)
        return s


class GraphAdapter:
    """Runtime-owned plan adapter; the eval transport never guesses query meanings."""

    def __init__(self, policy, model, task_spec, oracle, device, plan_path, plan):
        import importlib

        module, name = plan["runtime_factory"].split(":", 1)
        if not module.startswith("tgr."):
            raise ValueError("runtime factory must be TGR owned")
        factory = getattr(importlib.import_module(module), name)
        self.runtime = factory(
            plan_path=str(plan_path), device=device, oracle=oracle, task_spec=task_spec
        )
        for name in (
            "reset",
            "begin_chunk",
            "after_prefix",
            "export_state",
            "restore_state",
        ):
            if not callable(getattr(self.runtime, name, None)):
                raise ValueError("runtime missing " + name)
        if list(getattr(self.runtime, "node_keys", [])) != plan["node_keys"]:
            raise ValueError(
                "runtime candidate query order differs from frozen training plan"
            )
        self.policy = policy
        self.model = model
        self.oracle = oracle
        self.plan = plan
        self.reset()

    def reset(self):
        self.episode = None
        self.runtime.reset(None)

    def export(self):
        return {"episode": self.episode, "runtime": self.runtime.export_state()}

    def restore(self, state):
        if state is None:
            self.reset()
            return
        self.episode = state["episode"]
        self.runtime.restore_state(state["runtime"])

    def infer(self, raw):
        if raw["task"] != PROMPT:
            raise ValueError("hotdog prompt changed from frozen baseline")
        episode = raw["session"]
        if self.episode != episode:
            if self.episode is not None:
                raise ValueError("new episode requires explicit session start")
            self.runtime.reset(episode)
            self.episode = episode
        # Defense at the wire boundary, independent of runtime implementation.
        WireProvider(self.oracle).snapshot(
            raw,
            __import__("tgr.state", fromlist=["EntityRegistry"]).EntityRegistry(),
            episode,
            float(raw["sim_time"]),
        )
        c = self.runtime.begin_chunk(raw)
        if list(self.runtime.node_keys) != self.plan["node_keys"]:
            raise ValueError("runtime query identity order changed")
        if c.queries.shape[1] != len(self.plan["node_keys"]):
            raise ValueError("candidate plan query count mismatch")
        if not self.oracle and (c.oracle or not c.restricted):
            raise ValueError("restricted condition provenance mismatch")
        self.policy._sample_kwargs["conditions"] = c
        try:
            result = self.policy.infer(model_input(raw))
        finally:
            self.policy._sample_kwargs.pop("conditions", None)
        decision = self.model.take_decision()
        self.runtime.after_prefix(decision)
        self.last_task_trace = None
        if decision is not None:
            from .task import STATES

            scores = decision["next"].detach().float().cpu()
            probabilities = scores.softmax(-1)[0].tolist()
            selected = int(decision["selected_index"].detach().cpu()[0])
            manager = self.runtime.bridge.runtime.environments[0]["task_manager"]
            # Current skill probabilities are learned; PENDING/ACTIVE is procedural.
            if any(value not in (0, 1) for value in decision["plan_states"].cpu().tolist()[0]):
                raise ValueError("decoder procedural plan states must be PENDING/ACTIVE")
            self.last_task_trace = {
                "session": episode,
                "sim_time": float(raw["sim_time"]),
                "chunk": self.runtime.chunk,
                "selected_index": selected,
                "selected_candidate": self.plan["node_keys"][selected],
                "candidate_order": list(self.plan["node_keys"]),
                "current_skill_probabilities": probabilities,
                "next_probabilities": probabilities,  # legacy wire alias, current skill selection
                "probability_source": "actual_current_skill_model_output",
                "procedural_plan_states": {
                    key: STATES[int(value)]
                    for key, value in zip(
                        self.plan["node_keys"],
                        decision["plan_states"].cpu().tolist()[0],
                    )
                },
                "status_source": "procedural_pending_feedback_v2",
                "task_manager_states": dict(manager.states),
                "verified_goal_evidence": {
                    key: dict(manager.evidence.get(key, {}))
                    for key in manager.node_goals
                },
                "memory_l2": float(decision["memory"].detach().float().norm().cpu()),
                "source": "actual_model_current_skill_selection",
                "oracle_upper_bound": self.oracle,
                "scope": "current annotated skill selection; procedural plan states; trusted goal evidence; official task score independent",
            }
        return np.asarray(result["actions"], np.float32)


# Audited R1Pro legacy layout, owned copy; no active imports from pi05 tools.
INDICES = {
    "base_qvel": slice(0, 3),
    "arm_left_qpos": slice(3, 10),
    "arm_left_qvel": slice(10, 17),
    "eef_left_pos": slice(17, 20),
    "eef_left_quat": slice(20, 24),
    "gripper_left_qpos": slice(24, 26),
    "gripper_left_qvel": slice(26, 28),
    "arm_right_qpos": slice(28, 35),
    "arm_right_qvel": slice(35, 42),
    "eef_right_pos": slice(42, 45),
    "eef_right_quat": slice(45, 49),
    "gripper_right_qpos": slice(49, 51),
    "gripper_right_qvel": slice(51, 53),
    "trunk_qpos": slice(53, 57),
    "trunk_qvel": slice(57, 61),
}


def install_compat():
    for name in ("omnigibson", "omnigibson.learning", "omnigibson.learning.utils"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    m = types.ModuleType("omnigibson.learning.utils.eval_utils")
    m.PROPRIOCEPTION_INDICES = {"R1Pro": INDICES}
    sys.modules[m.__name__] = m


def model_input(raw):
    from openpi.policies.b1k_policy import extract_state_from_proprio

    s = raw["state"]
    p = np.full(61, np.nan, np.float32)
    for a, b in (
        ("base_qvel", "base_qvel"),
        ("trunk_qpos", "trunk_qpos"),
        ("arm_left_qpos", "left_arm"),
        ("arm_right_qpos", "right_arm"),
        ("gripper_left_qpos", "left_gripper"),
        ("gripper_right_qpos", "right_gripper"),
    ):
        p[INDICES[a]] = np.asarray(s[b], np.float32)
    if not np.isfinite(extract_state_from_proprio(p)).all():
        raise ValueError("invalid proprio")
    x = {"observation/state": p, "prompt": raw["task"]}
    for a, b in (
        ("head_rgb", "egocentric_camera"),
        ("left_wrist_rgb", "wrist_image_left"),
        ("right_wrist_rgb", "wrist_image_right"),
    ):
        im = np.asarray(raw["images"][a])
        if im.dtype != np.uint8 or im.shape != (3, 224, 224):
            raise ValueError("expected decoded uint8 CHW224")
        x["observation/" + b] = im.transpose(1, 2, 0)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--delta", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", default="tri_graph")
    p.add_argument("--port", default=18817, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--oracle-upper-bound", action="store_true")
    p.add_argument(
        "--task-spec",
        type=Path,
        default=ROOT / "data/task_definitions/cook_hot_dogs.bddl",
    )
    a = p.parse_args()
    import torch, websockets
    from .factory import create

    a.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(a.protocol.read_text())
    if bool(protocol["oracle_inputs"]) != a.oracle_upper_bound:
        raise ValueError("explicit Oracle flag/protocol mismatch")
    norm = list((a.checkpoint / "assets").rglob("norm_stats.json"))
    if len(norm) != 1 or sha(norm[0]) != protocol["norm_stats_sha256"]:
        raise ValueError("checkpoint normalization mismatch")
    base_sha = sha(a.checkpoint / "model.safetensors")
    plan = json.loads(a.plan.read_text())
    payload = checkpoint_gate(
        a.delta, a.mode, base_sha, sha(norm[0]), protocol, plan, sha(a.plan)
    )
    # Verify all base artifacts against independent frozen protocol identity.
    h = hashlib.sha256()
    known = {
        "model.safetensors": "7e257666d835f6af701de493676a6c86a0421b2efc737a0f911d782b7a09f635",
        "config.json": "a4ae208203adfdd64c5fdbd4b0dc257e4ebbc82e464cb146dd0377051b25fc0a",
    }
    if (
        protocol["base_weight_sha256"]
        != "7d3972b84f04b1e1bfe80c722036c8aac139dae4c99b75973a49ea5288f47174"
    ):
        raise ValueError("unexpected paired base bundle")
    for name, expected in known.items():
        value = base_sha if name == "model.safetensors" else sha(a.checkpoint / name)
        if value != expected:
            raise ValueError("base artifact changed: " + name)
        h.update(name.encode())
        h.update(bytes.fromhex(value))
    h.update(sha(norm[0]).encode())
    frozen = sources()
    artifact_paths = [ROOT / relative for relative in payload["metadata"]["implementation_source_sha256"]] + [
        a.delta / "model.safetensors",
        a.delta / "metadata.json",
        a.plan,
        a.checkpoint / "model.safetensors",
        a.checkpoint / "config.json",
        norm[0],
    ]
    artifact_stats = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in artifact_paths
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "base_model": base_sha,
                "base_bundle": h.hexdigest(),
                "delta": sha(a.delta / "model.safetensors"),
                "plan": sha(a.plan),
                "source": frozen,
                "protocol": protocol,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.set_num_threads(4)
    install_compat()
    policy, model = create(str(a.checkpoint), a.mode, a.device, str(a.delta))
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    adapter = GraphAdapter(
        policy,
        model,
        a.task_spec.read_text(),
        a.oracle_upper_bound,
        a.device,
        a.plan,
        plan,
    )

    def seed(n):
        torch.manual_seed(n)
        adapter.reset()

    def get_rng():
        return {
            "cpu": torch.get_rng_state().numpy(),
            "cuda": (
                torch.cuda.get_rng_state(a.device).cpu().numpy()
                if a.device.startswith("cuda")
                else None
            ),
        }

    def set_rng(s):
        torch.set_rng_state(torch.from_numpy(np.array(s["cpu"], copy=True)))
        if s["cuda"] is not None:
            torch.cuda.set_rng_state(
                torch.from_numpy(np.array(s["cuda"], copy=True)), a.device
            )

    store = SessionStore(
        a.output / "durable_state.msgpack",
        identity,
        seed,
        get_rng,
        set_rng,
        adapter.export,
        adapter.restore,
    )
    ready = {
        "transport": "tgr_durable_v1",
        "weight_sha256": identity,
        "graph_checkpoint_sha256": sha(a.delta / "model.safetensors"),
        "graph_checkpoint_trained": True,
        "base_model_sha256": base_sha,
        "norm_stats_sha256": sha(norm[0]),
        "protocol": protocol,
        "source_hashes": frozen,
        "source_changed_during_run": False,
        "oracle_upper_bound": a.oracle_upper_bound,
        "comparison": (
            "privileged Oracle Upper Bound; NOT information-matched RGB comparison"
            if a.oracle_upper_bound
            else "restricted observation mode"
        ),
        "condition_schema": plan["condition_schema"],
        "plan_sha256": sha(a.plan),
        "plan_node_keys": plan["node_keys"],
        "checkpoint_selection": payload["metadata"]["validated_selection"],
        "implementation_identity": {key: payload["metadata"][key] for key in ("architecture_version", "feature_schema", "reachability_revision", "task_feedback_revision")},
        "implementation_source_sha256": payload["metadata"]["implementation_source_sha256"],
        "completed_training_budget": payload["metadata"].get("completed_training_budget", payload["metadata"]),
        "task_status_source": "procedural_pending_feedback_v2",
        "current_skill_probability_source": "actual_model_output",
        "pid": os.getpid(),
        "port": a.port,
        "prompt": PROMPT,
        "reachability_provider_capabilities": getattr(
            adapter.runtime,
            "reachability_capabilities",
            {
                "ik": "UNKNOWN",
                "collision": "UNKNOWN",
                "base_pose_feedback": "UNKNOWN",
                "policy": "no guessed distance feasibility; no hardcoded basegoto",
            },
        ),
        "base_frozen": False,
        "expert_lora": False,
        "training_scope": "full_backbone_and_graph",
        "inference_weights_frozen": True,
    }
    tmp = a.output / "server_ready.tmp"
    tmp.write_text(json.dumps(ready, indent=2))
    tmp.replace(a.output / "server_ready.json")
    lock = asyncio.Lock()

    async def handler(ws):
        await ws.send(packb(ready))
        async for message in ws:
            try:
                raw = decode(message)
                async with lock:
                    assert_frozen(frozen)
                    if any(
                        (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != stamp
                        for p, stamp in artifact_stats.items()
                    ):
                        raise ValueError(
                            "frozen model/plan/normalizer artifact changed"
                        )
                    if raw["op"] == "start":
                        if raw["seed"] != 42:
                            raise ValueError("paired action seed must be42")
                        reply = store.start(raw["session"], raw["seed"])
                    elif raw["op"] == "resume":
                        reply = store.resume(raw["session"])
                    elif raw["op"] == "infer":

                        def run():
                            t = time.monotonic()
                            actions = adapter.infer(raw)
                            # Called only for a new idempotent request. Cached
                            # reply retries do not create duplicate prediction rows.
                            if adapter.last_task_trace is not None:
                                trace = dict(
                                    adapter.last_task_trace,
                                    request_id=int(raw["id"]),
                                    checkpoint_sha256=ready["graph_checkpoint_sha256"],
                                )
                                with (a.output / "task_progress.jsonl").open("a") as f:
                                    f.write(json.dumps(trace) + "\n")
                            return actions, time.monotonic() - t

                        reply = await asyncio.to_thread(
                            store.infer,
                            raw["session"],
                            int(raw["id"]),
                            hashlib.sha256(message).hexdigest(),
                            run,
                        )
                        with (a.output / "generation_audit.jsonl").open("a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "time": time.time(),
                                        "session": raw["session"],
                                        "id": raw["id"],
                                        "cached": reply["cached"],
                                        "inference_seconds": reply["inference_seconds"],
                                        "oracle": a.oracle_upper_bound,
                                    }
                                )
                                + "\n"
                            )
                    else:
                        raise ValueError("unknown operation")
                await ws.send(packb(reply))
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                try:
                    await ws.send(packb({"error": str(e)}))
                except websockets.exceptions.ConnectionClosed:
                    break

    async def run():
        async with websockets.serve(
            handler,
            "127.0.0.1",
            a.port,
            max_size=32 * 1024**2,
            compression=None,
            ping_interval=None,
        ):
            await asyncio.Future()

    try:
        asyncio.run(run())
    finally:
        receipt = dict(ready, source_changed_during_run=sources() != frozen)
        (a.output / "server_final.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
