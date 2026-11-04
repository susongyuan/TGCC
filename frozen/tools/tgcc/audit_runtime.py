"""Offline real-demonstration interface replay, not a robot task evaluation."""

from pathlib import Path
import sys, os, json, copy

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "code/src"),
    str(ROOT / "code/packages/openpi-client/src"),
]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "TGR_GEOMETRY_CACHE_PATH",
    "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3",
)
import torch, numpy as np
from dataclasses import asdict
from tgr.factory import create
from tgr.data import LazySequenceDataset
from tgr.build_batches import (
    task_fields,
    named_proprio,
    robot_fields_from_named,
    ee_world_from_named,
)
from tgr.runtime_bridge import snapshot_from_decoded_task_info
from tgr.state import EntityRegistry
from tgr.serve import PROMPT
from tgcc.policy import AdapterPolicy
from tgcc.data import conditions_from_dict, move_tree
from tgcc.robot import ActionNormalizer, action_spec
from tgcc.train_system import TrainSystem
from tgcc.serve import TGCCGraphAdapter
from tgcc.checkpoint import sources


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    device = "cuda:0"
    base = "/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"
    parent_path = "/media/data5/ssy/TGR/experiments/tc_esgr_v2_approx_full_5ep_20261024_02/selected_step5000_user_stop"
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    cfg = json.loads((ROOT / "configs/tgcc/default.json").read_text())
    dataset = LazySequenceDataset(
        ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl",
        base,
        ROOT / "reports/data_plan_vocabulary.json",
    )
    index = 50
    batch = dataset[index]
    row = dataset.rows[index]
    data = dataset.current_data
    policy, parent = create(base, "tri_graph", device, parent_path)
    parent.requires_grad_(False)
    parent.eval()
    from openpi.models.model import Observation

    item = move_tree(batch["sequence"][0], device)
    conditions = conditions_from_dict(item["conditions"])
    obs = Observation.from_dict(item["observation"])
    noise = parent.base.sample_noise((1, 32, 32), device)
    adapter = AdapterPolicy(parent).eval()
    original = parent.sample_actions(
        device, obs, noise=noise.clone(), conditions=conditions
    )
    adapted = adapter.sample_actions(
        device, obs, noise=noise.clone(), conditions=conditions
    )
    if not torch.equal(original, adapted):
        raise AssertionError(
            "Zero-gated complete Euler rollout differs from selected parent"
        )
    system = (
        TrainSystem(parent, action_spec(ROOT), ActionNormalizer(base), cfg)
        .to(device)
        .eval()
    )
    graph = TGCCGraphAdapter(
        policy,
        parent,
        (ROOT / "data/task_definitions/cook_hot_dogs.bddl").read_text(),
        True,
        device,
        ROOT / "reports/data_plan_vocabulary.json",
        plan,
        system,
        system.normalizer,
        {"analytic_contract_gate_enabled": True, "action_energy_gate_enabled": False},
        cfg,
    )
    registry = EntityRegistry()
    frames = row["sequence_steps"]
    raws = []
    for k, frame in enumerate(frames):
        r = data.records[frame]
        fields = named_proprio(r["observation.state"])
        now = float(r["timestamp"])
        decoded = task_fields(data.meta["task_obs_keys"], r["observation.task_info"])
        snapshot = snapshot_from_decoded_task_info(
            decoded,
            registry,
            "offline-demonstration-replay",
            now,
            ee_world_from_named(fields),
        )
        robot = robot_fields_from_named(fields, now)
        robot["compute_reachability"] = False
        raw = dict(
            task=PROMPT,
            session="offline-demonstration-replay",
            id=k,
            sim_time=now,
            graph_snapshot=asdict(snapshot),
            robot_fields=robot,
            state=dict(
                base_qvel=fields["base_qvel"],
                trunk_qpos=fields["trunk_qpos"],
                left_arm=fields["arm_left_qpos"],
                right_arm=fields["arm_right_qpos"],
                left_gripper=fields["gripper_left_qpos"],
                right_gripper=fields["gripper_right_qpos"],
            ),
            images={
                target: batch["sequence"][k]["observation"]["image"][name][0]
                .numpy()
                .transpose(2, 0, 1)
                for name, target in [
                    ("base_0_rgb", "head_rgb"),
                    ("left_wrist_0_rgb", "left_wrist_rgb"),
                    ("right_wrist_0_rgb", "right_wrist_rgb"),
                ]
            },
        )
        if k:
            actual = (
                data.records.columns["action"][frames[0] : frame]
                .astype(np.float32)
                .copy()
            )
            raw["execution_receipt"] = dict(
                start_time=float(data.records[frames[0]]["timestamp"]),
                end_time=now,
                actions=actual,
                step_count=len(actual),
                previous_request_id=0,
                source="native_controller_commands_after_successful_env_step",
                demonstration_replay=True,
            )
        raws.append(raw)
    first = graph.infer(raws[0])
    saved = graph.export()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state()
    second = graph.infer(raws[1])
    trace = copy.deepcopy(graph.last_task_trace)
    graph.restore(saved)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    repeated = graph.infer(raws[1])
    if not np.array_equal(second, repeated):
        raise AssertionError(
            "Controller graph/history resume changed next generated chunk"
        )
    if trace["tgcc"]["retrospective"]["actual_executed_steps"] != 32:
        raise AssertionError("Actual demo execution receipt not used")
    # Exercise actual TaskManager commit after pending-prefix transaction closes.
    graph.controller.trace["shadow"] = False
    graph.controller.commit_task(graph.runtime)
    out = dict(
        status="passed",
        offline_replay_only=True,
        not_a_closed_loop_task_result=True,
        optimizer_updates=0,
        demo_episode=row["episode_id"],
        frames=frames,
        full_euler_zero_gate_max_difference=float((original - adapted).abs().max()),
        exact_controller_resume_action_equality=True,
        actual_demonstrated_prefix_used=True,
        task_manager_commit_after_prefix=True,
        candidate_shapes=[list(first.shape), list(second.shape)],
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        source_sha256=sources(ROOT),
    )
    target = ROOT / "runs/interface_audit/runtime_replay_gate.json"
    target.write_text(json.dumps(out, indent=2))
    print(
        json.dumps({k: v for k, v in out.items() if k != "source_sha256"}), flush=True
    )
    dataset.close()


if __name__ == "__main__":
    main()
