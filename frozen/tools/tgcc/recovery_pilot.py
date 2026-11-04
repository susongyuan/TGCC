#!/usr/bin/env python3
"""Bounded, no-video native simulation collector on train instances only.

The pilot measures real restore, rollout, and verification latency. The native
controller corrects a small base XY displacement; this does not establish any
trained BASE_ADJUST, grasp recovery, or hot-dog task completion capability.
"""

import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tgcc.recovery_collection import (
    body_velocity_to_world_target,
    certificate_dict,
    certify_position_return,
    certify_joint_return,
    timing_summary,
    velocity_to_controller_input,
)
from tgcc.collection import GraphCollectionRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--recovery-steps", type=int, default=96)
    parser.add_argument("--max-texture-mips", type=int, default=10)
    parser.add_argument("--scenario-offset", type=int, default=0)
    parser.add_argument("--replay-actions", type=Path)
    parser.add_argument("--max-injections", type=int, default=12)
    parser.add_argument(
        "--subsystem", choices=("base", "left_arm", "right_arm"), default="base"
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Disable robot cameras/material loading; state-only critic data, no visual imitation targets",
    )
    args = parser.parse_args()
    if (
        not 0 <= args.instance < 200
        or not 1 <= args.trials <= 300
        or not 1 <= args.recovery_steps <= 180
    ):
        parser.error(
            "pilot requires train instance 0..199, 1..12 trials, 1..180 corrective steps"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "schema": "tgcc.geometry_recovery_pilot.v1",
        "status": "starting",
        "mode": "train",
        "instance": args.instance,
        "video": False,
        "task_score": None,
        "trials": [],
        "new_human_demonstrations": 0,
        "pid": os.getpid(),
        "scope": "native_controller_base_xy_return",
        "rgb_recorded": not args.state_only,
        "subsystem": args.subsystem,
        "balanced": args.balanced,
    }

    def write():
        report["elapsed_wall_seconds"] = time.perf_counter() - started
        path = args.output / "report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    def stage(value):
        report["status"] = value
        write()
        print(
            json.dumps(
                {
                    "status": value,
                    "elapsed_wall_seconds": report["elapsed_wall_seconds"],
                }
            ),
            flush=True,
        )

    evaluator = og = None
    stage("importing")
    try:
        import numpy as np
        import torch

        torch.set_num_threads(4)
        from omegaconf import OmegaConf

        spec = importlib.util.spec_from_file_location(
            "eval_client", ROOT / "deployment/tgcc_eval_client.py"
        )
        client = importlib.util.module_from_spec(spec)
        sys.modules["eval_client"] = client
        spec.loader.exec_module(client)
        import omnigibson as og
        from omnigibson.macros import gm

        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False
        client.install_wrapper()
        from omnigibson.eval.evaluator import Evaluator, DEFAULT_ROBOT_CONFIG_PATH
        from omnigibson.eval.utils.eval_utils import seed_everything

        seed_everything(0)
        stage("launching_simulator")
        og.launch()
        import carb

        settings = carb.settings.get_settings()
        settings.set_bool(
            "/rtx-transient/resourcemanager/texturestreaming/enabled", True
        )
        settings.set_float(
            "/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.25
        )
        # NVIDIA RTX Common: cap source texture resolution, retaining materials,
        # lighting and RGB. This is recorded; it is not bit-identical imagery.
        settings.set_int(
            "/rtx-transient/resourcemanager/maxMipCount", args.max_texture_mips
        )
        settings.set_int(
            "/rtx-transient/resourcemanager/texturestreaming/streamingBudgetMB", 50
        )
        report["renderer_profile"] = dict(
            max_texture_mips=args.max_texture_mips,
            streaming_budget_mb=50,
            texture_memory_fraction=0.25,
            materials_enabled=not args.state_only,
            bit_identical_to_original_renderer=False,
        )
        if args.state_only:
            settings.set_bool("/app/renderer/skipMaterialLoading", True)
        # Match the previously used low-memory evaluator profile. Headless
        # alone does not disable the active viewport's render product.
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.updates_enabled = False
        except ImportError:
            pass
        robot_config = OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH)
        robot_config.sensor_config.VisionSensor.sensor_kwargs.image_height = 224
        robot_config.sensor_config.VisionSensor.sensor_kwargs.image_width = 224
        if args.state_only:
            robot_config.obs_modalities = ["proprio"]
            robot_config.eval.camera_sensor_names = {}
        cfg = OmegaConf.create(
            {
                "env_wrapper": {
                    "_target_": "eval_client.RGBResolutionWrapper",
                    "image_size": 224,
                    "head_image_size": 224,
                    "wrist_image_size": 224,
                },
                "policy_name": "tgcc-recovery-pilot",
                "model": {
                    "_target_": "omnigibson.eval.policies.LocalPolicy",
                    "action_dim": None,
                },
                "headless": True,
                "partial_scene_load": True,
                "max_steps": None,
                "write_video": False,
                "video_view": "robot",
                "mode": "train",
                "seed": 0,
                "task": {"name": "cook_hot_dogs"},
                "robot": robot_config,
            }
        )
        stage("loading_train_scene")
        evaluator = Evaluator(cfg)
        evaluator.reset()
        evaluator.load_task_instance(args.instance)
        if not args.state_only:
            # RGBResolutionWrapper preserves the policy RGB path. Depth is
            # added only to this recorder and never changes model inputs.
            for sensor in evaluator.robot.sensors.values():
                if hasattr(sensor, "image_height"):
                    sensor.add_modality("depth_linear")
                    sensor.load_observation_space()
        evaluator.reset()
        robot = evaluator.robot
        contract = client.audit_controller_contract(robot)
        indices = {k: client.array(v) for k, v in robot.controller_action_idx.items()}
        report["native_controller_contract"] = contract
        report["startup_wall_seconds"] = time.perf_counter() - started
        stage("verifying_state_restore")

        def geometry():
            p, q = robot.get_position_orientation()
            feedback = robot._get_proprioception_dict()
            objects = {}
            for name, obj in evaluator.env.task.object_scope.items():
                if (
                    obj is not None
                    and hasattr(obj, "get_position_orientation")
                    and obj is not robot
                ):
                    try:
                        op, oq = obj.get_position_orientation()
                        objects[name] = {
                            "position": client.array(op).tolist(),
                            "quaternion_xyzw": client.array(oq).tolist(),
                        }
                    except (AttributeError, NotImplementedError):
                        pass
            return {
                "base_position": client.array(p).tolist(),
                "base_quaternion_xyzw": client.array(q).tolist(),
                "joint_positions": client.array(robot.get_joint_positions()).tolist(),
                "objects": objects,
                "sim_time": float(og.sim.current_time),
                "left_arm": client.array(feedback["arm_left_qpos"]).tolist(),
                "right_arm": client.array(feedback["arm_right_qpos"]).tolist(),
            }

        def measured_hold():
            fields = robot._get_proprioception_dict()
            state = {
                "left_arm": client.array(fields["arm_left_qpos"]),
                "right_arm": client.array(fields["arm_right_qpos"]),
                "trunk_qpos": client.array(fields["trunk_qpos"]),
                "left_gripper": client.array(fields["gripper_left_qpos"]),
                "right_gripper": client.array(fields["gripper_right_qpos"]),
            }
            parts, _ = client.hold_missing({}, state)
            return parts

        parts = measured_hold()

        recorder = None
        registry = extractor = None
        recorded_steps = 0
        pending_actions = []
        pending_annotations = []
        current_annotation = None
        source_episode = None
        source_split = "train"
        simulator_origin = None

        def capture():
            if args.state_only:
                feedback = robot._get_proprioception_dict()
                payload = dict(
                    images={},
                    state={
                        k: client.vector(feedback[src], size, k)
                        for k, (src, size) in client.STATE_KEYS.items()
                    },
                    task=client.TASKS["cook_hot_dogs"],
                    frequency=30.0,
                    embodiment_type=client.EMBODIMENT,
                    rgb_available=False,
                )
            else:
                payload = client.observation(
                    robot,
                    evaluator.obs,
                    evaluator.robot_camera_names,
                    client.TASKS["cook_hot_dogs"],
                )
            actual_simulator_time = float(og.sim.current_time)
            now = actual_simulator_time - simulator_origin
            if abs(now - recorded_steps / 30) > 2e-4:
                raise ValueError("actual simulator time differs from executed step timeline")
            if not args.state_only:
                depth, cameras = {}, {}
                for role, camera_name in evaluator.robot_camera_names.items():
                    image = client.array(evaluator.obs[camera_name + "::depth_linear"])
                    if image.shape != (224, 224):
                        raise ValueError("unaligned camera depth dimensions")
                    valid = np.isfinite(image) & (image > 0)
                    if not valid.any():
                        raise ValueError("camera depth has no valid measured pixels")
                    depth[role] = dict(values=np.where(valid, image, 0).astype(np.float32),
                                       valid=valid, timestamp=now, units="metres_to_image_plane")
                    sensor = robot.sensors[camera_name.split("::")[1]]
                    position, quaternion = sensor.get_position_orientation()
                    cameras[role] = dict(timestamp=now,
                        intrinsic_matrix=client.array(sensor.intrinsic_matrix).tolist(),
                        position_world=client.array(position).tolist(),
                        quaternion_xyzw=client.array(quaternion).tolist(),
                        camera_axes="USD_negative_Z_forward_positive_Y_up",
                        source="actual_render_product_and_live_camera_pose")
                payload["depth"] = depth
                payload["camera_calibration"] = cameras
                payload["rgb_exposure_timestamp"] = now
                payload["sensor_acquisition"] = dict(actual_simulator_time=actual_simulator_time,
                    relative_origin=simulator_origin, step=recorded_steps,
                    method="post_env_step_render_without_extra_physics")
            snap = extractor(payload, registry, recorder.session, now)
            from dataclasses import asdict

            payload.update(
                session=recorder.session,
                sim_time=now,
                graph_snapshot=asdict(snap),
                control_contract=contract,
                robot_fields=client.live_robot_fields(
                    robot,
                    robot._get_proprioception_dict(),
                    now,
                    extractor.geometry_spec,
                ),
                collection_provenance=dict(source_episode_id=source_episode,
                    source_split=source_split, source_commands_are_not_labels=True,
                    snapshot_labels_privileged=True, physics_steps=recorded_steps),
            )
            recorder.capture(payload, executed_native=pending_actions or None,
                execution_annotations=pending_annotations or None)
            pending_actions.clear()
            pending_annotations.clear()

        def step_native(action):
            nonlocal recorded_steps
            with og.sim.render_on_step(not args.state_only):
                obs, _, terminated, truncated, _ = evaluator.env.step(
                    torch.as_tensor(action, device=og.sim.device), n_render_iterations=1
                )
            evaluator.obs = evaluator._preprocess_obs(obs)
            if recorder is not None:
                recorded_steps += 1
                pending_actions.append(action.tolist())
                pending_annotations.append(deepcopy(current_annotation) if current_annotation else
                    dict(source_step=None, command_source="geometry_controller"))
                if len(pending_actions) == 32:
                    capture()
            return bool(terminated or truncated), action.tolist()

        def step(base):
            parts["lower_body"][:3] = velocity_to_controller_input(base, contract)
            action = client.action_vector(parts, indices, robot.action_dim)
            return step_native(action)

        def begin_recording(label):
            nonlocal recorder, registry, extractor, recorded_steps, simulator_origin
            registry = client.EntityRegistry()
            extractor = client.OmniGibsonExtractor(
                evaluator.env,
                arm_mapping={"left": "left", "right": "right"},
                geometry_assets=client.NominalGeometryAssets(
                    ROOT / "deployment/geometry_assets.json"
                ),
            )
            recorder = GraphCollectionRecorder(
                args.output / f"graphs_{label}",
                ROOT / "reports/data_plan_vocabulary.json",
                f"collector-train-{args.instance}-{label}",
                oracle=True,
            )
            recorded_steps = 0
            simulator_origin = float(og.sim.current_time)
            pending_actions.clear()
            pending_annotations.clear()
            capture()

        def displacement(a, b):
            return math.dist(a["base_position"][:2], b["base_position"][:2])

        def unrelated_motion(a, b):
            return max(
                (
                    math.dist(v["position"], b["objects"][k]["position"])
                    for k, v in a["objects"].items()
                    if k in b["objects"]
                ),
                default=0.0,
            )

        snapshot = deepcopy(og.sim.dump_state(serialized=False))
        before_restore = geometry()
        restore_start = time.perf_counter()
        og.sim.load_state(deepcopy(snapshot), serialized=False)
        terminal, _ = step(
            [0, 0, 0]
        )  # Required by native API before checking spatial states.
        restored = geometry()
        restore_error = displacement(before_restore, restored)
        joint_error = float(
            np.max(
                np.abs(
                    np.asarray(before_restore["joint_positions"])
                    - restored["joint_positions"]
                )
            )
        )
        report["restore_check"] = {
            "wall_seconds": time.perf_counter() - restore_start,
            "base_xy_error_m": restore_error,
            "joint_max_error_rad": joint_error,
            "object_max_error_m": unrelated_motion(before_restore, restored),
            "verified": not terminal
            and restore_error <= 0.002
            and joint_error <= 0.01
            and unrelated_motion(before_restore, restored) <= 0.002,
            "physics_cache_step_executed": True,
        }
        if not report["restore_check"]["verified"]:
            stage("blocked_restore_verification")
            return
        torch.save(snapshot, args.output / "native_initial_state.pt")
        if args.replay_actions:
            from tgcc.replay_collection import reexecute
            from tgcc.robot import action_spec
            from omnigibson.controllers.controller_base import IsGraspingState

            source = np.load(args.replay_actions, allow_pickle=False)
            source_episode = str(source["episode_id"].item()) if "episode_id" in source else "450010"
            source_split = str(source["split"].item()) if "split" in source else "train"
            report["source_episode_id"] = source_episode
            report["source_split"] = source_split
            report["source_actions_sha256"] = __import__("hashlib").sha256(args.replay_actions.read_bytes()).hexdigest()
            native = np.asarray(source["actions"], np.float32)
            limits = action_spec(ROOT)
            if (
                native.ndim != 2
                or native.shape[1] != 23
                or not np.isfinite(native).all()
                or (
                    (native < np.asarray(limits.lower, np.float32))
                    | (native > np.asarray(limits.upper, np.float32))
                ).any()
            ):
                raise ValueError(
                    "replay native control limits do not match live audited schema"
                )
            report.update(
                scope="new_demonstration_command_reexecution",
                original_outcomes_reused=False,
                training_split=source_split,
                initial_base_difference_m=float(
                    np.linalg.norm(
                        np.asarray(geometry()["base_position"])
                        - source["initial_base_world"][:3, 3]
                    )
                ),
            )
            initial_geometry = geometry()
            base_pose = client.transform(np.asarray(initial_geometry["base_position"]),
                                         np.asarray(initial_geometry["base_quaternion_xyzw"]))
            relative_rotation = np.asarray(source["initial_base_world"][:3, :3]).T @ base_pose[:3, :3]
            report["initial_base_rotation_difference_rad"] = float(np.arccos(np.clip((np.trace(relative_rotation)-1)/2,-1,1)))
            if "initial_object_positions_json" in source:
                expected_objects = json.loads(source["initial_object_positions_json"].item())
                differences = {name:float(np.linalg.norm(np.asarray(position)-initial_geometry["objects"][name]["position"]))
                               for name,position in expected_objects.items() if name in initial_geometry["objects"]}
                if len(differences) != len(expected_objects):
                    raise ValueError("initial object mapping does not match source demonstration")
                report["initial_object_position_differences_m"] = differences
            if (report["initial_base_difference_m"] > 0.05
                    or report["initial_base_rotation_difference_rad"] > 0.04
                    or max(report.get("initial_object_position_differences_m", {}).values(), default=0)>0.05):
                stage("blocked_initial_context_mismatch")
                return
            begin_recording("replay")

            def feedback():
                values = {}
                for name, obj in evaluator.env.task.object_scope.items():
                    if name.startswith("hotdog.") and obj is not None:
                        for side in ("left", "right"):
                            value = robot.is_grasping(arm=side, candidate_obj=obj)
                            values[name, side] = (
                                True
                                if value == IsGraspingState.TRUE
                                else False
                                if value == IsGraspingState.FALSE
                                else None
                            )
                return values

            def execute(action):
                return step_native(action)[0]

            def progress(value):
                report["replay_progress"] = value
                write()

            def changed():
                if pending_actions:
                    capture()

            def before_action(annotation):
                nonlocal current_annotation
                if annotation["first_intervention_step"] and pending_actions:
                    capture()
                current_annotation = dict(annotation, source_episode_id=source_episode,
                    source_split=source_split, command_source="demonstration_command_reexecution")

            stage("reexecuting_demonstration_commands")
            result = reexecute(
                native, execute, feedback, progress=progress, on_state_change=changed
                , on_before_action=before_action, max_injections=args.max_injections
            )
            if pending_actions:
                capture()
            report["replay_result"] = result
            stage("completed")
            return
        for trial in range(args.trials):
            if args.balanced:
                args.subsystem = ("base", "left_arm", "right_arm")[trial % 3]
            stage(f"collecting_trial_{trial}")
            trial_start = time.perf_counter()
            recorder = None
            og.sim.load_state(deepcopy(snapshot), serialized=False)
            terminal, _ = step([0, 0, 0])
            origin = geometry()
            parts = measured_hold()
            begin_recording(f"{trial:03d}")
            actions, states = [], [origin]
            scenario = args.scenario_offset + (trial // 3 if args.balanced else trial)
            direction = (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (0.7071, 0.7071),
                (0.7071, -0.7071),
                (-0.7071, 0.7071),
                (-0.7071, -0.7071),
            )[scenario % 8]
            speed = (0.025, 0.035, 0.045, 0.055, 0.065)[(scenario // 8) % 5]
            perturbation_steps = (16, 20, 24)[(scenario // 40) % 3]
            pulse = [speed * direction[0], speed * direction[1], 0.0]
            if args.subsystem != "base":
                from tgcc.robot import action_spec

                limits = action_spec(ROOT)
                offset = 7 if args.subsystem == "left_arm" else 15
                joint = scenario % 7
                delta = (0.025, 0.04, 0.055, 0.07, 0.085)[(scenario // 14) % 5] * (
                    1 if (scenario // 7) % 2 == 0 else -1
                )
                target = np.asarray(origin[args.subsystem], np.float32)
                if (
                    not limits.lower[offset + joint]
                    <= target[joint] + delta
                    <= limits.upper[offset + joint]
                ):
                    delta = -delta
                if (
                    not limits.lower[offset + joint]
                    <= target[joint] + delta
                    <= limits.upper[offset + joint]
                ):
                    raise ValueError("no valid bounded arm perturbation at this joint")
                target[joint] += delta
                parts[args.subsystem] = target
                pulse = [0.0, 0.0, 0.0]
                perturbation_steps = (20, 28)[(scenario // 70) % 2]
            for _ in range(perturbation_steps):
                terminal, action = step(pulse)
                actions.append(
                    {"phase": "controlled_deviation", "native_action23": action}
                )
                states.append(geometry())
                if terminal:
                    break
            perturbed = geometry()

            def error(state):
                if args.subsystem == "base":
                    return displacement(origin, state)
                return float(
                    np.max(
                        np.abs(
                            np.asarray(origin[args.subsystem]) - state[args.subsystem]
                        )
                    )
                )

            initial_error = error(perturbed)
            if args.subsystem != "base":
                parts[args.subsystem] = np.asarray(origin[args.subsystem], np.float32)
            count = 0
            steady = 0
            if not terminal:
                for _ in range(args.recovery_steps):
                    current = geometry()
                    command = body_velocity_to_world_target(
                        current["base_position"],
                        current["base_quaternion_xyzw"],
                        origin["base_position"][:2],
                    )
                    if args.subsystem != "base":
                        command = [0.0, 0.0, 0.0]
                    terminal, action = step(command)
                    count += 1
                    actions.append(
                        {"phase": "feedback_correction", "native_action23": action}
                    )
                    states.append(geometry())
                    steady = (
                        steady + 1
                        if error(states[-1])
                        < (0.005 if args.subsystem == "base" else 0.01)
                        else 0
                    )
                    if terminal or steady >= 3:
                        break
            terminal_after, action = step([0, 0, 0])
            actions.append({"phase": "stop", "native_action23": action})
            final = geometry()
            states.append(final)
            if pending_actions:
                capture()
            if args.subsystem == "base":
                certificate_data = certificate_dict(
                    certify_position_return(
                        initial_error_m=initial_error,
                        final_error_m=displacement(origin, final),
                        executed_steps=count,
                        maximum_unrelated_object_displacement_m=max(
                            unrelated_motion(origin, s) for s in states
                        ),
                        restore_verified=report["restore_check"]["verified"],
                        terminal=terminal or terminal_after,
                    )
                )
            else:
                certificate_data = certify_joint_return(
                    initial_error,
                    error(final),
                    count,
                    max(unrelated_motion(origin, s) for s in states),
                    restore_verified=report["restore_check"]["verified"],
                    terminal=terminal or terminal_after,
                )
            row = {
                "trial": trial,
                "wall_seconds": time.perf_counter() - trial_start,
                "certificate": certificate_data,
                "actions": actions,
                "states": states,
                "target_frame": "world" if args.subsystem == "base" else "joint",
                "subsystem": args.subsystem,
                "scenario_id": f"train-{args.instance}:{args.subsystem}-return-{scenario}",
                "perturbation_velocity_m_s": pulse,
                "perturbation_steps": perturbation_steps,
                "target_xy_m": origin["base_position"][:2],
                "joint_target_rad": origin.get(args.subsystem)
                if args.subsystem != "base"
                else None,
                "label_provenance": "actual_native_execution_and_geometric_verification",
                "training_use": "geometry_recovery_only_requires_separate_dataset_adapter",
            }
            (args.output / f"trial_{trial:03d}.json").write_text(json.dumps(row) + "\n")
            report["trials"].append(
                {k: v for k, v in row.items() if k not in ("actions", "states")}
            )
            report["timing"] = timing_summary(report["trials"])
            write()
        stage("completed")
    except Exception:
        report["error"] = traceback.format_exc()
        stage("failed")
        raise
    finally:
        if evaluator is not None:
            evaluator.env.close()
        if og is not None and getattr(og, "sim", None) is not None:
            og.shutdown()


if __name__ == "__main__":
    main()
