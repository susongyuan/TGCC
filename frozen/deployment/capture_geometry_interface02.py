"""Independent zero-policy instance301 native interface capture.

Uses the existing local OmniGibson evaluator/reset/renderer and frozen owned
deployment client only. No policy inference, action step, recording, TGR import
or geometry/catalog repair. Initialization reset/settling is kept explicit.
"""
import argparse
import hashlib
import inspect
import json
import os
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import numpy as np
from PIL import Image
import eval_client as client


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    note = {"status": "prepared", "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy_actions_executed": 0, "policy_inference_calls": 0,
            "video": False, "task_score": None}

    def write(name, value):
        (output / name).write_text(json.dumps(plain(value), indent=2) + "\n")

    def status(stage):
        note.update(status=stage, elapsed_wall_seconds=time.monotonic() - start)
        write("progress.json", note)
        print(json.dumps({"stage": stage, "elapsed_seconds": note["elapsed_wall_seconds"]}), flush=True)

    status("importing_original_simulator")
    evaluator = None
    og = None
    try:
        import torch
        from omegaconf import OmegaConf
        import omnigibson as og
        from omnigibson.macros import gm
        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False
        client.install_wrapper()
        from omnigibson.eval.evaluator import Evaluator, DEFAULT_ROBOT_CONFIG_PATH, resolve_instance_ids
        from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED

        seed_everything(DEFAULT_EVAL_SEED)
        status("launching_original_simulator")
        og.launch()
        import carb
        settings = carb.settings.get_settings()
        budget_key = "/rtx-transient/resourcemanager/texturestreaming/memoryBudget"
        streaming_key = "/rtx-transient/resourcemanager/texturestreaming/enabled"
        before = {budget_key: settings.get(budget_key), streaming_key: settings.get(streaming_key)}
        settings.set_bool(streaming_key, True)
        settings.set_float(budget_key, 0.25)
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        viewport_note = {"present": viewport is not None}
        if viewport is not None:
            viewport_note["previous_updates_enabled"] = bool(viewport.updates_enabled)
            viewport.updates_enabled = False
            viewport_note["updates_enabled"] = bool(viewport.updates_enabled)
        render = {"before": before, "actual": {budget_key: settings.get(budget_key),
                  streaming_key: settings.get(streaming_key)}, "viewport": viewport_note,
                  "scope": "original baseline renderer profile; texture residency may differ from uncapped images"}

        robot_config = OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH)
        robot_config.sensor_config.VisionSensor.sensor_kwargs.image_height = 720
        robot_config.sensor_config.VisionSensor.sensor_kwargs.image_width = 720
        cfg = OmegaConf.create({
            "env_wrapper": {"_target_": "eval_client.RGBResolutionWrapper",
                            "image_size": 224, "head_image_size": 720, "wrist_image_size": 480},
            "policy_name": "tc-esgr-snapshot",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True, "partial_scene_load": True, "max_steps": None,
            "write_video": False, "video_view": "robot", "mode": "public_test",
            "seed": DEFAULT_EVAL_SEED, "task": {"name": "cook_hot_dogs"}, "robot": robot_config})
        status("creating_native_evaluator")
        evaluator = Evaluator(cfg)
        instance = resolve_instance_ids("cook_hot_dogs", [0], mode="public_test")[0]
        assert instance == 301, instance
        # Exact successful original probe order; this is a new standalone
        # environment and never resets another active task.
        status("initial_reset_before_instance_load")
        evaluator.reset()
        status("loading_instance301")
        evaluator.load_task_instance(instance)
        status("final_native_reset")
        evaluator.reset()
        assert evaluator._video_path is None
        status("reading_zero_policy_native_interfaces")
        robot = evaluator.robot
        native_time_before = float(og.sim.current_time)
        native_step_before = int(og.sim.current_time_step_index)
        feedback = robot._get_proprioception_dict()
        payload = client.observation(robot, evaluator.obs, evaluator.robot_camera_names,
                                     client.TASKS["cook_hot_dogs"])
        # These files preserve native dtypes/shapes; no raw256 layout is guessed.
        native_named = {key: client.array(value) for key, value in feedback.items()}
        np.savez(output / "native_proprio_named.npz", **native_named)
        actor_state = {key: value for key, value in payload["state"].items()}
        np.savez(output / "actor_named_state.npz", **actor_state)
        vectors = {key: client.array(value) for key, value in evaluator.obs.items()
                   if "proprio" in key or "cam_rel_poses" in key}
        np.savez(output / "native_evaluator_vectors.npz", **vectors)
        fields_order = list(robot._proprio_obs)
        configured_concat = np.concatenate([native_named[key].reshape(-1) for key in fields_order])
        np.save(output / "native_configured_proprio_concat.npy", configured_concat)
        control_idx = client.array(robot.base_control_idx).astype(int)
        base_idx = client.array(robot.base_idx).astype(int)
        qpos = client.array(robot.get_joint_positions())
        qvel = client.array(robot.get_joint_velocities())
        yaw = qpos[base_idx][5]
        raw_base = qvel[control_idx]
        once = np.stack([np.cos(yaw)*raw_base[0]+np.sin(yaw)*raw_base[1],
                         -np.sin(yaw)*raw_base[0]+np.cos(yaw)*raw_base[1],raw_base[2]])
        width = {side: native_named["gripper_"+side+"_qpos"].sum(keepdims=True)
                 for side in ("left","right")}
        write("native_policy_input_provenance.json", {
            "source": "same post-reset native _get_proprioception_dict and evaluator.obs",
            "native_named": {key: {"dtype":str(v.dtype),"shape":list(v.shape)}
                             for key,v in native_named.items()},
            "actor_state": {key: {"dtype":str(v.dtype),"shape":list(v.shape)}
                            for key,v in actor_state.items()},
            "native_evaluator_vectors": {key: {"dtype":str(v.dtype),"shape":list(v.shape)}
                                        for key,v in vectors.items()},
            "configured_proprio_field_order": fields_order,
            "configured_proprio_widths": {key:int(native_named[key].size) for key in fields_order},
            "configured_proprio_concat_shape":list(configured_concat.shape),
            "configured_proprio_concat_dtype":str(configured_concat.dtype),
            "base_velocity": {
                "source_native_method":"robot._get_base_qvel_for_proprioception",
                "native_base_control_indices":control_idx.tolist(),
                "native_base_6dof_indices":base_idx.tolist(),
                "native_joint_base_qpos6":qpos[base_idx].tolist(),
                "native_joint_base_qvel3":raw_base.tolist(),
                "native_yaw_joint":float(yaw),
                "one_rotation_formula":"[cos(yaw)*vx+sin(yaw)*vy,-sin(yaw)*vx+cos(yaw)*vy,vyaw]",
                "one_rotation_numeric":once.tolist(),
                "reported_named_base_qvel":native_named["base_qvel"].tolist(),
                "max_difference":float(np.max(np.abs(once-native_named["base_qvel"])))},
            "policy_state_gripper1":{side:{"width_sum":value.tolist(),"dtype":str(value.dtype),
                "shape":list(value.shape),"source":"original B1k input sums two native finger qpos",
                "not_an_action_command":True} for side,value in width.items()},
            "geometry_finger_qpos2":{side:native_named["gripper_"+side+"_qpos"].tolist()
                                     for side in ("left","right")},
            "policy_inference_calls":0,"policy_actions_executed":0,
            "casts_for_comparison_performed":False,
            "raw256_column_inference":False})
        errors = []
        warnings = []
        assets_path = Path(__file__).parent / "geometry_assets.json"
        assets = client.NominalGeometryAssets(assets_path)
        object_records = {}
        scope = evaluator.env.task.object_scope
        required = ("hotdog.n.02_1", "hotdog.n.02_2",
                    "electric_refrigerator.n.01_1", "microwave.n.02_1")

        def object_record(name, obj, cabinet=False):
            record = {"instance_name": name, "present": obj is not None,
                      "source": "live_loaded_scene_native_API", "task_timestamp": 0.0,
                      "native_sim_time": native_time_before, "cabinet": cabinet}
            if obj is None:
                errors.append(name + ":absent")
                return record
            try:
                pos, quat = obj.get_position_orientation()
                root = client.transform(client.array(pos), client.array(quat))
                category = getattr(obj, "category", None)
                model = getattr(obj, "model", None)
                scale = client.array(obj.scale)
                record.update(category=category, model=model, scale=scale.tolist(),
                              fixed_base=getattr(obj, "fixed_base", None),
                              visual_only=bool(getattr(obj, "visual_only", False)),
                              root_world=root.tolist())
                spec = assets.object_spec(obj)
                record["model_in_catalog"] = spec is not None
                if spec is None:
                    errors.append(name + ":actual_model_not_in_catalog")
                    return record
                center, attrs = assets.centered(root, spec)
                record.update(nominal_spec=spec, nominal_center_world=center.tolist(),
                              nominal_center_formula="p_W_root + R_W_root @ (scale * native_center_offset)",
                              canonical_pose_kind="nominal_bbox_center")
                native_bbox = client.array(obj.native_bbox)
                native_offset = client.array(obj.scaled_bbox_center_in_base_frame)
                record["native_bbox_API"] = native_bbox.tolist()
                record["native_scaled_center_offset_API"] = native_offset.tolist()
                record["native_scaled_offset_semantics"] = "inverse bbox-center-to-root placement offset"
                record["native_root_to_bbox_scaled_offset"] = (-native_offset).tolist()
                record["native_base_link_offset_API"] = client.array(obj.base_link_offset).tolist()
                row = assets.records[(category, model)]
                eerr = float(np.max(np.abs(native_bbox - np.asarray(row["native_bbox_extent_m"]))))
                cerr = float(np.max(np.abs(-native_offset - np.asarray(spec["center_offset_m"]))))
                record["catalog_native_extent_max_difference_m"] = eerr
                record["catalog_scaled_center_offset_max_difference_m"] = cerr
                if eerr > 1e-5 or cerr > 1e-5:
                    errors.append(name + ":native_nominal_geometry_differs_from_catalog")
                lo, hi = (client.array(v) for v in obj.aabb)
                record["measured_world_AABB"] = [lo.tolist(), hi.tolist()]
                record["AABB_center_world"] = ((lo + hi) / 2).tolist()
                record["nominal_minus_AABB_center_m"] = (center[:3, 3] - (lo + hi) / 2).tolist()
                metadata = client.ROOT / "shared/omnigibson/behavior-1k-assets/objects" / category / model / "misc/metadata.json"
                if metadata.is_file():
                    record["actual_local_metadata_sha256"] = sha(metadata)
                    record["catalog_metadata_sha256"] = row["metadata_source_sha256"]
                    record["metadata_bytes_match_catalog"] = sha(metadata) == row["metadata_source_sha256"]
                    if not record["metadata_bytes_match_catalog"]:
                        warnings.append(name + ":metadata_bytes_differ; numeric_native_geometry_checked_separately")
                if cabinet and (record["fixed_base"] is not True or record["visual_only"]):
                    errors.append(name + ":cabinet_not_fixed_physical_object")
            except Exception as exc:
                record["API_error"] = type(exc).__name__ + ":" + str(exc)
                errors.append(name + ":native_object_API_error")
            return record

        for name in required:
            object_records[name] = object_record(name, scope.get(name))
        scene = getattr(evaluator.env, "scene", None)
        if scene is None:
            scenes = getattr(evaluator.env, "scenes", [])
            scene = scenes[0] if len(scenes) == 1 else None
        for name in assets.static_names:
            try:
                obj = scene.object_registry("name", name)
                object_records[name] = object_record(name, obj, cabinet=True)
            except Exception as exc:
                object_records[name] = {"present": False, "API_error": type(exc).__name__ + ":" + str(exc)}
                errors.append(name + ":scene_registry_API_error")

        fingers = {}
        for side in ("left", "right"):
            try:
                key = "gripper_" + side + "_qpos"
                names = list(robot.finger_joint_names[side])
                expected = [side + "_gripper_finger_joint1", side + "_gripper_finger_joint2"]
                indices = client.array(robot.gripper_control_idx[side]).astype(int)
                values = client.vector(feedback[key], 2, key)
                all_positions = client.array(robot.get_joint_positions())
                named = [float(all_positions[list(robot.joints.keys()).index(n)]) for n in names]
                fingers[side] = {"proprio_key": key, "native_names": names, "expected_names": expected,
                    "native_control_indices": indices.tolist(), "qpos": values.tolist(),
                    "named_joint_qpos": named, "native_names_match": names == expected,
                    "named_values_match": bool(np.allclose(values, named, atol=1e-7, rtol=0)),
                    "coordinate_units": "metres (original prismatic finger coordinates)",
                    "source": "same_zero_policy_native_robot_feedback"}
                if names != expected or not fingers[side]["named_values_match"]:
                    errors.append(side + ":finger_order_or_values_incompatible")
            except Exception as exc:
                fingers[side] = {"API_error": type(exc).__name__ + ":" + str(exc)}
                errors.append(side + ":finger_proprio_API_error")

        # Preserve all inspected evidence before any strict adapter failure.
        for key, image in payload["images"].items():
            Image.fromarray(image.transpose(1, 2, 0)).save(output / (key + ".png"))
        write("interface_probe.json", {"objects": object_records, "fingers": fingers,
              "errors": errors, "warnings": warnings, "interface_status": "READ_IN_PROGRESS",
              "physical_verified": False, "scope": "native interfaces only; no actions"})
        note["errors"] = errors
        robot_fields = client.live_robot_fields(robot, feedback, 0.0)
        extractor = client.OmniGibsonExtractor(
            evaluator.env, arm_mapping={"left": "left", "right": "right"}, geometry_assets=assets)
        snapshot = None
        try:
            snapshot = extractor(payload, client.EntityRegistry(),
                                 "source_snapshot_instance301_nominal_interface02", 0.0)
            robot_fields["geometry_spec"] = extractor.geometry_spec
            write("snapshot.json", asdict(snapshot))
        except Exception as exc:
            errors.append("current_deployment_extractor:" + type(exc).__name__ + ":" + str(exc))

        for key, image in payload["images"].items():
            Image.fromarray(image.transpose(1, 2, 0)).save(output / (key + ".png"))
        expected_shapes = {"head_rgb": [3, 720, 720], "left_wrist_rgb": [3, 480, 480],
                           "right_wrist_rgb": [3, 480, 480]}
        actual_shapes = {k: list(v.shape) for k, v in payload["images"].items()}
        if actual_shapes != expected_shapes:
            errors.append("native_camera_dimensions_incompatible")
        native_time_after = float(og.sim.current_time)
        native_step_after = int(og.sim.current_time_step_index)
        if native_time_after != native_time_before or native_step_after != native_step_before:
            errors.append("simulation_advanced_during_readonly_capture")
        for key, value, expected in (
            ("physics_hz", 1 / og.sim.get_physics_dt(), 120.0),
            ("action_hz", 1 / og.sim.get_sim_step_dt(), 30.0),
            ("render_hz", 1 / og.sim.get_rendering_dt(), 30.0),
        ):
            if not np.isclose(value, expected, atol=1e-6, rtol=0):
                errors.append(key + ":native_frequency_incompatible")
        write("robot_fields.json", robot_fields)
        write("interface_probe.json", {"objects": object_records, "fingers": fingers,
              "errors": errors, "warnings": warnings, "interface_status": "PASS" if not errors else "FAILED",
              "nominal_center_not_AABB_center": True, "physical_verified": False,
              "scope": "native field/model/scale/center/registry interface validation only; no actions or task score"})

        native_root = Path(og.__file__).resolve().parent
        source_files = [native_root / x for x in ("objects/dataset_object.py", "objects/usd_object.py",
                        "robots/robot.py", "simulator.py", "eval/evaluator.py")]
        provenance = {
            "source": "actual_omnigibson_initial_nominal_interface02",
            "reference_configuration": "exact old successful probe config; same existing local environment and prior probe appdata cache",
            "reference_cache_causality": "cache reuse is a tested initialization variation, not an established OOM cause",
            "reference_manifest": json.loads((Path(__file__).parent / "reference_manifest.json").read_text()),
            "instance_id": instance, "task": "cook_hot_dogs", "mode": "public_test",
            "environment_seed": int(DEFAULT_EVAL_SEED), "policy_actions_executed": 0,
            "policy_inference_calls": 0, "video": False, "task_score": None, "not_a_task_score": True,
            "sim_time": 0.0, "timestamp_source": "native evaluator task step0 after final reset",
            "native_sim_clock_before": native_time_before, "native_sim_clock_after": native_time_after,
            "native_sim_step_before": native_step_before, "native_sim_step_after": native_step_after,
            "capture_queries_advanced_simulation": native_time_after != native_time_before or native_step_after != native_step_before,
            "camera_observation_source": "evaluator.obs from the same final native reset; no extra render",
            "per_camera_hardware_timestamps_available": False,
            "reset_sequence": ["native_evaluator.reset", "load_instance301", "native_evaluator.reset"],
            "initialization_physics_settling_is_not_policy_action": True,
            "base_world": robot_fields["base_world"], "joints": robot_fields["joints"],
            "camera_sizes": actual_shapes, "camera_names": evaluator.robot_camera_names,
            "physics_hz": 1 / og.sim.get_physics_dt(), "action_hz": 1 / og.sim.get_sim_step_dt(),
            "render_hz": 1 / og.sim.get_rendering_dt(), "renderer_memory_profile": render,
            "native_original_source_sha256": {str(p): sha(p) for p in source_files if p.is_file()},
            "staged_source_sha256": {p.name: sha(p) for p in (Path(__file__), Path(client.__file__), assets_path)},
            "local_environment": str(Path(os.sys.executable).resolve()),
            "asset_root": str(client.ROOT / "shared/omnigibson"),
            "interface_status": "PASS" if not errors else "FAILED", "errors": errors, "warnings": warnings,
            "snapshot_written": snapshot is not None, "physical_verified": False,
            "wall_seconds": time.monotonic() - start}
        write("provenance.json", provenance)
        print(json.dumps({"interface_status": provenance["interface_status"], "errors": errors,
                          "warnings": warnings, "snapshot_written": snapshot is not None}), flush=True)
        if errors:
            note["errors"] = errors
            status("FAILED_INTERFACE")
        else:
            status("COMPLETE_ZERO_POLICY_CAPTURE")
    except Exception as exc:
        note["fatal_error"] = type(exc).__name__ + ":" + str(exc)
        note["traceback"] = traceback.format_exc()
        status("FAILED_INITIALIZATION_OR_CAPTURE")
        raise
    finally:
        cleanup = []
        if evaluator is not None:
            try:
                evaluator.env.close()
            except Exception as exc:
                cleanup.append("env.close:" + type(exc).__name__ + ":" + str(exc))
        if og is not None:
            try:
                og.shutdown()
            except Exception as exc:
                cleanup.append("og.shutdown:" + type(exc).__name__ + ":" + str(exc))
        note["cleanup_errors"] = cleanup
        note["cleanup_completed_utc"] = datetime.now(timezone.utc).isoformat()
        write("progress.json", note)


if __name__ == "__main__":
    main()
