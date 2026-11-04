"""Source-owned snapshot-only BEHAVIOR probe, no policy actions or video."""

import os, json, time
from pathlib import Path
import numpy as np
from PIL import Image
import eval_client as client


def main():
    root = client.ROOT
    output = Path(os.environ["TC_ESGR_VIZ_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from omegaconf import OmegaConf
    import omnigibson as og
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    client.install_wrapper()
    from omnigibson.eval.evaluator import (
        Evaluator,
        DEFAULT_ROBOT_CONFIG_PATH,
        resolve_instance_ids,
    )
    from omnigibson.eval.utils.eval_utils import seed_everything, DEFAULT_EVAL_SEED

    seed_everything(DEFAULT_EVAL_SEED)
    og.launch()
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx-transient/resourcemanager/texturestreaming/enabled", True)
    settings.set_float(
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.25
    )
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.updates_enabled = False
    robot = OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH)
    robot.sensor_config.VisionSensor.sensor_kwargs.image_height = 720
    robot.sensor_config.VisionSensor.sensor_kwargs.image_width = 720
    cfg = OmegaConf.create(
        {
            "env_wrapper": {
                "_target_": "eval_client.RGBResolutionWrapper",
                "image_size": 224,
                "head_image_size": 720,
                "wrist_image_size": 480,
            },
            "policy_name": "tc-esgr-snapshot",
            "model": {
                "_target_": "omnigibson.eval.policies.LocalPolicy",
                "action_dim": None,
            },
            "headless": True,
            "partial_scene_load": True,
            "max_steps": None,
            "write_video": False,
            "video_view": "robot",
            "mode": "public_test",
            "seed": DEFAULT_EVAL_SEED,
            "task": {"name": "cook_hot_dogs"},
            "robot": robot,
        }
    )
    evaluator = Evaluator(cfg)
    try:
        instance = resolve_instance_ids("cook_hot_dogs", [0], mode="public_test")[0]
        evaluator.reset()
        evaluator.load_task_instance(instance)
        evaluator.reset()
        payload = client.observation(
            evaluator.robot,
            evaluator.obs,
            evaluator.robot_camera_names,
            client.TASKS["cook_hot_dogs"],
        )
        registry = client.EntityRegistry()
        extractor = client.OmniGibsonExtractor(
            evaluator.env, arm_mapping={"left": "left", "right": "right"}
        )
        snapshot = extractor(payload, registry, "source_snapshot_instance301", 0.0)
        from dataclasses import asdict

        for key, image in payload["images"].items():
            Image.fromarray(image.transpose(1, 2, 0)).save(output / (key + ".png"))
        (output / "snapshot.json").write_text(json.dumps(asdict(snapshot), indent=2))
        fields = evaluator.robot._get_proprioception_dict()
        bp, bq = evaluator.robot.get_position_orientation()
        info = {
            "source": "actual_omnigibson_initial_snapshot",
            "instance_id": instance,
            "task": "cook_hot_dogs",
            "policy_actions_executed": 0,
            "video": False,
            "not_a_task_score": True,
            "sim_time": 0.0,
            "base_world": client.transform(client.array(bp), client.array(bq)).tolist(),
            "joints": {
                "trunk_qpos": client.array(fields["trunk_qpos"]).tolist(),
                "left_arm_qpos": client.array(fields["arm_left_qpos"]).tolist(),
                "right_arm_qpos": client.array(fields["arm_right_qpos"]).tolist(),
            },
            "camera_sizes": {k: list(v.shape) for k, v in payload["images"].items()},
            "physics_hz": 1 / og.sim.get_physics_dt(),
            "action_hz": 1 / og.sim.get_sim_step_dt(),
            "render_hz": 1 / og.sim.get_rendering_dt(),
        }
        (output / "provenance.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2), flush=True)
    finally:
        evaluator.env.close()
        og.shutdown()


if __name__ == "__main__":
    main()
