"""Native R1Pro control contract and checkpoint-owned action normalization."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .specs import RobotActionSpec


def action_spec(root):
    root = Path(root)
    layout = json.loads((root / "reports/data_action_layout.json").read_text())
    joints = {
        j.attrib["name"]: j
        for j in ET.parse(root / "data/robot/r1pro.urdf").getroot().findall("joint")
    }
    names = ["base_x", "base_y", "base_yaw"] + [f"torso_joint{i}" for i in range(1, 5)]
    names += [f"left_arm_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    names += [f"right_arm_joint{i}" for i in range(1, 8)] + ["right_gripper"]
    # Native base commands are controller INPUTS, scaled to .75/.75/1 m/s internally.
    base = layout["controller_config"]["base"]["command_input_limits"]
    lower, upper, known, units = [], [], [], []
    for i, name in enumerate(names):
        if i < 3:
            lower.append(base[0][i])
            upper.append(base[1][i])
            known.append(True)
            units.append("controller_input; output_scale=" + str([0.75, 0.75, 1.0][i]))
        elif name in joints:
            limits = joints[name].find("limit")
            lower.append(float(limits.attrib["lower"]))
            upper.append(float(limits.attrib["upper"]))
            known.append(True)
            units.append("rad; absolute_position")
        else:
            lower.append(-1.0)
            upper.append(1.0)
            known.append(True)
            units.append("smooth_gripper_controller_input")
    groups = {
        "base": tuple(range(3)),
        "left": tuple(range(7, 14)),
        "right": tuple(range(15, 22)),
        "coupled": (3, 4, 5, 6, 14, 22),
    }
    return RobotActionSpec(
        tuple(names), groups, tuple(units), tuple(lower), tuple(upper), tuple(known)
    )


class ActionNormalizer:
    """Exact pi05_b1k-base action transform after verifying no delta/reorder.

    This fast path is checked against every real item's official transformed
    actions. Any future config introducing delta/reorder fails construction.
    """

    def __init__(self, checkpoint):
        from tgr.factory import ensure_legacy_imports

        ensure_legacy_imports()
        from openpi.training.config import get_config

        cfg = get_config("pi05_b1k-base").data
        if (
            cfg.delta_action_mask
            or cfg.model_delta_action_mask
            or cfg.rearrange_action_indices
            or cfg.subsample_action_stride != 1
        ):
            raise ValueError(
                "Fast native normalization unsupported for changed transforms"
            )
        paths = list(Path(checkpoint).glob("assets/**/norm_stats.json"))
        if len(paths) != 1:
            raise ValueError("Ambiguous checkpoint action statistics")
        stats = json.loads(paths[0].read_text())["norm_stats"]
        self.stats = stats

    def transform(self, x, kind="actions"):
        s = self.stats[kind]
        if isinstance(x, torch.Tensor):
            lo = torch.as_tensor(s["q01"], device=x.device, dtype=x.dtype)[
                ..., : x.shape[-1]
            ]
            hi = torch.as_tensor(s["q99"], device=x.device, dtype=x.dtype)[
                ..., : x.shape[-1]
            ]
        else:
            lo = np.asarray(s["q01"])[..., : x.shape[-1]]
            hi = np.asarray(s["q99"])[..., : x.shape[-1]]
        return (x - lo) / (hi - lo + 1e-6) * 2.0 - 1.0

    def inverse(self, x, kind="actions"):
        s = self.stats[kind]
        lo = torch.as_tensor(s["q01"], device=x.device, dtype=x.dtype)[
            ..., : x.shape[-1]
        ]
        hi = torch.as_tensor(s["q99"], device=x.device, dtype=x.dtype)[
            ..., : x.shape[-1]
        ]
        return (x + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
