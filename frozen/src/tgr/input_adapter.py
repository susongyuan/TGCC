"""Official PT50 tensor inputs from VERIFIED named fields, never raw256 slices.

This module does not decode video, raw proprio layouts, graphs or task predicates.
"""

from pathlib import Path
from collections.abc import Mapping
import hashlib
import numpy as np
import torch
from .factory import ensure_legacy_imports
from .checkpoint import digest

FIELD_WIDTHS = {
    "base_qvel": 3,
    "trunk_qpos": 4,
    "arm_left_qpos": 7,
    "arm_right_qpos": 7,
    "gripper_left_qpos": 2,
    "gripper_right_qpos": 2,
}
CAMERAS = {
    "head_rgb": "observation/egocentric_camera",
    "left_wrist_rgb": "observation/wrist_image_left",
    "right_wrist_rgb": "observation/wrist_image_right",
}


def official_input_transform(checkpoint):
    """Reconstruct EXACT default create_trained_policy input pipeline, NO model load."""
    ensure_legacy_imports()
    from openpi.training import config, checkpoints
    from openpi import transforms

    cfg = config.get_config("pi05_b1k-base")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    if not data.use_quantile_norm:
        raise ValueError("PT50 checkpoint quantile pipeline required")
    stats = checkpoints.load_norm_stats(Path(checkpoint) / "assets", data.asset_id)
    return transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )


def legacy_state_from_named(named_proprio):
    """Only the six fields actually read by B1kInputs, in audited legacy positions.

    Caller must decode named fields from the actual episode's full layout and
    provide base_qvel in baseline-native control frame. No FK/frame guessing here.
    Unused slots stay NaN and cannot accidentally stand in for observed EE data.
    """
    if not isinstance(named_proprio, Mapping):
        raise ValueError("requires VERIFIED named fields; raw256 vectors forbidden")
    from .serve import INDICES

    result = np.full(61, np.nan, dtype=np.float32)
    for key, width in FIELD_WIDTHS.items():
        if key not in named_proprio:
            raise ValueError("missing verified field: " + key)
        value = np.asarray(named_proprio[key], dtype=np.float32)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError("invalid named field: " + key)
        result[INDICES[key]] = value
    return result


class OfficialInputAdapter:
    def __init__(
        self,
        checkpoint,
        *,
        proprio_layout_sha256,
        action_layout_sha256,
        policy=None,
        input_transform=None
    ):
        for proof in (proprio_layout_sha256, action_layout_sha256):
            if (
                not isinstance(proof, str)
                or len(proof) != 64
                or any(c not in "0123456789abcdef" for c in proof)
            ):
                raise ValueError(
                    "audited proprio/action layout artifact SHA256 required"
                )
        self.checkpoint = Path(checkpoint)
        norms = list(self.checkpoint.glob("assets/**/norm_stats.json"))
        if len(norms) != 1:
            raise ValueError("ambiguous checkpoint normalizer")
        self.normalization_sha256 = digest(norms[0])
        if policy is not None and input_transform is not None:
            raise ValueError("choose policy OR input_transform")
        self.transform = (
            policy._input_transform
            if policy is not None
            else (
                input_transform
                if input_transform is not None
                else official_input_transform(checkpoint)
            )
        )
        self.proof = {
            "adapter_revision": digest(Path(__file__)),
            "proprio_layout_sha256": proprio_layout_sha256,
            "action_layout_sha256": action_layout_sha256,
            "normalization": "checkpoint_owned_quantile",
            "raw256_decoding_performed": False,
            "action_order": "native23 unchanged; official transforms own normalization/delta/reorder",
            "camera_resize": "official openpi_client resize_with_pad; uint8 RGB; no image normalization invented",
        }

    def __call__(self, *, named_proprio, rgb, actions, task):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("explicit task instruction required")
        action = np.asarray(actions, dtype=np.float32)
        if action.shape != (32, 23) or not np.isfinite(action).all():
            raise ValueError("expected finite native23 32-step teacher actions")
        data = {
            "observation/state": legacy_state_from_named(named_proprio),
            "prompt": task,
            "actions": action.copy(),
        }
        from openpi_client.image_tools import resize_with_pad

        for key, target in CAMERAS.items():
            if key not in rgb:
                raise ValueError("missing current RGB: " + key)
            image = np.asarray(rgb[key])
            if image.dtype != np.uint8 or image.ndim != 3:
                raise ValueError("RGB must be uint8 CHW or HWC")
            if image.shape[-1] != 3:
                if image.shape[0] != 3:
                    raise ValueError("RGB channel count")
                image = image.transpose(1, 2, 0)
            data[target] = np.ascontiguousarray(resize_with_pad(image, 224, 224))
        transformed = self.transform(data)
        normalized = np.asarray(transformed.pop("actions"), dtype=np.float32)
        if (
            normalized.shape != (32, 32)
            or not np.isfinite(normalized).all()
            or np.count_nonzero(normalized[:, 23:])
        ):
            raise ValueError("official normalized action/padding contract mismatch")
        state = np.asarray(transformed["state"])
        if (
            state.shape != (32,)
            or not np.isfinite(state).all()
            or np.count_nonzero(state[23:])
        ):
            raise ValueError("official state32/padding contract mismatch")

        def tensor(x):
            if isinstance(x, dict):
                return {key: tensor(value) for key, value in x.items()}
            value = np.array(x, copy=True)
            if value.dtype.kind in "OUS":
                raise ValueError("unconsumed prompt/object in official model inputs")
            return torch.from_numpy(value)[None, ...]

        observation = tensor(transformed)
        return {
            "observation": observation,
            "actions": torch.from_numpy(normalized.copy())[None, ...],
            "normalization_sha256": self.normalization_sha256,
            "provenance": dict(self.proof),
        }


def make_raw_transform(adapter, decode_named):
    """Bridge LazyDemonstration.transform(raw). decode_named(raw_state256) is OWNED
    by the data-layout auditor and must use episode metadata; never inferred here.
    """

    def transform(raw):
        return adapter(
            named_proprio=decode_named(raw["state"]),
            rgb=raw["images"],
            actions=raw["actions"],
            task=raw["task"],
        )

    return transform


def prepare_training_inputs(
    *,
    named_proprio,
    images,
    actions,
    task,
    checkpoint,
    proprio_layout_sha256=None,
    action_layout_sha256=None,
    adapter=None
):
    """Shared explicit data/runtime bridge; caller owns audited layout proofs.

    Accepts data camera names or runtime *_rgb names. Named body-frame velocity
    is passed through once, without any extra frame transform. No raw256 decode.
    Pass an existing OfficialInputAdapter to reuse the checkpoint input pipeline.
    """
    if adapter is None:
        adapter = OfficialInputAdapter(
            checkpoint,
            proprio_layout_sha256=proprio_layout_sha256,
            action_layout_sha256=action_layout_sha256,
        )
    else:
        if not isinstance(adapter, OfficialInputAdapter):
            raise ValueError("OfficialInputAdapter required")
        if Path(checkpoint).resolve() != adapter.checkpoint.resolve():
            raise ValueError("adapter checkpoint mismatch")
        for key, value in (
            ("proprio_layout_sha256", proprio_layout_sha256),
            ("action_layout_sha256", action_layout_sha256),
        ):
            if value is not None and adapter.proof[key] != value:
                raise ValueError("adapter layout proof mismatch: " + key)
    if not isinstance(images, Mapping):
        raise ValueError("current RGB mapping required")
    aliases = {
        "head": "head_rgb",
        "left_wrist": "left_wrist_rgb",
        "right_wrist": "right_wrist_rgb",
    }
    rgb = {}
    for key, value in images.items():
        canonical = aliases.get(key, key)
        if canonical not in CAMERAS:
            raise ValueError("unknown RGB camera: " + key)
        if canonical in rgb:
            raise ValueError("duplicate RGB camera: " + canonical)
        rgb[canonical] = value
    return adapter(named_proprio=named_proprio, rgb=rgb, actions=actions, task=task)
