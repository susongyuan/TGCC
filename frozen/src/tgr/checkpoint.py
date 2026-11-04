"""TC-ESGR full-model inference artifacts; pristine base checkpoint stays read-only."""

import hashlib, json, os
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def save_inference(state, directory, metadata, aliases=None, original_dtypes=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    aliases = aliases or {}
    output = {}
    for name, value in state.items():
        if name in aliases:
            continue
        dtype = (
            original_dtypes.get(name, torch.float32) if original_dtypes else value.dtype
        )
        output[name] = value.detach().cpu().to(dtype).contiguous().clone()
    temp = directory / "model.tmp.safetensors"
    save_file(output, str(temp))
    os.replace(temp, directory / "model.safetensors")
    metadata = dict(
        metadata,
        format="tc_esgr_full_v2",
        aliases=aliases,
        model_sha256=digest(directory / "model.safetensors"),
    )
    temp = directory / "metadata.tmp"
    temp.write_text(json.dumps(metadata, indent=2))
    temp.replace(directory / "metadata.json")
    return metadata


def load_full(model, directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (
        metadata.get("format") != "tc_esgr_full_v2"
        or metadata.get("architecture_version") != "tc_esgr_v2"
    ):
        raise ValueError("V2 full-model checkpoint required; no legacy/LoRA fallback")
    if digest(directory / "model.safetensors") != metadata["model_sha256"]:
        raise ValueError("checkpoint changed")
    state = load_file(str(directory / "model.safetensors"), device="cpu")
    for alias, canonical in metadata.get("aliases", {}).items():
        state[alias] = state[canonical]
    model.load_state_dict(state, strict=True)
    model.trained = bool(
        (metadata.get("training_complete") or (metadata.get("training_stopped_by_user") is True
            and metadata.get("user_stop_evaluation", {}).get("selected_optimizer_step") == metadata.get("optimizer_steps") == 5000))
        and metadata.get("validated_selection")
    )
    model.provenance = metadata
    return metadata
