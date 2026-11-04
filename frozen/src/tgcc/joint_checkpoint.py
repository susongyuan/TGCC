"""Atomic FSDP continuation and independently verifiable full-policy exports."""

from pathlib import Path
import json, os, time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
from .checkpoint import digest, rng_state, restore_rng


def migrate_legacy(path, model, optimizer, rank, expected_parent, expected_sha):
    """Explicit recipe migration, not bypassing exact-source resume validation."""
    path = Path(path)
    md = json.loads((path / "metadata.json").read_text())
    if (
        md["new_updates"] != 100
        or md["parent_model_sha256"] != expected_parent
        or md["checkpoint_sha256"] != expected_sha
    ):
        raise ValueError("Only the audited complete step100 may be migrated")
    if digest(path / "tgcc.safetensors") != expected_sha:
        raise ValueError("Legacy checkpoint corrupted")
    old = torch.load(path / "resume.pt", weights_only=False, map_location="cpu")
    if old["new_updates"] != 100 or old["data_cursor"] != 4800:
        raise ValueError("Legacy optimizer/cursor counter mismatch")
    # Weights were strictly loaded BEFORE sharding. Restore the same named Adam
    # moments onto each FSDP parameter's documented distribution.
    from torch.distributed.tensor import DTensor, distribute_tensor

    for old_group, module, new_group in zip(
        old["optimizer"]["param_groups"],
        (model.tgcc.critic, model.tgcc.adapters),
        optimizer.param_groups[:2],
    ):
        parameters = list(module.parameters())
        if len(parameters) != len(old_group["params"]):
            raise ValueError("Legacy parameter ordering changed")
        for identifier, p in zip(old_group["params"], parameters):
            if identifier not in old["optimizer"]["state"]:
                continue
            restored = {}
            for k, value in old["optimizer"]["state"][identifier].items():
                if isinstance(value, torch.Tensor) and k != "step":
                    value = value.to(device=p.device, dtype=p.dtype)
                    if isinstance(p, DTensor):
                        value = distribute_tensor(value, p.device_mesh, p.placements)
                restored[k] = value
            optimizer.state[p] = restored
        for k in ("betas", "eps", "weight_decay", "amsgrad", "maximize"):
            new_group[k] = old_group[k]
    restore_rng(
        torch.load(path / f"rng_rank{rank}.pt", weights_only=False, map_location="cpu")
    )
    return (
        100,
        4800,
        dict(
            legacy_checkpoint=str(path),
            legacy_sha256=expected_sha,
            legacy_source_sha256=md["source_sha256"],
            policy="explicit_user_authorized_objective_and_recipe_migration",
            retained="all critic/adapter weights, named Adam moments, update counter, cursor and rank RNG",
            changed="stage budget, task negatives, no OOD loss, analytic memory/recovery; microbatch may change",
            bitwise_old_recipe_continuation=False,
        ),
    )


def _rank0_action(action, rank):
    """Publish filesystem errors to every rank before the next collective."""
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError("Collective checkpoint filesystem failure: " + error[0])


def _group_value(value, decode=False):
    # Adam's betas are tuples; JSON must retain that type for exact resumes.
    if decode:
        if isinstance(value, dict) and set(value) == {"__tuple__"}:
            return tuple(_group_value(x, True) for x in value["__tuple__"])
        if isinstance(value, list):
            return [_group_value(x, True) for x in value]
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_group_value(x) for x in value]}
    if isinstance(value, list):
        return [_group_value(x) for x in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("Unsupported optimizer group value: " + type(value).__name__)


def _named_optimizer(model, optimizer):
    """Read Adam state without DCP's implicit zero-gradient optimizer.step()."""
    names = {p: n for n, p in model.named_parameters()}
    states, schema, groups = {}, {}, []
    seen = set()
    for group in optimizer.param_groups:
        parameters = []
        for p in group["params"]:
            if p not in names or p in seen:
                raise ValueError("Optimizer parameters must name each model parameter once")
            seen.add(p)
            name = names[p]
            parameters.append(name)
            actual = optimizer.state.get(p, {})
            if not actual:
                continue
            states[name] = dict(actual)
            schema[name] = {}
            for key, value in actual.items():
                if isinstance(value, torch.Tensor):
                    if key != "step" and tuple(value.shape) != tuple(p.shape):
                        raise ValueError("Adam moment shape differs from its parameter")
                    schema[name][key] = dict(kind="tensor", dtype=str(value.dtype),
                        shape=list(value.shape), parameter_layout=key != "step")
                elif type(value) in (int, float, bool):
                    schema[name][key] = dict(kind=type(value).__name__)
                else:
                    raise TypeError("Unsupported optimizer state: " + key)
        groups.append(dict(params=parameters, attributes={
            k: _group_value(v) for k, v in group.items() if k != "params"
        }))
    return dict(state=states), schema, groups


def _optimizer_template(model, optimizer, metadata):
    """Allocate only moments that genuinely existed in the saved optimizer."""
    names = dict(model.named_parameters())
    groups = metadata["optimizer_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count changed")
    parameter_names = {p: n for n, p in names.items()}
    for saved, current in zip(groups, optimizer.param_groups):
        if saved["params"] != [parameter_names[p] for p in current["params"]]:
            raise ValueError("Optimizer parameter ordering changed")
    state = {}
    for name, fields in metadata["optimizer_state_schema"].items():
        if name not in names:
            raise ValueError("Optimizer state references an unknown parameter")
        p = names[name]
        state[name] = {}
        for key, field in fields.items():
            if field["kind"] == "tensor":
                dtype = getattr(torch, field["dtype"].removeprefix("torch."))
                if field["parameter_layout"]:
                    if field["shape"] != list(p.shape):
                        raise ValueError("Optimizer moment shape changed")
                    # zeros_like preserves FSDP2 DTensor mesh and placements,
                    # including the unsharded ignored scalar adapter gates.
                    value = torch.zeros_like(p, dtype=dtype)
                else:
                    value = torch.zeros(field["shape"], dtype=dtype, device="cpu")
            else:
                constructors = dict(int=int, float=float, bool=bool)
                value = constructors[field["kind"]]()
            state[name][key] = value
    return dict(state=state)


def _payload_identity(path, world):
    """Hash every continuation tensor and every rank's random state.

    Export hashes alone do not protect a resume: its FP32 student and Adam
    states live in DCP, not in the inference safetensors files.
    """
    path = Path(path)
    distributed = path / "distributed"
    if not distributed.is_dir():
        raise ValueError("Missing distributed continuation checkpoint")
    inventory = list(distributed.rglob("*"))
    if distributed.is_symlink() or any(p.is_symlink() for p in inventory):
        raise ValueError("Checkpoint payload symlinks are forbidden")
    files = [p for p in inventory if p.is_file()]
    if not files:
        raise ValueError("Empty distributed continuation checkpoint")
    expected_rng = {f"rng_rank{rank}.pt" for rank in range(world)}
    actual_rng = {p.name for p in path.glob("rng_rank*.pt") if p.is_file()}
    if actual_rng != expected_rng:
        raise ValueError("Checkpoint rank RNG inventory changed")
    files.extend(path / name for name in sorted(expected_rng))
    if any(p.is_symlink() for p in files):
        raise ValueError("Checkpoint payload symlinks are forbidden")
    return {str(p.relative_to(path)): digest(p) for p in sorted(files)}


def save_joint(root, model, optimizer, step, metadata, rank, export=False):
    root = Path(root)
    final = root / f"step{step:05d}"
    tmp = root / f".step{step:05d}.building"
    def prepare():
        if final.exists():
            raise FileExistsError(final)
        if tmp.exists():
            # Preserve an interrupted write without mistaking it for a complete
            # checkpoint. The new write uses its own clean atomic directory.
            os.replace(tmp, root / (tmp.name + f".abandoned.{time.time_ns()}"))
        tmp.mkdir(parents=True, exist_ok=False)
    _rank0_action(prepare, rank)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    ms = get_model_state_dict(model, options=options)
    osd, optimizer_schema, optimizer_groups = _named_optimizer(model, optimizer)
    dcp.save(dict(model=ms, optimizer=osd), checkpoint_id=tmp / "distributed")
    error = None
    try:
        torch.save(rng_state(), tmp / f"rng_rank{rank}.pt")
    except Exception as exc:
        error = f"rank{rank}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Checkpoint RNG write failed: " + str(errors))
    md = dict(
        metadata,
        schema="tgcc.joint.v2",
        new_updates=step,
        optimizer_steps=metadata["config"].get("parent_step", 5000) + step,
        data_cursor=step * metadata["config"].get("effective_batch", 48),
        training_complete=step == metadata["config"]["new_updates"],
        parent_frozen=step <= metadata["config"]["critic_warmup_updates"],
        optimizer_checkpoint_format="named-lazy-v1",
        optimizer_state_schema=optimizer_schema,
        optimizer_groups=optimizer_groups,
    )
    # Full export is collective even though only rank0 receives CPU tensors.
    if export:
        state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        def write_export():
            tgcc = {
                k.removeprefix("tgcc."): v.contiguous().clone()
                for k, v in state.items()
                if k.startswith("tgcc.")
            }
            student = {
                k.removeprefix("student."): v.to(
                    torch.bfloat16
                    if k.removeprefix("student.").startswith("base.")
                    else torch.float32
                )
                .contiguous()
                .clone()
                for k, v in state.items()
                if k.startswith("student.")
                and k.removeprefix("student.") not in md["parent_aliases"]
            }
            save_file(tgcc, str(tmp / "tgcc.safetensors"))
            save_file(student, str(tmp / "student.safetensors"))
            md.update(
                checkpoint_sha256=digest(tmp / "tgcc.safetensors"),
                student_sha256=digest(tmp / "student.safetensors"),
            )
        _rank0_action(write_export, rank)
    def write_metadata():
        md["continuation_sha256"] = _payload_identity(tmp, metadata["world_size"])
        (tmp / "metadata.json").write_text(json.dumps(md, indent=2))
    _rank0_action(write_metadata, rank)
    def commit():
        os.replace(tmp, final)
        link = root / ".latest.new"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(final.name)
        os.replace(link, root / "latest")
    _rank0_action(commit, rank)
    return final


def resume_joint(path, model, optimizer, expected, rank):
    path = Path(path)
    md = json.loads((path / "metadata.json").read_text())
    step = md["new_updates"]
    if type(step) is not int or not 0 <= step <= expected["config"]["new_updates"] or md["data_cursor"] != step * expected["config"].get("effective_batch", 48):
        raise ValueError("Joint sample cursor/counter mismatch")
    if (
        md.get("schema") != "tgcc.joint.v2"
        or md.get("optimizer_steps") != expected["config"].get("parent_step", 5000) + step
        or md.get("training_complete") is not (step == expected["config"]["new_updates"])
        or md.get("parent_frozen") is not (step <= expected["config"]["critic_warmup_updates"])
    ):
        raise ValueError("Joint stage/cumulative/completion metadata mismatch")
    for key in (
        "source_sha256",
        "config",
        "parent_model_sha256",
        "normalization_sha256",
        "manifest_sha256",
        "plan_sha256",
        "world_size",
        "micro_batch",
        "grad_accum_steps",
    ):
        if md[key] != expected[key]:
            raise ValueError("Joint resume identity changed: " + key)
    if md.get("optimizer_checkpoint_format") != "named-lazy-v1":
        raise ValueError("Unsupported optimizer checkpoint format")
    if (
        not isinstance(md.get("continuation_sha256"), dict)
        or _payload_identity(path, expected["world_size"]) != md["continuation_sha256"]
    ):
        raise ValueError("Joint continuation checkpoint identity changed")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    osd = _optimizer_template(model, optimizer, md)
    ms = get_model_state_dict(model, options=options)
    state = dict(model=ms, optimizer=osd)
    dcp.load(state, checkpoint_id=path / "distributed")
    set_model_state_dict(model, model_state_dict=state["model"], options=options)
    names = dict(model.named_parameters())
    optimizer.state.clear()
    for name, fields in state["optimizer"]["state"].items():
        optimizer.state[names[name]] = fields
    for saved, current in zip(md["optimizer_groups"], optimizer.param_groups):
        for key, value in saved["attributes"].items():
            current[key] = _group_value(value, True)
    restore_rng(
        torch.load(path / f"rng_rank{rank}.pt", weights_only=False, map_location="cpu")
    )
    return md["new_updates"], md["data_cursor"]


def load_student(parent, path, metadata):
    path = Path(path)
    if metadata.get("schema") == "tgcc.joint.v2":
        if digest(path / "student.safetensors") != metadata["student_sha256"]:
            raise ValueError("Joint student identity changed")
        state = load_file(str(path / "student.safetensors"))
        for alias, canonical in metadata["parent_aliases"].items():
            state[alias] = state[canonical]
        parent.load_state_dict(state, strict=True)
    return parent
