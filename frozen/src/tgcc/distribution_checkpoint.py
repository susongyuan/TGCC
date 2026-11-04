"""Atomic complete checkpoints for the success-only distribution recipe."""

import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
    get_model_state_dict, set_model_state_dict)
from safetensors.torch import save_file, load_file

from .checkpoint import digest, rng_state, restore_rng
from .joint_checkpoint import (_named_optimizer, _optimizer_template,
    _group_value, _payload_identity, _rank0_action)


IDENTITY_KEYS = ("source_sha256", "config", "parent_model_sha256", "normalization_sha256",
    "demonstrations_manifest_sha256", "feature_codec_sha256", "retrieval_index_sha256",
    "feature_artifact_sha256", "encoder_model_sha256", "world_size", "micro_batch",
    "grad_accum_steps", "distribution_config", "retrieval_adapter_config", "plan_sha256",
    "audit_disposable", "policy_export_dtypes", "policy_cache_identity_sha256")


def save_distribution(root, model, optimizer, step, metadata, rank, *, export=False):
    root = Path(root)
    final = root / f"step{step:05d}"
    tmp = root / f".step{step:05d}.building"
    def prepare():
        root.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(final)
        if tmp.exists():
            os.replace(tmp, root / (tmp.name + f".abandoned.{time.time_ns()}"))
        tmp.mkdir()
    _rank0_action(prepare, rank)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    optimizer_state, optimizer_schema, optimizer_groups = _named_optimizer(model, optimizer)
    if metadata['config'].get('checkpoint_storage')=='native_zstd':
        from torch.distributed.checkpoint import FileSystemWriter
        from torch.distributed.checkpoint._extension import ZStandard
        if not ZStandard.is_available():raise RuntimeError('Native lossless zstd storage unavailable')
        dcp.save(dict(model=state, optimizer=optimizer_state),
            storage_writer=FileSystemWriter(tmp/'distributed',_extensions=[ZStandard()]))
    else:
        dcp.save(dict(model=state, optimizer=optimizer_state), checkpoint_id=tmp / "distributed")
    error = None
    try:
        torch.save(rng_state(), tmp / f"rng_rank{rank}.pt")
    except Exception as exc:
        error = f"rank{rank}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Distribution checkpoint RNG write failed: " + str(errors))
    warmup = 0 if metadata.get('audit_phase')=='joint' else metadata["config"]["critic_warmup_updates"]
    policy_steps = max(0, step - warmup)
    policy_initialization=metadata.get('policy_initialization_updates',5000)
    md = dict(metadata, schema="tgcc.distribution_joint.v1", new_updates=step,
        data_cursor=step * metadata["config"]["effective_batch"],
        training_complete=step == metadata["config"]["new_updates"],
        policy_new_updates=policy_steps, policy_cumulative_updates=policy_initialization + policy_steps,
        cumulative_schedule_steps=policy_initialization + step,
        optimizer_checkpoint_format="named-lazy-v1", optimizer_state_schema=optimizer_schema,
        optimizer_groups=optimizer_groups)
    if metadata['config'].get('schema') == 'tgcc.full_success_distribution.recipe.v1':
        md.update(full_model_updates=step,
            critic_actual_total_updates=metadata['config']['critic_initialization_actual_updates'] + step,
            fresh_critic_only_updates=0)
    if metadata['config'].get('schema') == 'tgcc.corrected_full_success_distribution.recipe.v1':
        md.update(full_model_updates=policy_steps,
            critic_actual_total_updates=metadata['config']['critic_initialization_actual_updates']+step,
            fresh_critic_only_updates=min(step,warmup),execution_target=metadata['execution_target'],
            feature_schema=metadata['feature_schema'],memory_unroll_blocks=metadata['memory_unroll_blocks'])
    if export:
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        def write_export():
            auxiliary = {k: v.contiguous().clone() for k, v in full.items()
                         if k.startswith(("critic.", "retrieval."))}
            policy = {k.removeprefix("student."): v.to(getattr(torch,
                md["policy_export_dtypes"].get(k.removeprefix("student."), str(v.dtype)).removeprefix("torch.")))
                .contiguous().clone() for k, v in full.items()
                if k.startswith("student.") and k.removeprefix("student.") not in md["parent_aliases"]}
            save_file(auxiliary, str(tmp / "distribution.safetensors"))
            save_file(policy, str(tmp / "student.safetensors"))
            md.update(distribution_sha256=digest(tmp / "distribution.safetensors"),
                      student_sha256=digest(tmp / "student.safetensors"))
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


CONTINUATION_SOURCE_WHITELIST = (
    "code/src/openpi/models_pytorch/gemma_pytorch.py",
    "tools/tgcc/train_full_distribution.py",
    "src/tgcc/distribution_checkpoint.py",
    "configs/tgcc/full_distribution.json",
    "tests/tgcc/test_joint_attention.py",
    "tests/tgcc/test_declared_continuation.py",
)
CONTINUATION_CONFIG_KEYS = ("attention_implementation",)


def declared_continuation(md, expected, resumed_from=None):
    """Return the identity keys whose difference is a declared kernel-only continuation.

    The checkpoint and the running tree may differ only in CONTINUATION_SOURCE_WHITELIST files
    and CONTINUATION_CONFIG_KEYS; any other source or config difference raises. When a
    difference is accepted it is appended to expected['continuation_declared_changes'] so every
    later checkpoint records the exact provenance.
    """
    old_src = md.get("source_sha256") or {}; new_src = expected.get("source_sha256") or {}
    old_cfg = md.get("config") or {}; new_cfg = expected.get("config") or {}
    if old_src == new_src and old_cfg == new_cfg:
        return set()
    changed_files = sorted(k for k in set(old_src) | set(new_src) if old_src.get(k) != new_src.get(k))
    undeclared = [k for k in changed_files if k not in CONTINUATION_SOURCE_WHITELIST]
    if undeclared:
        raise ValueError("Undeclared source change on resume: " + ",".join(undeclared))
    changed_keys = sorted(k for k in set(old_cfg) | set(new_cfg) if old_cfg.get(k) != new_cfg.get(k))
    undeclared_keys = [k for k in changed_keys if k not in CONTINUATION_CONFIG_KEYS]
    if undeclared_keys:
        raise ValueError("Undeclared config change on resume: " + ",".join(undeclared_keys))
    record = dict(resumed_from=None if resumed_from is None else str(resumed_from), resumed_step=md.get("new_updates"),
                  files={k: dict(checkpoint=old_src.get(k), running=new_src.get(k)) for k in changed_files},
                  config={k: dict(checkpoint=old_cfg.get(k), running=new_cfg.get(k)) for k in changed_keys},
                  reason="kernel-only continuation: same data, recipe, optimizer and weights; attention kernel changed")
    expected.setdefault("continuation_declared_changes", []).append(record)
    declared = set()
    if changed_files: declared.add("source_sha256")
    if changed_keys: declared.add("config")
    return declared


def resume_distribution(path, model, optimizer, expected, rank):
    path = Path(path)
    md = json.loads((path / "metadata.json").read_text())
    step = md["new_updates"]
    if expected['config'].get('schema') == 'tgcc.full_success_distribution.recipe.v1':
        if (md.get('full_model_updates') != step or md.get('fresh_critic_only_updates') != 0 or
                md.get('critic_actual_total_updates') != expected['config']['critic_initialization_actual_updates'] + step):
            raise ValueError('Full-model actual provenance/counters changed')
    warmup=0 if expected.get('audit_phase')=='joint' else expected['config']['critic_warmup_updates']
    policy_steps = max(0, step - warmup)
    policy_initialization=expected.get('policy_initialization_updates',5000)
    if expected['config'].get('schema')=='tgcc.corrected_full_success_distribution.recipe.v1':
        for key in ('student_initialization_sha256','feature_library_sha256','memory_burnin_blocks','audit_phase'):
            if key not in expected or md.get(key)!=expected[key]:
                raise ValueError('Corrected initialization/library/unroll identity changed: '+key)
        if (md.get('full_model_updates')!=policy_steps or md.get('fresh_critic_only_updates')!=min(step,warmup)
                or md.get('critic_actual_total_updates')!=expected['config']['critic_initialization_actual_updates']+step
                or md.get('feature_schema')!=expected['feature_schema']
                or md.get('execution_target')!=expected['execution_target']
                or md.get('memory_unroll_blocks')!=expected['memory_unroll_blocks']
                or md.get('policy_initialization_updates')!=policy_initialization):
            raise ValueError('Corrected delta/memory/counter provenance changed')
    if (md.get("schema") != "tgcc.distribution_joint.v1" or type(step) is not int or
            not 0 <= step <= expected["config"]["new_updates"] or
            md.get("data_cursor") != step * expected["config"]["effective_batch"] or
            md.get("policy_new_updates") != policy_steps or
            md.get("policy_cumulative_updates") != policy_initialization + policy_steps or
            md.get("training_complete") is not (step == expected["config"]["new_updates"])):
        raise ValueError("Distribution checkpoint recipe/cursor/counter mismatch")
    declared = declared_continuation(md, expected, resumed_from=path)
    for key in IDENTITY_KEYS:
        if key in declared:
            continue
        if md.get(key) != expected.get(key) or key not in expected:
            raise ValueError("Distribution resume identity changed: " + key)
    if (md.get("optimizer_checkpoint_format") != "named-lazy-v1" or
            _payload_identity(path, expected["world_size"]) != md.get("continuation_sha256")):
        raise ValueError("Distribution continuation payload identity changed")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = dict(model=get_model_state_dict(model, options=options),
                 optimizer=_optimizer_template(model, optimizer, md))
    dcp.load(state, checkpoint_id=path / "distributed")
    set_model_state_dict(model, model_state_dict=state["model"], options=options)
    names = dict(model.named_parameters())
    optimizer.state.clear()
    for name, fields in state["optimizer"]["state"].items():
        optimizer.state[names[name]] = fields
    for saved, current in zip(md["optimizer_groups"], optimizer.param_groups):
        for key, value in saved["attributes"].items():
            current[key] = _group_value(value, True)
    restore_rng(torch.load(path / f"rng_rank{rank}.pt", weights_only=False, map_location="cpu"))
    return step, md["data_cursor"]


def load_distribution_critic(critic, path, expected, regression=None):
    """Reuse a trained delta critic from a formal corrected checkpoint after identity checks.

    Checks: feature schema, execution target, distribution config (modes and widths), library,
    normalization, codec and index identities, formal (non-disposable) provenance, continuation
    payload identity, and an optional fixed input-output regression receipt. Any mismatch raises;
    nothing is silently re-initialized.
    """
    import torch.distributed.checkpoint as dcp
    from .joint_checkpoint import _payload_identity
    path = Path(path); md = json.loads((path / "metadata.json").read_text())
    for key in ("feature_schema", "execution_target", "distribution_config", "feature_library_sha256",
                "normalization_sha256", "feature_codec_sha256", "retrieval_index_sha256"):
        if key not in expected or md.get(key) != expected[key]:
            raise ValueError("Critic reuse identity mismatch: " + key)
    if md.get("audit_disposable") is not False or md.get("training_complete") not in (False, True):
        raise ValueError("Critic reuse requires a formal checkpoint")
    if _payload_identity(path, md["world_size"]) != md.get("continuation_sha256"):
        raise ValueError("Critic source payload identity changed")
    state = {"model": {"critic." + name: torch.zeros_like(value, device="cpu") for name, value in critic.state_dict().items()}}
    dcp.load(state, checkpoint_id=str(path / "distributed"))
    critic.load_state_dict({k.removeprefix("critic."): v for k, v in state["model"].items()}, strict=True)
    receipt = dict(source=str(path), source_critic_actual_total_updates=md.get("critic_actual_total_updates"),
                   source_new_updates=md.get("new_updates"), continuation_sha256=md.get("continuation_sha256"),
                   source_sha256_files=len(md.get("source_sha256", {})), regression=None)
    if regression is not None:
        want = json.loads(Path(regression).read_text())
        got = critic_regression_signature(critic)
        for key, value in want["signature"].items():
            if abs(float(got[key]) - float(value)) > 1e-4 * max(1.0, abs(float(value))):
                raise ValueError("Critic fixed input-output regression failed: " + key)
        receipt["regression"] = dict(path=str(regression), matched=sorted(want["signature"]))
    return receipt


def critic_regression_signature(critic):
    """Deterministic outputs of the critic on a fixed synthetic batch (CPU, fp32)."""
    c = critic.config; g = torch.Generator().manual_seed(20261031)
    batch = dict(context=torch.randn(8, c.context_width, generator=g), actions=torch.randn(8, 32, 23, generator=g),
                 action_mask=torch.ones(8, 32, dtype=torch.bool), after_features=torch.randn(8, c.after_width, generator=g),
                 after_feature_mask=torch.ones(8, c.after_width, dtype=torch.bool), next_index=torch.arange(8) % c.planning_classes)
    was_training = critic.training; critic.eval()
    with torch.no_grad():
        device = next(critic.parameters()).device
        out = critic({k: v.to(device) for k, v in batch.items()})
    critic.train(was_training)
    return {k: float(v.float().sum().cpu()) for k, v in out.items()}


def load_distribution_student(parent, path, metadata):
    path = Path(path)
    if metadata.get("schema") != "tgcc.distribution_joint.v1":
        raise ValueError("Wrong distribution checkpoint schema")
    if digest(path / "student.safetensors") != metadata["student_sha256"]:
        raise ValueError("Distribution student identity changed")
    state = load_file(str(path / "student.safetensors"))
    for alias, canonical in metadata["parent_aliases"].items():
        state[alias] = state[canonical]
    parent.load_state_dict(state, strict=True)
    return parent
