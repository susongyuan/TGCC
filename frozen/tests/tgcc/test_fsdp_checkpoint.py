"""Exercise the actual DCP path with two CPU Gloo ranks.

FSDP2 execution itself requires the GPU preflight. These small replicated models
test the same public DCP get/set/load/save calls used by its checkpoint code,
including Adam's deliberately lazy state for the policy during critic warmup.
"""

import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from tgcc.checkpoint import rng_state
from tgcc.joint_checkpoint import resume_joint, save_joint
import tgcc.joint_checkpoint as checkpoint_api


class CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student = torch.nn.Linear(3, 2)
        self.tgcc = torch.nn.Module()
        self.tgcc.critic = torch.nn.Linear(3, 2)
        self.tgcc.adapters = torch.nn.Linear(2, 2)
        self.tgcc.gate = torch.nn.Parameter(torch.zeros(()))


def optimizer(model):
    return torch.optim.AdamW([
        dict(name="critic", params=list(model.tgcc.critic.parameters()), lr=.003),
        dict(name="adapters", params=list(model.tgcc.adapters.parameters()) + [model.tgcc.gate], lr=0.),
        dict(name="student", params=list(model.student.parameters()), lr=0.),
    ], betas=(.9, .95), weight_decay=.01, foreach=False)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        if hasattr(a, "to_local"):
            if not hasattr(b, "to_local") or a.placements != b.placements:
                return False
            return torch.equal(a.to_local(), b.to_local())
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def metadata(world):
    return dict(
        source_sha256={"fixture": "cpu-gloo-checkpoint"},
        config=dict(new_updates=6000, critic_warmup_updates=1000),
        parent_model_sha256="original-full-step5000", normalization_sha256="fixed-normalizer",
        manifest_sha256="fixture-manifest", plan_sha256="fixture-plan",
        world_size=world, micro_batch=1, grad_accum_steps=48 // world, parent_aliases={},
    )


def joint_update(model, opt):
    opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:
        group["lr"] = .001
    # FSDP unshards for a real forward; this CPU test directly addresses DTensor
    # parameters. Keep replicated and distributed scalar loss graphs separate
    # instead of creating an unsupported mixed-Tensor backward operation.
    for distributed in (False, True):
        parameters = [p for p in model.parameters()
            if hasattr(p, "to_local") is distributed]
        if parameters:
            sum((p - .2).square().sum() for p in parameters).backward()
    opt.step()


def checkpoint_worker(rank, world, init_file, root, step, sharded=False):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
        world_size=world, timeout=timedelta(seconds=45))
    result = dict(rank=rank, step=step)
    try:
        def make_model():
            model = CheckpointModel()
            if sharded:
                from torch.distributed.device_mesh import init_device_mesh
                from torch.distributed.tensor import distribute_tensor, Shard

                mesh = init_device_mesh("cpu", (world,))
                for name, p in model.student.named_parameters():
                    setattr(model.student, name, torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Shard(0)])
                    ))
            return model

        torch.manual_seed(901)
        model = make_model()
        opt = optimizer(model)
        if step:
            if step > 1000:
                joint_update(model, opt)
            else:
                opt.zero_grad(set_to_none=True)
                sum(p.square().sum() for p in model.tgcc.critic.parameters()).backward()
                opt.step()
            opt.zero_grad(set_to_none=True)
        # Per-rank RNG is deliberately different despite replicated weights.
        torch.manual_seed(100 + rank)
        random.seed(200 + rank)
        np.random.seed(300 + rank)
        original_state = copy.deepcopy(model.state_dict())
        original_optimizer = copy.deepcopy(opt.state_dict())
        original_rng = rng_state()
        if rank == 0 and step == 1:
            pending = Path(root) / "checkpoints" / ".step00001.building"
            pending.mkdir(parents=True)
            (pending / "interrupted.txt").write_text("preserve interrupted evidence")
            (pending.parent / ".latest.new").symlink_to("obsolete-incomplete")
        dist.barrier()
        saved = save_joint(Path(root) / "checkpoints", model, opt, step, metadata(world), rank, export=step > 0)
        result["save_model_unchanged"] = equal_tree(original_state, model.state_dict())
        result["save_optimizer_unchanged"] = equal_tree(original_optimizer, opt.state_dict())
        result["save_rng_unchanged"] = equal_tree(original_rng, rng_state())
        result["atomic_final"] = saved.is_dir() and not (saved.parent / f".step{step:05d}.building").exists()
        result["latest_complete"] = (saved.parent / "latest").resolve() == saved.resolve()
        result["interrupted_write_preserved"] = step != 1 or any(
            p.read_text() == "preserve interrupted evidence"
            for p in saved.parent.glob(".step00001.building.abandoned.*/interrupted.txt")
        )
        if step > 0:
            exported_student = load_file(str(saved / "student.safetensors"))
            exported_tgcc = load_file(str(saved / "tgcc.safetensors"))
            full_student = {
                k: v.full_tensor() if hasattr(v, "full_tensor") else v
                for k, v in model.student.state_dict().items()
            }
            result["export_exact"] = equal_tree(exported_student, full_student) and equal_tree(exported_tgcc, model.tgcc.state_dict())
        else:
            result["export_exact"] = True
        expected_draws = (random.random(), np.random.rand(4), torch.rand(4))
        restored = make_model()
        restored_opt = optimizer(restored)
        random.seed(700); np.random.seed(700); torch.manual_seed(700)
        try:
            resumed_step, cursor = resume_joint(saved, restored, restored_opt, metadata(world), rank)
            result["resume_ok"] = True
            result["correct_cursor"] = (resumed_step, cursor) == (step, step * 48)
            result["model_exact"] = equal_tree(original_state, restored.state_dict())
            result["optimizer_exact"] = equal_tree(original_optimizer, restored_opt.state_dict())
            actual_draws = (random.random(), np.random.rand(4), torch.rand(4))
            result["rng_exact"] = equal_tree(expected_draws, actual_draws)
            # Restore the unmodified reference optimizer even if save had a
            # side effect, so a new policy parameter must really start at t=1.
            opt.load_state_dict(original_optimizer)
            joint_update(model, opt)
            joint_update(restored, restored_opt)
            result["next_model_exact"] = equal_tree(model.state_dict(), restored.state_dict())
            result["next_optimizer_exact"] = equal_tree(opt.state_dict(), restored_opt.state_dict())
            result["policy_step_counter_exact"] = all(
                int(restored_opt.state[p]["step"]) == (2 if step > 1000 else 1)
                for p in restored.student.parameters()
            )
            # Reject changed metadata/payload before DCP, model, optimizer or
            # RNG mutation. Every rank runs the same read-only rejection path.
            md = json.loads((saved / "metadata.json").read_text())
            result["payload_inventory_complete"] = (
                any(k.startswith("distributed/") for k in md["continuation_sha256"])
                and all(f"rng_rank{r}.pt" in md["continuation_sha256"] for r in range(world))
            )
            changed_accum = dict(metadata(world), grad_accum_steps=99)
            try:
                resume_joint(saved, restored, restored_opt, changed_accum, rank)
                result["changed_accum_rejected"] = False
            except ValueError as exc:
                result["changed_accum_rejected"] = "grad_accum_steps" in str(exc)
        except BaseException as exc:
            result["resume_ok"] = False
            result["resume_error"] = f"{type(exc).__name__}: {exc}"
        try:
            save_joint(Path(root) / "checkpoints", model, opt, step, metadata(world), rank)
            result["existing_checkpoint_collectively_rejected"] = False
        except RuntimeError as exc:
            result["existing_checkpoint_collectively_rejected"] = "FileExistsError" in str(exc)
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("step", [0, 1, 1001], ids=["fresh-unused-adam", "critic-only-adam", "joint-adam-moments"])
@pytest.mark.parametrize("sharded", [False, True], ids=["replicated", "cpu-dtensor-shards"])
def test_two_rank_dcp_preserves_lazy_optimizer_rng_and_exact_continuation(tmp_path, step, sharded):
    mp.spawn(checkpoint_worker, args=(2, str(tmp_path / "gloo_init"), str(tmp_path), step, sharded), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    checks = (
        "save_model_unchanged", "save_optimizer_unchanged", "save_rng_unchanged",
        "atomic_final", "latest_complete", "resume_ok", "correct_cursor",
        "model_exact", "optimizer_exact", "rng_exact", "next_model_exact",
        "next_optimizer_exact", "policy_step_counter_exact",
        "export_exact",
        "interrupted_write_preserved", "existing_checkpoint_collectively_rejected",
        "payload_inventory_complete", "changed_accum_rejected",
    )
    failures = [{"rank": r["rank"], "failed": [k for k in checks if not r.get(k)],
        "resume_error": r.get("resume_error")} for r in results if not all(r.get(k) for k in checks)]
    assert not failures, json.dumps(failures, indent=2)


def test_invalid_cursor_is_rejected_before_model_optimizer_or_rng_mutation(tmp_path, monkeypatch):
    md = dict(metadata(2), new_updates=1, data_cursor=49)
    (tmp_path / "metadata.json").write_text(json.dumps(md))

    def forbidden_state_preparation(*args, **kwargs):
        pytest.fail("Invalid cursor reached DCP/optimizer preparation before validation")

    monkeypatch.setattr(checkpoint_api, "get_model_state_dict", forbidden_state_preparation)
    model = CheckpointModel()
    opt = optimizer(model)
    original_weights = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(opt.state_dict())
    original_rng = rng_state()
    with pytest.raises(ValueError, match="cursor|counter"):
        resume_joint(tmp_path, model, opt, metadata(2), 0)
    assert equal_tree(original_weights, model.state_dict())
    assert equal_tree(original_optimizer, opt.state_dict())
    assert equal_tree(original_rng, rng_state())


@pytest.mark.parametrize("changed", ["rng", "distributed", "inventory", "optimizer_steps", "training_complete", "parent_frozen"])
def test_resume_identity_rejects_payload_and_stage_changes_before_loading(tmp_path, monkeypatch, changed):
    root = tmp_path / "checkpoint"
    (root / "distributed").mkdir(parents=True)
    (root / "distributed/.metadata").write_bytes(b"fixture-dcp")
    for rank in range(2):
        (root / f"rng_rank{rank}.pt").write_bytes(b"fixture-rng")
    md = dict(metadata(2), schema="tgcc.joint.v2", new_updates=1000,
        optimizer_steps=6000, data_cursor=48000, training_complete=False,
        parent_frozen=True, optimizer_checkpoint_format="named-lazy-v1")
    md["continuation_sha256"] = checkpoint_api._payload_identity(root, 2)
    if changed == "rng":
        (root / "rng_rank1.pt").write_bytes(b"changed-rng")
    elif changed == "distributed":
        (root / "distributed/.metadata").write_bytes(b"changed-dcp")
    elif changed == "inventory":
        (root / "distributed/untracked.distcp").write_bytes(b"extra-tensor")
    elif changed == "optimizer_steps":
        md["optimizer_steps"] = 5999
    elif changed == "training_complete":
        md["training_complete"] = True
    elif changed == "parent_frozen":
        md["parent_frozen"] = False
    (root / "metadata.json").write_text(json.dumps(md))

    def forbidden(*args, **kwargs):
        pytest.fail("Changed checkpoint reached mutable DCP loading")

    monkeypatch.setattr(checkpoint_api, "get_model_state_dict", forbidden)
    model = CheckpointModel()
    opt = optimizer(model)
    weights = copy.deepcopy(model.state_dict())
    opt_before = copy.deepcopy(opt.state_dict())
    rng_before = rng_state()
    with pytest.raises(ValueError, match="identity|metadata"):
        resume_joint(root, model, opt, metadata(2), 0)
    assert equal_tree(weights, model.state_dict())
    assert equal_tree(opt_before, opt.state_dict())
    assert equal_tree(rng_before, rng_state())
