"""Exercise the current distribution checkpoint, not the legacy critic schema."""

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
from tgcc.distribution_checkpoint import IDENTITY_KEYS, resume_distribution, save_distribution
from test_fsdp_checkpoint import equal_tree


class DistributionCheckpointModel(torch.nn.Module):
    def __init__(self, sharded=False):
        super().__init__()
        self.student = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(3, 2)
        self.retrieval = torch.nn.Linear(2, 2)
        if sharded:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import distribute_tensor, Shard
            mesh = init_device_mesh("cpu", (dist.get_world_size(),))
            for name, parameter in self.student.named_parameters():
                setattr(self.student, name, torch.nn.Parameter(
                    distribute_tensor(parameter.detach(), mesh, [Shard(0)])))


def metadata(world):
    md = {key: "distribution-fixture-" + key for key in IDENTITY_KEYS}
    md.update(config=dict(new_updates=6000, critic_warmup_updates=1000, effective_batch=48),
        source_sha256={"fixture": "current-distribution-code"},
        world_size=world, micro_batch=1, grad_accum_steps=48 // world,
        audit_disposable=True, parent_aliases={}, policy_export_dtypes={}, schema="tgcc.distribution_joint.v1")
    return md


def optimizer(model):
    return torch.optim.AdamW([
        dict(name="critic", params=list(model.critic.parameters()), lr=.003),
        dict(name="retrieval", params=list(model.retrieval.parameters()), lr=0.),
        dict(name="action_expert", params=list(model.student.parameters()), lr=0.),
    ], betas=(.9, .95), weight_decay=.01, foreach=False)


def update(model, opt, joint):
    opt.zero_grad(set_to_none=True)
    target = random.random() + float(np.random.rand()) + float(torch.rand(()))
    for group in opt.param_groups:
        group["lr"] = .001 if joint or group["name"] == "critic" else 0.
    parameters = list(model.parameters()) if joint else list(model.critic.parameters())
    for sharded in (False, True):
        selected = [p for p in parameters if hasattr(p, "to_local") is sharded]
        if selected:
            sum((p - target).square().sum() for p in selected).backward()
    opt.step()


def worker(rank, world, init_file, root, step, sharded):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file,
        rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(902)
        random.seed(902)
        np.random.seed(902)
        model = DistributionCheckpointModel(sharded)
        opt = optimizer(model)
        if step:
            update(model, opt, step > 1000)
        original_model = copy.deepcopy(model.state_dict())
        original_opt = copy.deepcopy(opt.state_dict())
        random.seed(100 + rank); np.random.seed(200 + rank); torch.manual_seed(300 + rank)
        before_rng = rng_state()
        saved = save_distribution(Path(root) / "checkpoints", model, opt, step,
            metadata(world), rank, export=step > 0)
        checks = dict(save_model_exact=equal_tree(original_model, model.state_dict()),
            save_optimizer_exact=equal_tree(original_opt, opt.state_dict()),
            save_rng_exact=equal_tree(before_rng, rng_state()),
            atomic_latest=(saved.parent / "latest").resolve() == saved.resolve())
        if step:
            def full(value):
                return value.full_tensor() if hasattr(value, "full_tensor") else value
            student = {name: full(value) for name, value in model.student.state_dict().items()}
            auxiliary = {name: full(value) for name, value in model.state_dict().items()
                if name.startswith(("critic.", "retrieval."))}
            checks["full_exports_exact"] = (equal_tree(student, load_file(str(saved / "student.safetensors")))
                and equal_tree(auxiliary, load_file(str(saved / "distribution.safetensors"))))
        else:
            checks["full_exports_exact"] = True
        update(model, opt, True)
        continued_model, continued_opt, continued_rng = (copy.deepcopy(model.state_dict()),
            copy.deepcopy(opt.state_dict()), rng_state())
        restored = DistributionCheckpointModel(sharded)
        restored_opt = optimizer(restored)
        resumed = resume_distribution(saved, restored, restored_opt, metadata(world), rank)
        checks.update(cursor_phase_exact=resumed == (step, step * 48),
            model_exact=equal_tree(original_model, restored.state_dict()),
            named_adam_exact=equal_tree(original_opt, restored_opt.state_dict()),
            rank_rng_exact=equal_tree(before_rng, rng_state()))
        update(restored, restored_opt, True)
        checks.update(next_update_model_exact=equal_tree(continued_model, restored.state_dict()),
            next_update_named_adam_exact=equal_tree(continued_opt, restored_opt.state_dict()),
            next_update_rank_rng_exact=equal_tree(continued_rng, rng_state()),
            policy_adam_step_exact=all(int(restored_opt.state[p]["step"]) == (2 if step > 1000 else 1)
                for p in restored.student.parameters()))
        try:
            resume_distribution(saved, restored, restored_opt,
                dict(metadata(world), grad_accum_steps=99), rank)
            checks["changed_accumulation_rejected"] = False
        except ValueError:
            checks["changed_accumulation_rejected"] = True
        (Path(root) / f"rank{rank}.json").write_text(json.dumps(checks))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("step", [0, 1, 1002], ids=["fresh-lazy-adam", "critic-only", "joint-phase"])
@pytest.mark.parametrize("sharded", [False, True], ids=["replicated", "cpu-dtensor"])
def test_current_distribution_dcp_exact_complete_next_update(tmp_path, step, sharded):
    mp.spawn(worker, args=(2, str(tmp_path / "gloo_init"), str(tmp_path), step, sharded),
        nprocs=2, join=True)
    checks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all(all(row.values()) for row in checks), json.dumps(checks, indent=2)


def test_bad_distribution_cursor_is_rejected_before_dcp_loading(tmp_path, monkeypatch):
    import tgcc.distribution_checkpoint as api
    md = dict(metadata(2), new_updates=1002, data_cursor=3 * 48,
        policy_new_updates=2, policy_cumulative_updates=5002, training_complete=False)
    (tmp_path / "metadata.json").write_text(json.dumps(md))
    monkeypatch.setattr(api, "get_model_state_dict", lambda *args, **kwargs: pytest.fail("Reached mutable DCP path"))
    model = DistributionCheckpointModel()
    with pytest.raises(ValueError, match="cursor|counter"):
        resume_distribution(tmp_path, model, optimizer(model), metadata(2), 0)
