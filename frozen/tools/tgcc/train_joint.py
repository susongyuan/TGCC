"""1000 critic warmup + 5000 FULL joint updates. FSDP2; native actions unchanged."""

from pathlib import Path
import os, sys, json, time, random, math, shutil, signal

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "code/src"),
    str(ROOT / "code/packages/openpi-client/src"),
]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "TGR_GEOMETRY_CACHE_PATH",
    "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3",
)
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from tgr.factory import create, ensure_legacy_imports

ensure_legacy_imports()
from tgr.data import LazySequenceDataset
from tgcc.robot import action_spec, ActionNormalizer
from tgcc.data import DemonstrationMetadata
from tgcc.joint import JointSystem, optimizer_groups, learning_rates
from tgcc.joint_checkpoint import save_joint, resume_joint
from tgcc.checkpoint import digest, sources, rng_state, restore_rng
from tgcc.readiness import require_supervision
from train import planned_order, BASE, PARENT, PARENT_SHA, identity_list


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def synchronize_and_clip(model, replicated, maximum, world, device):
    # FSDP2 cannot shard the two existing 0D adapter gates. Keep their exact
    # checkpoint shape and explicitly average gradients; do not redesign them.
    for p in replicated:
        if p.grad is not None:
            dist.all_reduce(p.grad)
            p.grad.div_(world)
    norm2 = torch.zeros((), device=device, dtype=torch.float64)
    for p in model.parameters():
        if p.grad is not None:
            value = local(p.grad).double().square().sum()
            norm2 += value / world if p in replicated else value
    dist.all_reduce(norm2)
    norm = norm2.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite distributed gradient")
    scale = (maximum / (norm + 1e-6)).clamp(max=1).float()
    for p in model.parameters():
        if p.grad is not None:
            local(p.grad).mul_(scale)
    return norm


def report_gradients(model, device):
    values = {
        k: torch.zeros((), device=device)
        for k in (
            "critic",
            "adapters",
            "vision",
            "vlm",
            "expert",
            "action_io",
            "graphs",
            "memory",
        )
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        k = (
            "critic"
            if name.startswith("tgcc.critic")
            else "adapters"
            if name.startswith("tgcc.adapters")
            else "vision"
            if "vision_tower" in name
            else "expert"
            if "gemma_expert" in name
            else "vlm"
            if "paligemma" in name
            else "action_io"
            if name.startswith("student.base.")
            else "graphs"
        )
        v = local(p.grad).float().square().sum()
        values[k] += v
        if name.startswith("student.decoder.memory"):
            values["memory"] += v
    for k, v in values.items():
        dist.all_reduce(v)
        values[k] = float(v.sqrt())
    return values


def log(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, default=str) + "\n")
    print(json.dumps(value, default=str), flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--audit", action="store_true")
    a.add_argument("--micro-batch", type=int, default=1)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--resume", type=Path)
    args = a.parse_args()
    cfg = json.loads((ROOT / "configs/tgcc/default.json").read_text())
    assert (
        cfg["new_updates"] == 6000
        and cfg["critic_warmup_updates"] == 1000
        and cfg["joint_updates"] == 5000
        and cfg["effective_batch"] == 48
    )
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])
    if not 1 <= world <= 4 or 48 % (world * args.micro_batch):
        raise ValueError("Verified multi-GPU effective batch48 required")
    device = torch.device("cuda", gpu)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    source = sources(ROOT)
    if not args.audit:
        gate = json.loads(
            (ROOT / "runs/interface_audit/joint_training_readiness.json").read_text()
        )
        if gate["status"] != "passed" or gate["source_sha256"] != source:
            raise ValueError("Joint readiness gate/source mismatch")
        label_gate = gate["gates"]["labels"]
        label_path = Path(label_gate["report_path"])
        if digest(label_path) != label_gate["report_sha256"]:
            raise ValueError("Supervision report identity changed")
        label_report = json.loads(label_path.read_text())
        if label_report["source_sha256"] != source:
            raise ValueError("Supervision evidence is not audited for the frozen source")
        require_supervision(label_report)
    assert digest(Path(PARENT) / "model.safetensors") == PARENT_SHA
    parent_md = json.loads((Path(PARENT) / "metadata.json").read_text())
    for n, h in parent_md["implementation_source_sha256"].items():
        if digest(ROOT / n) != h:
            raise ValueError("Tested original implementation changed: " + n)
    print(
            json.dumps(dict(rank=rank, stage="loading_step5000_student_direct_finetune")),
        flush=True,
    )
    _, student = create(BASE, "tri_graph", "cpu", PARENT)
    student.requires_grad_(True)
    student.to(torch.float32).to(device)
    student.eval()
    normalizer = ActionNormalizer(BASE)
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    model = JointSystem(student, None, action_spec(ROOT), normalizer, cfg, plan).to(
        device
    )
    model.tgcc.critic.ood.requires_grad_(False)
    replicated = {p for p in model.parameters() if p.ndim == 0}
    fully_shard(
        model,
        ignored_params=replicated,
        mesh=init_device_mesh("cuda", (world,)),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
    )
    opt = torch.optim.AdamW(
        optimizer_groups(model, cfg),
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
        foreach=False,
        fused=False,
    )
    step, cursor = 0, 0
    migration = dict(
        policy="user_authorized_fresh_revised_critic",
        discarded_legacy_new_updates=100,
        original_policy_parent_preserved=PARENT_SHA,
        warmup_target=1000,
        joint_target=5000,
    )
    data = LazySequenceDataset(
        ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl",
        BASE,
        ROOT / "reports/data_plan_vocabulary.json",
    )
    dataset = DemonstrationMetadata(data, normalizer)
    if data.adapter.normalization_sha256 != parent_md["normalization_sha256"]:
        raise ValueError("Normalizer identity")
    metadata = dict(
        config=cfg,
        source_sha256=source,
        parent_checkpoint=PARENT,
        parent_model_sha256=PARENT_SHA,
        normalization_sha256=data.adapter.normalization_sha256,
        parent_aliases=parent_md.get("aliases", {}),
        manifest_sha256=digest(data.manifest_path),
        plan_sha256=digest(data.plan_path),
        world_size=world,
        micro_batch=args.micro_batch,
        grad_accum_steps=48 // (world * args.micro_batch),
        migration=migration,
        training_scope="1000 critic warmup plus 5000 full policy/graph/critic/adapter updates",
        graph_source="oracle",
        record_video=False,
        legacy_ood="storage-only, frozen, no forward/loss/gate",
        teacher="disabled; direct full finetune from original selected step5000",
        critic_gradient_boundary="policy features detached; flow and skill CE update full student",
        train_sequences=51714,
        validation_sequences=4940,
    )
    if args.resume:
        step, cursor = resume_joint(args.resume, model, opt, metadata, rank)
    model_rng = rng_state()
    order = planned_order(data.rows, 6000 * 48, cfg["seed"])
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        sampler=order[cursor + rank :: world],
        num_workers=args.workers,
        collate_fn=identity_list,
        generator=torch.Generator().manual_seed(cfg["seed"] + rank),
        multiprocessing_context="spawn" if args.workers else None,
        persistent_workers=bool(args.workers),
    )
    iterator = iter(loader)
    restore_rng(model_rng)
    if rank == 0:
        (args.output / "run_identity.json").write_text(json.dumps(metadata, indent=2))
    if args.audit:
        # Preflight optimizer update is confined to this disposable process.
        sample = [
            dataset[
                next(
                    i
                    for i, r in enumerate(data.rows)
                    if r["split"] == "train" and r["sequence_steps"][0] >= 1024
                )
            ]
        ]
        audits = []
        for phase in (100, 1000):
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, counts = model(sample, phase)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite audit loss")
            loss.backward()
            grads = report_gradients(model, device)
            if phase < 1000 and any(
                grads[k] for k in ("vision", "vlm", "expert", "action_io", "graphs")
            ):
                raise AssertionError("Warmup changed parent gradients")
            if phase >= 1000:
                if any(
                    grads[k] <= 0 or not math.isfinite(grads[k])
                    for k in (
                        "critic",
                        "vision",
                        "vlm",
                        "expert",
                        "action_io",
                        "graphs",
                        "memory",
                    )
                ):
                    raise AssertionError("Missing full-policy gradient: " + str(grads))
                for group, lr in zip(opt.param_groups, learning_rates(phase, cfg)):
                    group["lr"] = lr
                synchronize_and_clip(model, replicated, cfg["grad_clip"], world, device)
                opt.step()
            torch.cuda.synchronize()
            audits.append(
                dict(
                    phase_new_step=phase,
                    seconds=time.monotonic() - started,
                    loss=float(loss),
                    parts={k: float(v) for k, v in parts.items()},
                    valid_counts=counts,
                    gradients=grads,
                    peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    disposable_optimizer_updates=int(phase >= 1000),
                )
            )
        (args.output / f"joint_audit_rank{rank}.json").write_text(
            json.dumps(
                dict(
                    status="passed",
                    source_sha256=source,
                    audits=audits,
                    migration=migration,
                    formal_optimizer_updates=0,
                ),
                indent=2,
            )
        )
        print(
            json.dumps(dict(rank=rank, status="joint_audit_passed", audits=audits)),
            flush=True,
        )
        dist.destroy_process_group()
        return
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()
    while step < 6000:
        model.train()
        opt.zero_grad(set_to_none=True)
        for group, lr in zip(opt.param_groups, learning_rates(step, cfg)):
            group["lr"] = lr
        sums = {}
        counts = {}
        tick = time.monotonic()
        for micro in range(metadata["grad_accum_steps"]):
            samples = next(iterator)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, valid = model(samples, step)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint loss")
            (loss / metadata["grad_accum_steps"]).backward()
            for k, v in parts.items():
                sums[k] = (
                    sums.get(k, 0.0) + float(v.detach()) / metadata["grad_accum_steps"]
                )
            for k, v in valid.items():
                counts[k] = counts.get(k, 0) + int(v)
        if step < 1000:
            for module in (model.student, model.tgcc.adapters):
                for p in module.parameters():
                    p.grad = None
            # A zero-masked loss must not advance Adam or weight decay for the
            # handoff head before its supervised phase starts.
            for p in model.tgcc.critic.handoff.parameters():
                p.grad = None
        grad = synchronize_and_clip(model, replicated, cfg["grad_clip"], world, device)
        opt.step()
        step += 1
        cursor += 48
        if step % 10 == 0 or step in (1, 1000, 1001, 6000):
            names = sorted(sums)
            values = torch.tensor([sums[k] for k in names], device=device)
            dist.all_reduce(values)
            sums = {k: float(v) / world for k, v in zip(names, values)}
            names = sorted(counts)
            values = torch.tensor(
                [counts[k] for k in names], device=device, dtype=torch.long
            )
            dist.all_reduce(values)
            counts = {k: int(v) for k, v in zip(names, values)}
            if rank == 0:
                log(
                    args.output / "train.jsonl",
                    dict(
                        new_update=step,
                        cumulative_step=5000 + step,
                        phase="critic_warmup" if step <= 1000 else "full_joint",
                        seconds=time.monotonic() - tick,
                        elapsed_seconds=time.monotonic() - started,
                        losses=sums,
                        global_valid_counts=counts,
                        gradient_norm=float(
                            grad.full_tensor() if hasattr(grad, "full_tensor") else grad
                        ),
                        lrs=[g["lr"] for g in opt.param_groups],
                        sequences=cursor,
                        equivalent_passes=cursor / 51714,
                        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    ),
                )
        if step in cfg["checkpoint_new_steps"]:
            saved_rng = rng_state()
            model.eval()
            ids = [i for i, r in enumerate(data.rows) if r["split"] == "validation"]
            vals = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                validation_samples = math.ceil(16 / world) * world
                for n in np.linspace(0, len(ids) - 1, validation_samples).astype(int)[
                    rank::world
                ]:
                    _, p, _ = model([dataset[ids[n]]], min(step, 5999))
                    for k, v in p.items():
                        vals[k] = vals.get(k, 0.0) + float(v) / (
                            validation_samples / world
                        )
            restore_rng(saved_rng)
            if rank == 0:
                log(
                    args.output / "validation.jsonl",
                    dict(
                        new_update=step,
                        rank0_losses=vals,
                        examples=validation_samples,
                        test_instances_used=False,
                    ),
                )
        stopping = torch.tensor(int(stop[0]), device=device)
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if step % 250 == 0 or step == 6000 or bool(stopping):
            if sources(ROOT) != source:
                raise RuntimeError("Frozen source changed")
            if shutil.disk_usage(args.output).free < 55 * 2**30:
                raise OSError("Insufficient space for atomic full-state checkpoint")
            save_joint(
                args.output / "checkpoints",
                model,
                opt,
                step,
                metadata,
                rank,
                export=step == 6000,
            )
            if rank == 0:
                for old in (args.output / "checkpoints").glob("step*"):
                    if old.name != f"step{step:05d}":
                        shutil.rmtree(old)
            dist.barrier()
        if bool(stopping):
            break
    if rank == 0 and step == 6000:
        (args.output / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    new_updates=6000,
                    cumulative_steps=11000,
                    sequences=cursor,
                    checkpoint=str(args.output / "checkpoints/step06000"),
                ),
                indent=2,
            )
        )
    model.tap.close()
    data.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
