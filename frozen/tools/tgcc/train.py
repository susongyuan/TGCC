"""Exactly 5,000 new updates; --audit runs real gradients without optimizer.step."""

from pathlib import Path
import os, sys, json, time, math, random, argparse, shutil, contextlib

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
import numpy as np
import torch
import torch.distributed as dist
import resource

soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard_limit), hard_limit))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tgr.factory import create, ensure_legacy_imports

ensure_legacy_imports()  # spawn workers must install the same audited layout shim before unpickling transforms
from tgr.data import LazySequenceDataset
from tgcc.robot import action_spec, ActionNormalizer
from tgcc.data import DemonstrationMetadata
from tgcc.policy import PolicyFeatures
from tgcc.features import extract_pair
from tgcc.train_system import TrainSystem
from tgcc.checkpoint import (
    sources,
    digest,
    save_checkpoint,
    load_checkpoint,
    rng_state,
    restore_rng,
)
from tgcc.feature_cache import FeatureCache, CachedDemonstrations, slice_observation
from tgcc.features import concatenate_observations
import hashlib
from dataclasses import asdict

BASE = "/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"
PARENT = "/media/data5/ssy/TGR/experiments/tc_esgr_v2_approx_full_5ep_20261024_02/selected_step5000_user_stop"
PARENT_SHA = "80436905542b0ce75d77d17cf908b29846f5497b58189177d406daebdd3e1239"


def identity_list(x):
    return x


def planned_order(rows, count, seed):
    indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
    blocks = []
    for i in indices:
        if (
            not blocks
            or len(blocks[-1]) >= 64
            or rows[blocks[-1][0]]["episode_id"] != rows[i]["episode_id"]
        ):
            blocks.append([])
        blocks[-1].append(i)
    out = []
    epoch = 0
    while len(out) < count:
        rng = random.Random(seed + epoch)
        order = list(range(len(blocks)))
        rng.shuffle(order)
        for i in order:
            out.extend(blocks[i])
        epoch += 1
    return out[:count]


def rates(step, cfg):
    def ratio(s, total):
        if s < cfg["warmup"]:
            return (s + 1) / cfg["warmup"]
        return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (
            1 + math.cos(math.pi * (s - cfg["warmup"]) / (total - cfg["warmup"]))
        )

    return cfg["critic_lr"] * ratio(step, 5000), cfg["adapter_lr"] * ratio(
        step - 4000, 1000
    ) if step >= 4000 else 0.0


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def main():
    # Keep legacy helpers importable for audit/provenance, but this executable
    # may not consume the revised config with the obsolete frozen5000 recipe.
    if json.loads((ROOT / "configs/tgcc/default.json").read_text()).get("schema", "").startswith("tgcc.joint."):
        raise RuntimeError("Legacy frozen training is superseded; use the audited tools/tgcc/train_joint.py")
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true")
    p.add_argument("--micro-batch", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume")
    p.add_argument("--output", default=str(ROOT / "runs/tgcc_cook_hot_dogs_20261027"))
    args = p.parse_args()
    cfg = json.loads((ROOT / "configs/tgcc/default.json").read_text())
    if (
        cfg["new_updates"] != 5000
        or cfg["parent_step"] != 5000
        or cfg["effective_batch"] != 48
    ):
        raise ValueError("Fixed user-authorized continuation budget changed")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 4:
        raise ValueError("User permits at most four training GPUs")
    if 48 % (world * args.micro_batch):
        raise ValueError("Effective batch must remain 48")
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    source_identity = sources(ROOT)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    required = ["complete_label_coverage.json", "step0_real_batch_gate.json"]
    if not args.audit:
        for name in required:
            if not (ROOT / "runs/interface_audit" / name).exists():
                raise ValueError("Missing pretraining gate: " + name)
        gate = json.loads(
            (ROOT / "runs/interface_audit/training_readiness.json").read_text()
        )
        if gate.get("status") != "passed" or gate["source_sha256"] != source_identity:
            raise ValueError("Readiness source identity changed")
    if shutil.disk_usage(output).free < 50 * 2**30:
        raise OSError("Training disk floor: 50 GiB")
    if rank == 0 and digest(Path(PARENT) / "model.safetensors") != PARENT_SHA:
        raise ValueError("Wrong selected parent")
    parent_metadata = json.loads((Path(PARENT) / "metadata.json").read_text())
    for name, expected in parent_metadata["implementation_source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Tested parent implementation changed: " + name)
    if world > 1:
        dist.barrier()
    _, parent = create(BASE, "tri_graph", str(device), PARENT)
    parent.requires_grad_(False)
    parent.eval()
    normalizer = ActionNormalizer(BASE)
    spec = action_spec(ROOT)
    system = TrainSystem(parent, spec, normalizer, cfg).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": system.critic.parameters(),
                "lr": cfg["critic_lr"],
                "name": "critic",
            },
            {"params": system.adapters.parameters(), "lr": 0.0, "name": "adapters"},
        ],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    steps = 0
    cursor = 0
    continuation_rng = None
    if args.resume:
        resume_metadata = json.loads((Path(args.resume) / "metadata.json").read_text())
        if (
            resume_metadata["world_size"] != world
            or resume_metadata["micro_batch"] != args.micro_batch
        ):
            raise ValueError(
                "Exact continuation requires the saved world size and micro batch"
            )
        steps, cursor = load_checkpoint(
            args.resume, system, optimizer, source_identity, rank
        )
        continuation_rng = rng_state()
    else:
        # New-module initialization varies by rank only AFTER DDP broadcasts it.
        torch.manual_seed(cfg["seed"] + rank)
        np.random.seed(cfg["seed"] + rank)
        random.seed(cfg["seed"] + rank)
    model = (
        DDP(system, device_ids=[local], broadcast_buffers=False)
        if world > 1
        else system
    )
    dataset = LazySequenceDataset(
        ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl",
        BASE,
        ROOT / "reports/data_plan_vocabulary.json",
    )
    if dataset.adapter.normalization_sha256 != parent_metadata["normalization_sha256"]:
        raise ValueError("The selected parent's normalization changed")
    if args.resume and (
        resume_metadata["manifest_sha256"] != digest(dataset.manifest_path)
        or resume_metadata["plan_sha256"] != digest(dataset.plan_path)
        or resume_metadata["parent_model_sha256"] != PARENT_SHA
    ):
        raise ValueError("Resume data, plan, or parent identity changed")
    data = DemonstrationMetadata(dataset, normalizer)
    plan = json.loads((ROOT / "reports/data_plan_vocabulary.json").read_text())
    order = planned_order(dataset.rows, 5000 * 48, cfg["seed"])
    if cursor != steps * 48:
        raise ValueError("Resume sample counter mismatch")
    cache_identity = hashlib.sha256(
        json.dumps(
            dict(
                parent=PARENT_SHA,
                source=source_identity,
                quantization="critic_graph_context_fp16_v1",
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    feature_cache = FeatureCache(ROOT / "cache/tgcc_frozen_features", cache_identity)

    def make_loader(consumed, use_cache):
        indices = order[consumed + rank :: world]
        if args.audit:
            first = next(r["episode_id"] for r in dataset.rows if r["split"] == "train")
            episode_indices = [
                i for i, r in enumerate(dataset.rows) if r["episode_id"] == first
            ]
            indices = [
                episode_indices[int(i)]
                for i in np.linspace(0, len(episode_indices) - 1, args.micro_batch)
            ]
        return DataLoader(
            CachedDemonstrations(data, feature_cache, use_cache),
            batch_size=args.micro_batch,
            sampler=indices,
            num_workers=args.workers,
            collate_fn=identity_list,
            pin_memory=True,
            prefetch_factor=2 if args.workers else None,
            multiprocessing_context="spawn" if args.workers else None,
            persistent_workers=bool(args.workers),
            generator=torch.Generator().manual_seed(cfg["seed"] + rank),
        )

    loader = make_loader(cursor, steps < 4000 and not args.audit)
    iterator = iter(loader)
    tap = PolicyFeatures(parent)
    accum = 48 // (world * args.micro_batch)
    if continuation_rng is not None:
        restore_rng(continuation_rng)
    metadata = dict(
        parent_checkpoint=PARENT,
        parent_model_sha256=PARENT_SHA,
        normalization_sha256=dataset.adapter.normalization_sha256,
        source_sha256=source_identity,
        config=cfg,
        world_size=world,
        micro_batch=args.micro_batch,
        grad_accum_steps=accum,
        train_sequences=51714,
        validation_sequences=4940,
        graph_source="oracle",
        record_video=False,
        manifest_sha256=digest(dataset.manifest_path),
        plan_sha256=digest(dataset.plan_path),
        robot_action_spec=asdict(spec),
        graph_feature_spec=asdict(system.critic.spec),
        parent_aliases=parent_metadata.get("aliases", {}),
        environment=dict(
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
        ),
        trainable_parameters=sum(p.numel() for p in system.parameters()),
        parent_frozen=True,
        critic_parameters=sum(p.numel() for p in system.critic.parameters()),
        adapter_parameters=sum(p.numel() for p in system.adapters.parameters()),
        transition_label="real post-execution contract satisfaction; UNKNOWN masked; not a learned dynamics simulator",
        negative_label="verified native controller/joint-limit violations only",
        current_action_in_history=False,
        prior_skill_annotation_as_input=False,
        base_origin_commit="9f4dc9e701e6c6a8870f5a17bb4e5c98aaee2320",
    )
    if rank == 0:
        (output / "run_identity.json").write_text(json.dumps(metadata, indent=2))

    def one_gradient(step, entries):
        fresh = [e for e in entries if e["cached"] is None]
        flow = []
        if fresh:
            b, a, n, flow = extract_pair(
                parent,
                [e["sample"] for e in fresh],
                tap,
                device,
                plan,
                with_flow=step >= 4000,
            )
            for i, e in enumerate(fresh):
                e["cached"] = dict(
                    before=slice_observation(b, i, compact=True),
                    after=slice_observation(a, i, compact=True),
                    native=n[i : i + 1].detach().cpu(),
                )
                if step < 4000 and not args.audit:
                    feature_cache.save(e["index"], **e["cached"])
        if step >= 4000 and len(fresh) != len(entries):
            raise ValueError(
                "Flow adapters require live prefix KV, not cached critic features"
            )
        before = concatenate_observations(
            [slice_observation(e["cached"]["before"], 0, device) for e in entries]
        )
        after = concatenate_observations(
            [slice_observation(e["cached"]["after"], 0, device) for e in entries]
        )
        native = torch.cat([e["cached"]["native"] for e in entries]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts, counts = model(before, after, native, step, flow)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite TGCC loss")
        return loss, parts, counts

    if args.audit:
        samples = next(iterator)
        audits = []
        for phase in (0, 1000, 4000):
            # Audits must rerun the live parent path at each phase, not reuse an
            # entry mutated by the preceding dry check.
            entries = [
                dict(index=e["index"], cached=None, sample=e["sample"]) for e in samples
            ]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            loss, parts, counts = one_gradient(phase, entries)
            loss.backward()
            torch.cuda.synchronize()
            grads = {
                k: sum(
                    float(p.grad.float().square().sum())
                    for p in module.parameters()
                    if p.grad is not None
                )
                ** 0.5
                for k, module in [
                    ("critic", system.critic),
                    ("adapters", system.adapters),
                ]
            }
            if any(p.grad is not None for p in parent.parameters()):
                raise AssertionError("Frozen parent acquired gradients")
            if phase == 4000 and float(parts["teacher_consistency"]) != 0.0:
                raise AssertionError("Zero-gate teacher mismatch")
            audits.append(
                dict(
                    phase_new_step=phase,
                    seconds=time.monotonic() - t,
                    peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    loss=float(loss),
                    parts={k: float(v) for k, v in parts.items()},
                    valid_counts=counts,
                    gradient_norms=grads,
                    optimizer_updates=0,
                )
            )
        if rank == 0:
            (output / f"dry_gradient_rank{rank}.json").write_text(
                json.dumps(
                    dict(status="passed", metadata=metadata, audits=audits), indent=2
                )
            )
        else:
            (output / f"dry_gradient_rank{rank}.json").write_text(
                json.dumps(dict(status="passed", audits=audits), indent=2)
            )
        print(
            json.dumps(
                {"rank": rank, "status": "dry_gradient_passed", "audits": audits}
            ),
            flush=True,
        )
        if world > 1:
            dist.destroy_process_group()
        return
    started = time.monotonic()
    while steps < 5000:
        if steps == 4000 and loader.dataset.use_cache:
            del iterator, loader
            loader = make_loader(cursor, False)
            iterator = iter(loader)
        system.train()
        parent.eval()
        optimizer.zero_grad(set_to_none=True)
        for group, lr in zip(optimizer.param_groups, rates(steps, cfg)):
            group["lr"] = lr
        sums = {}
        valids = {}
        start = time.monotonic()
        for micro in range(accum):
            samples = next(iterator)
            sync = (
                model.no_sync()
                if world > 1 and micro + 1 < accum
                else contextlib.nullcontext()
            )
            with sync:
                loss, parts, counts = one_gradient(steps, samples)
                (loss / accum).backward()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) / accum
            for k, v in counts.items():
                valids[k] = valids.get(k, 0) + v
        # Inactive heads must not accumulate Adam state or weight decay merely
        # because their masked loss produces a zero gradient tensor.
        if steps < 1000:
            for p in system.critic.handoff.parameters():
                p.grad = None
        if steps < 4000:
            for p in system.adapters.parameters():
                p.grad = None
        grad = torch.nn.utils.clip_grad_norm_(
            system.parameters(), cfg["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        steps += 1
        cursor += 48
        # Schedule is a pure function of the persisted exact update counter.
        log_update = steps <= 5 or steps % 10 == 0
        if log_update and world > 1:
            names = sorted(sums)
            values = torch.tensor([sums[k] for k in names], device=device)
            dist.all_reduce(values)
            sums = {k: float(v) / world for k, v in zip(names, values)}
            names = sorted(valids)
            values = torch.tensor(
                [valids[k] for k in names], device=device, dtype=torch.long
            )
            dist.all_reduce(values)
            valids = {k: int(v) for k, v in zip(names, values)}
        if rank == 0 and log_update:
            row = dict(
                new_update=steps,
                cumulative_step=5000 + steps,
                phase="critic"
                if steps <= 1000
                else ("critic_handoff" if steps <= 4000 else "critic_adapters"),
                seconds=time.monotonic() - start,
                elapsed_seconds=time.monotonic() - started,
                losses=sums,
                global_valid_counts=valids,
                gradient_norm=float(grad),
                lrs=[g["lr"] for g in optimizer.param_groups],
                sequences=cursor,
                equivalent_passes=cursor / 51714,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            append(output / "train.jsonl", row)
        if steps % 100 == 0:
            if sources(ROOT) != source_identity:
                raise RuntimeError("Source changed during formal training")
            if shutil.disk_usage(output).free < 50 * 2**30:
                raise OSError("Training disk floor")
            if steps in cfg["checkpoint_new_steps"]:
                state_before_validation = rng_state()
                system.eval()
                validation_parts = {}
                validation_counts = {}
                validation_indices = [
                    i for i, r in enumerate(dataset.rows) if r["split"] == "validation"
                ]
                validation_count = 16 * world
                selected = [
                    validation_indices[int(i)]
                    for i in np.linspace(
                        0, len(validation_indices) - 1, validation_count
                    )
                ]
                with torch.no_grad():
                    for vi in selected[rank::world]:
                        sample = data[vi]
                        loss, parts, counts = one_gradient(
                            steps, [dict(index=vi, cached=None, sample=sample)]
                        )
                        for k, v in parts.items():
                            validation_parts[k] = (
                                validation_parts.get(k, 0.0) + float(v) / 16
                            )
                        for k, v in counts.items():
                            validation_counts[k] = validation_counts.get(k, 0) + v
                restore_rng(state_before_validation)
                system.train()
                parent.eval()
                if rank == 0:
                    append(
                        output / "validation.jsonl",
                        dict(
                            new_update=steps,
                            sequences=validation_count,
                            rank0_losses=validation_parts,
                            rank0_valid_counts=validation_counts,
                            optimizer_updates=0,
                            test_instances_used=False,
                        ),
                    )
            save_checkpoint(
                output / "checkpoints", system, optimizer, steps, cursor, metadata, rank
            )
            if rank == 0:
                keep = set(cfg["checkpoint_new_steps"] + [steps, steps - 100])
                for pth in (output / "checkpoints").glob("step*"):
                    if pth.is_dir() and int(pth.name[4:]) not in keep:
                        shutil.rmtree(pth)
    if rank == 0:
        (output / "TRAINING_COMPLETE.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    new_updates=steps,
                    cumulative_steps=10000,
                    sequences=cursor,
                    checkpoint=str(output / "checkpoints/step05000"),
                ),
                indent=2,
            )
        )
    tap.close()
    dataset.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
