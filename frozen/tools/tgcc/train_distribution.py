"""Train the success-only density heads and retrieval-conditioned Action Expert.

Formal updates require a current-source/data readiness artifact. --audit uses
real successful data and complete effective-batch updates in a disposable run;
it does not start or advance the formal 1000+5000 schedule.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import signal
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"), str(ROOT / "code/packages/openpi-client/src")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TGR_GEOMETRY_CACHE_PATH", "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3")

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, register_fsdp_forward_method
from tgr.factory import create, ensure_legacy_imports
ensure_legacy_imports()
from tgr.data import LazySequenceDataset
from tgcc.robot import ActionNormalizer
from tgcc.retrieval_data import RetrievalDataset
from tgcc.retrieval_adapter import RetrievalAdapter
from tgcc.policy_anchor_cache import PolicyAnchorCache
from tgcc.distribution_data import (DistributionTrainingData,
    move_distribution_batch, successful_anchor_order, make_distribution_loader)
from tgcc.distribution_joint import (DistributionJointSystem, distribution_optimizer_groups,
    distribution_learning_rates, effective_distribution_loss)
from tgcc.distribution_checkpoint import save_distribution, resume_distribution
from tgcc.distribution_calibration import distribution_readiness
from tgcc.checkpoint import digest, sources
from tgcc.joint_checkpoint import _named_optimizer
from train import BASE, PARENT, PARENT_SHA


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def gradient_report(model, device):
    result = {key: torch.zeros((), device=device) for key in
              ("critic", "action", "execution", "planning", "retrieval", "action_expert", "frozen")}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        value = local(p.grad).float().square().sum()
        key = "critic" if name.startswith("critic.") else "retrieval" if name.startswith("retrieval.") else "action_expert" if p.requires_grad else "frozen"
        result[key] += value
        for head, prefix in (("action", "critic.action_density"), ("execution", "critic.execution_density"), ("planning", "critic.planning")):
            if name.startswith(prefix):
                result[head] += value
    for key in result:
        dist.all_reduce(result[key])
        result[key] = float(result[key].sqrt())
    return result


def clip(model, device, maximum):
    squared = torch.zeros((), device=device, dtype=torch.float64)
    for p in model.parameters():
        if p.grad is not None:
            squared += local(p.grad).double().square().sum()
    dist.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite success-only distributed gradient")
    scale = (maximum / (norm + 1e-6)).clamp(max=1).float()
    for p in model.parameters():
        if p.grad is not None:
            local(p.grad).mul_(scale)
    return float(norm)


def state_fingerprint(model, optimizer):
    h = hashlib.sha256()
    _, schema, groups = _named_optimizer(model, optimizer)
    h.update(json.dumps(dict(schema=schema, groups=groups), sort_keys=True).encode())
    for name, p in model.named_parameters():
        h.update(name.encode())
        value = local(p).detach().cpu().contiguous()
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        if p in optimizer.state:
            for key, value in sorted(optimizer.state[p].items()):
                h.update(key.encode())
                if isinstance(value, torch.Tensor):
                    h.update(local(value).detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                else:
                    h.update(str(value).encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tgcc/distribution.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/hotdog_lazy_seq2_manifest.jsonl")
    parser.add_argument("--readiness", type=Path, default=ROOT / "runs/distribution_readiness/joint_training_readiness.json")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-steady-updates", type=int, default=1,
        help="Complete consecutive joint updates measured in the disposable audit only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--policy-cache", type=Path,
        help="Complete identity-bound original prepared first-policy anchors; required for formal training")
    args = parser.parse_args()
    if args.audit_steady_updates < 1 or (not args.audit and args.audit_steady_updates != 1):
        raise ValueError("Extra steady updates are allowed only in disposable audits")
    if not args.audit and args.policy_cache is None:
        raise ValueError("Formal training requires the complete prepared first-policy cache")
    cfg = json.loads(args.config.read_text())
    if cfg["new_updates"] != cfg["critic_warmup_updates"] + cfg["joint_updates"]:
        raise ValueError("Warmup/joint/global update counters disagree")
    if cfg.get("checkpoint_every", 250) < 1 or cfg.get("keep_last_checkpoints", 2) < 1:
        raise ValueError("At least one recent complete checkpoint must be retained")
    rank, world, gpu = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    effective = cfg["effective_batch"]
    if not 1 <= world <= 4 or args.micro_batch < 1 or effective % (world * args.micro_batch):
        raise ValueError("At most four GPUs and an exact effective batch are required")
    device = torch.device("cuda", gpu)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    source = sources(ROOT)
    library = RetrievalDataset(args.feature_library)
    ident = library.identity
    if ident.get("temporary_incomplete") and not args.audit:
        raise ValueError("Incomplete smoke support is allowed only for disposable GPU audits")
    if digest(Path(PARENT) / "model.safetensors") != PARENT_SHA:
        raise ValueError("Original tested step5000 identity changed")
    parent_md = json.loads((Path(PARENT) / "metadata.json").read_text())
    for name, sha in parent_md["implementation_source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Original tested implementation changed: " + name)
    normalizer = ActionNormalizer(BASE)
    raw = LazySequenceDataset(args.manifest, BASE, ROOT / "reports/data_plan_vocabulary.json")
    if (digest(args.manifest) != ident["demonstrations_manifest_sha256"] or
            raw.adapter.normalization_sha256 != ident["normalization_sha256"]):
        raise ValueError("Owned demonstration/normalizer feature identity changed")
    policy_cache = (PolicyAnchorCache(args.policy_cache, library, raw, normalizer)
                    if args.policy_cache is not None else None)
    dimensions = dict(context_width=ident["context_width"], after_width=ident["after_width"],
        planning_classes=ident["planning_classes"], hidden=cfg["hidden"], horizon=32, action_width=23,
        action_mixtures=cfg["action_mixtures"], execution_mixtures=cfg["execution_mixtures"],
        minimum_scale=cfg["minimum_scale"], maximum_scale=cfg["maximum_scale"])
    _, student = create(BASE, "tri_graph", "cpu", PARENT)
    student.eval()
    policy_export_dtypes = {name: str(value.dtype) for name, value in student.state_dict().items()}
    adapter = RetrievalAdapter(expert_width=student.base.action_in_proj.out_features,
        context_width=ident["context_width"], after_width=ident["after_width"])
    model = DistributionJointSystem(student, adapter, dimensions, cfg)
    # Keep frozen parent weights in their official inference dtype. Only Adam
    # parameters need FP32 master weights; replicating the frozen parent avoids
    # gathering several GB during every critic-only warmup/microbatch.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    model.to(device)
    metadata = dict(schema="tgcc.distribution_joint.v1", config=cfg, source_sha256=source,
        parent_checkpoint=PARENT, parent_model_sha256=PARENT_SHA, parent_aliases=parent_md.get("aliases", {}),
        encoder_checkpoint=PARENT, distribution_config=dimensions, retrieval_adapter_config=adapter.record(),
        policy_export_dtypes=policy_export_dtypes,
        world_size=world, micro_batch=args.micro_batch, grad_accum_steps=effective // (world * args.micro_batch),
        plan_sha256=digest(raw.plan_path), feature_library_sha256=digest(args.feature_library / "identity.json"),
        feature_library=str(args.feature_library.resolve()), demonstration_manifest=str(args.manifest.resolve()),
        policy_cache=str(args.policy_cache.resolve()) if args.policy_cache else None,
        policy_cache_identity_sha256=policy_cache.identity_sha256 if policy_cache else None,
        audit_disposable=args.audit,
        feature_artifact_sha256=ident["artifact_sha256"],
        training_unit="one32step_prefix_plus_actual_after_exposure",
        training_anchors=len(library.train_indices), validation_anchors=len(library.validation_indices),
        policy_trainable_names=model.policy_trainable_names,
        sampler="shuffled_episode_groups_each_pass_chronological_anchors_within_episode",
        warmup_loader="immutable_feature_arrays_only_no_RGB_no_policy_prefix_no_retrieval_query",
        rng_policy="identical_model_initialization_then_independent_seed_plus_rank_training_streams",
        fsdp_layout="separate_critic_retrieval_and_velocity_only_student; frozen_parent_replicated_official_dtype",
        frozen_modules=["vision", "VLM", "three_graphs", "semantic_decoder", "retrieval_index_encoder"],
        labels="original successful actions, actual after features, annotated subtask selection",
        failed_or_synthetic_targets_used=False, task_success_probability=False, failure_classifier_trained=False)
    for key in ("feature_codec_sha256", "retrieval_index_sha256", "demonstrations_manifest_sha256", "normalization_sha256", "encoder_model_sha256"):
        metadata[key] = ident[key]
    data_gate = distribution_readiness(metadata, ident)
    if data_gate["status"] != "passed":
        raise ValueError("Data readiness failed: " + "; ".join(data_gate["failures"]))
    if not args.audit:
        gate = json.loads(args.readiness.read_text())
        if (gate["status"] != "passed" or gate["source_sha256"] != source or
                gate["feature_library_sha256"] != metadata["feature_library_sha256"] or
                any(g["status"] != "passed" for g in gate["gates"].values())):
            raise ValueError("Formal training requires current successful source/data/GPU gates")
    mesh = init_device_mesh("cuda", (world,))
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    frozen = {p for p in model.student.parameters() if not p.requires_grad}
    fully_shard(model.critic, mesh=mesh, mp_policy=precision)
    fully_shard(model.retrieval, mesh=mesh, mp_policy=precision)
    fully_shard(model.student, mesh=mesh, mp_policy=precision, ignored_params=frozen)
    # Prefix directly touches only frozen/replicated modules. The velocity
    # method is the sole Action Expert route and must execute FSDP hooks even
    # though it is called as a named method rather than student(...).
    register_fsdp_forward_method(model.student, "velocity")
    fully_shard(model, mesh=mesh, mp_policy=precision, ignored_params=frozen)
    opt = torch.optim.AdamW(distribution_optimizer_groups(model, cfg), betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"], foreach=False, fused=False)
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    random.seed(cfg["seed"] + rank)
    step, cursor = resume_distribution(args.resume, model, opt, metadata, rank) if args.resume else (0, 0)
    order = successful_anchor_order(library.rows, cfg["new_updates"] * effective, cfg["seed"])
    consumed = cursor
    loader_phase = step >= cfg["critic_warmup_updates"]
    def make_loader(position, include_policy):
        phase_data = DistributionTrainingData(library, raw, normalizer, include_policy=include_policy,
            policy_cache=policy_cache)
        return make_distribution_loader(phase_data, order, position, rank, world,
            args.micro_batch, args.workers if include_policy else 0, cfg["seed"])
    loader, iterator = make_loader(cursor, loader_phase)
    if rank == 0:
        (args.output / "run_identity.json").write_text(json.dumps(metadata, indent=2))

    def update(phase):
        nonlocal loader, iterator, loader_phase, consumed
        include_policy = phase >= cfg["critic_warmup_updates"]
        if include_policy != loader_phase:
            # Restore precisely the next global training unit, independent of
            # old worker prefetch. Iterator construction cannot consume model RNG.
            iterator = None
            loader = None
            loader, iterator = make_loader(consumed, include_policy)
            loader_phase = include_policy
        model.train()
        opt.zero_grad(set_to_none=True)
        for group, lr in zip(opt.param_groups, distribution_learning_rates(phase, cfg)):
            group["lr"] = lr
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        sums, counts = {}, {}
        batches = [next(iterator) for _ in range(metadata["grad_accum_steps"])]
        heads = ("action", "execution", "planning")
        denominator = torch.tensor([sum(int(batch["density"].get(head + "_valid",
            torch.ones(len(batch["density"]["context"]), dtype=torch.bool)).sum())
            for batch in batches) for head in heads], dtype=torch.long, device=device)
        # Exactly the same collective exists on every rank, including a rank
        # whose planning labels are all UNKNOWN. Targets remain masked locally.
        dist.all_reduce(denominator)
        global_counts = {head: int(value) for head, value in zip(heads, denominator)}
        batch_support_ids = []
        for cpu_batch in batches:
            batch_support_ids.extend(cpu_batch["support_ids"])
            batch = move_distribution_batch(cpu_batch, device)
            consumed += world * args.micro_batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, parts, valid = model(batch, phase)
                loss = effective_distribution_loss(parts, valid, global_counts, world,
                    metadata["grad_accum_steps"], cfg["loss_weights"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite density/flow training loss")
            loss.backward()
            for key, value in parts.items():
                weight = valid[key] * world / max(global_counts[key], 1) if key in heads else 1 / metadata["grad_accum_steps"]
                sums[key] = sums.get(key, 0.) + float(value.detach()) * weight
            for key, value in valid.items():
                counts[key] = counts.get(key, 0) + value
        if phase < cfg["critic_warmup_updates"]:
            for module in (model.student, model.retrieval):
                for p in module.parameters():
                    p.grad = None
        gradients = gradient_report(model, device) if args.audit else None
        norm = clip(model, device, cfg["grad_clip"])
        opt.step()
        loss_keys = sorted(sums)
        values = torch.tensor([sums[k] for k in loss_keys], device=device)
        dist.all_reduce(values)
        sums = {k: float(value) / world for k, value in zip(loss_keys, values)}
        count_keys = sorted(counts)
        values = torch.tensor([counts[k] for k in count_keys], device=device, dtype=torch.long)
        dist.all_reduce(values)
        counts = {k: int(value) for k, value in zip(count_keys, values)}
        torch.cuda.synchronize()
        return dict(phase_new_step=phase, seconds=time.monotonic() - started,
            effective_prefix_batch=effective, micro_batch=args.micro_batch,
            grad_accum_steps=metadata["grad_accum_steps"], losses=sums, global_valid_counts=counts,
            loss_aggregation="global effective-batch means of known targets; FSDP world average compensated",
            local_support_ids=batch_support_ids if args.audit else None,
            gradients=gradients, gradient_norm=norm, peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)

    if args.audit:
        records = []
        phases = [0, cfg["critic_warmup_updates"]] + list(range(
            cfg["critic_warmup_updates"] + 1,
            cfg["critic_warmup_updates"] + 1 + args.audit_steady_updates))
        for phase in phases:
            if phase == cfg["critic_warmup_updates"]:
                # A disposable audit skips 999 critic updates, but positions its
                # sampler at the true joint-phase cursor. Never label a joint
                # checkpoint as warmup step3 merely because only 3 calls ran.
                consumed = phase * effective
                loader, iterator = make_loader(consumed, True)
                loader_phase = True
            report = update(phase)
            g = report["gradients"]
            if g["frozen"] != 0 or any(g[k] <= 0 for k in ("critic", "action", "execution", "planning")):
                raise AssertionError("Frozen or missing density gradient: " + str(g))
            if phase == 0 and (g["action_expert"] or g["retrieval"]):
                raise AssertionError("Warmup changed Action Expert/adapter")
            if phase >= cfg["critic_warmup_updates"] and (g["action_expert"] <= 0 or g["retrieval"] <= 0):
                raise AssertionError("Missing real Action Expert/retrieval gradient")
            records.append(report)
            if rank == 0:
                print(json.dumps(dict(stage="disposable_audit_update", **report)), flush=True)
        before = state_fingerprint(model, opt)
        logical_step = cfg["critic_warmup_updates"] + 1 + args.audit_steady_updates
        if consumed != logical_step * effective:
            raise AssertionError("Audit logical data cursor does not match joint phase")
        checkpoint = save_distribution(args.output / "audit_checkpoint", model, opt, logical_step, metadata, rank)
        if rank == 0:
            print(json.dumps(dict(stage="disposable_audit_checkpoint_saved", checkpoint=str(checkpoint))), flush=True)
        draws = (random.random(), np.random.rand(), torch.rand(3, device=device))
        with torch.no_grad():
            for p in model.critic.parameters():
                local(p).add_(.25)
                break
        resumed_step, resumed_cursor = resume_distribution(checkpoint, model, opt, metadata, rank)
        restored = (random.random(), np.random.rand(), torch.rand(3, device=device))
        if before != state_fingerprint(model, opt) or draws[:2] != restored[:2] or not torch.equal(draws[2], restored[2]):
            raise AssertionError("Actual sharded optimizer/RNG checkpoint roundtrip mismatch")
        if (resumed_step, resumed_cursor) != (logical_step, consumed):
            raise AssertionError("Actual checkpoint phase/cursor roundtrip mismatch")
        # The first continuation and the restored continuation consume the
        # same complete global batch, including rank RNG/dropout/noise, rather
        # than merely comparing the state immediately after DCP loading.
        resume_distribution(checkpoint, model, opt, metadata, rank)
        loader, iterator = make_loader(resumed_cursor, True)
        consumed = resumed_cursor
        uninterrupted = update(logical_step)
        continued_fingerprint = state_fingerprint(model, opt)
        from tgcc.checkpoint import rng_state
        continued_rng = rng_state()
        resume_distribution(checkpoint, model, opt, metadata, rank)
        loader, iterator = make_loader(resumed_cursor, True)
        consumed = resumed_cursor
        restarted = update(logical_step)
        if (continued_fingerprint != state_fingerprint(model, opt) or
                uninterrupted["local_support_ids"] != restarted["local_support_ids"] or
                uninterrupted["losses"] != restarted["losses"]):
            raise AssertionError("Restored complete next-update model/Adam/data differs")
        final_rng = rng_state()
        if (continued_rng["python"] != final_rng["python"] or
                continued_rng["numpy"][0] != final_rng["numpy"][0] or
                not np.array_equal(continued_rng["numpy"][1], final_rng["numpy"][1]) or
                continued_rng["numpy"][2:] != final_rng["numpy"][2:] or
                not torch.equal(continued_rng["torch"], final_rng["torch"]) or
                not torch.equal(continued_rng["cuda"], final_rng["cuda"])):
            raise AssertionError("Restored complete next-update RNG differs")
        records.extend((uninterrupted, restarted))
        result = dict(status="passed", source_sha256=source,
            feature_library_sha256=metadata["feature_library_sha256"], audits=records,
            actual_optimizer_rng_roundtrip=True, actual_next_complete_update_exact=True,
            actual_logical_phase_cursor_roundtrip=True, checkpoint_logical_step=logical_step,
            skipped_disposable_schedule_updates=cfg["critic_warmup_updates"] - 1,
            formal_optimizer_updates=0, disposable_optimizer_update_calls=len(phases) + 2,
            steady_joint_update_count=args.audit_steady_updates)
        (args.output / f"distribution_audit_rank{rank}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        # Stop video/prefetch workers while Python and the CUDA process group
        # still have a normal lifetime. Interpreter shutdown is too late for
        # native decoder/worker threads, even after a successful audit report.
        iterator = None
        loader = None
        raw.close()
        dist.barrier()
        dist.destroy_process_group()
        return
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    started = time.monotonic()
    while step < cfg["new_updates"]:
        report = update(step)
        step += 1
        cursor += effective
        if step % 10 == 0 or step in (1, cfg["critic_warmup_updates"], cfg["critic_warmup_updates"] + 1):
            report.update(new_update=step, policy_new_updates=max(0, step-cfg["critic_warmup_updates"]),
                sampled_prefixes=cursor, equivalent_anchor_passes=cursor / len(library.train_indices),
                elapsed_seconds=time.monotonic()-started)
            if rank == 0:
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
                print(json.dumps(report), flush=True)
        stop = torch.tensor(int(stopped[0]), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if step % cfg.get("checkpoint_every", 250) == 0 or step == cfg["new_updates"] or bool(stop):
            if sources(ROOT) != source:
                raise ValueError("Frozen source changed during formal training")
            save_distribution(args.output / "checkpoints", model, opt, step, metadata, rank,
                              export=step == cfg["new_updates"])
            if rank == 0:
                root = args.output / "checkpoints"
                checkpoints = sorted(p for p in root.glob("step[0-9][0-9][0-9][0-9][0-9]") if p.is_dir() and not p.is_symlink())
                for old in checkpoints[:-cfg.get("keep_last_checkpoints", 2)]:
                    owned = json.loads((old / "metadata.json").read_text())
                    if (owned.get("schema") != "tgcc.distribution_joint.v1" or
                            owned.get("source_sha256") != source or
                            owned.get("feature_library_sha256") != metadata["feature_library_sha256"]):
                        raise ValueError("Refusing to prune a foreign checkpoint")
                    shutil.rmtree(old)
            dist.barrier()
        if bool(stop):
            break
    if rank == 0 and step == cfg["new_updates"]:
        (args.output / "TRAINING_COMPLETE.json").write_text(json.dumps(dict(new_updates=step,
            policy_new_updates=cfg["joint_updates"], policy_cumulative_updates=5000+cfg["joint_updates"],
            elapsed_seconds=time.monotonic()-started), indent=2))
    iterator = None
    loader = None
    raw.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
