"""Full TC-ESGR V2 / pi fine-tuning with FSDP, exact sampler state, and real data."""

import argparse, json, os, random, signal, time, math, shutil, socket, hashlib, threading, traceback
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
TRAIN_INFRA_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TRAIN_CACHE_ENV = {
    "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "cache/torch_inductor"),
    "TRITON_CACHE_DIR": str(ROOT / "cache/triton"),
    "HF_HOME": str(ROOT / "cache/huggingface"),
    "HF_HUB_CACHE": str(ROOT / "cache/huggingface/hub"),
}
os.environ.update(TRAIN_CACHE_ENV)

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions,
)
from .encoders import GraphBatch
from .model import Conditions
from .factory import create
from .checkpoint import digest, save_inference
from .training import full_parameter_groups, objective_vector
from .data import LazySequenceDataset, collate_sequences
from .prefetch import CPUPrefetch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class GlobalSampler:
    """One deterministic global stream; never drop a tail when crossing epochs."""

    def __init__(self, indices, seed):
        self.indices = list(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("sampler requires nonempty unique training indices")
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.order = []
        self.indices_sha256 = hashlib.sha256(json.dumps(self.indices).encode()).hexdigest()
        self.last_take_audit = {}
        self.shuffle()

    def shuffle(self):
        self.order = self.indices.copy()
        random.Random(self.seed + self.epoch).shuffle(self.order)

    @property
    def consumed_samples(self):
        return self.epoch * len(self.indices) + self.position

    def take(self, n):
        if n <= 0:
            raise ValueError("sample count must be positive")
        result, epochs = [], []
        before = self.state()
        while len(result) < n:
            count = min(n - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position:self.position + count])
            epochs.extend([self.epoch] * count)
            self.position += count
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self.shuffle()
        duplicates = sorted({idx for idx in result if result.count(idx) > 1})
        self.last_take_audit = {
            "start": before, "end": self.state(), "sample_epochs": epochs,
            "crossed_epoch": len(set(epochs)) > 1,
            "duplicate_indices_within_batch": duplicates,
            "duplicate_count_within_batch": len(result) - len(set(result)),
            "same_epoch_duplicates": len(set(zip(epochs, result))) != len(result),
        }
        if self.last_take_audit["same_epoch_duplicates"]:
            raise AssertionError("a training sequence was repeated within the same epoch")
        return result

    def state(self):
        return dict(version=2, epoch=self.epoch, position=self.position, seed=self.seed,
                    dataset_size=len(self.indices), indices_sha256=self.indices_sha256,
                    consumed_samples=self.consumed_samples, completed_epochs=self.epoch)

    def restore(self, state):
        if state.get("version") != 2:
            raise ValueError("legacy tail-dropping sampler state cannot claim full coverage")
        if state["seed"] != self.seed or state["dataset_size"] != len(self.indices) or state["indices_sha256"] != self.indices_sha256:
            raise ValueError("sampler seed or training index identity mismatch")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        if self.epoch < 0 or not 0 <= self.position < len(self.indices):
            raise ValueError("invalid sampler position")
        if state["consumed_samples"] != self.consumed_samples or state["completed_epochs"] != self.epoch:
            raise ValueError("sampler coverage accounting mismatch")
        self.shuffle()


def training_budget(dataset_size, global_batch, minimum_epochs):
    if dataset_size <= 0 or global_batch not in (12, 24, 48) or minimum_epochs < 5:
        raise ValueError("formal budget requires a nonempty dataset, batch 12/24/48, and >=5 epochs")
    samples = dataset_size * minimum_epochs
    updates = math.ceil(samples / global_batch)
    return dict(required_epochs=minimum_epochs, train_sequences=dataset_size,
                required_samples=samples, required_optimizer_steps=updates,
                scheduled_samples=updates * global_batch,
                final_batch_overrun_samples=updates * global_batch - samples)


def resume_batch_transition(metadata, phase, global_batch, micro_batch, world_size):
    """Only preformal checkpoints may change the global batch on resume."""
    if global_batch not in (12, 24, 48) or global_batch % (micro_batch * world_size):
        raise ValueError("invalid resumed batch partition")
    if metadata["phase"] == "formal":
        if phase != "formal":
            raise ValueError("cannot turn formal training back into preformal phase")
        if metadata["global_batch"] != global_batch:
            raise ValueError("global batch cannot change after formal coverage begins")
    old = (metadata["global_batch"], metadata["micro_batch"], metadata["world_size"])
    new = (global_batch, micro_batch, world_size)
    if old == new and metadata["phase"] == phase:
        return None
    return dict(from_phase=metadata["phase"], to_phase=phase,
                from_global_batch=old[0], to_global_batch=new[0],
                from_micro_batch=old[1], to_micro_batch=new[1],
                from_world_size=old[2], to_world_size=new[2],
                optimizer_steps=metadata["optimizer_steps"],
                sampler_samples_before=metadata["sampler"]["consumed_samples"],
                formal_sampler_reset=phase == "formal" and metadata["phase"] != "formal",
                preservation="model/optimizer/existing-rank RNG restored; new-rank RNG deterministic",
                budget_scope="preformal samples never count toward formal epochs")


def audit_formal_resume(metadata, global_batch):
    """Reject counter drift or accidental inclusion of preformal samples."""
    if metadata["phase"] != "formal":
        return
    updates = metadata["optimizer_steps"] - metadata["formal_start_optimizer_steps"]
    samples = metadata["sampler"]["consumed_samples"]
    if updates != metadata["formal_optimizer_steps"] or samples != updates * global_batch or samples != metadata["formal_consumed_samples"]:
        raise ValueError("formal optimizer/sampler coverage counters disagree")
    if metadata.get("training_complete") and samples < metadata["formal_budget"]["required_samples"]:
        raise ValueError("formal completion predates the minimum five-epoch sample budget")


class StageHeartbeat:
    """Native progress timer, independent of optimizer log writes and GPU work."""

    def __init__(self, output, rank):
        self.output, self.rank = output, rank
        self.lock, self.done = threading.Lock(), threading.Event()
        self.value = dict(stage="distributed_initialization", stage_started_at=time.time(),
                          last_progress_at=time.time(), optimizer_step=0, pid=os.getpid(),
                          training_infra_sha256=TRAIN_INFRA_SHA256)
        if rank == 0:
            threading.Thread(target=self._run, daemon=True).start()

    def update(self, stage, **fields):
        if self.rank:
            return
        with self.lock:
            now = time.time()
            if self.value["stage"] != stage:
                self.value["stage_started_at"] = now
            self.value.update(stage=stage, last_progress_at=now, **fields)
            atomic_json(self.output / "heartbeat.json", dict(self.value, time=now))

    def _run(self):
        while not self.done.wait(15):
            with self.lock:
                atomic_json(self.output / "heartbeat.json", dict(self.value, time=time.time()))

    def close(self):
        self.done.set()


ACTIVE_OUTPUT = None
ACTIVE_HEARTBEAT = None

def heldout_indices(rows, limit=128):
    groups = {}
    for i, row in enumerate(rows):
        if row["split"] == "validation":
            groups.setdefault(row["episode_id"], []).append(i)
    count = math.ceil(limit / len(groups))
    picked = []
    for layer in range(count):
        for values in groups.values():
            j = round(layer * (len(values) - 1) / max(count - 1, 1))
            picked.append(values[j])
    return picked[:limit]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


def batch_loss(model, batch, validation=False):
    from openpi.models.model import Observation

    memory = None
    total = None
    parts_sum = {}
    decisions = []
    previous_time = None
    items = batch["sequence"]
    for i, item in enumerate(items):
        c = item["conditions"]
        condition = Conditions(
            **{
                key: GraphBatch(**c[key])
                for key in ("global_graph", "local_graph", "task_graph")
            },
            task_condition=c["task_condition"],
            queries=c["queries"],
            legal=c["legal"],
            memory=memory,
            candidate_mask=c.get("candidate_mask"),
            event_changed=c.get("event_changed"),
            task_candidate_slots=c.get("task_candidate_slots"),
            oracle=True,
            restricted=False
        )
        times = item["timestamps"] if "timestamps" in item else [item["timestamp"]]
        current_time = torch.tensor(
            times, dtype=torch.float64, device=item["actions"].device
        )
        if decisions:
            from .task_feedback import roll_forward_task_graph

            condition.task_graph = roll_forward_task_graph(
                condition.task_graph,
                decisions[-1],
                condition.task_candidate_slots,
                current_time - previous_time,
            )
        previous_time = current_time
        result = model(
            Observation.from_dict(item["observation"]),
            item["actions"],
            conditions=condition,
            validation=validation,
        )
        value, parts = objective_vector(result, item["labels"])
        total = value if total is None else total + value
        for key, v in parts.items():
            parts_sum[key] = parts_sum.get(key, 0) + v
        memory = result["decision"]["memory"]
        decisions.append(result["decision"])
    return (
        total / len(items),
        {k: v / len(items) for k, v in parts_sum.items()},
        decisions,
    )


def initialize_optimizer(opt):
    # Allocate resumable optimizer slots without changing parameters or counting an update.
    rates = [group["lr"] for group in opt.param_groups]
    for group in opt.param_groups:
        group["lr"] = 0.0
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    for group, lr in zip(opt.param_groups, rates):
        group["lr"] = lr
    for state in opt.state.values():
        if "step" in state:
            state["step"].zero_()


def grad_audit(model, device):
    groups = {
        name: torch.zeros((), device=device)
        for name in ("vlm", "siglip", "expert", "action_io", "graph", "task_memory")
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gradient = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
        value = gradient.float().abs().sum()
        if "vision_tower" in name:
            groups["siglip"] += value
        elif "gemma_expert" in name:
            groups["expert"] += value
        elif "base.paligemma" in name:
            groups["vlm"] += value
        elif name.startswith("base.") or ".base." in name:
            groups["action_io"] += value
        else:
            groups["graph"] += value
        if name.startswith("decoder.memory") or ".decoder.memory" in name:
            groups["task_memory"] += value
    for value in groups.values():
        dist.all_reduce(value)
    return {k: float(v) for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--steps", type=int, help="absolute cumulative optimizer target; formal default derives >=5 epoch budget")
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--implementation-identity", type=Path)
    parser.add_argument("--checkpoint-free-gib", type=float, default=50)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--global-batch", type=int, default=48, choices=(12, 24, 48))
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--graph-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=200)
    parser.add_argument("--validation-sequences", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume-checkpoint-after", type=int, default=0, help="save an additional resume checkpoint after N real resumed updates")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-audit-every", type=int, default=100)
    parser.add_argument("--phase", default="formal")
    parser.add_argument("--control", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    a = parser.parse_args()
    import importlib.util
    atomic_spec = importlib.util.spec_from_file_location("tgr_atomic_training_checkpoint", ROOT / "scripts/atomic_training_checkpoint.py")
    atomic_helper = importlib.util.module_from_spec(atomic_spec)
    atomic_spec.loader.exec_module(atomic_helper)
    atomic_helper.assert_launch_state(a.output, a.resume)
    space_spec = importlib.util.spec_from_file_location("tgr_checkpoint_space_budget", ROOT / "scripts/checkpoint_space_budget.py")
    space_helper = importlib.util.module_from_spec(space_spec)
    space_spec.loader.exec_module(space_helper)
    storage_spec = importlib.util.spec_from_file_location("tgr_lossless_dcp_storage", ROOT / "scripts/lossless_dcp_storage.py")
    storage_helper = importlib.util.module_from_spec(storage_spec)
    storage_spec.loader.exec_module(storage_helper)
    selection_spec = importlib.util.spec_from_file_location("tgr_validation_checkpoint_selection", ROOT / "scripts/validation_checkpoint_selection.py")
    selection_helper = importlib.util.module_from_spec(selection_spec)
    selection_spec.loader.exec_module(selection_helper)
    cache_spec = importlib.util.spec_from_file_location("tgr_geometry_cache_launch_env", ROOT / "scripts/geometry_cache_launch_env.py")
    cache_helper = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache_helper)
    selected_cache_env = cache_helper.selected_cache_environment(a.output, required=a.phase == "formal")
    os.environ.update(selected_cache_env)
    global ACTIVE_OUTPUT, ACTIVE_HEARTBEAT
    ACTIVE_OUTPUT = a.output
    identity_path = a.implementation_identity or a.output / "implementation_identity.json"
    implementation = json.loads(identity_path.read_text())
    revision_keys = ("architecture_version", "feature_schema", "reachability_revision", "task_feedback_revision")
    if any(not implementation.get(key) for key in revision_keys):
        raise ValueError("complete implementation identity required before training")
    if a.phase == "formal" and (a.global_batch != 48 or a.micro_batch not in (2, 4)):
        raise ValueError("this formal run requires global_batch48 and micro_batch2 or4")
    if a.phase == "formal" and a.seed != 42:
        raise ValueError("formal training seed must be 42")
    if a.phase != "formal" and a.steps is None:
        raise ValueError("preformal phase requires an explicit absolute --steps target")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if not 1 <= world <= 4 or a.global_batch % (a.micro_batch * world):
        raise ValueError(
            "global_batch must equal micro_batch * world * integer accumulation"
        )
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=True)
    for cache_path in TRAIN_CACHE_ENV.values():
        Path(cache_path).mkdir(parents=True, exist_ok=True)
    heartbeat = StageHeartbeat(a.output, rank)
    ACTIVE_HEARTBEAT = heartbeat
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    heartbeat.update("loading_model", phase=a.phase)
    policy, raw = create(a.checkpoint, "tc_esgr", "cpu")
    original_dtypes = {k: v.dtype for k, v in raw.state_dict().items()}
    aliases = {}
    seen = {}
    for name, p in raw.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            aliases[name] = seen[id(p)]
        else:
            seen[id(p)] = name
    raw.requires_grad_(True)
    raw.train()
    raw.to(torch.float32).to(device)
    groups, names = full_parameter_groups(raw, a.base_lr, a.graph_lr)
    parameter_count = sum(p.numel() for p in raw.parameters())
    mesh = init_device_mesh("cuda", (world,))
    fully_shard(
        raw,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        ),
    )
    model = raw
    groups, names = full_parameter_groups(model, a.base_lr, a.graph_lr)
    opt = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False, fused=False)
    heartbeat.update("initializing_optimizer")
    initialize_optimizer(opt)
    from .input_adapter import OfficialInputAdapter

    adapter = OfficialInputAdapter(
        a.checkpoint,
        proprio_layout_sha256=digest(ROOT / "reports/data_proprio_layout.json"),
        action_layout_sha256=digest(ROOT / "reports/data_action_layout.json"),
        policy=policy,
    )
    heartbeat.update("initializing_data")
    data = LazySequenceDataset(a.manifest, a.checkpoint, a.plan, adapter)
    valdata = LazySequenceDataset(a.manifest, a.checkpoint, a.plan, adapter)
    train_indices = [i for i, row in enumerate(data.rows) if row["split"] == "train"]
    sampler = GlobalSampler(train_indices, a.seed)
    budget = training_budget(len(train_indices), a.global_batch, a.minimum_epochs)
    val_indices = heldout_indices(data.rows, a.validation_sequences)

    cpu_pool = None
    cpu_pipeline_source_sha256 = None
    cpu_pipeline_parent_rng_unchanged = None
    if a.cpu_workers:
        if a.micro_batch > 4 or not selected_cache_env:
            raise ValueError("CPU spawn pool requires micro<=4 and validated production geometry cache")
        from .cpu_loader import ParallelCPUSequenceLoader
        before_cpu_rng = torch.get_rng_state()
        before_cuda_rng = torch.cuda.get_rng_state(device)
        before_python_rng = random.getstate()
        cpu_pool = ParallelCPUSequenceLoader(a.manifest, a.checkpoint, a.plan, selected_cache_env,
            workers=a.cpu_workers, max_batch=4, max_worker_rss_gib=6)
        cpu_pipeline_parent_rng_unchanged = dict(
            torch_cpu=torch.equal(before_cpu_rng, torch.get_rng_state()),
            torch_cuda=torch.equal(before_cuda_rng, torch.cuda.get_rng_state(device)),
            python=before_python_rng == random.getstate())
        if not all(cpu_pipeline_parent_rng_unchanged.values()):
            cpu_pool.close()
            raise RuntimeError("CPU pipeline constructor changed parent RNG")
        cpu_pipeline_source_sha256 = digest(ROOT / "src/tgr/cpu_loader.py")

    def load(indices, source=data):
        if source is data and cpu_pool is not None:
            batch = cpu_pool.load(indices)
            if batch.get("sample_rows") != [source.rows[index] for index in indices]:
                raise ValueError("CPU pipeline returned a different sample order/source row")
            return batch
        return collate_sequences([source[i] for i in indices])

    queue = CPUPrefetch(load)
    norms = list(Path(a.checkpoint).glob("assets/**/norm_stats.json"))
    external_sources = [ROOT / "reports/asset_bounds_readonly_audit.json", ROOT / "reports/robot_gripper_proxy_source.json"]
    required_modules = [ROOT / "src/tgr/geometry_data.py", ROOT / "src/tgr/approx_reachability.py"]
    if any(not path.is_file() for path in external_sources + required_modules):
        raise ValueError("native bbox/reachability modules and both geometry catalogs must exist")
    identity = {
        **implementation,  # Metadata passthrough; existing whole-identity SHA remains the gate.
        **{key: implementation[key] for key in revision_keys},
        "training_infra_sha256": TRAIN_INFRA_SHA256,
        "implementation_identity_sha256": digest(identity_path),
        "implementation_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src/tgr").glob("*.py")) if path.name not in ("train.py", "prefetch.py", "checkpoint.py", "serve.py", "transport.py", "evaluate.py", "cpu_loader.py")} | {str(path.relative_to(ROOT)): digest(path) for path in external_sources},
        "training_cache_environment": TRAIN_CACHE_ENV,
        "geometry_cache_environment": selected_cache_env,
        "cpu_workers": a.cpu_workers,
        "cpu_pipeline_source_sha256": cpu_pipeline_source_sha256,
        "cpu_pipeline_start_method": "spawn" if cpu_pool else "serial",
        "cpu_pipeline_initialization": cpu_pool.initialization if cpu_pool else [],
        "cpu_pipeline_cold_init_seconds": cpu_pool.cold_init_seconds if cpu_pool else 0.0,
        "cpu_pipeline_parent_rng_unchanged": cpu_pipeline_parent_rng_unchanged,
        "cpu_pipeline_cpu_only_proof": "independent worker initializer CUDA_VISIBLE_DEVICES empty and CUDA uninitialized; every returned Tensor CPU",
        "cpu_pipeline_bounds": dict(workers=a.cpu_workers, max_pending_batches=1, max_samples_per_micro=4, max_worker_rss_gib=6),
        "input_transform_source": "original_checkpoint_owned_readonly_native_transforms",
        "geometry_catalog_paths": [str(path.relative_to(ROOT)) for path in external_sources],
        "task_manager_trainable_parameters": 0,
        "task_status_classifier_present": False,
        "training_objective": "native_flow_plus_0.01_current_skill_cross_entropy",
        "state_goal_loss_fields": "legacy_zero_not_trained",
        "next_macro_f1_semantics": "current_annotated_skill_candidate_selection",
        "training_scope": "full_backbone_and_graph",
        "base_frozen": False,
        "expert_lora": False,
        "normalization_sha256": digest(norms[0]),
        "base_origin_sha256": digest(Path(a.checkpoint) / "model.safetensors"),
        "manifest_sha256": digest(a.manifest),
        "plan_sha256": digest(a.plan),
        "condition_schema": "tgr_static_skill_candidates_v1",
        "runtime_factory": "tgr.runtime_bridge:make_eval_runtime",
        "oracle_upper_bound": True,
        "graph_source": "oracle",
        "global_batch": a.global_batch,
        "micro_batch": a.micro_batch,
        "world_size": world,
        "grad_accum_steps": a.global_batch // (a.micro_batch * world),
        "base_lr": a.base_lr,
        "graph_lr": a.graph_lr,
        "trainable_parameters": parameter_count,
        "trainable_parameter_names": names,
        "train_episodes": sorted({data.rows[i]["episode_id"] for i in train_indices}),
        "validation_episodes": sorted(
            {data.rows[i]["episode_id"] for i in val_indices}
        ),
        "test_instances_used": False,
        "phase": a.phase,
        "validation_indices": val_indices,
        "formal_budget": budget,
        "sampler_version": 2,
        "checkpoint_free_gib": max(50, a.checkpoint_free_gib),
        "base_checkpoint": a.checkpoint,
        "aliases": aliases,
    }
    step = 0
    best = float("inf")
    best_eligible = False
    initial_flow = None
    stale = 0
    formal_start = 0 if a.phase == "formal" else None
    preformal_samples = 0
    phase_start = 0
    transition = None
    state_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    if a.resume:
        heartbeat.update("loading_resume")
        meta = json.loads((a.resume / "metadata.json").read_text())
        transition = resume_batch_transition(meta, a.phase, a.global_batch, a.micro_batch, world)
        for key in (
            "manifest_sha256",
            "plan_sha256",
            "architecture_version",
            "normalization_sha256",
            "feature_schema",
            "reachability_revision",
            "task_feedback_revision",
            "implementation_identity_sha256",
            "implementation_source_sha256",
        ):
            if meta[key] != identity[key]:
                raise ValueError("resume identity mismatch: " + key)
        ms, osd = get_state_dict(model, opt, options=state_options)
        state = {"model": ms, "optimizer": osd}
        dcp.load(state, checkpoint_id=a.resume / "distributed")
        set_state_dict(
            model,
            opt,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=state_options,
        )
        step = meta["optimizer_steps"]
        best_eligible = selection_helper.restored_best_eligibility(meta, a.phase)
        feedback_changed = (
            meta.get("task_feedback_revision") != identity["task_feedback_revision"]
        )
        sampler.restore(meta["sampler"])
        audit_formal_resume(meta, a.global_batch)
        formal_start = meta.get("formal_start_optimizer_steps")
        preformal_samples = meta.get("preformal_consumed_samples", 0) - (sampler.consumed_samples if meta["phase"] != "formal" else 0)
        stale = meta.get("stale_validation_checks", 0)
        phase_start = meta.get("phase_start_optimizer_steps", 0)
        if meta["phase"] == "formal" and a.phase != "formal":
            raise ValueError("cannot turn formal training back into preformal phase")
        if meta["phase"] != a.phase:
            phase_start = step
        if a.phase == "formal" and meta["phase"] != "formal":
            preformal_samples += sampler.consumed_samples
            formal_start = step
            best = float("inf")  # Preformal checkpoints cannot become the formal selected candidate.
            best_eligible = False
            sampler = GlobalSampler(train_indices, a.seed)
            if rank == 0:
                atomic_json(a.output / "formal_transition.json", dict(time=time.time(), optimizer_steps=step, previous_phase=meta["phase"], preformal_consumed_samples=preformal_samples, formal_sampler=sampler.state(), reason="preserve parameters/optimizer/RNG; restart formal coverage from seed42"))
        best = float("inf") if a.phase == "formal" and meta["phase"] != "formal" else meta["best_flow"]
        initial_flow = meta.get("initial_flow")
        if feedback_changed or meta.get("validation_indices") != val_indices:
            best = float("inf")
            best_eligible = False
            initial_flow = None
            reference_path = a.output / "untrained_reference.json"
            if feedback_changed and reference_path.exists():
                reference = json.loads(reference_path.read_text())
                if (
                    reference["task_feedback_revision"]
                    != identity["task_feedback_revision"]
                    or reference["manifest_sha256"] != identity["manifest_sha256"]
                    or reference["plan_sha256"] != identity["plan_sha256"]
                ):
                    raise ValueError("revised initial reference identity mismatch")
                initial_flow = reference["metrics"]["flow"]
                identity["initial_reference_sha256"] = digest(reference_path)
                if rank == 0:
                    atomic_json(
                        a.output / "initial_validation.json",
                        dict(
                            reference["metrics"],
                            reference_scope="untrained revised V2 model",
                            reference_optimizer_steps=0,
                        ),
                    )
        rng_path = a.resume / ("rng_rank_" + str(rank) + ".pt")
        if rng_path.exists():
            rng = torch.load(rng_path, weights_only=True, map_location="cpu")
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"], device)
            if "python" in rng:
                random.setstate(rng["python"])
            identity["rng_restore_mode"] = "saved_existing_rank"
        else:
            torch.manual_seed(a.seed + rank * 100003 + step)
            random.seed(a.seed + rank * 100003 + step)
            identity["rng_restore_mode"] = "deterministic_new_rank_on_expansion"
    if transition:
        transition.update(time=time.time(), status="restored", rng_restore_mode=identity.get("rng_restore_mode"))
        identity["batch_transition"] = transition
        if rank == 0:
            with (a.output / "batch_transitions.jsonl").open("a") as stream:
                stream.write(json.dumps(transition) + "\n")
    if a.phase == "formal":
        required_target = formal_start + budget["required_optimizer_steps"]
        if a.steps is not None and a.steps < required_target:
            raise ValueError(f"formal target {a.steps} would omit complete five-epoch coverage; require >= {required_target}")
        a.steps = a.steps or required_target
    a.schedule_steps = a.schedule_steps or budget["required_optimizer_steps"]
    identity.update(schedule_steps=a.schedule_steps, target_optimizer_steps=a.steps,
                    formal_start_optimizer_steps=formal_start, phase_start_optimizer_steps=phase_start)

    def coverage():
        formal_samples = sampler.consumed_samples if a.phase == "formal" else 0
        return dict(formal_start_optimizer_steps=formal_start,
                    formal_optimizer_steps=step - formal_start if formal_start is not None else 0,
                    phase_start_optimizer_steps=phase_start,
                    formal_consumed_samples=formal_samples,
                    preformal_consumed_samples=preformal_samples + (sampler.consumed_samples if a.phase != "formal" else 0),
                    completed_epochs=formal_samples // len(train_indices),
                    coverage_budget_met=formal_samples >= budget["required_samples"],
                    formal_budget=budget)

    if rank == 0:
        atomic_json(a.output / "config.json", identity)
    cache = {}
    cache_bytes = 0

    def validate():
        nonlocal cache_bytes
        heartbeat.update("validation", optimizer_step=step, validation_completed=0, validation_total=len(val_indices))
        model.eval()
        sums = torch.zeros(5, device=device)
        confusion = torch.zeros(23, 23, device=device, dtype=torch.long)
        local_indices = val_indices[rank::world]
        # Every rank executes the same number of FSDP forwards; the final padded
        # entries contribute zero weight rather than duplicating validation samples.
        slots = math.ceil(len(val_indices) / world)
        with torch.no_grad(), torch.random.fork_rng(devices=[local]):
            torch.manual_seed(a.seed + 100003)
            for offset in range(slots):
                valid = offset < len(local_indices)
                idx = local_indices[offset] if valid else local_indices[-1]
                torch.manual_seed(a.seed + 100003 + idx * 31)
                if idx in cache:
                    cpu = cache[idx]
                else:
                    cpu = load([idx], valdata)

                    def size(value):
                        if isinstance(value, torch.Tensor):
                            return value.numel() * value.element_size()
                        if isinstance(value, dict):
                            return sum(size(item) for item in value.values())
                        if isinstance(value, (tuple, list)):
                            return sum(size(item) for item in value)
                        return 0

                    amount = size(cpu)
                    if cache_bytes + amount < 1024**3:
                        cache[idx] = cpu
                        cache_bytes += amount
                batch = move(cpu, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, parts, decisions = batch_loss(model, batch, True)
                if valid:
                    sums[:4] += torch.stack(
                        [parts[k].sum() for k in ("flow", "state", "next", "goal")]
                    )
                    sums[4] += 1
                    for item, decision in zip(batch["sequence"], decisions):
                        mask = item["labels"]["next_mask"]
                        truth = item["labels"]["next"][mask]
                        pred = decision["next"].argmax(-1)[mask]
                        for t, p in zip(truth.tolist(), pred.tolist()):
                            confusion[t, p] += 1
                del batch
                heartbeat.update("validation", optimizer_step=step, validation_completed=min((offset + 1) * world, len(val_indices)))
        dist.all_reduce(sums)
        dist.all_reduce(confusion)
        precision = confusion.diag() / (confusion.sum(0).clamp_min(1))
        recall = confusion.diag() / (confusion.sum(1).clamp_min(1))
        present = confusion.sum(1) > 0
        f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-9))[
            present
        ].mean()
        model.train()
        if not torch.isfinite(sums).all() or not torch.isfinite(f1):
            raise FloatingPointError("nonfinite validation loss or skill F1")
        return dict(
            zip(("flow", "state", "next", "goal"), (sums[:4] / sums[4]).tolist())
        ) | {
            "next_macro_f1": float(f1),
            "samples": int(sums[4]),
            "validation_episodes": len(identity["validation_episodes"]),
            "test_instances_used": False,
            **{key: identity[key] for key in revision_keys},
            "implementation_source_sha256": identity["implementation_source_sha256"],
        }

    def check_save_space(stage):
        heartbeat.update(stage, optimizer_step=step)
        plan = None
        if rank == 0:
            try:
                plan = space_helper.save_space_plan(a.output, stage, parameter_count=parameter_count,
                    minimum_free_gib=max(50, a.checkpoint_free_gib))
            except Exception as error:
                plan = dict(stage=stage, allowed=False, error=str(error))
            plan.update(time=time.time(), optimizer_step=step, check="before_write_peak_budget")
            atomic_json(a.output / "checkpoint_space_latest.json", plan)
            with (a.output / "checkpoint_space_audit.jsonl").open("a") as stream:
                stream.write(json.dumps(plan) + "\n")
        enough = torch.tensor(int(rank != 0 or plan["allowed"]), device=device)
        dist.all_reduce(enough, op=dist.ReduceOp.MIN)
        if not enough:
            raise OSError("checkpoint peak reserve blocked; current checkpoint remains intact: " + json.dumps(plan))

    def check_written_floor(stage):
        heartbeat.update(stage, optimizer_step=step)
        detail = None
        if rank == 0:
            try:
                free = space_helper.assert_actual_free_floor(a.output, max(50, a.checkpoint_free_gib))
                detail = dict(stage=stage, allowed=True, free_bytes=free)
            except Exception as error:
                detail = dict(stage=stage, allowed=False, error=str(error))
            detail.update(time=time.time(), optimizer_step=step, check="complete_temp_before_replace")
            atomic_json(a.output / "checkpoint_space_latest.json", detail)
            with (a.output / "checkpoint_space_audit.jsonl").open("a") as stream:
                stream.write(json.dumps(detail) + "\n")
        enough = torch.tensor(int(rank != 0 or detail["allowed"]), device=device)
        dist.all_reduce(enough, op=dist.ReduceOp.MIN)
        if not enough:
            raise OSError("completed temporary checkpoint free floor blocked; current checkpoint remains intact: " + json.dumps(detail))

    def save_checkpoint(reason, complete=False, selection=None):
        nonlocal best
        meta = {
            **identity,
            "optimizer_steps": step,
            "checkpoint_storage_format": "native_stream.zstd/1_lossless",
            "optimizer_precision_changed": False,
            **coverage(),
            "sampler": sampler.state(),
            "early_stop_patience_used": False,
            "stale_validation_checks": stale,
            "best_flow": best,
            "best_selection_eligible": best_eligible,
            "best_is_eligible": best_eligible,
            "initial_flow": initial_flow,
            "reason": reason,
            "training_complete": complete,
            "validated_selection": selection,
            "world_size": world,
        }
        check_save_space("checkpoint_space_check")
        heartbeat.update("saving_checkpoint", optimizer_step=step)
        temp = a.output / "resume.tmp"
        if rank == 0:
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir()
        dist.barrier()
        ms, osd = get_state_dict(model, opt, options=state_options)
        dcp.save({"model": ms, "optimizer": osd}, storage_writer=storage_helper.writer(temp / "distributed"))
        torch.save(
            {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device), "python": random.getstate()},
            temp / ("rng_rank_" + str(rank) + ".pt"),
        )
        if rank == 0:
            atomic_json(temp / "metadata.json", meta)
        dist.barrier()
        check_written_floor("checkpoint_written_space_check")
        if rank == 0:
            atomic_helper.commit_complete_directory(temp, a.output / "resume", a.output / "resume.old",
                required_files=("metadata.json", "distributed/.metadata"))
        dist.barrier()
        return meta

    if initial_flow is None:
        initial = validate()
        initial_flow = initial["flow"]
        if rank == 0:
            atomic_json(a.output / "initial_validation.json", initial)
    pending = None
    log = (a.output / ("rank_" + str(rank) + ".jsonl")).open("a", buffering=1)
    resume_start_step = step
    try:
        while step < a.steps:
            heartbeat.update("training", optimizer_step=step, **coverage())
            tick = time.monotonic()
            indices = sampler.take(a.global_batch)
            pieces = [
                indices[
                    offset + rank * a.micro_batch : offset + (rank + 1) * a.micro_batch
                ]
                for offset in range(0, a.global_batch, world * a.micro_batch)
            ]
            opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            parts_sum = torch.zeros(4, device=device)
            wait_time = 0.0
            queue_wait_seconds = 0.0
            h2d_submit_seconds = 0.0
            loader_wall_seconds = 0.0
            micro_batch_timings = []
            queue.submit(0, pieces[0])
            for micro, part in enumerate(pieces):
                started = time.monotonic()
                cpu, load_seconds = queue.take(micro, part)
                take_done = time.monotonic()
                batch = move(cpu, device)
                move_done = time.monotonic()
                wait_time += move_done - started
                queue_wait_seconds += take_done - started
                # Host call duration, including any host blocking; this does not
                # measure CUDA transfer completion or add a GPU synchronization.
                h2d_submit_seconds += move_done - take_done
                # Background loader wall time overlaps GPU work; do not add it
                # to queue/H2D intervals as if the pipeline were sequential.
                loader_wall_seconds += load_seconds
                # Snapshot the corresponding completed load before submitting the next future.
                pipeline_metrics = dict(cpu_pool.last_load_metrics) if cpu_pool is not None else None
                micro_batch_timings.append(dict(
                    micro_index=micro, local_sequences_count=len(part), local_sample_indices=list(part),
                    cpu_pipeline_metrics=pipeline_metrics,
                    queue_wait_seconds=take_done - started,
                    h2d_submit_seconds=move_done - take_done,
                    loader_wall_seconds=load_seconds,
                ))
                if micro + 1 < len(pieces):
                    queue.submit(micro + 1, pieces[micro + 1])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vector, parts, _ = batch_loss(model, batch)
                loss = vector.sum() * world / a.global_batch
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite full-tuning loss")
                loss.backward()
                loss_sum += vector.detach().sum()
                parts_sum += torch.stack(
                    [parts[k].detach().sum() for k in ("flow", "state", "next", "goal")]
                )
                del batch, loss, vector, parts
            audit = (
                grad_audit(model, device)
                if step < 10 or step % a.gradient_audit_every == 0
                else None
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if hasattr(norm, "full_tensor"):
                norm = norm.full_tensor()
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            factor = (
                min(1.0, (step - (formal_start or 0) + 1) / max(1, a.warmup))
                * 0.5
                * (
                    1
                    + math.cos(math.pi * min(step - (formal_start or 0), a.schedule_steps) / a.schedule_steps)
                )
            )
            opt.param_groups[0]["lr"] = a.base_lr * factor
            opt.param_groups[1]["lr"] = a.graph_lr * factor
            opt.step()
            step += 1
            torch.cuda.synchronize(device)
            dist.all_reduce(loss_sum)
            dist.all_reduce(parts_sum)
            elapsed = time.monotonic() - tick
            duration = torch.tensor(elapsed, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            record = dict(
                step=step,
                time=time.time(),
                **coverage(),
                **{key: identity[key] for key in revision_keys},
                implementation_source_sha256=identity["implementation_source_sha256"],
                training_infra_sha256=TRAIN_INFRA_SHA256,
                cpu_workers=a.cpu_workers,
                cpu_pipeline_source_sha256=cpu_pipeline_source_sha256,
                cpu_pipeline_start_method="spawn" if cpu_pool else "serial",
                cpu_pipeline_parent_rng_unchanged=cpu_pipeline_parent_rng_unchanged,
                cpu_pipeline_initialization=cpu_pool.initialization if cpu_pool else [],
                rng_restore_mode=identity.get("rng_restore_mode"),
                phase=a.phase,
                world_size=world,
                micro_batch=a.micro_batch,
                global_batch=a.global_batch,
                grad_accum_steps=len(pieces),
                loss=float(loss_sum / a.global_batch),
                parts=dict(
                    zip(
                        ("flow", "state", "next", "goal"),
                        (parts_sum / a.global_batch).tolist(),
                    )
                ),
                step_seconds=float(duration),
                data_wait_seconds=wait_time,
                queue_wait_seconds=queue_wait_seconds,
                h2d_submit_seconds=h2d_submit_seconds,
                h2d_submit_semantics="host submit/call duration; not GPU transfer completion",
                loader_wall_seconds=loader_wall_seconds,
                loader_wall_semantics="sum of consumed background loads; overlaps GPU work",
                timed_micro_batches_count=len(micro_batch_timings),
                micro_batch_timings=micro_batch_timings,
                samples_per_second=a.global_batch / float(duration),
                grad_norm=float(norm),
                gradient_audit=audit,
                allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
                sample_indices=indices,
                sampler_audit=sampler.last_take_audit,
                local_sample_indices=[index for part in pieces for index in part],
                epoch=sampler.epoch,
                position=sampler.position,
            )
            log.write(json.dumps(record) + "\n")
            heartbeat.update("training", optimizer_step=step, **coverage())
            if rank == 0:
                atomic_json(
                    a.output / "status.json",
                    dict(
                        status="training",
                        **record,
                        training_complete=False,
                        base_origin_unchanged=True
                    ),
                )
            selection = None
            if step % a.validation_every == 0 or step == a.steps:
                selection = validate()
                if rank == 0:
                    with (a.output / "validation.jsonl").open("a") as f:
                        f.write(json.dumps(dict(step=step, **selection)) + "\n")
                improved = selection_helper.should_replace_best(selection, best, best_eligible, initial_flow, a.phase)
                stale = 0 if improved else stale + 1
                if improved:
                    best = selection["flow"]
                    best_eligible = selection_helper.selection_eligible(selection, initial_flow, a.phase)
                    check_save_space("inference_checkpoint_space_check")
                    state = get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True, cpu_offload=True
                        ),
                    )
                    if rank == 0:
                        save_inference(
                            state,
                            a.output / "best.tmp",
                            dict(
                                identity,
                                optimizer_steps=step,
                                **coverage(),
                                best_selection_eligible=best_eligible,
                                best_is_eligible=best_eligible,
                                evaluation_authorized=a.phase == "formal" and step == a.steps and coverage()["coverage_budget_met"] and best_eligible,
                                training_complete=step == a.steps and (a.phase != "formal" or coverage()["coverage_budget_met"]),
                                validated_selection=dict(
                                    split="validation", step=step, **selection
                                )
                            ),
                            aliases,
                            original_dtypes,
                        )
                    dist.barrier()
                    check_written_floor("inference_checkpoint_written_space_check")
                    if rank == 0:
                        atomic_helper.commit_complete_directory(a.output / "best.tmp", a.output / "best", a.output / "best.old",
                            required_files=("metadata.json", "model.safetensors"))
                    dist.barrier()
            # Validation chooses a retained checkpoint; it never shortens the five-epoch budget.
            request = stop[0]
            if rank == 0 and a.control and a.control.exists():
                request = request or json.loads(a.control.read_text()).get(
                    "checkpoint_stop", False
                )
            flag = torch.tensor(int(request), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            resume_probe_boundary = a.resume_checkpoint_after > 0 and step == resume_start_step + a.resume_checkpoint_after
            if step % a.checkpoint_every == 0 or step == a.steps or flag or resume_probe_boundary:
                save_checkpoint(
                    "requested_boundary" if flag else "periodic",
                    step == a.steps and (a.phase != "formal" or coverage()["coverage_budget_met"]),
                    selection,
                )
            if flag:
                if rank == 0:
                    atomic_json(
                        a.output / "status.json",
                        dict(
                            status="checkpointed_stop",
                            optimizer_steps=step,
                            **coverage(),
                            training_complete=False,
                            checkpoint=str(a.output / "resume"),
                        ),
                    )
                break
        if step == a.steps:
            if a.phase == "formal" and not coverage()["coverage_budget_met"]:
                raise AssertionError("formal training cannot complete before five full epochs")
            heartbeat.update("training_complete", optimizer_step=step, **coverage())
            if rank == 0:
                meta_path = a.output / "best/metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["training_complete"] = True
                    meta["completed_training_budget"] = coverage()
                    meta["completed_phase"] = a.phase
                    meta["evaluation_authorized"] = a.phase == "formal" and coverage()["coverage_budget_met"] and best_eligible
                    atomic_json(meta_path, meta)
                atomic_json(
                    a.output / "status.json",
                    dict(
                        identity,
                        status="training_complete",
                        optimizer_steps=step,
                        training_complete=True,
                        validated_early_stop=False,
                        best_flow=best,
                        initial_flow=initial_flow,
                        selected_checkpoint=str(a.output / "best"),
                        **coverage(),
                    ),
                )
    finally:
        heartbeat.close()
        try:
            queue.close()
        finally:
            if cpu_pool is not None:
                try:
                    cpu_pool.close()
                finally:
                    if cpu_pool.shutdown_report is not None:
                        atomic_json(a.output / ("cpu_pipeline_shutdown_rank_" + str(rank) + ".json"),
                            dict(cpu_pool.shutdown_report, optimizer_steps=step,
                                 cpu_pipeline_source_sha256=cpu_pipeline_source_sha256))
        data.close()
        valdata.close()
        log.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ACTIVE_OUTPUT:
            detail = dict(time=time.time(), rank=int(os.environ.get("RANK", 0)),
                          type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                          cuda_oom=isinstance(error, torch.cuda.OutOfMemoryError),
                          nonfinite=isinstance(error, FloatingPointError),
                          last_heartbeat=ACTIVE_HEARTBEAT.value if ACTIVE_HEARTBEAT else None)
            if torch.cuda.is_initialized():
                detail["cuda_memory"] = torch.cuda.memory_stats()
            atomic_json(ACTIVE_OUTPUT / ("failure_rank_" + os.environ.get("RANK", "0") + ".json"), detail)
            if ACTIVE_HEARTBEAT:
                ACTIVE_HEARTBEAT.update("training_error", error=str(error))
                ACTIVE_HEARTBEAT.close()
        raise
