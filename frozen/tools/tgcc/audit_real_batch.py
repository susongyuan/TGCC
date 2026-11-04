from pathlib import Path
import json, os, sys, time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "code/src"),
    str(ROOT / "code/packages/openpi-client/src"),
]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TGR_GEOMETRY_CACHE_PATH"] = (
    "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3"
)
from tgr.data import LazySequenceDataset
import torch

torch.set_num_threads(4)
out = ROOT / "runs/interface_audit"
out.mkdir(parents=True, exist_ok=True)
p = "/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"
d = LazySequenceDataset(
    "/media/data5/ssy/TGR/reports/hotdog_lazy_seq2_manifest.jsonl",
    p,
    ROOT / "reports/data_plan_vocabulary.json",
)
t = time.monotonic()
batch = d[0]


def shapes(v):
    if isinstance(v, torch.Tensor):
        return {"shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {k: shapes(x) for k, x in v.items()}
    if isinstance(v, list):
        return [shapes(x) for x in v[:2]]
    return v


(out / "real_batch_schema.json").write_text(
    json.dumps(shapes(batch), indent=2, default=str)
)
print("DATASET_OK", time.monotonic() - t, flush=True)
# Validate parent loading and matched-noise flow behavior on actual demonstration exposure.
from tgr.factory import create
from tgr.model import Conditions
from tgr.encoders import GraphBatch
from openpi.models.model import Observation
from tgcc.policy import AdapterPolicy, PolicyFeatures

_, parent = create(
    p,
    "tri_graph",
    "cuda:0",
    "/media/data5/ssy/TGR/experiments/tc_esgr_v2_approx_full_5ep_20261024_02/selected_step5000_user_stop",
)
parent.eval()


def move(x):
    if isinstance(x, torch.Tensor):
        return x.to("cuda:0")
    if isinstance(x, dict):
        return {k: move(v) for k, v in x.items()}
    return x


item = move(batch["sequence"][0])
c = item["conditions"]
cond = Conditions(
    **{k: GraphBatch(**c[k]) for k in ("global_graph", "local_graph", "task_graph")},
    **{
        k: v
        for k, v in c.items()
        if k not in ("global_graph", "local_graph", "task_graph")
    },
)
obs = Observation.from_dict(item["observation"])
actions = item["actions"]
noise = parent.base.sample_noise(actions.shape, actions.device)
timeval = torch.full((1,), 0.5, device="cuda:0")
tap = PolicyFeatures(parent)
wrapped = AdapterPolicy(parent)
torch.cuda.reset_peak_memory_stats()
t = time.monotonic()
result = wrapped.train_pair(obs, cond, actions, noise, timeval)
torch.cuda.synchronize()
assert result["teacher_consistency"].item() == 0.0, result
result["flow"].backward()
assert wrapped.local.gate.grad is not None
r = {
    "status": "passed",
    "parent_step": 5000,
    "same_noise_time_teacher_max_loss": result["teacher_consistency"].item(),
    "adapter_gate_gradient": float(wrapped.local.gate.grad),
    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    "seconds": time.monotonic() - t,
    "captured_graph_shapes": {
        k: [list(x.shape) for x in v] for k, v in tap.values.items()
    },
    "decision_keys": list(result["decision"]),
}
(out / "step0_real_batch_gate.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r), flush=True)
