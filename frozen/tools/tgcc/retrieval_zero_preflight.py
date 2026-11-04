"""Real parent/official-transform zero-gate check; no physics and no training."""

import argparse
import json
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"), str(ROOT / "code/packages/openpi-client/src")]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TGR_GEOMETRY_CACHE_PATH", "/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3")

import numpy as np
import torch

from tgr.data import LazySequenceDataset
from tgr.factory import create, ensure_legacy_imports
from tgcc.checkpoint import digest
from tgcc.data import conditions_from_dict, move_tree
from tgcc.retrieval_adapter import RetrievalAdapter, RetrievalContext
from tgcc.retrieval_data import RetrievalDataset
from tgcc.distribution_runtime import DistributionController
from tgcc.robot import ActionNormalizer


def main():
    ensure_legacy_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "parent", "manifest", "plan", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    library = RetrievalDataset(args.library)
    dataset = LazySequenceDataset(args.manifest, args.checkpoint, args.plan)
    index = next(i for i, row in enumerate(dataset.rows) if row["split"] == "train")
    sample = dataset[index]
    policy, parent = create(str(args.checkpoint), "tri_graph", args.device, str(args.parent))
    parent.eval().requires_grad_(False)
    normalizer = ActionNormalizer(args.checkpoint)
    query = next(i for i, row in enumerate(library.rows)
                 if row["split"] == "train" and row["episode_id"] == dataset.rows[index]["episode_id"])
    record = library[query]
    reference, distance = library.query(record["context"], exclude_episode_id=record["episode_id"])[0]
    if reference["episode_id"] == record["episode_id"]:
        raise ValueError("Preflight reference episode exclusion failed")
    device = torch.device(args.device)
    context = RetrievalContext(normalizer.transform(torch.as_tensor(reference["actual_native"], device=device)[None]).float(),
                               torch.as_tensor(reference["action_mask"], device=device, dtype=torch.bool)[None],
                               torch.as_tensor(reference["context"], device=device)[None],
                               torch.as_tensor(reference["after_features"], device=device)[None],
                               torch.ones(1, device=device, dtype=torch.bool),
                               torch.as_tensor(reference["after_feature_mask"], device=device, dtype=torch.bool)[None])
    adapter = RetrievalAdapter(parent.base.action_in_proj.out_features,
                               len(record["context"]), len(record["after_features"])).to(device).eval()
    from openpi.models.model import Observation
    item = move_tree(sample["sequence"][0], device)
    observation = Observation.from_dict(item["observation"])
    conditions = conditions_from_dict(item["conditions"])
    start = time.monotonic()
    with torch.no_grad():
        prefix = parent.prefix(observation, conditions, False)
        state, valid, cache, original_tokens, decision = prefix
        noise = parent.base.sample_noise((1, 32, 32), device)
        def generate(tokens):
            actions = noise.clone()
            for step in range(10):
                when = torch.full((1,), 1 - step / 10, device=device)
                actions = actions - 0.1 * parent.velocity(state, valid, cache, tokens, actions, when)
            return actions
        rng_before = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(device).clone())
        zero_tokens = adapter(original_tokens, context)
        rng_after = dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(device).clone())
        original = generate(original_tokens)
        zero = generate(zero_tokens)
        equal = torch.equal(original, zero)
        state_np = state[0].cpu().numpy()
        native_original = np.asarray(policy._output_transform(dict(state=state_np, actions=original[0].cpu().numpy()))["actions"], np.float32)
        native_zero = np.asarray(policy._output_transform(dict(state=state_np, actions=zero[0].cpu().numpy()))["actions"], np.float32)
        # Exercise the real output-transform receipt bytes, including their
        # rounding difference from the fast torch normalization path.
        fast = normalizer.inverse(zero[..., :23])[0].cpu().numpy()
        fast_difference = float(np.max(np.abs(fast - native_zero)))
        receipt_controller = object.__new__(DistributionController)
        receipt_controller.previous = dict(native_actions=native_zero.copy(), request_id=1, timestamp=0.0)
        receipt_raw = dict(sim_time=16 / 30, execution_receipt=dict(
            step_count=16, actions=native_zero[:16].copy(), previous_request_id=1,
            start_time=0.0, end_time=16 / 30,
            source="native_controller_commands_after_successful_env_step"))
        acknowledged, count = receipt_controller._receipt(receipt_raw)
        actual_receipt_accepted = count == 16 and np.array_equal(acknowledged, native_zero[:16])
        for gate in adapter.gates.values():
            gate.fill_(0.01)
        changed_tokens = adapter(original_tokens, context)
        changed = generate(changed_tokens)
        changed_difference = float((changed - original).abs().max())
    torch.cuda.synchronize(device)
    report = dict(schema="tgcc.real_retrieval_zero_gate_preflight.v1", status="passed" if equal else "failed",
                  original_normalized_actions_equal=equal,
                  native_official_output_equal=bool(np.array_equal(native_original, native_zero)),
                  normalized_max_difference=float((original - zero).abs().max()),
                  suffix_tokens_same_object=zero_tokens is original_tokens,
                  rng_unchanged=all(torch.equal(rng_before[k], rng_after[k]) for k in rng_before),
                  fast_vs_official_native_max_difference=fast_difference,
                  actual_official_prefix_receipt_accepted=actual_receipt_accepted,
                  nonzero_gate_max_action_difference=changed_difference,
                  nonzero_gate_changes_action=changed_difference > 0,
                  source_episode=record["episode_id"], reference_episode=reference["episode_id"],
                  source_episode_excluded=True, retrieval_distance=distance,
                  parent_sha256=digest(args.parent / "model.safetensors"),
                  python_executable=sys.executable, torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  library_directory=str(args.library.resolve()),
                  source_sha256={name: digest(ROOT / name) for name in (
                      "tools/tgcc/retrieval_zero_preflight.py", "src/tgcc/retrieval_adapter.py",
                      "src/tgcc/distribution_runtime.py", "src/tgcc/robot.py",
                      "src/tgcc/retrieval_data.py", "src/tgr/factory.py", "src/tgr/model.py",
                      "src/tgr/encoders.py", "src/tgr/data.py", "src/tgr/serve.py",
                      "code/src/openpi/policies/policy.py", "code/src/openpi/policies/b1k_policy.py",
                      "code/src/openpi/transforms.py", "code/src/openpi/models_pytorch/pi0_pytorch.py")},
                  retrieval_identity=library.identity,
                  elapsed_seconds=time.monotonic() - start, training_started=False,
                  simulator_started=False, task_success=None)
    if not (report["native_official_output_equal"] and report["rng_unchanged"]
            and report["actual_official_prefix_receipt_accepted"]
            and report["nonzero_gate_changes_action"]):
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
