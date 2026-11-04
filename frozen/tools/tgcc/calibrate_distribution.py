"""Held-out-success likelihood thresholds; no failed-example performance claim."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "code/src"), str(ROOT / "code/packages/openpi-client/src")]

import numpy as np
import torch
from safetensors.torch import load_file
from tgcc.distribution_critic import DistributionCritic
from tgcc.distribution_calibration import (calibrate_success_scores, bind_calibration,
    distribution_readiness)
from tgcc.checkpoint import digest, sources
from tgcc.retrieval_data import RetrievalDataset
from tgcc.robot import ActionNormalizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-library", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--base-checkpoint", type=Path,
        default=Path("/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32"))
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    if (metadata.get("schema") != "tgcc.distribution_joint.v1" or
            metadata.get("audit_disposable") is not False or not metadata.get("training_complete") or
            metadata["new_updates"] != metadata["config"]["new_updates"] or
            metadata["source_sha256"] != sources(ROOT) or
            digest(args.checkpoint / "distribution.safetensors") != metadata["distribution_sha256"] or
            digest(args.checkpoint / "student.safetensors") != metadata["student_sha256"]):
        raise ValueError("Only the final frozen success-distribution checkpoint may be calibrated")
    library = RetrievalDataset(args.feature_library)
    if digest(args.feature_library / "identity.json") != metadata["feature_library_sha256"]:
        raise ValueError("Held-out feature library changed")
    gate = distribution_readiness(metadata, library.identity)
    if gate["status"] != "passed":
        raise ValueError("Calibration data identities not ready: " + str(gate["failures"]))
    normalizer = ActionNormalizer(args.base_checkpoint)
    normalization_files = list(args.base_checkpoint.glob("assets/**/norm_stats.json"))
    if len(normalization_files) != 1 or digest(normalization_files[0]) != metadata["normalization_sha256"]:
        raise ValueError("Checkpoint-owned action normalization changed")
    model = DistributionCritic(metadata["distribution_config"]).to(args.device).eval()
    weights = load_file(str(args.checkpoint / "distribution.safetensors"))
    model.load_state_dict({k.removeprefix("critic."): value for k, value in weights.items()
                          if k.startswith("critic.")}, strict=True)
    records = []
    ids = library.validation_indices.tolist()
    for start in range(0, len(ids), args.batch_size):
        samples = [library[i] for i in ids[start:start + args.batch_size]]
        def tensor(name, dtype=None):
            return torch.as_tensor(np.stack([s[name] for s in samples]), dtype=dtype, device=args.device)
        actions = np.stack([normalizer.transform(np.asarray(s["actual_native"], np.float32)) for s in samples]).astype(np.float32)
        batch = dict(context=tensor("context", torch.float32), actions=torch.from_numpy(actions).to(args.device),
            action_mask=tensor("action_mask", torch.bool), after_features=tensor("after_features", torch.float32),
            next_index=tensor("next_index", torch.long))
        if samples[0].get("after_feature_mask") is not None:
            batch["after_feature_mask"] = tensor("after_feature_mask", torch.bool)
        output = model(batch)
        for i, sample in enumerate(samples):
            records.append(dict(episode_id=sample["episode_id"], split=sample["split"],
                sequence_steps=sample["sequence_steps"], action_nll=float(output["action_nll"][i]),
                execution_nll=float(output["execution_nll"][i]), planning_nll=float(output["planning_nll"][i]),
                executed_steps=int(np.asarray(sample["action_mask"], bool).sum()),
                action_valid=True, execution_valid=True, planning_valid=bool(sample["next_mask"])))
    cfg = metadata["config"]
    result = calibrate_success_scores(records, library.identity["index_episodes"],
        alpha=cfg["calibration_alpha"], minimum_episodes=cfg["calibration_minimum_episodes"])
    result = bind_calibration(result, metadata, args.checkpoint / "distribution.safetensors")
    result["student_sha256"] = metadata["student_sha256"]
    result["feature_library_sha256"] = metadata["feature_library_sha256"]
    result["calibration_scope"] = "actual32step_change_with_before_and_after_known_mask; full planning vocabulary"
    (args.checkpoint / "calibration.json").write_text(json.dumps(result, indent=2))
    (args.checkpoint / "calibration_samples.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
