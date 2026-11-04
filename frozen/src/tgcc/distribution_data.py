"""Hash-bound successful support anchors to likelihood and original flow targets."""

import numpy as np
import torch

from .retrieval_data import RetrievalDataset, digest
from .retrieval_adapter import RetrievalContext


class DistributionTrainingData:
    def __init__(self, library, demonstrations, normalizer, *, include_policy=True, policy_cache=None):
        self.library = library if isinstance(library, RetrievalDataset) else RetrievalDataset(library)
        self.demonstrations, self.normalizer = demonstrations, normalizer
        self.include_policy = include_policy
        self.policy_cache = policy_cache
        identity=self.library.identity
        if ('demonstrations_manifest_sha256' in identity and
                digest(demonstrations.manifest_path)!=identity['demonstrations_manifest_sha256']):
            raise ValueError('Distribution data manifest identity changed')
        if ('plan_sha256' in identity and digest(demonstrations.plan_path)!=identity['plan_sha256']):
            raise ValueError('Distribution data train-owned plan identity changed')

    def __len__(self):
        return len(self.library)

    def normalized_actions(self, sample):
        native = np.asarray(sample["actual_native"], np.float32)
        if native.shape != (32, 23):
            raise ValueError("Support must contain the exact actual native32 prefix")
        normalized = np.asarray(self.normalizer.transform(native), np.float32)
        if sample.get("actions") is not None and not np.allclose(sample["actions"], normalized, atol=3e-6, rtol=2e-6):
            raise ValueError("Support normalized action target changed")
        return normalized

    def __getitem__(self, index):
        support = self.library[index]
        manifest_index = int(support["manifest_index"])
        original = self.demonstrations.rows[manifest_index]
        if (str(original["episode_id"]) != str(support["episode_id"]) or
                original["split"] != support["split"] or
                original["sequence_steps"] != support["sequence_steps"]):
            raise ValueError("Frozen support/original demonstration ownership mismatch")
        for key in ('source_sha256','plan_sha256'):
            if key in support and original.get(key)!=support[key]:
                raise ValueError('Frozen support/original source identity mismatch: '+key)
        actions = self.normalized_actions(support)
        density = dict(context=np.asarray(support["context"], np.float32), actions=actions,
            action_mask=np.asarray(support["action_mask"], bool),
            after_features=np.asarray(support["after_features"], np.float32),
            next_index=np.asarray(support["next_index"], np.int64),
            planning_valid=np.asarray(support["next_mask"], bool),
            action_valid=np.asarray(True), execution_valid=np.asarray(True))
        if support.get("after_feature_mask") is not None:
            density["after_feature_mask"] = np.asarray(support["after_feature_mask"], bool)
        if not self.include_policy:
            # The immutable feature build already checks the original official
            # action pipeline. Warmup must not decode RGB or open an episode.
            return dict(density=density, query_episode_id=str(support["episode_id"]),
                        support_id=int(index), sequence_steps=support["sequence_steps"])
        policy = (self.policy_cache.get(index) if self.policy_cache is not None
                  else self.demonstrations[manifest_index])
        official = policy["sequence"][0]["actions"][0, :, :23].numpy()
        if not np.allclose(official, actions, atol=3e-6, rtol=2e-6):
            raise ValueError("Density target differs from original successful imitation target")
        if abs(float(policy["sequence"][0]["timestamp"]) - support["before_timestamp"]) > 2e-4:
            raise ValueError("Policy target and frozen current context time mismatch")
        if policy["normalization_sha256"] != self.library.identity["normalization_sha256"]:
            raise ValueError("Policy/frozen feature normalization identity changed")
        # The original loader validates the owned before/after pair. Transfer
        # only its first policy exposure to the GPU; the frozen after-feature
        # remains exclusively an execution-density target.
        policy = dict(policy, sequence=policy["sequence"][:1])
        reference, distance = self.library.query(support["context"],
            exclude_episode_id=support["episode_id"])[0]
        if reference["split"] != "train" or str(reference["episode_id"]) == str(support["episode_id"]):
            raise ValueError("Reference must be a different training source episode")
        policy["policy_imitation_allowed"] = True
        policy["source_kind"] = "demonstration"
        return dict(density=density, policy_sample=policy,
            reference=dict(normalized_actions=self.normalized_actions(reference),
                action_mask=np.asarray(reference["action_mask"], bool),
                before_features=np.asarray(reference["context"], np.float32),
                after_features=np.asarray(reference["after_features"], np.float32),
                after_feature_mask=np.asarray(reference["after_feature_mask"], bool), valid=np.asarray(True)),
            query_episode_id=str(support["episode_id"]),
            reference_episode_id=str(reference["episode_id"]), distance=distance,
            support_id=int(index), sequence_steps=support["sequence_steps"])


def collate_distribution(samples):
    if not samples:
        raise ValueError("Empty distribution batch")
    def tensors(records):
        keys = records[0].keys()
        if any(record.keys() != keys for record in records):
            raise ValueError("Mixed density feature/mask schemas")
        return {key: torch.as_tensor(np.stack([record[key] for record in records])) for key in keys}
    if all("reference" not in s for s in samples):
        return dict(density=tensors([s["density"] for s in samples]),
            query_episode_ids=[s["query_episode_id"] for s in samples],
            support_ids=[s["support_id"] for s in samples], references=[])
    if any("reference" not in s for s in samples):
        raise ValueError("Mixed warmup/policy-loader phases")
    reference = RetrievalContext(**tensors([s["reference"] for s in samples]))
    return dict(density=tensors([s["density"] for s in samples]),
        policy_samples=[s["policy_sample"] for s in samples], references=[reference],
        query_episode_ids=[s["query_episode_id"] for s in samples],
        reference_episode_ids=[[s["reference_episode_id"] for s in samples]],
        support_ids=[s["support_id"] for s in samples])


def move_distribution_batch(batch, device):
    result = dict(batch)
    result["density"] = {key: value.to(device) for key, value in batch["density"].items()}
    result["references"] = [reference.to(device) for reference in batch["references"]]
    return result


def successful_anchor_order(rows, total, seed):
    groups = {}
    for index, row in enumerate(rows):
        if row["split"] == "train":
            groups.setdefault(str(row["episode_id"]), []).append(index)
    if not groups:
        raise ValueError("No successful training anchors")
    for episode in groups:
        groups[episode].sort(key=lambda i: rows[i]["sequence_steps"][0])
    episodes = sorted(groups)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < total:
        for episode in rng.permutation(episodes).tolist():
            out.extend(groups[episode])
    return out[:total]


def initialize_distribution_worker(worker_id):
    """Close native video readers before a loader worker's normal process exit."""
    from multiprocessing.util import Finalize
    from torch.utils.data import get_worker_info

    worker = get_worker_info()
    demonstrations = getattr(worker.dataset, "demonstrations", None)
    close = getattr(demonstrations, "close", None)
    if close is not None:
        # multiprocessing workers use os._exit; ordinary atexit callbacks do
        # not reliably run. Its registered finalizers run before os._exit.
        Finalize(None, close, exitpriority=20)


def make_distribution_loader(dataset, order, position, rank, world, micro_batch, workers, seed):
    """Rebuild a phase/resume loader without consuming the model RNG stream."""
    from torch.utils.data import DataLoader
    from .checkpoint import rng_state, restore_rng

    state = rng_state()
    loader = DataLoader(dataset, batch_size=micro_batch, sampler=order[position + rank::world],
        num_workers=workers, collate_fn=collate_distribution,
        worker_init_fn=initialize_distribution_worker if workers else None,
        generator=torch.Generator().manual_seed(seed + rank),
        multiprocessing_context="spawn" if workers else None, persistent_workers=False)
    iterator = iter(loader)
    restore_rng(state)
    return loader, iterator
