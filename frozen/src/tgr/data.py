"""Fail-closed temporal dataset manifest checks; never synthesize oracle from RGB."""

import hashlib, json


def validate_manifest(rows, split_by="episode_id"):
    required = {
        "episode_id",
        "scene_id",
        "step_id",
        "timestamp",
        "object_ids",
        "action_horizon",
        "split",
        "graph_source",
        "graph_timestamp",
    }
    splits = {}
    last = {}
    for row in rows:
        if required - set(row):
            raise ValueError("missing synchronized provenance fields")
        if row["graph_timestamp"] > row["timestamp"]:
            raise ValueError("future graph leak")
        if row["graph_source"] not in ("oracle", "observation"):
            raise ValueError("unknown graph provenance")
        key = row[split_by]
        if key in splits and splits[key] != row["split"]:
            raise ValueError("episode/scene split leakage")
        splits[key] = row["split"]
        e = row["episode_id"]
        if e in last and row["timestamp"] <= last[e]:
            raise ValueError("non-monotonic episode")
        last[e] = row["timestamp"]
    return True


class LazySequenceDataset:
    """Server-only seq2 manifests. One episode/readers cached, no tensor disk cache.

    rows are the JSONL descriptors consumed by train.BatchSource. __getitem__
    returns B1 causal sequences; collate_sequences creates optional B>1 batches.
    Source hashes are verified when an episode enters the bounded RAM cache.
    """

    def __init__(self, manifest_path, checkpoint, plan_path, adapter=None):
        from pathlib import Path
        from .build_batches import sha256

        self.manifest_path = Path(manifest_path)
        self.rows = [
            json.loads(x)
            for x in self.manifest_path.read_text().splitlines()
            if x.strip()
        ]
        if not self.rows:
            raise ValueError("empty lazy manifest")
        validate_manifest(self.rows)
        self.checkpoint = checkpoint
        self.plan_path = Path(plan_path)
        self.plan_sha = sha256(self.plan_path)
        if adapter is None:
            from .input_adapter import OfficialInputAdapter

            owner = Path(__file__).resolve().parents[2]
            adapter = OfficialInputAdapter(
                checkpoint,
                proprio_layout_sha256=sha256(
                    owner / "reports/data_proprio_layout.json"
                ),
                action_layout_sha256=sha256(owner / "reports/data_action_layout.json"),
            )
        self.adapter = adapter
        self.current_episode = None
        self.current_data = None
        self.current_reader = None
        for row in self.rows:
            if row.get("schema") != "tgr.lazy_seq2.v1" or row.get("sequence_steps") != [
                row["step_id"],
                row["step_id"] + 32,
            ]:
                raise ValueError("lazy causal sequence schema")
            if row.get("plan_sha256") != self.plan_sha:
                raise ValueError("lazy plan SHA mismatch")
            if row["split"] not in ("train", "validation"):
                raise ValueError("lazy train/validation only")

    def __len__(self):
        return len(self.rows)

    def close(self):
        if self.current_reader is not None:
            self.current_reader.close()
        self.current_episode = None
        self.current_data = None
        self.current_reader = None

    def __getitem__(self, index):
        from pathlib import Path
        from .build_batches import (
            sha256,
            make_demonstration,
            pack_sequence,
            validate_batch,
        )

        row = self.rows[index]
        ep = str(row["episode_id"])
        if self.current_episode != ep:
            self.close()
            root = Path(row["dataset_root"])
            for path, digest in row["source_sha256"].items():
                if sha256(root / path) != digest:
                    raise ValueError("lazy source changed: " + path)
            self.current_data, self.current_reader = make_demonstration(
                root, ep, row["split"], self.checkpoint, self.plan_path, self.adapter
            )
            self.current_episode = ep
        data = self.current_data
        if data.split != row["split"]:
            raise ValueError("cached episode split mismatch")
        try:
            items = [data[data.anchors.index(frame)] for frame in row["sequence_steps"]]
        except Exception as error:
            raise ValueError(
                f'episode={ep} sequence={row["sequence_steps"]}: {error}'
            ) from error
        for item in items:
            if item["provenance"]["plan_sha256"] != self.plan_sha:
                raise ValueError("runtime plan mismatch")
        batch = pack_sequence(items)
        validate_batch(batch, row, data.norm_sha)
        batch["sample_rows"] = [dict(row)]
        return batch


def collate_sequences(samples):
    """Batch independent same-length B1 sequences with masked history padding.

    Padding never invents exposures: per_sample_history stores original unpadded
    provenance; padded graph frames have valid=false. Memory resets per sequence.
    """
    import torch

    if not samples:
        raise ValueError("empty sequence collation")
    if len(samples) == 1:
        return samples[0]
    length = len(samples[0]["sequence"])
    norm = samples[0]["normalization_sha256"]
    common = dict(
        samples[0].get("provenance", samples[0]["sequence"][0].get("provenance", {}))
    )
    for sample in samples:
        proof = sample.get("provenance", sample["sequence"][0].get("provenance", {}))
        for key in ("plan_sha256", "condition_schema", "feature_schema", "query_order"):
            if proof.get(key) != common.get(key):
                raise ValueError("mixed sequence proof: " + key)
    if any(
        len(s["sequence"]) != length or s["normalization_sha256"] != norm
        for s in samples
    ):
        raise ValueError("mixed sequence length/normalizer")

    def tree(values):
        if isinstance(values[0], dict):
            return {k: tree([v[k] for v in values]) for k in values[0]}
        if all(v is None for v in values):
            return None
        if isinstance(values[0], torch.Tensor):
            return torch.cat(values, 0)
        if all(v == values[0] for v in values):
            return values[0]
        raise ValueError("non-tensor observation mismatch")

    sequence = []
    for step in range(length):
        items = [s["sequence"][step] for s in samples]
        conditions = {}
        for key in ("global_graph", "local_graph", "task_graph"):
            graphs = [i["conditions"][key] for i in items]
            max_t = max(g["features"].shape[1] for g in graphs)
            padded = []
            for g in graphs:
                p = dict(g)
                gap = max_t - g["features"].shape[1]
                if gap:
                    for field in (
                        "features",
                        "semantics",
                        "relations",
                        "valid",
                        "roles",
                        "relevance",
                        "pair_nodes",
                    ):
                        if g.get(field) is None:
                            continue
                        x = g[field]
                        z = torch.zeros(
                            (x.shape[0], gap, *x.shape[2:]),
                            dtype=x.dtype,
                            device=x.device,
                        )
                        p[field] = torch.cat([z, x], 1)
                padded.append(p)
            conditions[key] = tree(padded)
        for key in items[0]["conditions"]:
            if key not in conditions:
                conditions[key] = tree([i["conditions"][key] for i in items])
        provenance = dict(items[0]["provenance"])
        provenance["per_sample_history"] = [
            {
                "history_timestamps": i["provenance"]["history_timestamps"],
                "history_node_ids": i["provenance"]["history_node_ids"],
                "rgb_exposures": i["provenance"].get("rgb_exposures", {}),
            }
            for i in items
        ]
        provenance["episode_ids"] = [i["episode_id"] for i in items]
        sequence.append(
            {
                "observation": tree([i["observation"] for i in items]),
                "actions": tree([i["actions"] for i in items]),
                "conditions": conditions,
                "labels": tree([i["labels"] for i in items]),
                "normalization_sha256": norm,
                "provenance": provenance,
                "episode_ids": [i["episode_id"] for i in items],
                "step_ids": [i["step_id"] for i in items],
                "timestamps": [i["timestamp"] for i in items],
                "has_graph_supervision": any(i["has_graph_supervision"] for i in items),
            }
        )
    for sample in samples:
        items = sample["sequence"]
        if any(
            a["episode_id"] != b["episode_id"]
            or a["timestamp"] >= b["timestamp"]
            or a["step_id"] >= b["step_id"]
            for a, b in zip(items, items[1:])
        ):
            raise ValueError("noncausal input collation")
    return {
        "normalization_sha256": norm,
        "sequence": sequence,
        "provenance": common,
        "sample_rows": [r for s in samples for r in s.get("sample_rows", [])],
        "episode_ids": [s["episode_id"] for s in samples],
        "supervision_scope": "partial labels; independent episode memory per batch row",
    }
