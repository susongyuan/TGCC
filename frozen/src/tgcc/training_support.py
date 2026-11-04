"""All original successful training units with a separate immutable reference index."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .checkpoint import digest


class DenseTrainingSupport:
    schema = "tgcc.dense_success_training_support.v1"

    def __init__(self, path, reference_library, demonstrations):
        self.directory = Path(path)
        self.reference_library = reference_library
        self.identity = json.loads((self.directory / "identity.json").read_text())
        ident = self.identity
        ref = reference_library.identity
        if (ident.get("schema") != self.schema or ident.get("status") != "complete" or
                ident.get("reference_feature_library_sha256") != digest(reference_library.directory / "identity.json") or
                ident.get("reference_index_expanded", True) or ident.get("codec_refitted", True) or
                ident.get("failed_or_synthetic_targets_used", True)):
            raise ValueError("Dense training support must use the unchanged successful reference index/codec")
        for key in ("demonstrations_manifest_sha256", "normalization_sha256", "plan_sha256",
                "feature_codec_sha256", "retrieval_index_sha256", "encoder_model_sha256",
                "context_width", "after_width", "planning_classes"):
            if ident.get(key) != ref.get(key) or key not in ref:
                raise ValueError("Dense support/reference identity mismatch: " + key)
        if (digest(demonstrations.manifest_path) != ident["demonstrations_manifest_sha256"] or
                digest(demonstrations.plan_path) != ident["plan_sha256"] or
                demonstrations.adapter.normalization_sha256 != ident["normalization_sha256"]):
            raise ValueError("Dense training original demonstration/plan/normalizer changed")
        for filename in ("samples.npz", "rows.jsonl"):
            if digest(self.directory / filename) != ident.get("artifact_sha256", {}).get(filename):
                raise ValueError("Dense training support artifact changed: " + filename)
        self.rows = [json.loads(s) for s in (self.directory / "rows.jsonl").read_text().splitlines() if s]
        expected = [i for i, row in enumerate(demonstrations.rows) if row["split"] == "train"]
        actual = [int(row["manifest_index"]) for row in self.rows]
        if sorted(actual) != expected or len(actual) != ident.get("train_samples"):
            raise ValueError("Dense support must cover every original training manifest unit exactly once")
        for row in self.rows:
            original = demonstrations.rows[int(row["manifest_index"])]
            if (row["split"] != "train" or str(row["episode_id"]) != str(original["episode_id"]) or
                    row["sequence_steps"] != original["sequence_steps"] or
                    row["source_sha256"] != original["source_sha256"] or
                    row["plan_sha256"] != original["plan_sha256"] or
                    abs(row["before_timestamp"] - original["timestamp"]) > 2e-4 or
                    abs(row["after_timestamp"] - row["before_timestamp"] - 32/30) > 2e-4):
                raise ValueError("Dense training original unit ownership/time changed")
        with np.load(self.directory / "samples.npz", allow_pickle=False) as stored:
            self.arrays = {key: stored[key] for key in stored.files}
        shapes = dict(context=(len(self.rows), ident["context_width"]),
            after_features=(len(self.rows), ident["after_width"]),
            after_feature_mask=(len(self.rows), ident["after_width"]),
            actual_native=(len(self.rows), 32, 23), actions=(len(self.rows), 32, 23),
            action_mask=(len(self.rows), 32), next_index=(len(self.rows),), next_mask=(len(self.rows),))
        if any(key not in self.arrays or self.arrays[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("Dense support target/context/mask dimensions changed")
        if (any(len(value) != len(self.rows) for value in self.arrays.values()) or
                any(self.arrays[key].dtype != np.bool_ for key in ("action_mask", "after_feature_mask", "next_mask")) or
                not self.arrays["action_mask"].all() or
                not np.isfinite(self.arrays["context"]).all() or
                not np.isfinite(self.arrays["actions"]).all() or
                not np.isfinite(self.arrays["actual_native"]).all() or
                not np.isfinite(self.arrays["after_features"][self.arrays["after_feature_mask"]]).all()):
            raise ValueError("Dense support requires actual complete finite32 success units and reliable masks")
        self.train_indices = np.arange(len(self.rows), dtype=np.int64)
        self.validation_indices = np.empty(0, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = deepcopy(self.rows[int(index)])
        row.update({key: value[int(index)].copy() for key, value in self.arrays.items()})
        return row

    def query(self, context, **kwargs):
        # Retrieval remains the original 5888-anchor training-only reference
        # index. Dense query units are never added to its search candidates.
        return self.reference_library.query(context, **kwargs)
