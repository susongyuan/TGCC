import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tgcc.checkpoint import digest
from tgcc.policy_anchor_cache import (PolicyAnchorCache, SCHEMA, anchor_sources,
    equal_prepared, source_signature, write_policy_anchor)


class OriginalPolicyFixture:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0
        self.adapter = SimpleNamespace(normalization_sha256="n" * 64)

    def __getitem__(self, index):
        self.calls += 1
        return dict(normalization_sha256="n" * 64,
            sequence=[dict(timestamp=torch.tensor(float(i)),
                observation=dict(image=torch.full((1, 2, 2, 3), i, dtype=torch.uint8)),
                actions=torch.zeros(1, 32, 32),
                conditions=dict(history=torch.tensor([i-1]), valid=torch.tensor([True])))
                for i in (0, 1)])


class LibraryFixture:
    def __init__(self, rows):
        self.rows = rows
        self.train_indices = [0]
        self.identity = dict(normalization_sha256="n" * 64)

    def __getitem__(self, index):
        return dict(self.rows[index], actual_native=np.zeros((32, 23), np.float32))


def cache_fixture(tmp_path, monkeypatch):
    from tgcc import policy_anchor_cache as module
    source = tmp_path / "original.json"
    source.write_text("original immutable demonstration")
    videos = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        path = tmp_path / (camera + ".mp4")
        path.write_bytes((camera + " original RGB bytes").encode())
        videos[camera] = path
    raw_row = dict(dataset_root=str(tmp_path), episode_id="1", split="train",
        source_sha256={source.name: digest(source)})
    support = dict(manifest_index=0, episode_id="1", split="train", sequence_steps=[0, 32],
        before_timestamp=0., video_sha256={c: digest(p) for c, p in videos.items()},
        rgb_exposures=[{c: dict(source_path=str(p)) for c, p in videos.items()}] * 2)
    library, raw = LibraryFixture([support]), OriginalPolicyFixture([raw_row])
    normalizer = SimpleNamespace(transform=lambda x: x)
    header = dict(schema=SCHEMA, normalization_sha256="n" * 64,
        train_anchor_indices=[0], train_anchor_count=1,
        source_files_sha256=anchor_sources(library, raw, 0), loader_source_sha256={"fixture": "a" * 64})
    monkeypatch.setattr(module, "cache_header", lambda *_: header)
    output = tmp_path / "cache"
    entry = write_policy_anchor(output, library, raw, normalizer, 0)
    identity = dict(header, status="complete", entries={"0": entry},
        source_file_signatures={p: source_signature(p) for p in header["source_files_sha256"]})
    (output / "identity.json").write_text(json.dumps(identity))
    return PolicyAnchorCache(output, library, raw, normalizer), raw, source, videos, identity


def test_prepared_cache_retains_exact_first_inputs_and_never_uses_after_or_redecodes(tmp_path, monkeypatch):
    cache, raw, *_ = cache_fixture(tmp_path, monkeypatch)
    original = raw[0]
    expected = dict(original, sequence=original["sequence"][:1])
    calls = raw.calls
    prepared = cache.get(0)
    assert equal_prepared(expected, prepared) and raw.calls == calls
    assert len(prepared["sequence"]) == 1
    prepared["sequence"][0]["actions"].add_(7)
    assert equal_prepared(expected, cache.get(0))


def test_prepared_cache_corruption_is_rejected_before_deserialization(tmp_path, monkeypatch):
    cache, *_ = cache_fixture(tmp_path, monkeypatch)
    path = cache.path / cache.identity["entries"]["0"]["path"]
    path.write_bytes(path.read_bytes() + b"corruption")
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("Deserialized corrupt payload"))
    with pytest.raises(ValueError, match="corrupt"):
        cache.get(0)


@pytest.mark.parametrize("kind", ["original", "RGB"])
def test_prepared_cache_source_mutation_blocks_hit(tmp_path, monkeypatch, kind):
    cache, _, original, videos, _ = cache_fixture(tmp_path, monkeypatch)
    source = original if kind == "original" else videos["head"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="original source changed"):
        cache.get(0)


def test_prepared_cache_missing_anchor_and_incomplete_identity_block_training(tmp_path, monkeypatch):
    cache, raw, _, _, identity = cache_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Non-training|missing"):
        cache.get(1)
    changed = copy.deepcopy(identity)
    changed["status"] = "incomplete"
    (cache.path / "identity.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incomplete"):
        PolicyAnchorCache(cache.path, cache.library, raw, SimpleNamespace())
