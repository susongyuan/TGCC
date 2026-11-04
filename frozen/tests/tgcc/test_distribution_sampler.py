import random
from types import SimpleNamespace

import numpy as np
import torch

from tgcc.distribution_data import (DistributionTrainingData, make_distribution_loader,
    successful_anchor_order)
from tgcc.retrieval_data import RetrievalDataset


def rows():
    return [dict(episode_id=str(ep), split="train", sequence_steps=[step, step+32])
            for ep in (1, 2, 3) for step in (64, 0, 32)] + [
                dict(episode_id="99", split="validation", sequence_steps=[0, 32])]


def test_shuffled_episode_sampler_uniform_passes_chronological_cache_locality_and_exact_cursor():
    source = rows()
    order = successful_anchor_order(source, 27, 42)
    assert order == successful_anchor_order(source, 27, 42)
    for start in (0, 9, 18):
        cycle = order[start:start+9]
        assert sorted(cycle) == list(range(9))
        for episode_start in (0, 3, 6):
            block = [source[i] for i in cycle[episode_start:episode_start+3]]
            assert len({r["episode_id"] for r in block}) == 1
            assert [r["sequence_steps"][0] for r in block] == [0, 32, 64]
    cursor, world = 12, 3
    partitions = [order[cursor+rank::world] for rank in range(world)]
    assert [x for group in zip(*partitions) for x in group] == order[cursor:]


def warmup_data():
    library = object.__new__(RetrievalDataset)
    library.identity = dict(normalization_sha256="a" * 64)
    library.rows = [dict(manifest_index=0, episode_id="1", split="train", sequence_steps=[0, 32],
        before_timestamp=0., after_timestamp=32/30)]
    library.arrays = dict(context=np.zeros((1, 8), np.float32), actual_native=np.zeros((1, 32, 23), np.float32),
        actions=np.zeros((1, 32, 23), np.float32), action_mask=np.ones((1, 32), bool),
        after_features=np.zeros((1, 6), np.float32), next_index=np.zeros(1, np.int64), next_mask=np.ones(1, bool))
    def forbidden_query(*args, **kwargs):
        raise AssertionError("Warmup must not execute a nearest-neighbor query")
    library.query = forbidden_query
    class NoRGBDemonstrations:
        rows = library.rows
        def __getitem__(self, index):
            raise AssertionError("Warmup must not read RGB or construct a policy prefix")
    normalizer = SimpleNamespace(transform=lambda x: x)
    return DistributionTrainingData(library, NoRGBDemonstrations(), normalizer, include_policy=False)


def test_warmup_reads_only_immutable_features_without_rgb_or_retrieval():
    sample = warmup_data()[0]
    assert "policy_sample" not in sample and "reference" not in sample
    assert sample["density"]["actions"].shape == (32, 23)


def test_phase_loader_restart_resumes_next_global_unit_without_consuming_model_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    _, iterator = make_distribution_loader(warmup_data(), [0] * 48, 12, 1, 4, 2, 0, 42)
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])
    batch = next(iterator)
    assert batch["support_ids"] == [0, 0]
    assert batch["references"] == []
