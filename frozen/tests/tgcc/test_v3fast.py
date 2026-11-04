"""v3fast: exactness of the visual feature cache, the focused detached comparison and prefix-only recomputation."""
import types, pytest, torch, numpy as np
from torch.nn import functional as F


def test_slice_prefix_cache_handles_legacy_tuples_objects_and_unknown():
    from tgcc.full_distribution_joint import slice_prefix_cache
    k = torch.arange(24.).view(4, 1, 3, 2); v = -k
    rows = torch.tensor([1, 3])
    out = slice_prefix_cache(((k, v), (k * 2, v * 2)), rows)
    assert isinstance(out, tuple) and len(out) == 2 and torch.equal(out[0][0], k[rows]) and torch.equal(out[1][1], v[rows] * 2)
    class Fake:
        def __init__(self, layers): self.layers = layers
        def to_legacy_cache(self): return tuple(self.layers)
        @classmethod
        def from_legacy_cache(cls, layers): return cls(list(layers))
    sliced = slice_prefix_cache(Fake([(k, v)]), rows)
    assert isinstance(sliced, Fake) and torch.equal(sliced.layers[0][0], k[rows])
    assert slice_prefix_cache(object(), rows) is None


def test_focused_comparison_equals_full_comparison_in_the_masked_mean():
    torch.manual_seed(0)
    errors = torch.rand(6); plain_full = torch.rand(6); focus = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.bool); fraction = 0.05
    reference = (F.relu(errors - (1 - fraction) * plain_full) * focus).mean()
    rows = focus.nonzero(as_tuple=True)[0]; plain_sub = torch.zeros_like(errors).index_put((rows,), plain_full[rows])
    assert torch.equal((F.relu(errors - (1 - fraction) * plain_sub) * focus).mean(), reference)
    empty = torch.zeros(6, dtype=torch.bool)
    assert (F.relu(errors - (1 - fraction) * torch.zeros_like(errors)) * empty).mean() == 0


class _CountingEncoder:
    def __init__(self): self.calls = []
    def __call__(self, observations):
        self.calls.append(len(observations))
        return np.stack([np.full(3456, float(obs['image']['a'][0, 0, 0, 0]), np.float32) for obs in observations])


def _obs(value, state=0.0):
    image = {k: torch.full((1, 224, 224, 3), value, dtype=torch.uint8) for k in ('a', 'b', 'c')}
    return dict(image=image, image_mask={k: torch.ones(1, dtype=torch.bool) for k in image}, state=torch.full((1, 32), state))


def test_visual_cache_reuses_identical_pixels_and_ignores_state(monkeypatch):
    import tgcc.unrolled_distribution_data as udd
    monkeypatch.setattr(udd, 'CAMERA_KEYS', ('a', 'b', 'c'), raising=False)
    import tgcc.retrieval_data as rd; monkeypatch.setattr(rd, 'CAMERA_KEYS', ('a', 'b', 'c'))
    prep = udd.UnrolledBatchPreparer.__new__(udd.UnrolledBatchPreparer)
    prep.encoder = _CountingEncoder(); prep.visual_cache = {}; prep.visual_cache_stats = dict(hits=0, misses=0, calls=0)
    first = prep.encode_visual([_obs(3), _obs(5), _obs(3, state=0.7)])
    assert prep.encoder.calls == [2] and first.shape == (3, 3456) and np.array_equal(first[0], first[2])
    second = prep.encode_visual([_obs(5, state=-1.0), _obs(9)])
    assert prep.encoder.calls == [2, 1] and np.array_equal(second[0], first[1]) and second[1][0] == 9
    assert prep.visual_cache_stats == dict(hits=2, misses=3, calls=2) and len(prep.visual_cache) == 3
    prep.visual_cache = None
    assert prep.encode_visual([_obs(11)])[0][0] == 11 and prep.encoder.calls == [2, 1, 1]


def test_policy_call_scope_checkpoints_only_the_prefix(monkeypatch):
    from tgcc.full_distribution_joint import FullDistributionSystem
    import torch.utils.checkpoint as ckpt
    used = []
    monkeypatch.setattr(ckpt, 'checkpoint', lambda f, *a, **k: (used.append(f.__name__), f(*a))[1])
    def prefix(x): return x + 1
    def velocity(x): return x * 2
    host = types.SimpleNamespace(training=True)
    call = FullDistributionSystem.policy_call
    x = torch.ones(1, requires_grad=True)
    host.recipe = dict(activation_checkpointing=True, activation_checkpointing_scope='prefix')
    assert call(host, prefix, x).item() == 2 and call(host, velocity, x).item() == 2 and used == ['prefix']
    host.recipe = dict(activation_checkpointing=True)
    call(host, velocity, x); assert used == ['prefix', 'velocity']
    host.recipe = dict(activation_checkpointing=True, activation_checkpointing_scope='bogus')
    with pytest.raises(ValueError):
        call(host, prefix, x)
