"""v2.1 recovery signal: effective-input perturbation, correction target, history targets, routing, acceptance."""
import numpy as np
import pytest
import torch
from tgcc.unrolled_distribution_data import perturbation_plan, correction_target, official_state_tokenizer, TASK_PROMPT, STATE_SLICE

SPEC = dict(block_fraction=0.4, state_noise_std=0.1)


def test_plan_is_deterministic_bounded_and_zero_when_unperturbed():
    a = perturbation_plan(42, 1234, 8, SPEC); b = perturbation_plan(42, 1234, 8, SPEC)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    flags, noise = a
    assert noise.shape == (8, 23) and flags.dtype == bool and not noise[~flags].any()
    other = perturbation_plan(42, 1235, 8, SPEC)
    assert not (np.array_equal(other[0], flags) and np.array_equal(other[1], noise))
    none = perturbation_plan(42, 1234, 8, None)
    assert not none[0].any() and not none[1].any()
    flags = np.concatenate([perturbation_plan(42, i, 8, SPEC)[0] for i in range(2000)])
    assert 0.37 < flags.mean() < 0.43
    with pytest.raises(ValueError):
        perturbation_plan(42, 1, 8, dict(block_fraction=1.5, state_noise_std=0.1))


def test_correction_target_is_actual_delta_after_clipping():
    clean = np.array([0.95, -0.95, 0.0], np.float64); noise = np.array([0.2, -0.2, 0.1], np.float32)
    perturbed, delta = correction_target(clean, noise)
    assert perturbed.dtype == np.float64 and delta.dtype == np.float32
    assert np.allclose(perturbed, [1.0, -1.0, 0.1]) and np.allclose(delta, [-0.05, 0.05, -0.1])
    assert np.abs(delta).max() < np.abs(noise).max()


def _fake_item(state23, tokens, mask):
    state32 = np.zeros(32, np.float64); state32[:23] = state23
    return dict(observation=dict(state=torch.as_tensor(state32[None]), tokenized_prompt=torch.as_tensor(np.asarray(tokens)[None]), tokenized_prompt_mask=torch.as_tensor(np.asarray(mask)[None])), labels={})


def test_perturb_item_skips_instead_of_raising_when_the_stored_tokens_cannot_be_reproduced():
    from tgcc.unrolled_distribution_data import UnrolledBatchPreparer
    prep = UnrolledBatchPreparer.__new__(UnrolledBatchPreparer); tok = official_state_tokenizer()
    rng = np.random.default_rng(11); state = rng.uniform(-0.9, 0.9, 23)
    good_tokens, good_mask = tok.tokenize(TASK_PROMPT, state)
    wrong_tokens, wrong_mask = tok.tokenize(TASK_PROMPT, np.clip(state + 0.3, -1, 1))
    noise = np.full(23, 0.25)
    item = _fake_item(state, wrong_tokens, wrong_mask)
    out, delta, effective, p23, skipped = prep.perturb_item(item, noise)
    assert skipped is True and effective is False and out is item and not delta.any() and np.array_equal(p23, state)
    item = _fake_item(state, good_tokens, good_mask)
    out, delta, effective, p23, skipped = prep.perturb_item(item, noise)
    assert skipped is False and effective is True and out is not item and delta.any()
    assert np.array_equal(out['observation']['tokenized_prompt'][0].numpy(), tok.tokenize(TASK_PROMPT, p23)[0])
    assert np.array_equal(item['observation']['tokenized_prompt'][0].numpy(), np.asarray(good_tokens))  # cached item untouched


def test_preparer_retokenize_preserves_the_stored_dtype():
    import types
    from tgcc.unrolled_distribution_data import UnrolledBatchPreparer
    tok = official_state_tokenizer(); holder = types.SimpleNamespace(_tokenizer=tok)
    edge = np.linspace(-1, 1, 257)[:-1]; state = np.full(23, 0.1, np.float64); state[:5] = edge[[40, 80, 120, 160, 200]] - 1e-9
    t64, _ = tok.tokenize(TASK_PROMPT, state); t32, _ = tok.tokenize(TASK_PROMPT, state.astype(np.float32))
    assert not np.array_equal(t64, t32)
    via_preparer, _ = UnrolledBatchPreparer.retokenize(holder, state)
    assert np.array_equal(via_preparer, t64)  # the preparer must reproduce the float64 tokenization exactly


def test_retokenization_must_use_the_stored_float64_state_not_a_float32_copy():
    tok = official_state_tokenizer(); rng = np.random.default_rng(3)
    flips = 0
    for _ in range(400):
        state = rng.uniform(-1, 1, 23)
        edge = np.linspace(-1, 1, 257)[:-1]
        state[:5] = edge[rng.integers(1, 255, 5)] - 1e-9  # values just below bin edges
        t64, _ = tok.tokenize(TASK_PROMPT, state); t32, _ = tok.tokenize(TASK_PROMPT, state.astype(np.float32))
        flips += int(not np.array_equal(t64, t32))
    assert flips > 0  # float32 rounding changes bins near edges: re-tokenization must keep the stored dtype


def test_official_tokenizer_changes_tokens_for_effective_noise_and_not_for_absorbed_noise():
    tok = official_state_tokenizer()
    state = np.linspace(-0.8, 0.8, 23).astype(np.float32)  # the official pipeline tokenizes the 23-dim normalized state
    clean_tokens, clean_mask = tok.tokenize(TASK_PROMPT, state)
    again_tokens, again_mask = tok.tokenize(TASK_PROMPT, state)
    assert np.array_equal(clean_tokens, again_tokens) and np.array_equal(clean_mask, again_mask)
    padded_tokens, _ = tok.tokenize(TASK_PROMPT, np.concatenate([state, np.zeros(9, np.float32)]))
    assert not np.array_equal(padded_tokens, clean_tokens)  # padding to 32 before tokenizing would not match the stored tokens
    perturbed = np.clip(state + 0.1, -1, 1)
    tokens, _ = tok.tokenize(TASK_PROMPT, perturbed)
    assert not np.array_equal(tokens, clean_tokens)
    tokens_tiny, _ = tok.tokenize(TASK_PROMPT, state + 1e-5)
    assert np.array_equal(tokens_tiny, clean_tokens)  # absorbed by 256-bin discretization: must be flagged ineffective


def test_query_state_slice_is_recomputed_through_codec_standardization():
    mean = np.random.default_rng(0).standard_normal(3097).astype(np.float32); std = np.full(3097, 2.0, np.float32)
    clean23 = np.linspace(-0.5, 0.5, 23).astype(np.float32); raw = np.zeros(3097, np.float32); raw[STATE_SLICE] = clean23
    standardized = (raw - mean) / std
    perturbed23 = np.clip(clean23 + 0.1, -1, 1)
    query = standardized.copy(); query[STATE_SLICE] = (perturbed23 - mean[STATE_SLICE]) / std[STATE_SLICE]
    raw2 = raw.copy(); raw2[STATE_SLICE] = perturbed23
    assert np.allclose(query, (raw2 - mean) / std)
    assert np.array_equal(query[:256], standardized[:256]) and np.array_equal(query[279:], standardized[279:])


def test_history_targets_use_only_the_past_and_are_valid_after_a_known_label():
    from tgcc.full_distribution_joint import history_targets, history_valid_count
    def item(label, known):
        return dict(labels=dict(next=torch.tensor([label]), next_mask=torch.tensor([known])))
    burnin = [item(2, True)]; window = [item(5, True), item(5, False), item(7, True)]
    targets, valid = history_targets(burnin, window, 10, 'cpu')
    assert targets.shape == (3, 1, 10) and valid.tolist() == [[True], [True], [True]]
    assert targets[0, 0].nonzero().flatten().tolist() == [2]
    assert targets[1, 0].nonzero().flatten().tolist() == [2, 5]
    assert targets[2, 0].nonzero().flatten().tolist() == [2, 5]  # the current block's own label never enters
    targets, valid = history_targets(None, [item(1, False), item(3, True), item(4, True)], 10, 'cpu')
    assert valid.tolist() == [[False], [False], [True]] and targets[2, 0].nonzero().flatten().tolist() == [3]
    sample = dict(policy=dict(sequence=[item(1, False), item(3, True), item(4, True)]), policy_burnin=None)
    assert history_valid_count(sample) == 1


def test_reference_acceptance_rejects_worse_assisted_error():
    from tgcc.full_distribution_joint import reference_benefit_ok
    assert reference_benefit_ok(plain=1.0, reference=0.8, replacement=0.95)
    assert not reference_benefit_ok(plain=1.0, reference=2.0, replacement=3.0)
    assert not reference_benefit_ok(plain=1.0, reference=0.9, replacement=0.8)


def test_memory_and_correction_routing_with_real_parameter_names_and_rates():
    from tgcc.full_distribution_joint import full_rates, memory_parameter, memory_width
    from tgcc.joint import policy_parameter_group
    assert memory_parameter('decoder.memory.weight_ih') and memory_parameter('student.decoder.memory_out.weight')
    assert not memory_parameter('decoder.next.weight') and policy_parameter_group('decoder.memory.weight_ih') == 'graphs'
    class D: memory = torch.nn.GRUCell(16, 16)
    class S: decoder = D()
    assert memory_width(S()) == 16
    recipe = dict(warmup=10, min_lr_ratio=0.1, new_updates=100, critic_lr=1e-4, retrieval_adapter_lr=1e-4, retrieval_gate_lr=3e-4,
                  vision_lr=1e-6, vlm_lr=1e-6, expert_lr=2e-6, action_io_lr=2e-6, graphs_lr=1e-5, memory_lr=1e-4, correction_lr=1e-4,
                  memory_unroll_blocks=8, critic_warmup_updates=20, joint_updates=80)
    rates = full_rates(50, recipe)
    assert len(rates) == 10 and rates[-1] == pytest.approx(rates[1]) and rates[-2] == pytest.approx(rates[1])


def test_masked_sums_normalize_to_means_over_valid_entries():
    values = torch.tensor([1., 2., 3., 4.]); mask = torch.tensor([1., 0., 1., 0.])
    local_sum = (values * mask).sum(); global_count = 2
    assert float(local_sum / global_count) == pytest.approx(values[mask.bool()].mean().item())
