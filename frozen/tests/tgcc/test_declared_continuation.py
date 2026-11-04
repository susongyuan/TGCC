"""Resume accepts only the declared kernel-only differences and records them."""
import copy
import pytest
from tgcc.distribution_checkpoint import declared_continuation, CONTINUATION_SOURCE_WHITELIST


def _pair():
    md = dict(new_updates=1100, source_sha256={"a.py": "1", CONTINUATION_SOURCE_WHITELIST[0]: "old"}, config=dict(seed=42))
    expected = copy.deepcopy(md); expected.pop("new_updates")
    return md, expected


def test_identical_identity_declares_nothing():
    md, expected = _pair()
    assert declared_continuation(md, expected, "ckpt") == set() and "continuation_declared_changes" not in expected


def test_whitelisted_kernel_change_is_declared_and_recorded():
    md, expected = _pair()
    expected["source_sha256"][CONTINUATION_SOURCE_WHITELIST[0]] = "new"
    expected["config"]["attention_implementation"] = "sdpa"
    assert declared_continuation(md, expected, "ckpt") == {"source_sha256", "config"}
    record = expected["continuation_declared_changes"][0]
    assert record["resumed_step"] == 1100 and record["resumed_from"] == "ckpt"
    assert record["files"] == {CONTINUATION_SOURCE_WHITELIST[0]: dict(checkpoint="old", running="new")}
    assert record["config"] == {"attention_implementation": dict(checkpoint=None, running="sdpa")}


def test_undeclared_source_or_config_change_is_refused():
    md, expected = _pair(); expected["source_sha256"]["a.py"] = "2"
    with pytest.raises(ValueError, match="Undeclared source change"):
        declared_continuation(md, expected, "ckpt")
    md, expected = _pair(); expected["config"]["seed"] = 7
    with pytest.raises(ValueError, match="Undeclared config change"):
        declared_continuation(md, expected, "ckpt")
