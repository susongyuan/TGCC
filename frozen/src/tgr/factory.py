"""Official checkpoint-owned pi input/output transforms and a V2 composition."""

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
from .model import TriGraphPolicy


def ensure_legacy_imports():
    import sys

    try:
        from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
    except ModuleNotFoundError as error:
        if (
            not (error.name or "").startswith("omnigibson")
            or "omnigibson" in sys.modules
        ):
            raise
        from .serve import install_compat

        install_compat()


def create(
    checkpoint, mode="baseline", device="cuda:0", delta=None, features=32, hidden=256
):
    ensure_legacy_imports()
    from openpi.training import config
    from openpi.policies.policy_config import create_trained_policy

    policy = create_trained_policy(
        config.get_config("pi05_b1k-base"), checkpoint, pytorch_device=device
    )
    if mode in ("baseline", "graph_shadow"):
        if delta:
            raise ValueError("pristine baseline never loads fine-tuned graph weights")
        return policy, policy._model
    if mode not in ("tc_esgr", "tri_graph"):
        raise ValueError("only baseline/shadow/TC-ESGR V2 are active")
    model = TriGraphPolicy(policy._model, features, hidden, "tc_esgr").to(device)
    if delta:
        from .checkpoint import load_full

        load_full(model, delta)
    model.eval()
    policy._model = model
    policy._sample_actions = model.sample_actions
    return policy, model
