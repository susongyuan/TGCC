"""Causal three-graph recording using the production graph runtime.

Controller collection has no neural decision or GRU state: those remain absent,
never copied from a future exposure or invented from demonstration labels.
"""

from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


class GraphCollectionRecorder:
    def __init__(self, output, plan_path, session, *, oracle, runtime=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if runtime is None:
            from tgr.runtime_bridge import EvalGraphRuntime

            runtime = EvalGraphRuntime(str(plan_path), device="cpu", oracle=oracle)
        self.runtime = runtime
        self.runtime.reset(session)
        self.session = session
        self.oracle = oracle
        self.previous = None
        self.index = 0
        self.plan_sha256 = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()

    def capture(self, raw, *, executed_native=None, execution_annotations=None, decision=None):
        """Record the current exposure and only the prefix actually executed.

        The caller must supply the exact production neural decision if one was
        run. Algorithmic controller samples use None, not ground-truth skills.
        """
        if raw["session"] != self.session:
            raise ValueError("collection session mismatch")
        now = float(raw["sim_time"])
        snapshot = raw["graph_snapshot"]
        if (
            snapshot["episode_id"] != self.session
            or float(snapshot["timestamp"]) != now
        ):
            raise ValueError("graph exposure identity/time mismatch")
        if bool(snapshot.get("oracle")) != self.oracle:
            raise ValueError("collection graph provenance mismatch")
        actions = None
        if self.previous is not None:
            before_time = self.previous["timestamp"]
            actions = np.asarray(executed_native, dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 23
                or not 1 <= len(actions) <= 32
                or not np.isfinite(actions).all()
            ):
                raise ValueError("exact executed native prefix of 1..32 steps required")
            if abs(now - before_time - len(actions) / 30) > 2e-4:
                raise ValueError(
                    "executed prefix and graph exposures are not time aligned"
                )
        elif executed_native is not None:
            raise ValueError("initial exposure cannot claim a prior execution")
        if execution_annotations is not None:
            if actions is None or len(execution_annotations) != len(actions):
                raise ValueError("one provenance annotation per actual executed step required")
            for step in execution_annotations:
                if step.get("source_step") is not None and step["source_step"] < 0:
                    raise ValueError("negative demonstration source step")
        conditions = self.runtime.begin_chunk(raw)
        # Snapshot BEFORE current decision updates task state: same causal order
        # as serving. No after-state evidence is written into previous conditions.
        record = dict(
            schema="tgcc.synchronized_exposure.v1",
            session=self.session,
            timestamp=now,
            raw=deepcopy(raw),
            conditions=asdict(conditions),
            plan_sha256=self.plan_sha256,
            oracle_upper_bound=self.oracle,
            neural_decision_present=decision is not None,
            policy_semantic_memory_present=conditions.memory is not None,
        )
        self.runtime.after_prefix(decision)
        record["runtime_after_commit"] = self.runtime.export_state()
        path = self.output / f"exposure_{self.index:05d}.pkl.gz"
        with gzip.open(path, "xb") as stream:
            pickle.dump(record, stream, protocol=5)
        receipt = dict(
            index=self.index,
            timestamp=now,
            file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if self.previous is not None:
            transition = dict(
                schema="tgcc.actual_prefix_transition.v1",
                before=self.previous,
                after=receipt,
                executed_native=actions.tolist(),
                executed_steps=len(actions),
                action_mask=[True] * len(actions) + [False] * (32 - len(actions)),
                simulated_outcome=False,
                outcome_source="actual_simulator_execution",
                task_success_label=None,
                recovery_success_label=None,
                training_adapter_ready=False,
                execution_annotations=deepcopy(execution_annotations),
            )
            with (self.output / "transitions.jsonl").open("a") as stream:
                stream.write(json.dumps(transition) + "\n")
        self.previous = receipt
        self.index += 1
        return receipt
