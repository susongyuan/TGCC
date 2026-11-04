from dataclasses import dataclass
import gzip
import json
import pickle

import numpy as np
import pytest

from tgcc.collection import GraphCollectionRecorder


@dataclass
class Conditions:
    global_graph: dict
    task_graph: dict
    local_graph: dict
    memory: object = None


class Runtime:
    def reset(self, session):
        self.task_version = 0

    def begin_chunk(self, raw):
        return Conditions(
            dict(time=raw["sim_time"]), dict(version=self.task_version), {}
        )

    def after_prefix(self, decision):
        self.task_version += 1

    def export_state(self):
        return dict(task_version=self.task_version)


def raw(time):
    return dict(
        session="train-ep",
        sim_time=time,
        graph_snapshot=dict(episode_id="train-ep", timestamp=time, oracle=True),
    )


def recorder(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    return GraphCollectionRecorder(
        tmp_path / "records", plan, "train-ep", oracle=True, runtime=Runtime()
    )


def test_partial_execution_and_causal_task_state_are_preserved(tmp_path):
    r = recorder(tmp_path)
    r.capture(raw(0))
    r.capture(raw(16 / 30), executed_native=np.zeros((16, 23)))
    transition = json.loads((r.output / "transitions.jsonl").read_text())
    assert transition["executed_steps"] == 16
    assert sum(transition["action_mask"]) == 16
    with gzip.open(r.output / "exposure_00000.pkl.gz", "rb") as f:
        before = pickle.load(f)
    with gzip.open(r.output / "exposure_00001.pkl.gz", "rb") as f:
        after = pickle.load(f)
    assert before["conditions"]["task_graph"]["version"] == 0
    assert after["conditions"]["task_graph"]["version"] == 1
    assert before["neural_decision_present"] is False
    assert transition["recovery_success_label"] is None
    assert transition["training_adapter_ready"] is False


@pytest.mark.parametrize("time,count", [(1, 16), (32 / 30, 33)])
def test_unexecuted_or_misaligned_prefix_never_becomes_a_transition(
    tmp_path, time, count
):
    r = recorder(tmp_path)
    r.capture(raw(0))
    with pytest.raises(ValueError):
        r.capture(raw(time), executed_native=np.zeros((count, 23)))
    assert not (r.output / "transitions.jsonl").exists()
