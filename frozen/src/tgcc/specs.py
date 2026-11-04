from dataclasses import dataclass
from typing import Mapping
import torch


@dataclass(frozen=True)
class RobotActionSpec:
    names: tuple
    groups: Mapping[str, tuple]
    units: tuple
    lower: tuple
    upper: tuple
    bounds_valid: tuple
    horizon: int = 32
    normalization: str = "checkpoint-owned quantile; native control ordering unchanged"

    def __post_init__(self):
        d = len(self.names)
        assert all(
            len(x) == d for x in (self.units, self.lower, self.upper, self.bounds_valid)
        )
        flattened = [i for g in self.groups.values() for i in g]
        if sorted(flattened) != list(range(d)):
            raise ValueError(
                "Groups must partition every control dimension, including trunk"
            )
        if any(
            v and not lo < hi
            for lo, hi, v in zip(self.lower, self.upper, self.bounds_valid)
        ):
            raise ValueError("Invalid bounds")

    @property
    def width(self):
        return len(self.names)

    def masks(self, device=None):
        out = torch.zeros(len(self.groups), self.width, dtype=torch.bool, device=device)
        for i, g in enumerate(self.groups.values()):
            out[i, list(g)] = True
        return out


@dataclass(frozen=True)
class GraphFeatureSpec:
    global_tokens: int = 8
    task_tokens: int = 4
    local_tokens: int = 8
    global_width: int = 2048
    task_width: int = 2048
    local_width: int = 1024
    semantic_width: int = 256
    feature_width: int = 32
    update_hz: float = 30 / 32
    history: int = 16
    max_age_seconds: float = 2.2
    global_frame: str = "world"
    local_frames: tuple = ("left_ee", "right_ee", "base", "target")
    update_event: str = "new observation at action-chunk boundary"


@dataclass
class CriticObservation:
    graph_tokens: tuple
    graph_masks: tuple
    graph_timestamps: tuple
    graph_confidences: tuple
    timestamp: torch.Tensor
    semantic_context: torch.Tensor
    contract_ids: torch.Tensor
    contract_entities: torch.Tensor
    contract_numeric: torch.Tensor
    contract_category: torch.Tensor
    contract_mask: torch.Tensor
    contract_values: torch.Tensor
    contract_valid: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    history_timestamps: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_action_mask: torch.Tensor
    robot_current: torch.Tensor | None = None
    task_ids: torch.Tensor | None = None
    skill_ids: torch.Tensor | None = None
    preserve_gripper_mask: torch.Tensor | None = None

    def validate(self):
        b = self.timestamp.shape[0]
        if len(self.graph_tokens) != 3:
            raise ValueError("Three graph types required")
        for t, m, ts, c in zip(
            self.graph_tokens,
            self.graph_masks,
            self.graph_timestamps,
            self.graph_confidences,
        ):
            if t.shape[:2] != m.shape or ts.shape != m.shape or c.shape != m.shape:
                raise ValueError("Graph provenance shape")
            if ((ts > self.timestamp[:, None]) & m).any():
                raise ValueError("Future graph exposure")
            if ((c < 0) | (c > 1)).any():
                raise ValueError("Invalid confidence")
            if (
                not torch.isfinite(t[m]).all()
                or not torch.isfinite(ts[m]).all()
                or not torch.isfinite(c).all()
            ):
                raise ValueError("Non-finite graph provenance/features")
        if (
            (self.history_timestamps >= self.timestamp[:, None]) & self.history_mask
        ).any():
            raise ValueError("History must precede current exposure")
        if self.candidate_actions.shape[:2] != self.candidate_action_mask.shape:
            raise ValueError("Action prefix mask")
        if self.contract_ids.shape[1] > 16:
            raise ValueError(
                "Contract must be explicitly partitioned, not silently truncated"
            )
        if not self.candidate_action_mask.any(-1).all():
            raise ValueError("No valid action prefix")
        if not torch.isfinite(self.candidate_actions[self.candidate_action_mask]).all():
            raise ValueError("Non-finite action")
        if not torch.isfinite(self.history[self.history_mask]).all():
            raise ValueError("Non-finite executed history")
        if not torch.isfinite(self.semantic_context).all():
            raise ValueError("Non-finite semantic context")
        if self.robot_current is not None and (
            self.robot_current.shape != (b, 64)
            or not torch.isfinite(self.robot_current).all()
        ):
            raise ValueError("Current robot feedback shape/finiteness")
        if not torch.isfinite(self.timestamp).all():
            raise ValueError("Non-finite exposure time")
        if (
            not self.contract_ids.shape
            == self.contract_mask.shape
            == self.contract_valid.shape
            == self.contract_values.shape
        ):
            raise ValueError("Contract clause masks")
        if (self.contract_valid & ~self.contract_mask).any():
            raise ValueError("Padding cannot be a known predicate")
        if not torch.isfinite(self.contract_numeric[self.contract_mask]).all():
            raise ValueError("Non-finite contract metadata")


@dataclass
class ExecutedTransition:
    before: CriticObservation
    after: CriticObservation
    executed_actions: torch.Tensor
    executed_mask: torch.Tensor
    start_time: torch.Tensor
    end_time: torch.Tensor

    def validate(self):
        self.before.validate()
        self.after.validate()
        if not (
            (self.start_time == self.before.timestamp)
            & (self.end_time == self.after.timestamp)
            & (self.end_time > self.start_time)
        ).all():
            raise ValueError("Execution observation alignment")
        if self.executed_actions.shape[:2] != self.executed_mask.shape:
            raise ValueError("Executed prefix mismatch")
        if not self.executed_mask.any(-1).all():
            raise ValueError("No executed actions")
        if not torch.isfinite(self.executed_actions[self.executed_mask]).all():
            raise ValueError("Non-finite executed command")
